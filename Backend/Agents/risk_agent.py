# risk_agent.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Literal, Tuple
import math
import time

Side = Literal["BUY", "SELL"]
Market = Literal["equity", "crypto"]

@dataclass
class RiskConfig:
    # Core limits
    max_risk_per_trade_pct: float = 1.0          # % of equity
    max_symbol_exposure_pct: float = 15.0        # % of equity
    max_market_exposure_pct: float = 60.0        # % of equity (e.g., total crypto exposure)
    max_open_positions: int = 8

    # Loss limits
    max_daily_loss_pct: float = 3.0              # if breached, block new trades

    # Trade quality gates
    min_rr: float = 1.5                          # minimum reward:risk
    min_sl_distance_pct: float = 0.3             # avoid too tight SL (noise)
    max_sl_distance_pct: float = 8.0             # avoid overly wide SL

    # Confidence scaling
    min_confidence: float = 0.45                 # below this, block
    confidence_position_scale: bool = True       # scale qty by confidence

    # Volatility scaling (pluggable, optional)
    vol_position_scale: bool = True
    max_volatility_pct: float = 6.0              # if vol > this, reduce or block (your choice)

    # Rounding / lot sizes
    equity_qty_step: int = 1                     # stocks: integer qty
    crypto_qty_step: float = 0.0001              # crypto: fractional step

    # Fees/slippage buffers (simple)
    slippage_bps: float = 5.0                    # 0.05%
    fee_bps: float = 2.0                         # 0.02%


@dataclass
class Position:
    symbol: str
    market: Market
    qty: float
    avg_price: float  # average entry
    side: Literal["LONG", "SHORT"] = "LONG"


@dataclass
class AccountState:
    equity_value: float                  # total account equity in INR (or base currency)
    cash_available: float                # available cash/margin
    open_positions: List[Position]
    day_pnl: float                       # realized + unrealized for day (can be approx)


@dataclass
class TradeRequest:
    symbol: str
    market: Market
    side: Side
    entry_price: float
    stop_loss: float
    take_profit: Optional[float] = None
    signal_confidence: float = 0.7
    timestamp: float = 0.0


@dataclass
class RiskDecision:
    allowed: bool
    qty: float
    risk_amount: float
    risk_pct: float
    reasons: List[str]
    order_plan: Dict
    limits_snapshot: Dict
    created_at: float


class RiskAgent:
    def __init__(self, config: RiskConfig):
        self.cfg = config

    # --------- Public API ---------
    def evaluate(self, req: TradeRequest, acct: AccountState, *,
                 volatility_pct: Optional[float] = None,
                 liquidity_score: Optional[float] = None) -> RiskDecision:
        """
        volatility_pct: optional (e.g., ATR% or 1h vol %)
        liquidity_score: optional (0-1), where 1 = excellent liquidity
        """
        reasons: List[str] = []

        # Basic sanity
        if req.entry_price <= 0 or req.stop_loss <= 0:
            return self._deny("Invalid price inputs", req, acct)

        if req.signal_confidence < self.cfg.min_confidence:
            return self._deny(f"Confidence too low ({req.signal_confidence:.2f} < {self.cfg.min_confidence:.2f})", req, acct)

        if len(acct.open_positions) >= self.cfg.max_open_positions:
            return self._deny("Max open positions reached", req, acct)

        # Daily loss circuit breaker
        daily_loss_pct = 0.0
        if acct.equity_value > 0:
            daily_loss_pct = (-acct.day_pnl / acct.equity_value) * 100.0 if acct.day_pnl < 0 else 0.0
        if daily_loss_pct >= self.cfg.max_daily_loss_pct:
            return self._deny(f"Daily loss limit breached ({daily_loss_pct:.2f}% >= {self.cfg.max_daily_loss_pct:.2f}%)", req, acct)

        # Stop-loss distance checks
        sl_dist_pct = self._stop_distance_pct(req)
        if sl_dist_pct < self.cfg.min_sl_distance_pct:
            return self._deny(f"Stop-loss too tight ({sl_dist_pct:.2f}% < {self.cfg.min_sl_distance_pct:.2f}%)", req, acct)
        if sl_dist_pct > self.cfg.max_sl_distance_pct:
            return self._deny(f"Stop-loss too wide ({sl_dist_pct:.2f}% > {self.cfg.max_sl_distance_pct:.2f}%)", req, acct)

        # RR check (if TP provided)
        if req.take_profit is not None:
            rr = self._rr(req)
            if rr < self.cfg.min_rr:
                return self._deny(f"RR too low ({rr:.2f} < {self.cfg.min_rr:.2f})", req, acct)
        else:
            reasons.append("No take-profit provided (RR check skipped)")

        # Exposure checks (current + proposed)
        symbol_exposure_pct = self._symbol_exposure_pct(acct, req.symbol, req.market)
        market_exposure_pct = self._market_exposure_pct(acct, req.market)

        # We'll compute qty later, but we can pre-block if already over limits
        if symbol_exposure_pct >= self.cfg.max_symbol_exposure_pct:
            return self._deny(f"Symbol exposure already high ({symbol_exposure_pct:.2f}% >= {self.cfg.max_symbol_exposure_pct:.2f}%)", req, acct)
        if market_exposure_pct >= self.cfg.max_market_exposure_pct:
            return self._deny(f"Market exposure already high ({market_exposure_pct:.2f}% >= {self.cfg.max_market_exposure_pct:.2f}%)", req, acct)

        # Volatility / liquidity optional filters
        vol_scale = 1.0
        if volatility_pct is not None and self.cfg.vol_position_scale:
            if volatility_pct > self.cfg.max_volatility_pct:
                # reduce size, not necessarily deny
                vol_scale = max(0.25, self.cfg.max_volatility_pct / max(volatility_pct, 1e-9))
                reasons.append(f"High volatility {volatility_pct:.2f}% -> position scaled x{vol_scale:.2f}")

        liq_scale = 1.0
        if liquidity_score is not None:
            if liquidity_score < 0.35:
                return self._deny(f"Liquidity too poor ({liquidity_score:.2f})", req, acct)
            if liquidity_score < 0.6:
                liq_scale = 0.7
                reasons.append(f"Moderate liquidity ({liquidity_score:.2f}) -> position scaled x{liq_scale:.2f}")

        # Risk-based position sizing
        max_risk_amount = acct.equity_value * (self.cfg.max_risk_per_trade_pct / 100.0)

        # adjust by confidence (optional)
        conf_scale = 1.0
        if self.cfg.confidence_position_scale:
            # map confidence [min_conf,1] to [0.5,1.0]
            conf_scale = self._map_confidence(req.signal_confidence)
            reasons.append(f"Confidence scale x{conf_scale:.2f}")

        effective_risk_amount = max_risk_amount * conf_scale * vol_scale * liq_scale

        qty = self._calc_qty_from_risk(req, effective_risk_amount)

        # Cash constraint
        notional = qty * req.entry_price
        if notional > acct.cash_available * 1.01:  # small buffer
            # downsize to available cash
            qty = max(0.0, (acct.cash_available / req.entry_price))
            qty = self._round_qty(qty, req.market)
            reasons.append("Downsized due to cash/margin limits")

        # Re-check exposure with proposed notional
        proposed_symbol_exposure_pct = symbol_exposure_pct + (notional / acct.equity_value * 100.0 if acct.equity_value else 0.0)
        proposed_market_exposure_pct = market_exposure_pct + (notional / acct.equity_value * 100.0 if acct.equity_value else 0.0)

        if proposed_symbol_exposure_pct > self.cfg.max_symbol_exposure_pct:
            # downsize to fit
            allowed_notional = (self.cfg.max_symbol_exposure_pct - symbol_exposure_pct) / 100.0 * acct.equity_value
            qty = self._round_qty(max(0.0, allowed_notional / req.entry_price), req.market)
            reasons.append("Downsized to satisfy symbol exposure limit")

        if proposed_market_exposure_pct > self.cfg.max_market_exposure_pct:
            allowed_notional = (self.cfg.max_market_exposure_pct - market_exposure_pct) / 100.0 * acct.equity_value
            qty = self._round_qty(max(0.0, allowed_notional / req.entry_price), req.market)
            reasons.append("Downsized to satisfy market exposure limit")

        if qty <= 0:
            return self._deny("Position size rounds to zero under constraints", req, acct)

        # Final risk estimate (approx incl. slippage/fees buffer)
        risk_amount = self._estimate_risk_amount(req, qty)
        risk_pct = (risk_amount / acct.equity_value * 100.0) if acct.equity_value else 0.0

        if risk_pct > self.cfg.max_risk_per_trade_pct * 1.05:
            return self._deny(f"Post-sizing risk too high ({risk_pct:.2f}%)", req, acct)

        order_plan = self._build_order_plan(req, qty)

        return RiskDecision(
            allowed=True,
            qty=qty,
            risk_amount=risk_amount,
            risk_pct=risk_pct,
            reasons=reasons,
            order_plan=order_plan,
            limits_snapshot=asdict(self.cfg),
            created_at=time.time()
        )

    # --------- Helpers ---------
    def _deny(self, reason: str, req: TradeRequest, acct: AccountState) -> RiskDecision:
        return RiskDecision(
            allowed=False,
            qty=0.0,
            risk_amount=0.0,
            risk_pct=0.0,
            reasons=[reason],
            order_plan={},
            limits_snapshot=asdict(self.cfg),
            created_at=time.time()
        )

    def _stop_distance_pct(self, req: TradeRequest) -> float:
        if req.side == "BUY":
            dist = (req.entry_price - req.stop_loss) / req.entry_price * 100.0
        else:
            dist = (req.stop_loss - req.entry_price) / req.entry_price * 100.0
        return abs(dist)

    def _rr(self, req: TradeRequest) -> float:
        if req.take_profit is None:
            return 0.0
        risk = abs(req.entry_price - req.stop_loss)
        reward = abs(req.take_profit - req.entry_price)
        return (reward / risk) if risk > 0 else 0.0

    def _map_confidence(self, conf: float) -> float:
        # maps [min_conf, 1] -> [0.5, 1.0]
        lo = self.cfg.min_confidence
        conf = max(lo, min(1.0, conf))
        return 0.5 + 0.5 * ((conf - lo) / max(1e-9, (1.0 - lo)))

    def _calc_qty_from_risk(self, req: TradeRequest, risk_amount: float) -> float:
        stop_dist = abs(req.entry_price - req.stop_loss)
        if stop_dist <= 0:
            return 0.0
        raw_qty = risk_amount / stop_dist
        return self._round_qty(raw_qty, req.market)

    def _round_qty(self, qty: float, market: Market) -> float:
        if market == "equity":
            return float(max(0, int(qty // self.cfg.equity_qty_step) * self.cfg.equity_qty_step))
        else:
            step = self.cfg.crypto_qty_step
            return math.floor(qty / step) * step

    def _estimate_risk_amount(self, req: TradeRequest, qty: float) -> float:
        # basic buffer for slippage + fees (bps)
        price = req.entry_price
        slip = price * (self.cfg.slippage_bps / 10000.0)
        fee = price * (self.cfg.fee_bps / 10000.0)
        buffered_entry = price + slip + fee if req.side == "BUY" else price - slip - fee

        stop_dist = abs(buffered_entry - req.stop_loss)
        return stop_dist * qty

    def _symbol_exposure_pct(self, acct: AccountState, symbol: str, market: Market) -> float:
        exposure = 0.0
        for p in acct.open_positions:
            if p.symbol == symbol and p.market == market:
                exposure += abs(p.qty * p.avg_price)
        return (exposure / acct.equity_value * 100.0) if acct.equity_value else 0.0

    def _market_exposure_pct(self, acct: AccountState, market: Market) -> float:
        exposure = 0.0
        for p in acct.open_positions:
            if p.market == market:
                exposure += abs(p.qty * p.avg_price)
        return (exposure / acct.equity_value * 100.0) if acct.equity_value else 0.0

    def _build_order_plan(self, req: TradeRequest, qty: float) -> Dict:
        # Keep it simple: entry + protective SL (+ optional TP)
        return {
            "symbol": req.symbol,
            "market": req.market,
            "side": req.side,
            "qty": qty,
            "entry": {"type": "MARKET", "price": None},
            "stop_loss": {"type": "SL-MARKET", "price": req.stop_loss},
            "take_profit": {"type": "LIMIT", "price": req.take_profit} if req.take_profit else None
        }
