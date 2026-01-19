#!/usr/bin/env python3
"""
Technical Agent for Multi-Asset Trading
Provides technical analysis signals with on-chain intelligence for crypto
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

try:
    # Try relative imports (for when imported as part of package)
    from Backend.services.crypto_nansen import CryptoDataProvider
    from indicator_engine import IndicatorEngine
    from scoring_engine import ScoringEngine
    from services.common_models import (
        TechnicalSignal, TechnicalIndicators, OnchainSignals,
        StrengthScores, RiskManagement, MarketCondition, Candle
    )
except ImportError:
    # Fallback to absolute imports (for direct script execution)
    import sys
    import os
    # Add parent directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from services.crypto_nansen import CryptoDataProvider
    from indicator_engine import IndicatorEngine
    from scoring_engine import ScoringEngine
    from services.common_models import (
        TechnicalSignal, TechnicalIndicators, OnchainSignals,
        StrengthScores, RiskManagement, MarketCondition, Candle
    )


class TechnicalAgent:
    """
    Technical Agent providing comprehensive technical analysis
    Supports both crypto (with Nansen) and traditional assets
    """

    def __init__(self):
        # Initialize components
        self.crypto_provider = CryptoDataProvider(None)
        self.indicator_engine = IndicatorEngine()
        self.scoring_engine = ScoringEngine()

        # Setup logging
        self.logger = logging.getLogger('TechnicalAgent')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("✅ Technical Agent initialized")

    def get_signal(self, symbol: str, asset_type: str = 'crypto',
                   timeframe: str = '1d', lookback: int = 365) -> TechnicalSignal:
        """
        Get comprehensive technical analysis signal

        Args:
            symbol: Asset symbol (e.g., 'BTC', 'ETH', 'RELIANCE')
            asset_type: 'crypto' or 'equity'
            timeframe: Timeframe for analysis ('1d', '1h', etc.)
            lookback: Days of historical data to analyze

        Returns:
            TechnicalSignal with complete analysis
        """
        try:
            self.logger.info(f"🔍 Analyzing {symbol} ({asset_type}) with {lookback} days of data")

            # Get price data
            candles = self._get_price_data(symbol, asset_type, timeframe, lookback)
            if not candles:
                return self._create_error_signal(symbol, asset_type, "No price data available")

            # Convert to DataFrame and compute indicators
            df = self.indicator_engine.candles_to_df(candles)
            df_with_indicators = self.indicator_engine.compute_indicators(df)

            if df_with_indicators.empty:
                return self._create_error_signal(symbol, asset_type, "Failed to compute indicators")

            # Extract latest indicators
            indicators = self.indicator_engine.get_latest_indicator_snapshot(df_with_indicators)
            current_price = float(df_with_indicators['close'].iloc[-1])

            # Get on-chain data (crypto only)
            onchain_signals = None
            if asset_type == 'crypto' and self.crypto_provider:
                try:
                    onchain_signals = self.crypto_provider.get_onchain_signals(symbol)
                    self.logger.info(f"✅ Retrieved on-chain data for {symbol}")
                except Exception as e:
                    self.logger.warning(f"⚠️  Failed to get on-chain data for {symbol}: {e}")
                    onchain_signals = OnchainSignals()

            # Compute strength scores
            strength_scores = self.scoring_engine.compute_strength_scores(indicators, onchain_signals)

            # Generate signal and bias
            bias, signal, confidence = self._decide_action(strength_scores.final_strength, asset_type)

            # Risk management
            risk_management = self._calculate_risk_management(
                current_price, indicators.atr, bias, df_with_indicators
            )

            # Market condition assessment
            market_condition = self._assess_market_condition(df_with_indicators)

            # Create comprehensive signal
            technical_signal = TechnicalSignal(
                symbol=symbol,
                asset_type=asset_type,
                timestamp=datetime.now(),
                current_price=current_price,
                bias=bias,
                signal=signal,
                confidence=confidence,
                strength_scores=strength_scores,
                technical_indicators=indicators,
                onchain_signals=onchain_signals,
                risk_management=risk_management,
                data_points=len(df_with_indicators),
                indicators_calculated=True,
                onchain_data_available=onchain_signals is not None
            )

            self.logger.info(f"✅ Generated {signal} signal for {symbol} with {confidence:.2f} confidence")
            return technical_signal

        except Exception as e:
            self.logger.error(f"❌ Error generating signal for {symbol}: {e}")
            return self._create_error_signal(symbol, asset_type, str(e))

    def _get_price_data(self, symbol: str, asset_type: str, timeframe: str, lookback: int) -> List:
        """Get price data based on asset type"""
        try:
            if asset_type == 'crypto':
                if self.crypto_provider:
                    return self.crypto_provider.get_candles(symbol, timeframe, lookback)
                else:
                    self.logger.error("❌ Crypto provider not available")
                    return []
            else:
                # For equity, we'd integrate with Zerodha/Angel
                self.logger.error("❌ Equity data provider not implemented")
                return []

        except Exception as e:
            self.logger.error(f"❌ Error getting price data for {symbol}: {e}")
            return []

    def _decide_action(self, strength: float, asset_type: str) -> tuple:
        """
        Convert strength score to bias, signal, and confidence
        """
        if strength >= 75:
            bias = "bullish"
            signal = "LONG" if asset_type == "crypto" else "BUY"
            confidence = min((strength - 75) / 25 + 0.8, 1.0)
        elif strength >= 60:
            bias = "bullish"
            signal = "HOLD"
            confidence = 0.6
        elif strength >= 40:
            bias = "neutral"
            signal = "HOLD"
            confidence = 0.5
        elif strength >= 25:
            bias = "bearish"
            signal = "HOLD"
            confidence = 0.6
        else:
            bias = "bearish"
            signal = "SHORT" if asset_type == "crypto" else "SELL"
            confidence = min((25 - strength) / 25 + 0.8, 1.0)

        return bias, signal, confidence

    def _calculate_risk_management(self, current_price: float, atr: float,
                                 bias: str, df: Any) -> RiskManagement:
        """Calculate stoploss and target levels using ATR"""
        try:
            if atr <= 0 or current_price <= 0:
                return RiskManagement()

            atr_multiplier = 1.5  # Standard ATR stop distance

            if bias == "bullish":
                stoploss = current_price - (atr * atr_multiplier)
                target1 = current_price + (atr * 2)  # 2:1 RR
                target2 = current_price + (atr * 3)  # 3:1 RR
            elif bias == "bearish":
                stoploss = current_price + (atr * atr_multiplier)
                target1 = current_price - (atr * 2)
                target2 = current_price - (atr * 3)
            else:
                # Neutral - use wider stops
                stoploss = None  # No stop for neutral positions
                target1 = current_price * 1.05  # 5% target
                target2 = current_price * 1.10  # 10% target

            # Calculate risk-reward ratio
            if stoploss and target1:
                if bias == "bullish":
                    rr_ratio = (target1 - current_price) / (current_price - stoploss)
                else:  # bearish
                    rr_ratio = (current_price - target1) / (stoploss - current_price)
            else:
                rr_ratio = None

            return RiskManagement(
                stoploss=stoploss,
                target1=target1,
                target2=target2,
                rr_ratio=rr_ratio
            )

        except Exception as e:
            self.logger.error(f"❌ Error calculating risk management: {e}")
            return RiskManagement()

    def _assess_market_condition(self, df: Any) -> MarketCondition:
        """Assess overall market condition"""
        try:
            # Calculate volatility (20-day rolling std of returns)
            returns_volatility = df['close'].pct_change().rolling(20).std().iloc[-1]

            # Trend assessment
            sma20 = df['close'].rolling(20).mean().iloc[-1]
            sma50 = df['close'].rolling(50).mean().iloc[-1]
            current_price = df['close'].iloc[-1]

            if current_price > sma20 > sma50:
                trend = "bullish"
            elif current_price < sma20 < sma50:
                trend = "bearish"
            else:
                trend = "sideways"

            # Volatility assessment
            avg_volatility = df['close'].pct_change().rolling(20).std().mean()
            if returns_volatility > avg_volatility * 1.5:
                volatility = "high"
            elif returns_volatility < avg_volatility * 0.7:
                volatility = "low"
            else:
                volatility = "medium"

            # Volume assessment (relative to average)
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            if volume_ratio > 1.5:
                volume = "high"
            elif volume_ratio < 0.7:
                volume = "low"
            else:
                volume = "normal"

            # Overall market condition
            if trend == "bullish" and volatility == "low" and volume == "high":
                overall = "bullish"
            elif trend == "bearish" and volatility == "high":
                overall = "bearish"
            else:
                overall = "neutral"

            return MarketCondition(
                volatility=volatility,
                trend=trend,
                volume=volume,
                overall=overall
            )

        except Exception as e:
            self.logger.error(f"❌ Error assessing market condition: {e}")
            return MarketCondition()

    def _create_error_signal(self, symbol: str, asset_type: str, error_msg: str) -> TechnicalSignal:
        """Create error signal when analysis fails"""
        return TechnicalSignal(
            symbol=symbol,
            asset_type=asset_type,
            timestamp=datetime.now(),
            current_price=0.0,
            bias="neutral",
            signal="HOLD",
            confidence=0.0,
            strength_scores=StrengthScores(),
            technical_indicators=TechnicalIndicators(),
            onchain_signals=None,
            risk_management=RiskManagement(),
            data_points=0,
            indicators_calculated=False,
            onchain_data_available=False
        )

    def get_market_condition(self, symbol: str, asset_type: str = 'crypto') -> MarketCondition:
        """Get current market condition assessment"""
        try:
            candles = self._get_price_data(symbol, asset_type, '1d', 100)
            if not candles:
                return MarketCondition()

            df = self.indicator_engine.candles_to_df(candles)
            df_with_indicators = self.indicator_engine.compute_indicators(df)

            return self._assess_market_condition(df_with_indicators)

        except Exception as e:
            self.logger.error(f"❌ Error getting market condition for {symbol}: {e}")
            return MarketCondition()


# Example usage and testing
if __name__ == "__main__":
    # Initialize agent (without Nansen for testing)
    agent = TechnicalAgent()

    # Test with BTC
    signal = agent.get_signal('BTC', asset_type='crypto', lookback=100)

    print(f"\n{'='*50}")
    print(f"Symbol: {signal.symbol}")
    print(f"Signal: {signal.signal}")
    print(f"Bias: {signal.bias}")
    print(f"Confidence: {signal.confidence:.2f}")
    print(f"Strength Score: {signal.strength_scores.final_strength:.1f}")
    print(f"Current Price: ${signal.current_price:.2f}")

    if signal.risk_management.stoploss:
        print(f"Stop Loss: ${signal.risk_management.stoploss:.2f}")
    if signal.risk_management.target1:
        print(f"Target 1: ${signal.risk_management.target1:.2f}")
    if signal.risk_management.rr_ratio:
        print(f"Risk-Reward Ratio: {signal.risk_management.rr_ratio:.2f}")

    # Market condition
    condition = agent.get_market_condition('BTC')
    print(f"\nMarket Condition: {condition.overall}")
    print(f"Trend: {condition.trend} | Volatility: {condition.volatility} | Volume: {condition.volume}")
    print(f"{'='*50}\n")