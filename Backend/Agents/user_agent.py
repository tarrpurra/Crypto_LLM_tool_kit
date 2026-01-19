import json
from typing import Dict, Any
from sqlalchemy.orm import Session
import sys
import os

# Add the Backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Database.connection import connection
from Database.model import Position, AccountState, RiskConfig, User

class UserAgent:
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.db_connection = connection()
        self.session = self.db_connection.Session()

    def fetch_user_data(self) -> Dict[str, Any]:
        """Fetch user data from the database and return it as JSON."""
        user_data = {
            "user_id": self.user_id,
            "positions": [],
            "account_state": None,
            "risk_config": None,
        }

        # Fetch user details
        user = self.session.query(User).filter_by(id=self.user_id).first()
        if user:
            user_data["username"] = user.username

        # Fetch positions
        positions = self.session.query(Position).all()
        for position in positions:
            user_data["positions"].append({
                "symbol": position.symbol,
                "market": position.market,
                "instrument_type": position.instrument_type,
                "qty": position.qty,
                "avg_price": position.avg_price,
                "side": position.side,
                "entry_time": position.entry_time.isoformat() if position.entry_time else None,
                "exit_time": position.exit_time.isoformat() if position.exit_time else None,
                "status": position.status,
            })

        # Fetch account state
        account_state = self.session.query(AccountState).order_by(AccountState.timestamp.desc()).first()
        if account_state:
            user_data["account_state"] = {
                "equity_value": account_state.equity_value,
                "cash_available": account_state.cash_available,
                "day_pnl": account_state.day_pnl,
                "timestamp": account_state.timestamp.isoformat() if account_state.timestamp else None,
            }

        # Fetch risk configuration
        risk_config = self.session.query(RiskConfig).first()
        if risk_config:
            user_data["risk_config"] = {
                "max_risk_per_trade_pct": risk_config.max_risk_per_trade_pct,
                "max_symbol_exposure_pct": risk_config.max_symbol_exposure_pct,
                "max_market_exposure_pct": risk_config.max_market_exposure_pct,
                "max_open_positions": risk_config.max_open_positions,
                "max_daily_loss_pct": risk_config.max_daily_loss_pct,
                "min_rr": risk_config.min_rr,
                "min_sl_distance_pct": risk_config.min_sl_distance_pct,
                "max_sl_distance_pct": risk_config.max_sl_distance_pct,
                "min_confidence": risk_config.min_confidence,
                "confidence_position_scale": risk_config.confidence_position_scale,
                "vol_position_scale": risk_config.vol_position_scale,
                "max_volatility_pct": risk_config.max_volatility_pct,
                "equity_qty_step": risk_config.equity_qty_step,
                "crypto_qty_step": risk_config.crypto_qty_step,
                "slippage_bps": risk_config.slippage_bps,
                "fee_bps": risk_config.fee_bps,
            }

        self.session.close()
        return user_data

    def get_user_data_json(self) -> str:
        """Return user data as a JSON string."""
        user_data = self.fetch_user_data()
        return json.dumps(user_data, indent=2)

# Example usage
if __name__ == "__main__":
    user_agent = UserAgent(user_id=1)
    user_data_json = user_agent.get_user_data_json()
    print(user_data_json)