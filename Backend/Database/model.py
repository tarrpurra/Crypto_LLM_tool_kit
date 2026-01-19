from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime

Base = declarative_base()

class RiskConfig(Base):
    __tablename__ = 'risk_config'
    id = Column(Integer, primary_key=True)
    max_risk_per_trade_pct = Column(Float, default=1.0)
    max_symbol_exposure_pct = Column(Float, default=15.0)
    max_market_exposure_pct = Column(Float, default=60.0)
    max_open_positions = Column(Integer, default=8)
    max_daily_loss_pct = Column(Float, default=3.0)
    min_rr = Column(Float, default=1.5)
    min_sl_distance_pct = Column(Float, default=0.3)
    max_sl_distance_pct = Column(Float, default=8.0)
    min_confidence = Column(Float, default=0.45)
    confidence_position_scale = Column(Boolean, default=True)
    vol_position_scale = Column(Boolean, default=True)
    max_volatility_pct = Column(Float, default=6.0)
    equity_qty_step = Column(Integer, default=1)
    crypto_qty_step = Column(Float, default=0.0001)
    slippage_bps = Column(Float, default=5.0)
    fee_bps = Column(Float, default=2.0)

class Position(Base):
    __tablename__ = 'positions'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    market = Column(String, nullable=False)
    instrument_type = Column(String, nullable=False)
    qty = Column(Float, nullable=False)
    avg_price = Column(Float, nullable=False)
    side = Column(String, default="LONG")
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    status = Column(String, default="OPEN")

class AccountState(Base):
    __tablename__ = 'account_state'
    id = Column(Integer, primary_key=True)
    equity_value = Column(Float, nullable=False)
    cash_available = Column(Float, nullable=False)
    day_pnl = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow)

class TradeRequest(Base):
    __tablename__ = 'trade_requests'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    market = Column(String, nullable=False)
    side = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=True)
    signal_confidence = Column(Float, default=0.7)
    timestamp = Column(DateTime, default=datetime.utcnow)

class RiskDecision(Base):
    __tablename__ = 'risk_decisions'
    id = Column(Integer, primary_key=True)
    trade_request_id = Column(Integer, ForeignKey('trade_requests.id'))
    allowed = Column(Boolean, nullable=False)
    qty = Column(Float, nullable=False)
    risk_amount = Column(Float, nullable=False)
    risk_pct = Column(Float, nullable=False)
    reasons = Column(String)
    order_plan = Column(String)
    limits_snapshot = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    trade_request = relationship("TradeRequest", backref="risk_decisions")

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    symbol = Column(String, nullable=False)
    market = Column(String, nullable=False)
    instrument_type = Column(String, nullable=False)
    side = Column(String, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    qty = Column(Float, nullable=False)
    status = Column(String, default="OPEN")
    profit_loss = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user = relationship("User", backref="trades")

class Asset(Base):
    __tablename__ = 'assets'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, unique=True, nullable=False)
    market = Column(String, nullable=False)
    name = Column(String, nullable=False)
    current_price = Column(Float, nullable=False)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AgentInteraction(Base):
    __tablename__ = 'agent_interactions'
    id = Column(Integer, primary_key=True)
    agent_name = Column(String, nullable=False)
    input_data = Column(String, nullable=False)
    output_data = Column(String, nullable=False)
    details = Column(String)
    recommendation = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class PnLSnapshot(Base):
    __tablename__ = 'pnl_snapshots'
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    portfolio_id = Column(String, nullable=False)
    equity_value = Column(Float, nullable=False)
    cash_available = Column(Float, nullable=False)
    day_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
