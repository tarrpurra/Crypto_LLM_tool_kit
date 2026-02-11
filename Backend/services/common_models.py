#!/usr/bin/env python3
"""
Common data models for the Technical Agent
Standardized models for candles, on-chain signals, and technical indicators
"""

from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class Candle(BaseModel):
    """Standardized candle model for both crypto and equity data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class OnchainSignals(BaseModel):
    """On-chain signals from Nansen API"""
    smart_money_inflow: float = 0.0
    exchange_inflow: float = 0.0
    exchange_outflow: float = 0.0
    whale_tx_count: int = 0
    whale_volume_24h: float = 0.0
    holder_concentration: float = 0.5  # 0-1 scale
    avg_token_age_days: float = 30.0
    onchain_sentiment: float = 0.5  # 0-1 scale
    recent_whale_alerts: int = 0
    avg_daily_volume: float = 1000000.0


class TechnicalIndicators(BaseModel):
    """Technical indicator values"""
    rsi: float = 50.0
    ema20: float = 0.0
    ema50: float = 0.0
    sma200: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0
    atr: float = 0.0
    obv: float = 0.0


class StrengthScores(BaseModel):
    """Component strength scores"""
    trend_score: float = 0.0
    momentum_score: float = 0.0
    volatility_score: float = 0.0
    volume_score: float = 0.0
    onchain_accumulation_score: float = 5.0  # 0-10 scale
    onchain_whale_score: float = 5.0  # 0-10 scale
    technical_strength: float = 50.0  # 0-100 scale
    onchain_score: float = 50.0  # 0-100 scale
    final_strength: float = 50.0  # 0-100 scale


class RiskManagement(BaseModel):
    """Risk management suggestions"""
    stoploss: Optional[float] = None
    target1: Optional[float] = None
    target2: Optional[float] = None
    rr_ratio: Optional[float] = None


class TechnicalSignal(BaseModel):
    """Complete technical analysis signal"""
    symbol: str
    asset_type: str  # 'crypto' or 'equity'
    timestamp: datetime
    current_price: float

    # Component signals
    bias: str  # 'bullish', 'bearish', 'neutral'
    signal: str  # 'BUY', 'SELL', 'HOLD' for equity; 'LONG', 'SHORT', 'HOLD' for crypto
    confidence: float  # 0-1 scale

    # Detailed scores
    strength_scores: StrengthScores
    technical_indicators: TechnicalIndicators
    onchain_signals: Optional[OnchainSignals] = None

    # Risk management
    risk_management: RiskManagement

    # Metadata
    data_points: int = 0
    indicators_calculated: bool = False
    onchain_data_available: bool = False


class MarketCondition(BaseModel):
    """Market condition assessment"""
    volatility: str  # 'low', 'medium', 'high'
    trend: str  # 'bullish', 'bearish', 'sideways'
    volume: str  # 'low', 'normal', 'high'
    overall: str  # 'bullish', 'bearish', 'neutral'