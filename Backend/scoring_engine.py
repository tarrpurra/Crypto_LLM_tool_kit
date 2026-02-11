#!/usr/bin/env python3
"""
Scoring Engine for Technical Analysis
Computes strength scores from technical indicators and on-chain metrics
"""

import logging
from typing import Dict, Any, Optional
try:
    # Try relative imports (for when imported as part of package)
    from .services.common_models import TechnicalIndicators, OnchainSignals, StrengthScores
except ImportError:
    # Fallback to absolute imports (for direct script execution)
    import sys
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, current_dir)

    from services.common_models import TechnicalIndicators, OnchainSignals, StrengthScores


class ScoringEngine:
    """Engine for computing technical strength scores"""

    def __init__(self):
        # Setup logging
        self.logger = logging.getLogger('ScoringEngine')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def score_rsi(self, rsi: float) -> int:
        """
        Score RSI (0-100 scale)
        Oversold (<30): Bullish, Overbought (>70): Bearish
        """
        if rsi < 20:
            return 10  # Extremely oversold
        elif rsi < 30:
            return 8   # Oversold
        elif rsi < 45:
            return 3   # Slightly oversold
        elif rsi <= 55:
            return 0   # Neutral
        elif rsi <= 70:
            return -3  # Slightly overbought
        elif rsi <= 80:
            return -8  # Overbought
        else:
            return -10 # Extremely overbought

    def score_trend(self, close: float, ema20: float, ema50: float, sma200: float) -> int:
        """
        Score trend based on moving averages alignment
        """
        score = 0

        # Price vs EMAs
        if close > ema20:
            score += 5
        if close > ema50:
            score += 5
        if close > sma200:
            score += 10

        # EMA alignment
        if ema20 > ema50:
            score += 5
        if ema50 > sma200:
            score += 5

        return score

    def score_macd(self, macd: float, macd_signal: float) -> int:
        """
        Score MACD signal
        """
        if macd > macd_signal and macd > 0:
            return 8   # Bullish crossover above zero
        elif macd > macd_signal and macd <= 0:
            return 4   # Bullish crossover below zero
        elif macd < macd_signal and macd < 0:
            return -8  # Bearish crossover below zero
        else:
            return -4  # Bearish crossover above zero

    def score_bollinger(self, close: float, bb_upper: float, bb_lower: float) -> int:
        """
        Score Bollinger Band position
        """
        if bb_upper == bb_lower:  # Avoid division by zero
            return 0

        position = (close - bb_lower) / (bb_upper - bb_lower)

        if position < 0.2:
            return 6   # Near lower band (potential bounce)
        elif position < 0.4:
            return 2   # Lower half
        elif position <= 0.6:
            return 0   # Middle
        elif position <= 0.8:
            return -2  # Upper half
        else:
            return -6  # Near upper band (potential reversal)

    def score_stochastic(self, stoch_k: float, stoch_d: float) -> int:
        """
        Score Stochastic Oscillator
        """
        if stoch_k < 20 and stoch_d < 20:
            return 5   # Oversold
        elif stoch_k > 80 and stoch_d > 80:
            return -5  # Overbought
        elif stoch_k > stoch_d and stoch_k < 80:
            return 2   # Bullish momentum
        elif stoch_k < stoch_d and stoch_k > 20:
            return -2  # Bearish momentum
        else:
            return 0   # Neutral

    def score_volume(self, obv: float, volume_trend: Optional[float] = None) -> int:
        """
        Score volume indicators
        OBV positive = accumulation, negative = distribution
        """
        # For simplicity, we'll use OBV slope as a proxy
        # In a real implementation, you'd calculate the trend
        if volume_trend is None:
            volume_trend = 0  # Neutral

        if volume_trend > 0.1:
            return 4   # Rising volume (bullish)
        elif volume_trend > 0:
            return 2   # Slightly rising
        elif volume_trend > -0.1:
            return 0   # Neutral
        else:
            return -4  # Falling volume (bearish)

    def score_volatility(self, atr: float, close: float) -> int:
        """
        Score volatility (ATR relative to price)
        """
        atr_ratio = atr / close

        if atr_ratio < 0.02:
            return 2   # Low volatility (consolidation)
        elif atr_ratio < 0.05:
            return 0   # Normal volatility
        else:
            return -2  # High volatility (caution)

    def calculate_accumulation_score(self, signals: OnchainSignals) -> float:
        """
        Calculate On-chain Accumulation Score (0-10)
        Measures institutional/smart money accumulation
        """
        score = 0.0

        # Safeguard: Handle None values in signals
        if signals.smart_money_inflow is None:
            signals.smart_money_inflow = 0.0
        if signals.exchange_inflow is None:
            signals.exchange_inflow = 0.0
        if signals.exchange_outflow is None:
            signals.exchange_outflow = 0.0
        if signals.holder_concentration is None:
            signals.holder_concentration = 0.5
        if signals.avg_token_age_days is None:
            signals.avg_token_age_days = 30.0

        # Smart money inflows (positive signal)
        if signals.smart_money_inflow > 1000000:  # $1M+ inflow
            score += 4
        elif signals.smart_money_inflow > 500000:
            score += 2

        # Exchange outflows (institutions taking off exchange = accumulation)
        net_exchange_flow = signals.exchange_outflow - signals.exchange_inflow
        if net_exchange_flow > 500000:
            score += 3
        elif net_exchange_flow > 100000:
            score += 1

        # Holder concentration (lower concentration = more distributed = positive)
        if signals.holder_concentration < 0.3:  # Well distributed
            score += 2
        elif signals.holder_concentration > 0.7:  # Highly concentrated (risky)
            score -= 1

        # Token age distribution (older holders = more committed)
        if signals.avg_token_age_days > 180:  # 6+ months average
            score += 1

        return max(0, min(10, score))

    def calculate_whale_score(self, signals: OnchainSignals) -> float:
        """
        Calculate Whale Activity Score (0-10)
        Measures whale transaction activity and sentiment
        """
        score = 5.0  # Neutral starting point

        # Whale transaction count (high activity = mixed signal)
        if signals.whale_tx_count > 50:  # Very active
            score -= 2  # Could indicate distribution
        elif signals.whale_tx_count > 20:
            score -= 1
        elif signals.whale_tx_count < 5:  # Low activity
            score += 1  # Potentially accumulating

        # Large transaction volume
        if signals.avg_daily_volume > 0:
            volume_ratio = signals.whale_volume_24h / signals.avg_daily_volume
            if volume_ratio > 0.5:  # Whales moving >50% of daily volume
                score -= 1  # High impact potential
            elif volume_ratio < 0.1:  # Low whale activity
                score += 1

        # On-chain sentiment
        if signals.onchain_sentiment > 0.7:
            score += 2
        elif signals.onchain_sentiment < 0.3:
            score -= 2

        # Real-time whale alerts (recent large transactions)
        if signals.recent_whale_alerts > 10:
            score -= 1  # Too much activity

        return max(0, min(10, score))

    def calculate_technical_strength(self, indicators: TechnicalIndicators) -> float:
        """
        Calculate overall technical strength score (0-100)
        """
        try:
            # Ensure all indicator values are valid numbers
            rsi = float(indicators.rsi or 50)
            ema20 = float(indicators.ema20 or 0)
            ema50 = float(indicators.ema50 or 0)
            sma200 = float(indicators.sma200 or 0)
            macd = float(indicators.macd or 0)
            macd_signal = float(indicators.macd_signal or 0)
            bb_upper = float(indicators.bb_upper or 0)
            bb_lower = float(indicators.bb_lower or 0)
            obv = float(indicators.obv or 0)
            
            # Component scores
            trend_score = self.score_trend(
                indicators.close if hasattr(indicators, 'close') else 0,
                ema20,
                ema50,
                sma200
            )

            rsi_score = self.score_rsi(rsi)

            macd_score = self.score_macd(macd, macd_signal)

            bollinger_score = self.score_bollinger(
                indicators.close if hasattr(indicators, 'close') else 0,
                bb_upper,
                bb_lower
            )

            # Weighted combination
            weights = {
                'trend': 0.4,      # Most important
                'rsi': 0.3,        # Momentum
                'macd': 0.2,       # Trend confirmation
                'bollinger': 0.1   # Volatility context
            }

            raw_score = (
                trend_score * weights['trend'] +
                rsi_score * weights['rsi'] +
                macd_score * weights['macd'] +
                bollinger_score * weights['bollinger']
            )

            # Normalize to 0-100 scale (assuming raw range of -40 to +40)
            strength = max(0, min(100, (raw_score + 40) * (100 / 80)))

            return strength

        except Exception as e:
            self.logger.error(f"❌ Error calculating technical strength: {e}")
            return 50.0  # Neutral fallback

    def calculate_onchain_score(self, signals: OnchainSignals) -> float:
        """
        Calculate combined on-chain score (0-100)
        """
        try:
            accumulation = self.calculate_accumulation_score(signals)
            whale_activity = self.calculate_whale_score(signals)

            # Weight: 60% accumulation (more important), 40% whale activity
            onchain_score = (0.6 * accumulation) + (0.4 * whale_activity)

            # Convert to 0-100 scale
            onchain_score_100 = onchain_score * 10  # Since individual scores are 0-10

            return onchain_score_100

        except Exception as e:
            self.logger.error(f"❌ Error calculating on-chain score: {e}")
            return 50.0  # Neutral fallback

    def calculate_final_strength(self, technical_strength: float, onchain_score: float) -> float:
        """
        Calculate final strength combining technical and on-chain analysis
        """
        # Technical = 60%, On-chain = 40%
        final_strength = (0.6 * technical_strength) + (0.4 * onchain_score)

        return max(0, min(100, final_strength))

    def compute_strength_scores(self,
                               indicators: TechnicalIndicators,
                               onchain_signals: Optional[OnchainSignals] = None) -> StrengthScores:
        """
        Compute all strength scores
        """
        try:
            # Technical strength
            technical_strength = self.calculate_technical_strength(indicators)

            # On-chain scores (if available)
            if onchain_signals:
                accumulation_score = self.calculate_accumulation_score(onchain_signals)
                whale_score = self.calculate_whale_score(onchain_signals)
                onchain_score = self.calculate_onchain_score(onchain_signals)
                final_strength = self.calculate_final_strength(technical_strength, onchain_score)
            else:
                accumulation_score = 5.0
                whale_score = 5.0
                onchain_score = 50.0
                final_strength = technical_strength

            return StrengthScores(
                trend_score=self.score_trend(
                    indicators.close if hasattr(indicators, 'close') else 0,
                    indicators.ema20, indicators.ema50, indicators.sma200
                ),
                momentum_score=self.score_rsi(indicators.rsi) + self.score_macd(indicators.macd, indicators.macd_signal),
                volatility_score=self.score_bollinger(
                    indicators.close if hasattr(indicators, 'close') else 0,
                    indicators.bb_upper, indicators.bb_lower
                ),
                volume_score=self.score_volume(indicators.obv),
                onchain_accumulation_score=accumulation_score,
                onchain_whale_score=whale_score,
                technical_strength=technical_strength,
                onchain_score=onchain_score,
                final_strength=final_strength
            )

        except Exception as e:
            self.logger.error(f"❌ Error computing strength scores: {e}")
            return StrengthScores()


# Example usage and testing
if __name__ == "__main__":
    from services.common_models import TechnicalIndicators, OnchainSignals

    engine = ScoringEngine()

    # Test technical indicators
    indicators = TechnicalIndicators(
        rsi=65,
        ema20=50000,
        ema50=49500,
        sma200=48000,
        macd=200,
        macd_signal=150,
        bb_upper=52000,
        bb_lower=48000,
        atr=1000,
        obv=1000000
    )

    # Test on-chain signals
    onchain = OnchainSignals(
        smart_money_inflow=750000,
        exchange_inflow=200000,
        exchange_outflow=800000,
        whale_tx_count=25,
        whale_volume_24h=50000000,
        holder_concentration=0.4,
        avg_token_age_days=120,
        onchain_sentiment=0.6,
        recent_whale_alerts=5,
        avg_daily_volume=200000000
    )

    # Compute scores
    scores = engine.compute_strength_scores(indicators, onchain)

    print("Technical Strength:", scores.technical_strength)
    print("On-chain Score:", scores.onchain_score)
    print("Final Strength:", scores.final_strength)
    print("Accumulation Score:", scores.onchain_accumulation_score)
    print("Whale Score:", scores.onchain_whale_score)