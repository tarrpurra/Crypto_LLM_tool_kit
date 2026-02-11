#!/usr/bin/env python3
"""
Technical Indicator Engine
Computes comprehensive technical indicators using pandas-ta
"""

import pandas as pd
import talib
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    # Try relative imports (for when imported as part of package)
    from .services.common_models import Candle, TechnicalIndicators
except ImportError:
    # Fallback to absolute imports (for direct script execution)
    import sys
    import os
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, current_dir)

    from services.common_models import Candle, TechnicalIndicators


class IndicatorEngine:
    """Engine for computing technical indicators from price data"""

    def __init__(self):
        # Setup logging
        self.logger = logging.getLogger('IndicatorEngine')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def candles_to_df(self, candles: List[Candle]) -> pd.DataFrame:
        """
        Convert list of Candle objects to pandas DataFrame
        """
        try:
            df = pd.DataFrame([candle.dict() for candle in candles])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.sort_index()

            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns: {required_cols}")

            self.logger.info(f"✅ Converted {len(df)} candles to DataFrame")
            return df

        except Exception as e:
            self.logger.error(f"❌ Error converting candles to DataFrame: {e}")
            return pd.DataFrame()

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute comprehensive technical indicators
        """
        try:
            if df.empty:
                self.logger.warning("⚠️  Empty DataFrame provided")
                return df

            df = df.copy()

            # Convert to numpy arrays for TA-Lib
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            open_price = df['open'].values.astype(float)
            volume = df['volume'].values.astype(float)

            # Trend Indicators
            # Moving Averages
            df['sma_5'] = talib.SMA(close, timeperiod=5)
            df['sma_10'] = talib.SMA(close, timeperiod=10)
            df['sma_20'] = talib.SMA(close, timeperiod=20)
            df['sma_50'] = talib.SMA(close, timeperiod=50)
            df['sma_200'] = talib.SMA(close, timeperiod=200)

            # Exponential Moving Averages
            df['ema_12'] = talib.EMA(close, timeperiod=12)
            df['ema_20'] = talib.EMA(close, timeperiod=20)
            df['ema_26'] = talib.EMA(close, timeperiod=26)
            df['ema_50'] = talib.EMA(close, timeperiod=50)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist

            # Momentum Indicators
            # RSI
            df['rsi'] = talib.RSI(close, timeperiod=14)

            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd

            # Williams %R
            df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)

            # Commodity Channel Index
            df['cci'] = talib.CCI(high, low, close, timeperiod=20)

            # Volatility Indicators
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle

            # Average True Range
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)

            # Rolling Volatility (Standard Deviation)
            df['volatility_20'] = df['close'].pct_change().rolling(20).std()
            df['volatility_50'] = df['close'].pct_change().rolling(50).std()

            # Volume Indicators
            # On Balance Volume
            df['obv'] = talib.OBV(close, volume)

            # Chaikin Money Flow (Accumulation/Distribution as approximation)
            df['cmf'] = talib.AD(high, low, close, volume)

            # Volume Moving Averages
            df['volume_sma_5'] = talib.SMA(volume, timeperiod=5)
            df['volume_sma_20'] = talib.SMA(volume, timeperiod=20)

            # Rate of Change
            df['roc_5'] = talib.ROC(close, timeperiod=5)
            df['roc_10'] = talib.ROC(close, timeperiod=10)

            # Momentum
            df['momentum_1'] = talib.MOM(close, timeperiod=1)
            df['momentum_5'] = talib.MOM(close, timeperiod=5)
            df['momentum_10'] = talib.MOM(close, timeperiod=10)

            # Price position within channels
            df['high_20'] = df['high'].rolling(20).max()
            df['low_20'] = df['low'].rolling(20).min()
            df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])

            # Returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = df['close'].pct_change().apply(lambda x: 0 if x == 0 else np.log(1 + x))

            # Handle NaN values created by indicators
            initial_rows = len(df)
            df_clean = df.dropna()

            if len(df_clean) == 0:
                # If all rows were dropped, fill NaN values with neutral defaults
                self.logger.warning(f"All {initial_rows} rows would be dropped by dropna, filling NaN with defaults")
                df = df.fillna({
                    'rsi': 50, 'ema20': df['close'].iloc[-1], 'ema50': df['close'].iloc[-1],
                    'sma200': df['close'].iloc[-1], 'macd': 0, 'macd_signal': 0,
                    'bb_upper': df['close'].iloc[-1] * 1.05, 'bb_lower': df['close'].iloc[-1] * 0.95,
                    'atr': df['close'].iloc[-1] * 0.02, 'obv': 1000000
                })
            else:
                df = df_clean

            self.logger.info(f"✅ Computed {len(df.columns) - 5} technical indicators")  # -5 for OHLCV
            self.logger.info(f"   Final dataset: {len(df)} records, {len(df.columns)} features")

            return df

        except Exception as e:
            self.logger.error(f"❌ Error computing indicators: {e}")
            return df

    def get_latest_indicator_snapshot(self, df: pd.DataFrame) -> TechnicalIndicators:
        """
        Extract the latest indicator values for scoring
        """
        try:
            if df.empty:
                self.logger.warning("⚠️  Empty DataFrame for indicator snapshot")
                return TechnicalIndicators()

            last_row = df.iloc[-1]
            close_price = float(last_row.get('close', 0))

            indicators = TechnicalIndicators(
                rsi=float(last_row.get('rsi', 50) or 50),
                ema20=float(last_row.get('ema_20', close_price) or close_price),
                ema50=float(last_row.get('ema_50', close_price) or close_price),
                sma200=float(last_row.get('sma_200', close_price) or close_price),
                macd=float(last_row.get('macd', 0) or 0),
                macd_signal=float(last_row.get('macd_signal', 0) or 0),
                bb_upper=float(last_row.get('bb_upper', close_price * 1.1) or close_price * 1.1),
                bb_lower=float(last_row.get('bb_lower', close_price * 0.9) or close_price * 0.9),
                atr=float(last_row.get('atr', 0) or 0),
                obv=float(last_row.get('obv', 0) or 0)
            )

            self.logger.info("✅ Extracted latest indicator snapshot")
            return indicators

        except Exception as e:
            self.logger.error(f"❌ Error extracting indicator snapshot: {e}")
            return TechnicalIndicators()

    def get_indicator_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for indicators
        """
        try:
            if df.empty:
                return {}

            summary = {
                'data_points': len(df),
                'date_range': {
                    'start': df.index.min().isoformat() if not df.empty else None,
                    'end': df.index.max().isoformat() if not df.empty else None
                },
                'price_range': {
                    'min': float(df['close'].min()),
                    'max': float(df['close'].max()),
                    'current': float(df['close'].iloc[-1])
                },
                'indicators_available': [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']],
                'volatility': {
                    'current_20': float(df['volatility_20'].iloc[-1]) if 'volatility_20' in df.columns else None,
                    'average_20': float(df['volatility_20'].mean()) if 'volatility_20' in df.columns else None
                }
            }

            return summary

        except Exception as e:
            self.logger.error(f"❌ Error generating indicator summary: {e}")
            return {}


# Example usage and testing
if __name__ == "__main__":
    from services.crypto_coinmarketcap import CryptoDataProvider

    # Initialize components
    engine = IndicatorEngine()

    # Mock data provider (replace with real implementation)
    provider = CryptoDataProvider("mock_api_key")

    # Get sample candles
    candles = provider.get_candles('BTC', timeframe='1d', lookback=100)

    if candles:
        # Convert to DataFrame
        df = engine.candles_to_df(candles)
        print(f"DataFrame shape: {df.shape}")

        # Compute indicators
        df_with_indicators = engine.compute_indicators(df)
        print(f"With indicators shape: {df_with_indicators.shape}")

        # Get latest snapshot
        snapshot = engine.get_latest_indicator_snapshot(df_with_indicators)
        print(f"RSI: {snapshot.rsi:.2f}")
        print(f"EMA20: {snapshot.ema20:.2f}")

        # Get summary
        summary = engine.get_indicator_summary(df_with_indicators)
        print(f"Summary: {summary}")
    else:
        print("No candle data available")