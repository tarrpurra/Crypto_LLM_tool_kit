#!/usr/bin/env python3
"""
Stock Data Service using Yahoo Finance

Fetches historical stock data and calculates comprehensive technical indicators
for Indian Nifty 50 companies and other global stocks.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import talib
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os

class StockService:
    """Service for fetching and processing stock market data."""

    def __init__(self):
        # Indian stock ticker mappings (Yahoo Finance format)
        self.indian_stocks = {
            'RELIANCE': 'RELIANCE.NS',
            'HDFCBANK': 'HDFCBANK.NS',
            'INFY': 'INFY.NS',
            'ICICIBANK': 'ICICIBANK.NS',
            'TCS': 'TCS.NS',
            'KOTAKBANK': 'KOTAKBANK.NS',
            'HINDUNILVR': 'HINDUNILVR.NS',
            'ITC': 'ITC.NS',
            'BAJFINANCE': 'BAJFINANCE.NS',
            'BHARTIARTL': 'BHARTIARTL.NS',
            'MARUTI': 'MARUTI.NS',
            'ASIANPAINT': 'ASIANPAINT.NS',
            'SBIN': 'SBIN.NS',
            'NTPC': 'NTPC.NS',
            'TITAN': 'TITAN.NS',
            'COALINDIA': 'COALINDIA.NS',
            'LT': 'LT.NS',
            'TATASTEEL': 'TATASTEEL.NS',
            'UPL': 'UPL.NS',
            'WIPRO': 'WIPRO.NS',
            'NIFTY': '^NSEI',  # Nifty 50 index
            'BANKNIFTY': '^NSEBANK'  # Bank Nifty
        }

        # Cache for API responses
        self.cache = {}
        self.cache_duration = 3600  # 1 hour

        # Setup logging
        self.logger = logging.getLogger('StockService')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _get_cached_data(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data if still valid."""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.cache_duration:
                return data
            else:
                del self.cache[key]
        return None

    def _set_cached_data(self, key: str, data: pd.DataFrame) -> None:
        """Cache data."""
        self.cache[key] = (data, datetime.now())

    def get_ticker_symbol(self, symbol: str) -> str:
        """Convert common symbol to Yahoo Finance ticker."""
        symbol = symbol.upper()

        # Check if it's already a Yahoo ticker
        if symbol.endswith('.NS') or symbol.startswith('^'):
            return symbol

        # Check Indian stocks mapping
        if symbol in self.indian_stocks:
            return self.indian_stocks[symbol]

        # Assume it's a US stock or international
        return symbol

    def fetch_historical_data(self, symbol: str, days: int = 730,
                            interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetch historical stock data from Yahoo Finance.

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'AAPL', 'NIFTY')
            days: Number of days of historical data
            interval: Data interval ('1d', '1h', '1wk', etc.)

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        cache_key = f"yfinance_{symbol}_{days}_{interval}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            self.logger.info(f"Returning cached data for {symbol}")
            return cached_data

        try:
            # Get Yahoo Finance ticker
            ticker_symbol = self.get_ticker_symbol(symbol)

            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            self.logger.info(f"Fetching {symbol} ({ticker_symbol}) data from Yahoo Finance...")
            self.logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

            # Download data
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True,  # Adjust for splits/dividends
                prepost=False
            )

            if data.empty:
                self.logger.warning(f"No data received for {symbol} ({ticker_symbol})")
                return None

            # Clean and format data
            data = data.reset_index()

            # Rename columns to match our format
            data = data.rename(columns={
                'Date': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # Select only required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in required_cols if col in data.columns]

            if len(available_cols) < 5:  # Need at least OHLC
                self.logger.error(f"Insufficient data columns for {symbol}")
                return None

            data = data[available_cols]
            data = data.set_index('timestamp')

            # Sort by timestamp
            data = data.sort_index()

            # Remove any duplicate indices
            data = data[~data.index.duplicated(keep='first')]

            self._set_cached_data(cache_key, data)

            self.logger.info(f"✅ Successfully fetched {len(data)} records for {symbol}")
            self.logger.info(f"   Date range: {data.index.min()} to {data.index.max()}")
            self.logger.info(f"   Price range: INR {data['close'].min():.2f} - INR {data['close'].max():.2f}")

            return data

        except Exception as e:
            self.logger.error(f"❌ Error fetching data for {symbol}: {e}")
            return None

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add comprehensive technical indicators to the data.

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with technical indicators
        """
        try:
            df = data.copy()

            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.warning("Missing required OHLCV columns for technical indicators")
                return df

            # Convert to numpy arrays for TA-Lib
            close = df['close'].values.astype(float)
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            open_price = df['open'].values.astype(float)
            volume = df['volume'].values.astype(float)

            # Moving Averages
            df['ma_5'] = talib.SMA(close, timeperiod=5)
            df['ma_10'] = talib.SMA(close, timeperiod=10)
            df['ma_20'] = talib.SMA(close, timeperiod=20)
            df['ma_50'] = talib.SMA(close, timeperiod=50)
            df['ma_200'] = talib.SMA(close, timeperiod=200)

            # Exponential Moving Averages
            df['ema_12'] = talib.EMA(close, timeperiod=12)
            df['ema_26'] = talib.EMA(close, timeperiod=26)

            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist

            # RSI
            df['rsi'] = talib.RSI(close, timeperiod=14)

            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd

            # Williams %R
            df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)

            # Commodity Channel Index
            df['cci'] = talib.CCI(high, low, close, timeperiod=14)

            # Average Directional Index
            df['adx'] = talib.ADX(high, low, close, timeperiod=14)
            df['adx_pos'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            df['adx_neg'] = talib.MINUS_DI(high, low, close, timeperiod=14)

            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle

            # Average True Range
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)

            # Volatility (Rolling Standard Deviation)
            df['volatility_20'] = df['close'].pct_change().rolling(20).std()
            df['volatility_50'] = df['close'].pct_change().rolling(50).std()

            # Rate of Change
            df['roc_5'] = talib.ROC(close, timeperiod=5)
            df['roc_10'] = talib.ROC(close, timeperiod=10)

            # Momentum
            df['momentum_1'] = talib.MOM(close, timeperiod=1)
            df['momentum_5'] = talib.MOM(close, timeperiod=5)
            df['momentum_10'] = talib.MOM(close, timeperiod=10)

            # Volume indicators
            df['obv'] = talib.OBV(close, volume)
            df['cmf'] = talib.AD(high, low, close, volume)  # Chaikin Money Flow approximation

            # Volume moving averages
            df['volume_ma_5'] = talib.SMA(volume, timeperiod=5)
            df['volume_ma_20'] = talib.SMA(volume, timeperiod=20)

            # Price position within range
            df['high_20'] = df['high'].rolling(20).max()
            df['low_20'] = df['low'].rolling(20).min()
            df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'])

            # Returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Remove NaN values created by indicators
            df = df.dropna()

            self.logger.info(f"Added {len(df.columns) - len(data.columns)} technical indicators")
            self.logger.info(f"Final dataset: {len(df)} records, {len(df.columns)} features")

            return df

        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {e}")
            return data

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a stock."""
        try:
            ticker_symbol = self.get_ticker_symbol(symbol)
            ticker = yf.Ticker(ticker_symbol)
            data = ticker.history(period='1d', interval='1m')

            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None

        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company information."""
        try:
            ticker_symbol = self.get_ticker_symbol(symbol)
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            return {
                'symbol': symbol,
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0)
            }

        except Exception as e:
            self.logger.error(f"Error getting company info for {symbol}: {e}")
            return None

# Example usage and testing
if __name__ == "__main__":
    service = StockService()

    # Test Nifty companies
    test_symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK']

    for symbol in test_symbols:
        print(f"\n=== Testing {symbol} ===")

        # Fetch data
        data = service.fetch_historical_data(symbol, days=365)
        if data is None:
            print(f"❌ Failed to fetch {symbol}")
            continue

        print(f"✅ Fetched {len(data)} records")

        # Add technical indicators
        data_with_indicators = service.add_technical_indicators(data)
        print(f"✅ Added indicators: {len(data_with_indicators.columns)} total columns")

        # Show sample
        print("Sample data with indicators:")
        print(data_with_indicators.head(3))

        # Get current price
        current_price = service.get_current_price(symbol)
        if current_price:
            print(f"💰 Current price: ₹{current_price:.2f}")

        # Get company info
        info = service.get_company_info(symbol)
        if info:
            print(f"🏢 Company: {info.get('name', 'N/A')}")
            print(f"📊 Sector: {info.get('sector', 'N/A')}")

        break  # Test only first symbol