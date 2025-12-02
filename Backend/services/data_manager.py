import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
import talib
from .crypto_service import CryptoService
from .kite_service import KiteService
from .stock_service import StockService

class DataManager:
    def __init__(self):
        self.crypto_service = CryptoService()
        self.kite_service = KiteService()
        self.stock_service = StockService()

        # Set up logging
        self.logger = logging.getLogger('DataManager')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Data cache
        self.data_cache = {}
        self.cache_duration = 3600  # 1 hour

    def _get_cached_data(self, key):
        """Get cached data if still valid."""
        if key in self.data_cache:
            data, timestamp = self.data_cache[key]
            if (datetime.now() - timestamp).seconds < self.cache_duration:
                return data
            else:
                del self.data_cache[key]
        return None

    def _set_cached_data(self, key, data):
        """Cache data."""
        self.data_cache[key] = (data, datetime.now())

    def get_asset_type(self, symbol):
        """
        Determine asset type and data source based on symbol.

        Args:
            symbol: Asset symbol

        Returns:
            str: 'crypto', 'stock_yahoo', 'stock_kite', or 'index'
        """
        symbol = symbol.upper()

        # Cryptocurrency symbols
        crypto_symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'LINK', 'LTC', 'XRP', 'BCH']
        if symbol in crypto_symbols or symbol.endswith(('BTC', 'ETH', 'USDT', 'USD')):
            return 'crypto'

        # Indian stock symbols (Yahoo Finance)
        indian_stocks = [
            'RELIANCE', 'HDFCBANK', 'INFY', 'ICICIBANK', 'TCS', 'KOTAKBANK',
            'HINDUNILVR', 'ITC', 'BAJFINANCE', 'BHARTIARTL', 'MARUTI', 'ASIANPAINT',
            'SBIN', 'NTPC', 'TITAN', 'COALINDIA', 'LT', 'TATASTEEL', 'UPL', 'WIPRO'
        ]
        if symbol in indian_stocks:
            return 'stock_yahoo'

        # Indices
        if symbol in ['NIFTY', '^NSEI', '^NSEBANK']:
            return 'stock_yahoo'

        # International stocks (Yahoo Finance)
        international_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        if symbol in international_stocks:
            return 'stock_yahoo'

        # Default to Yahoo Finance for unknown symbols
        return 'stock_yahoo'

    def fetch_historical_data(self, symbol, days=365, interval='1d'):
        """
        Unified method to fetch historical data for any asset.

        Args:
            symbol: Asset symbol (e.g., 'BTC', 'NIFTY', 'RELIANCE')
            days: Number of days of historical data
            interval: Time interval ('1d', '1h', '1m' for crypto, 'day', 'hour', 'minute' for stocks)

        Returns:
            pandas.DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{days}_{interval}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            self.logger.info(f"Returning cached data for {symbol}")
            return cached_data

        asset_type = self.get_asset_type(symbol)

        if asset_type == 'crypto':
            # Try Binance first (free), fallback to CoinAPI
            data = self.crypto_service.get_historical_data(symbol, days, interval, source='binance')
            if data is None or len(data) < 10:
                self.logger.info(f"Binance failed for {symbol}, trying CoinAPI")
                data = self.crypto_service.get_historical_data(symbol, days, interval, source='coinapi')
        elif asset_type == 'stock_yahoo':
            # Stock data from Yahoo Finance (free)
            data = self.stock_service.fetch_historical_data(symbol, days, interval)
        else:
            # Fallback to Kite for other stock data
            data = self.kite_service.get_historical_data(symbol, days, interval)

        if data is not None and not data.empty:
            self._set_cached_data(cache_key, data)
            self.logger.info(f"Successfully fetched {len(data)} records for {symbol}")
        else:
            self.logger.warning(f"Failed to fetch data for {symbol}")

        return data

    def get_current_price(self, symbol):
        """
        Get current price for any asset.

        Args:
            symbol: Asset symbol

        Returns:
            float: Current price
        """
        asset_type = self.get_asset_type(symbol)

        if asset_type == 'crypto':
            return self.crypto_service.get_current_price(symbol)
        elif asset_type == 'stock_yahoo':
            return self.stock_service.get_current_price(symbol)
        else:
            return self.kite_service.get_current_price(symbol)

    def prepare_ml_data(self, data, add_features=True):
        """
        Prepare data for ML models with cleaning and feature engineering.

        Args:
            data: Raw OHLCV DataFrame
            add_features: Whether to add technical indicators

        Returns:
            pandas.DataFrame: Cleaned and processed data
        """
        if data is None or data.empty:
            return None

        try:
            # Basic cleaning
            df = data.copy()
            df = df.dropna()  # Remove NaN values
            df = df[df['close'] > 0]  # Remove invalid prices
            df = df[df['volume'] >= 0]  # Remove negative volume

            if add_features:
                # Use stock service for comprehensive technical indicators
                df = self.stock_service.add_technical_indicators(df)

            # Sort by timestamp
            df = df.sort_index()

            self.logger.info(f"Prepared ML data with {len(df)} records and {len(df.columns)} features")
            return df

        except Exception as e:
            self.logger.error(f"Error preparing ML data: {e}")
            return None

    def _add_technical_indicators(self, df):
        """
        Add comprehensive technical indicators using TA-Lib.

        Args:
            df: OHLCV DataFrame

        Returns:
            pandas.DataFrame: DataFrame with additional features
        """
        try:
            # Convert to numpy arrays for TA-Lib
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            close = df['close'].values.astype(float)
            open_price = df['open'].values.astype(float)

            # Basic price-based indicators
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Moving averages
            df['ma_5'] = talib.SMA(close, timeperiod=5)
            df['ma_10'] = talib.SMA(close, timeperiod=10)
            df['ma_20'] = talib.SMA(close, timeperiod=20)
            df['ma_50'] = talib.SMA(close, timeperiod=50)
            df['ma_200'] = talib.SMA(close, timeperiod=200)

            # Exponential moving averages
            df['ema_12'] = talib.EMA(close, timeperiod=12)
            df['ema_26'] = talib.EMA(close, timeperiod=26)

            # MACD (Moving Average Convergence Divergence)
            macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist

            # RSI (Relative Strength Index)
            df['rsi'] = talib.RSI(close, timeperiod=14)

            # Stochastic Oscillator
            slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd

            # Williams %R
            df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)

            # ADX (Average Directional Index)
            df['adx'] = talib.ADX(high, low, close, timeperiod=14)
            df['adx_pos'] = talib.PLUS_DI(high, low, close, timeperiod=14)
            df['adx_neg'] = talib.MINUS_DI(high, low, close, timeperiod=14)

            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle

            # Average True Range (ATR)
            df['atr'] = talib.ATR(high, low, close, timeperiod=14)

            # Commodity Channel Index (CCI)
            df['cci'] = talib.CCI(high, low, close, timeperiod=20)

            # Rate of Change (ROC)
            df['roc_5'] = talib.ROC(close, timeperiod=5)
            df['roc_10'] = talib.ROC(close, timeperiod=10)

            # Momentum
            df['momentum_1'] = talib.MOM(close, timeperiod=1)
            df['momentum_5'] = talib.MOM(close, timeperiod=5)
            df['momentum_10'] = talib.MOM(close, timeperiod=10)

            # Volatility measures
            df['volatility_20'] = df['returns'].rolling(20).std()
            df['volatility_50'] = df['returns'].rolling(50).std()

            # Price channels
            df['high_20'] = df['high'].rolling(20).max()
            df['low_20'] = df['low'].rolling(20).min()

            # Volume indicators (if volume data exists)
            if 'volume' in df.columns and df['volume'].notna().any():
                volume = df['volume'].values.astype(float)

                # On Balance Volume
                df['obv'] = talib.OBV(close, volume)

                # Chaikin Money Flow
                df['cmf'] = talib.AD(high, low, close, volume)  # Accumulation/Distribution as proxy

                # Volume moving averages
                df['volume_ma_5'] = talib.SMA(volume, timeperiod=5)
                df['volume_ma_20'] = talib.SMA(volume, timeperiod=20)

            # Remove NaN values created by technical indicators
            df = df.dropna()

            self.logger.info(f"Added {len(df.columns) - 6} technical indicators using TA-Lib")  # 6 = original OHLCV columns
            return df

        except Exception as e:
            self.logger.error(f"Error adding technical indicators with TA-Lib: {e}")
            # Return original dataframe if indicators fail
            return df

    def validate_data(self, data, min_records=100):
        """
        Validate data quality for ML training.

        Args:
            data: DataFrame to validate
            min_records: Minimum required records

        Returns:
            dict: Validation results
        """
        if data is None or data.empty:
            return {"valid": False, "errors": ["No data provided"]}

        errors = []

        # Check minimum records
        if len(data) < min_records:
            errors.append(f"Insufficient data: {len(data)} records, minimum {min_records} required")

        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                errors.append(f"Missing required column: {col}")

        # Check for invalid values
        if (data['close'] <= 0).any():
            errors.append("Invalid close prices (zero or negative)")

        if (data['volume'] < 0).any():
            errors.append("Invalid volume (negative values)")

        # Check chronological order
        if not data.index.is_monotonic_increasing:
            errors.append("Data not in chronological order")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "record_count": len(data),
            "date_range": f"{data.index.min()} to {data.index.max()}"
        }

    def save_raw_data(self, data, symbol, filepath=None):
        """
        Save raw data as CSV for reference.

        Args:
            data: Raw DataFrame to save
            symbol: Asset symbol
            filepath: Optional custom filepath
        """
        if filepath is None:
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            os.makedirs(data_dir, exist_ok=True)
            filepath = os.path.join(data_dir, f"{symbol}_raw_data.csv")

        try:
            data.to_csv(filepath)
            self.logger.info(f"Saved raw data for {symbol} to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving raw data for {symbol}: {e}")

    def save_processed_csv(self, data, symbol, filepath=None):
        """
        Save processed data with technical indicators as CSV.

        Args:
            data: Processed DataFrame with technical indicators
            symbol: Asset symbol
            filepath: Optional custom filepath
        """
        if filepath is None:
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            os.makedirs(data_dir, exist_ok=True)
            filepath = os.path.join(data_dir, f"{symbol}_processed_data.csv")

        try:
            data.to_csv(filepath)
            self.logger.info(f"Saved processed data for {symbol} to {filepath} ({len(data.columns)} columns)")
        except Exception as e:
            self.logger.error(f"Error saving processed data for {symbol}: {e}")

    def save_data(self, data, symbol, filepath=None):
        """
        Save processed data to file.

        Args:
            data: DataFrame to save
            symbol: Asset symbol
            filepath: Optional custom filepath
        """
        if filepath is None:
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            os.makedirs(data_dir, exist_ok=True)

            # Try parquet first, fallback to CSV if pyarrow not available
            try:
                import pyarrow
                filepath = os.path.join(data_dir, f"{symbol}_processed.parquet")
            except ImportError:
                filepath = os.path.join(data_dir, f"{symbol}_processed.csv")

        try:
            if filepath.endswith('.parquet'):
                data.to_parquet(filepath)
            else:
                data.to_csv(filepath)
            self.logger.info(f"Saved data for {symbol} to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving data for {symbol}: {e}")

    def load_data(self, symbol, filepath=None):
        """
        Load processed data from file.

        Args:
            symbol: Asset symbol
            filepath: Optional custom filepath

        Returns:
            pandas.DataFrame: Loaded data
        """
        if filepath is None:
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')

            # Try parquet first, then CSV
            parquet_path = os.path.join(data_dir, f"{symbol}_processed.parquet")
            csv_path = os.path.join(data_dir, f"{symbol}_processed.csv")

            if os.path.exists(parquet_path):
                filepath = parquet_path
            elif os.path.exists(csv_path):
                filepath = csv_path
            else:
                self.logger.warning(f"No data file found for {symbol}")
                return None

        try:
            if filepath.endswith('.parquet'):
                data = pd.read_parquet(filepath)
            else:
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            self.logger.info(f"Loaded data for {symbol} from {filepath}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {e}")
            return None

    def get_training_data(self, symbol, days=365, use_cache=True):
        """
        Get complete training data pipeline for ML agent.

        Args:
            symbol: Asset symbol
            days: Historical data period
            use_cache: Whether to use cached data

        Returns:
            pandas.DataFrame: Processed training data
        """
        # Try to load from cache first
        if use_cache:
            cached_data = self.load_data(symbol)
            if cached_data is not None:
                validation = self.validate_data(cached_data)
                if validation['valid']:
                    self.logger.info(f"Using cached data for {symbol}")
                    return cached_data

        # Fetch fresh data
        self.logger.info(f"Fetching fresh data for {symbol}")
        raw_data = self.fetch_historical_data(symbol, days)

        if raw_data is None:
            return None

        # Process data
        processed_data = self.prepare_ml_data(raw_data)

        if processed_data is None:
            return None

        # Validate
        validation = self.validate_data(processed_data)
        if not validation['valid']:
            self.logger.error(f"Data validation failed for {symbol}: {validation['errors']}")
            return None

        # Save raw data as CSV for reference
        self.save_raw_data(raw_data, symbol)
    
        # Save processed data with technical indicators as CSV
        self.save_processed_csv(processed_data, symbol)
    
        # Save for future use
        self.save_data(processed_data, symbol)

        return processed_data

# Example usage
if __name__ == "__main__":
    manager = DataManager()

    # Test crypto data
    btc_data = manager.get_training_data('BTC', days=30)
    if btc_data is not None:
        print(f"BTC Training Data Shape: {btc_data.shape}")
        print(btc_data.head())
        print(f"Columns: {list(btc_data.columns)}")

    # Test stock data (requires Kite authentication)
    # nifty_data = manager.get_training_data('NIFTY', days=30)
    # if nifty_data is not None:
    #     print(f"NIFTY Training Data Shape: {nifty_data.shape}")
    #     print(nifty_data.head())