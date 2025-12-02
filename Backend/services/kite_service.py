import requests
import json
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
from kiteconnect import KiteConnect

class KiteService:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'api_keys.json')
        with open(config_path) as f:
            self.keys = json.load(f)

        self.api_key = self.keys.get('zerodha_api_key', '')
        self.api_secret = self.keys.get('zerodha_secret', '')

        # Kite Connect setup
        self.kite = None
        self.access_token = None

        # Load access token if exists
        self.load_access_token()

        # Set up logging
        self.logger = logging.getLogger('KiteService')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Cache for API responses
        self.cache = {}
        self.cache_duration = 300  # 5 minutes

        # Instrument tokens cache
        self.instrument_cache = {}

    def load_access_token(self):
        """Load saved access token."""
        token_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'kite_token.json')
        if os.path.exists(token_file):
            try:
                with open(token_file) as f:
                    token_data = json.load(f)
                self.access_token = token_data.get('access_token')
                if self.api_key and self.api_secret:
                    self.kite = KiteConnect(api_key=self.api_key)
                    self.kite.set_access_token(self.access_token)
                self.logger.info("Loaded saved access token")
            except Exception as e:
                self.logger.error(f"Error loading access token: {e}")

    def save_access_token(self, access_token):
        """Save access token for future use."""
        token_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'kite_token.json')
        os.makedirs(os.path.dirname(token_file), exist_ok=True)
        with open(token_file, 'w') as f:
            json.dump({'access_token': access_token}, f)
        self.logger.info("Saved access token")

    def authenticate(self, request_token):
        """
        Complete authentication with request token.

        Args:
            request_token: Request token from Kite login

        Returns:
            bool: Success status
        """
        try:
            if not self.api_key or not self.api_secret:
                self.logger.error("API key or secret not configured")
                return False

            self.kite = KiteConnect(api_key=self.api_key)
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            self.save_access_token(self.access_token)
            self.logger.info("Authentication successful")
            return True
        except Exception as e:
            self.logger.error(f"Authentication failed: {e}")
            return False

    def get_login_url(self):
        """Get Kite login URL for authentication."""
        if not self.api_key:
            return None
        self.kite = KiteConnect(api_key=self.api_key)
        return self.kite.login_url()

    def _get_cached_data(self, key):
        """Get cached data if still valid."""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.cache_duration:
                return data
            else:
                del self.cache[key]
        return None

    def _set_cached_data(self, key, data):
        """Cache API response."""
        self.cache[key] = (data, datetime.now())

    def get_instrument_token(self, symbol):
        """
        Get instrument token for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'NIFTY', 'RELIANCE')

        Returns:
            int: Instrument token
        """
        if symbol in self.instrument_cache:
            return self.instrument_cache[symbol]

        try:
            if not self.kite:
                self.logger.error("Kite not authenticated")
                return None

            # Get instruments
            instruments = self.kite.instruments()

            # Find instrument
            for instrument in instruments:
                if instrument['tradingsymbol'] == symbol and instrument['exchange'] == 'NSE':
                    token = instrument['instrument_token']
                    self.instrument_cache[symbol] = token
                    return token

            self.logger.warning(f"Instrument not found: {symbol}")
            return None

        except Exception as e:
            self.logger.error(f"Error getting instrument token for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol, days=365, interval='day'):
        """
        Fetch historical data from Kite.

        Args:
            symbol: Stock symbol (e.g., 'NIFTY', 'RELIANCE')
            days: Number of days of historical data
            interval: Time interval ('day', 'hour', 'minute')

        Returns:
            pandas.DataFrame with OHLCV data
        """
        if not self.kite or not self.access_token:
            self.logger.error("Kite not authenticated. Please authenticate first.")
            return None

        cache_key = f"kite_{symbol}_{days}_{interval}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            self.logger.info(f"Returning cached Kite data for {symbol}")
            return cached_data

        # Get instrument token
        instrument_token = self.get_instrument_token(symbol)
        if not instrument_token:
            return None

        # Map interval
        interval_map = {
            'minute': 'minute',
            'hour': 'hour',
            'day': 'day'
        }
        kite_interval = interval_map.get(interval, 'day')

        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        try:
            self.logger.info(f"Fetching {symbol} data from Kite for {days} days")

            # Fetch historical data
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=kite_interval
            )

            if not data:
                self.logger.warning(f"No data received from Kite for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['date'])
            df = df.set_index('timestamp')
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.sort_index()

            self._set_cached_data(cache_key, df)
            self.logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Error fetching Kite data for {symbol}: {e}")
            return None

    def get_current_price(self, symbol):
        """
        Get current price for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            float: Current price
        """
        try:
            if not self.kite:
                self.logger.error("Kite not authenticated")
                return None

            instrument_token = self.get_instrument_token(symbol)
            if not instrument_token:
                return None

            # Get quote
            quote = self.kite.quote(instrument_token)
            return quote[str(instrument_token)]['last_price']

        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None

    def get_market_data(self, symbols):
        """
        Get current market data for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            dict: Market data
        """
        try:
            if not self.kite:
                return {}

            instrument_tokens = []
            for symbol in symbols:
                token = self.get_instrument_token(symbol)
                if token:
                    instrument_tokens.append(token)

            if not instrument_tokens:
                return {}

            quotes = self.kite.quote(instrument_tokens)
            return quotes

        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    service = KiteService()

    # Check if authenticated
    if not service.kite or not service.access_token:
        print("Please authenticate first:")
        print("1. Get login URL:", service.get_login_url())
        print("2. Complete login and get request token")
        print("3. Call service.authenticate(request_token)")
    else:
        # Test with NIFTY
        nifty_data = service.get_historical_data('NIFTY', days=30)
        if nifty_data is not None:
            print(f"NIFTY Data shape: {nifty_data.shape}")
            print(nifty_data.head())
        else:
            print("Failed to fetch NIFTY data")