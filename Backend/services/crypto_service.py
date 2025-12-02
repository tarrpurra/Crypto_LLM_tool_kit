import requests
import json
import pandas as pd
import logging
from datetime import datetime, timedelta
import os
import time

class CryptoService:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'api_keys.json')
        with open(config_path) as f:
            self.keys = json.load(f)

        self.coinapi_key = self.keys.get('coinapi_key', '')
        self.binance_key = self.keys.get('binance_api_key', '')

        # API endpoints
        self.coinapi_base = "https://rest.coinapi.io/v1"
        self.binance_base = "https://api.binance.com/api/v3"

        # Set up logging
        self.logger = logging.getLogger('CryptoService')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Cache for API responses
        self.cache = {}
        self.cache_duration = 300  # 5 minutes

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

    def get_historical_data_coinapi(self, symbol, days=365, interval='1DAY'):
        """
        Fetch historical data from CoinAPI.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            days: Number of days of historical data
            interval: Time interval ('1DAY', '1HRS', '1MIN')

        Returns:
            pandas.DataFrame with OHLCV data
        """
        if not self.coinapi_key or self.coinapi_key == 'YOUR_COINAPI_KEY_HERE':
            self.logger.warning("CoinAPI key not configured")
            return None

        cache_key = f"coinapi_{symbol}_{days}_{interval}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            self.logger.info(f"Returning cached CoinAPI data for {symbol}")
            return cached_data

        # Map symbol to CoinAPI format
        symbol_map = {
            'BTC': 'BITSTAMP_SPOT_BTC_USD',
            'ETH': 'BITSTAMP_SPOT_ETH_USD',
            'BNB': 'BINANCE_SPOT_BNB_USDT',
            'ADA': 'BINANCE_SPOT_ADA_USDT'
        }

        api_symbol = symbol_map.get(symbol, f'BINANCE_SPOT_{symbol}_USDT')

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        url = f"{self.coinapi_base}/ohlcv/{api_symbol}/history"
        params = {
            'period_id': interval,
            'time_start': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'time_end': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
            'limit': 10000
        }
        headers = {'X-CoinAPI-Key': self.coinapi_key}

        try:
            self.logger.info(f"Fetching {symbol} data from CoinAPI for {days} days")
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not data:
                self.logger.warning(f"No data received from CoinAPI for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['time_period_start'])
            df = df.set_index('timestamp')
            df = df.rename(columns={
                'price_open': 'open',
                'price_high': 'high',
                'price_low': 'low',
                'price_close': 'close',
                'volume_traded': 'volume'
            })
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.sort_index()

            self._set_cached_data(cache_key, df)
            self.logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df

        except requests.RequestException as e:
            self.logger.error(f"CoinAPI request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing CoinAPI data: {e}")
            return None

    def get_historical_data_binance(self, symbol, days=365, interval='1d'):
        """
        Fetch historical data from Binance API.

        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            days: Number of days of historical data
            interval: Kline interval ('1d', '1h', '1m')

        Returns:
            pandas.DataFrame with OHLCV data
        """
        cache_key = f"binance_{symbol}_{days}_{interval}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            self.logger.info(f"Returning cached Binance data for {symbol}")
            return cached_data

        # Ensure symbol is in correct format
        if not symbol.endswith('USDT') and not symbol.endswith('USD'):
            symbol = f"{symbol}USDT"

        # Calculate timestamps
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

        url = f"{self.binance_base}/klines"
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }

        try:
            self.logger.info(f"Fetching {symbol} data from Binance for {days} days")
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if not data:
                self.logger.warning(f"No data received from Binance for {symbol}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df = df.astype(float)
            df = df.sort_index()

            self._set_cached_data(cache_key, df)
            self.logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df

        except requests.RequestException as e:
            self.logger.error(f"Binance request failed: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error processing Binance data: {e}")
            return None

    def get_historical_data(self, symbol, days=365, interval='1d', source='binance'):
        """
        Unified method to get historical data.

        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            days: Number of days of historical data
            interval: Time interval
            source: 'coinapi' or 'binance'

        Returns:
            pandas.DataFrame with OHLCV data
        """
        if source.lower() == 'coinapi':
            # Map interval
            interval_map = {'1d': '1DAY', '1h': '1HRS', '1m': '1MIN'}
            api_interval = interval_map.get(interval, '1DAY')
            return self.get_historical_data_coinapi(symbol, days, api_interval)
        else:
            return self.get_historical_data_binance(symbol, days, interval)

    def get_current_price(self, symbol, source='binance'):
        """
        Get current price for a symbol.

        Args:
            symbol: Crypto symbol

        Returns:
            float: Current price
        """
        try:
            if source.lower() == 'binance':
                if not symbol.endswith('USDT'):
                    symbol = f"{symbol}USDT"

                url = f"{self.binance_base}/ticker/price"
                params = {'symbol': symbol}
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                return float(data['price'])
            else:
                # CoinAPI current price
                if not self.coinapi_key or self.coinapi_key == 'YOUR_COINAPI_KEY_HERE':
                    return None

                symbol_map = {
                    'BTC': 'BITSTAMP_SPOT_BTC_USD',
                    'ETH': 'BITSTAMP_SPOT_ETH_USD'
                }
                api_symbol = symbol_map.get(symbol, f'BINANCE_SPOT_{symbol}_USDT')

                url = f"{self.coinapi_base}/exchangerate/{api_symbol}"
                headers = {'X-CoinAPI-Key': self.coinapi_key}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                data = response.json()
                return data.get('rate')

        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}: {e}")
            return None

# Example usage
if __name__ == "__main__":
    service = CryptoService()

    # Test Binance (free)
    btc_data = service.get_historical_data('BTC', days=30, source='binance')
    if btc_data is not None:
        print(f"BTC Data shape: {btc_data.shape}")
        print(btc_data.head())
    else:
        print("Failed to fetch BTC data")

    # Test CoinAPI (requires key)
    eth_data = service.get_historical_data('ETH', days=30, source='coinapi')
    if eth_data is not None:
        print(f"ETH Data shape: {eth_data.shape}")
        print(eth_data.head())
    else:
        print("Failed to fetch ETH data from CoinAPI")