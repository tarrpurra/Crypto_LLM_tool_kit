#!/usr/bin/env python3
"""
CoinMarketCap API client for crypto market data and analytics
Provides market data, metadata, and on-chain signals using CoinMarketCap API
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import os

from .common_models import Candle, OnchainSignals


class CoinMarketCapClient:
    """Client for CoinMarketCap API v2 cryptocurrency data and analytics"""

    def __init__(self):
        # Load API key from config file
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'api_keys.json')
        with open(config_path) as f:
            config = json.load(f)

        self.api_key = config.get('coinmarketcap_api_key', '')
        if not self.api_key or self.api_key == 'YOUR_COINMARKETCAP_API_KEY':
            raise ValueError("CoinMarketCap API key not found in config file (key: 'coinmarketcap_api_key')")

        self.base_url = "https://pro-api.coinmarketcap.com/v2"
        self.headers = {
            "X-CMC_PRO_API_KEY": self.api_key,
            "Accept": "application/json",
        }

        # Cache for API responses (5-minute cache)
        self.cache: Dict[str, Any] = {}
        self.cache_duration = 300  # 5 minutes

        # Setup logging
        self.logger = logging.getLogger('CoinMarketCapClient')
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    # ---------------------------------------------------------------------
    # Caching helpers
    # ---------------------------------------------------------------------

    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached data if still valid"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.cache_duration:
                return data
            else:
                del self.cache[key]
        return None

    def _set_cached(self, key: str, data: Dict) -> None:
        """Cache API response"""
        self.cache[key] = (data, datetime.now())

    # ---------------------------------------------------------------------
    # Low-level CoinMarketCap endpoints
    # ---------------------------------------------------------------------

    def _safe_fetch(self, fetch_func, data_type: str) -> Dict[str, Any]:
        """Safe fetch with error handling"""
        try:
            return fetch_func()
        except Exception as e:
            self.logger.error(f"❌ Error fetching {data_type}: {e}")
            return {}

    def get_cryptocurrency_quotes(self, symbol: str) -> Dict[str, Any]:
        """Get current quotes and market data for a cryptocurrency"""
        cache_key = f"quotes_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            endpoint = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/latest"
            params = {
                "symbol": symbol.upper(),
                "convert": "USD"
            }

            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                self._set_cached(cache_key, data)
                self.logger.info(f"✅ Quotes retrieved for {symbol}")
                return data

            self.logger.error(f"❌ API error {response.status_code}: {response.text}")
            return {}

        except Exception as e:
            self.logger.error(f"❌ Error in get_cryptocurrency_quotes: {e}")
            return {}

    def get_cryptocurrency_metadata(self, symbol: str) -> Dict[str, Any]:
        """Get metadata for a cryptocurrency"""
        cache_key = f"metadata_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            endpoint = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/info"
            params = {
                "symbol": symbol.upper()
            }

            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                self._set_cached(cache_key, data)
                self.logger.info(f"✅ Metadata retrieved for {symbol}")
                return data

            self.logger.error(f"❌ API error {response.status_code}: {response.text}")
            return {}

        except Exception as e:
            self.logger.error(f"❌ Error in get_cryptocurrency_metadata: {e}")
            return {}

    def get_cryptocurrency_market_pairs(self, symbol: str) -> Dict[str, Any]:
        """Get market pairs and exchange information for a cryptocurrency"""
        cache_key = f"market_pairs_{symbol}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            endpoint = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/market-pairs/latest"
            params = {
                "symbol": symbol.upper(),
                "convert": "USD"
            }

            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                self._set_cached(cache_key, data)
                self.logger.info(f"✅ Market pairs retrieved for {symbol}")
                return data

            self.logger.error(f"❌ API error {response.status_code}: {response.text}")
            return {}

        except Exception as e:
            self.logger.error(f"❌ Error in get_cryptocurrency_market_pairs: {e}")
            return {}

    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global cryptocurrency market metrics"""
        cache_key = "global_metrics"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            endpoint = "https://pro-api.coinmarketcap.com/v2/global-metrics/quotes/latest"
            params = {
                "convert": "USD"
            }

            response = requests.get(
                endpoint,
                headers=self.headers,
                params=params,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                self._set_cached(cache_key, data)
                self.logger.info("✅ Global metrics retrieved")
                return data

            self.logger.error(f"❌ API error {response.status_code}: {response.text}")
            return {}

        except Exception as e:
            self.logger.error(f"❌ Error in get_global_metrics: {e}")
            return {}

    # ---------------------------------------------------------------------
    # High-level aggregation into OnchainSignals
    # ---------------------------------------------------------------------
    def get_comprehensive_onchain_data(self, symbol: str) -> OnchainSignals:
        """Get all market and on-chain signals combined into a standardized format."""
        try:
            # Fetch all data (with error handling)
            quotes = self._safe_fetch(lambda: self.get_cryptocurrency_quotes(symbol), "quotes")
            metadata = self._safe_fetch(lambda: self.get_cryptocurrency_metadata(symbol), "metadata")
            market_pairs = self._safe_fetch(lambda: self.get_cryptocurrency_market_pairs(symbol), "market_pairs")
            global_metrics = self._safe_fetch(lambda: self.get_global_metrics(), "global_metrics")

            # Track data quality
            data_sources = [quotes, metadata, market_pairs, global_metrics]
            data_quality = sum(1 for d in data_sources if d.get('data') or d.get('data') is not None) / len(data_sources)

            # Initialize signals
            signals = OnchainSignals()

            # ============================================================
            # 1. Market Data - Extract comprehensive metrics
            # ============================================================
            if quotes.get('data'):
                try:
                    # CoinMarketCap returns data by symbol (uppercase)
                    symbol_key = symbol.upper()
                    if symbol_key in quotes['data']:
                        quote_data = quotes['data'][symbol_key][0]
                        usd_quote = quote_data['quote']['USD']

                        # Volume and market data
                        signals.avg_daily_volume = usd_quote.get('volume_24h', 1000000.0)
                        
                        # Sentiment based on price change
                        price_change = usd_quote.get('percent_change_24h', 0)
                        if price_change > 2:
                            signals.onchain_sentiment = 0.7  # bullish
                        elif price_change < -2:
                            signals.onchain_sentiment = 0.3  # bearish
                        else:
                            signals.onchain_sentiment = 0.5  # neutral
                except Exception as e:
                    self.logger.error(f"❌ Error parsing quotes data: {e}")

            # ============================================================
            # 2. Market Pairs - Exchange and liquidity data
            # ============================================================
            if market_pairs.get('data'):
                try:
                    market_data = market_pairs['data']
                    signals.exchange_inflow = 0.0
                    signals.exchange_outflow = 0.0
                    
                    # Calculate exchange related metrics from market pairs
                    if 'num_market_pairs' in market_data:
                        signals.whale_tx_count = market_data['num_market_pairs']
                    
                    # Estimate whale volume from total volume and market pairs
                    if quotes.get('data'):
                        symbol_key = symbol.upper()
                        if symbol_key in quotes['data']:
                            quote_data = quotes['data'][symbol_key][0]
                            usd_quote = quote_data['quote']['USD']
                            total_volume = usd_quote.get('volume_24h', 0)
                            signals.whale_volume_24h = total_volume * 0.1  # Estimate 10% is whale volume
                except Exception as e:
                    self.logger.error(f"❌ Error parsing market pairs data: {e}")

            # ============================================================
            # 3. Metadata - Token and project information
            # ============================================================
            if metadata.get('data'):
                try:
                    symbol_key = symbol.upper()
                    if symbol_key in metadata['data']:
                        meta_data = metadata['data'][symbol_key]
                        
                        # Holder concentration estimate (based on circulating supply info)
                        if 'circulating_supply' in meta_data and 'total_supply' in meta_data:
                            if meta_data['total_supply'] > 0:
                                supply_ratio = meta_data['circulating_supply'] / meta_data['total_supply']
                                signals.holder_concentration = min(1.0, supply_ratio * 2)  # Scale to 0-1
                        
                        # Token age estimate (based on launch date if available)
                        if 'date_added' in meta_data:
                            try:
                                launch_date = datetime.strptime(meta_data['date_added'], '%Y-%m-%dT%H:%M:%S.%fZ')
                                days_since_launch = (datetime.now() - launch_date).days
                                signals.avg_token_age_days = min(365, days_since_launch)  # Cap at 1 year
                            except:
                                pass
                except Exception as e:
                    self.logger.error(f"❌ Error parsing metadata: {e}")

            # ============================================================
            # 4. Global Metrics - Market conditions
            # ============================================================
            if global_metrics.get('data'):
                try:
                    global_data = global_metrics['data']
                    usd_global = global_data['quote']['USD']
                    
                    # Smart money inflow estimate based on global market conditions
                    if usd_global.get('total_volume_24h') > 0:
                        signals.smart_money_inflow = usd_global['total_volume_24h'] * 0.05  # Estimate 5% is smart money
                except Exception as e:
                    self.logger.error(f"❌ Error parsing global metrics: {e}")

            # ============================================================
            # 5. Estimated whale alerts (based on volatility and volume)
            # ============================================================
            if quotes.get('data'):
                try:
                    symbol_key = symbol.upper()
                    if symbol_key in quotes['data']:
                        quote_data = quotes['data'][symbol_key][0]
                        usd_quote = quote_data['quote']['USD']
                        
                        # Detect whale activity based on large price movements or volume spikes
                        price_volatility = abs(usd_quote.get('percent_change_24h', 0))
                        volume_change = abs(usd_quote.get('volume_change_24h', 0))
                        
                        if price_volatility > 5 or volume_change > 100:
                            signals.recent_whale_alerts = 1
                        elif price_volatility > 2 or volume_change > 50:
                            signals.recent_whale_alerts = 0
                        else:
                            signals.recent_whale_alerts = 0
                except Exception as e:
                    self.logger.error(f"❌ Error calculating whale alerts: {e}")

            self.logger.info(f"✅ On-chain signals generated for {symbol}")
            return signals

        except Exception as e:
            self.logger.error(f"❌ Error in get_comprehensive_onchain_data: {e}")
            return OnchainSignals()


class CryptoDataProvider:
    """Combined provider for crypto price data + CoinMarketCap market analytics"""

    def __init__(self, price_api_key: Optional[str]):
        self.coinmarketcap_client = CoinMarketCapClient()
        self.price_api_key = price_api_key

        # Setup logging
        self.logger = logging.getLogger('CryptoDataProvider')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def get_candles(self, symbol: str, timeframe: str, lookback: int) -> List[Candle]:
        """Get real price candles from Binance API. Returns empty list on failure."""
        # Normalize symbol - replace any slashes with nothing (e.g., "BTC/USDT" -> "BTCUSDT")
        symbol = symbol.replace('/', '').replace('\\', '').upper()
        try:
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w', '1M': '1M',
            }
            interval = interval_map.get(timeframe, '1d')

            end_time = datetime.now()
            start_time = end_time - timedelta(days=lookback)

            params = {
                'symbol': f"{symbol}USDT",
                'interval': interval,
                'startTime': int(start_time.timestamp() * 1000),
                'endTime': int(end_time.timestamp() * 1000),
                'limit': 1000,
            }

            response = requests.get(
                "https://api.binance.com/api/v3/klines",
                params=params,
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                candles: List[Candle] = []
                for item in data:
                    candles.append(Candle(
                        timestamp=datetime.fromtimestamp(item[0] / 1000),
                        open=float(item[1]),
                        high=float(item[2]),
                        low=float(item[3]),
                        close=float(item[4]),
                        volume=float(item[5]),
                    ))
                return candles
            else:
                self.logger.error(f"❌ Binance API error {response.status_code}: {response.text}")
                return []

        except Exception as e:
            self.logger.error(f"❌ Error getting candles from Binance: {e}")
            return []

    def get_onchain_signals(self, symbol: str) -> OnchainSignals:
        """Proxy into CoinMarketCapClient for TechnicalAgent."""
        return self.coinmarketcap_client.get_comprehensive_onchain_data(symbol)


if __name__ == "__main__":
    try:
        client = CoinMarketCapClient()
        signals_usdc = client.get_comprehensive_onchain_data('USDC')
        signals_btc = client.get_comprehensive_onchain_data('BTC')
        signals_eth = client.get_comprehensive_onchain_data('ETH')
        
        print("\n=== CoinMarketCap API Test ===")
        print(f"USDC Signals: {signals_usdc}")
        print(f"BTC Signals: {signals_btc}")
        print(f"ETH Signals: {signals_eth}")
    except Exception as e:
        print(f"Error: {e}")
