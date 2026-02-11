#!/usr/bin/env python3
"""
CoinDesk Data API Service
Provides comprehensive market data, historical candles, real-time prices, and news data
from CoinDesk Data API and News API
"""

import requests
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from enum import Enum

# Set up logging
logger = logging.getLogger('CoinDeskService')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TimeframeType(Enum):
    """Timeframe types for historical candle data."""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"


class CoinDeskService:
    """
    Service for fetching data from CoinDesk Data API.
    Provides market discovery, historical candles, real-time data, and news.
    """
    
    def __init__(self):
        """Initialize CoinDesk service with API configuration."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'api_keys.json')
        
        try:
            with open(config_path) as f:
                self.keys = json.load(f)
        except FileNotFoundError:
            logger.error(f"API keys config not found at {config_path}")
            self.keys = {}
        
        self.coindesk_api_key = self.keys.get('CoinDesk', '')
        self.newsapi_key = self.keys.get('newsapi_key', '')
        
        # API endpoints
        self.data_api_base = "https://data-api.coindesk.com"
        self.news_api_base = "https://newsapi.org/v2"
        self.websocket_url = "wss://data-streamer.coindesk.com"
        
        # Default market (Standard Digital Assets)
        self.default_market = "sda"
        
        # Cache for API responses
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 0.1  # 100ms between requests
        
        logger.info("✅ CoinDesk Service initialized")
    
    def _get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data if still valid."""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_duration:
                logger.debug(f"📦 Cache hit for {key}")
                return data
            else:
                del self.cache[key]
        return None
    
    def _set_cached_data(self, key: str, data: Any) -> None:
        """Cache API response."""
        self.cache[key] = (data, datetime.now())
    
    def _rate_limit(self, endpoint: str) -> None:
        """Implement rate limiting between requests."""
        if endpoint in self.last_request_time:
            elapsed = time.time() - self.last_request_time[endpoint]
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        self.last_request_time[endpoint] = time.time()
    
    def _make_request(self, method: str, url: str, params: Optional[Dict] = None,
                     headers: Optional[Dict] = None, timeout: int = 30) -> Dict:
        """
        Make HTTP request with error handling and rate limiting.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL to request
            params: Query parameters
            headers: HTTP headers
            timeout: Request timeout in seconds
            
        Returns:
            Response JSON or error dict
        """
        endpoint = url.split('/')[-1]
        self._rate_limit(endpoint)
        
        try:
            response = requests.request(
                method=method,
                url=url,
                params=params,
                headers=headers or {},
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request failed for {url}: {str(e)}")
            return {"error": str(e), "status": "failed"}
    
    # ==================== MARKET DISCOVERY ====================
    
    def get_available_markets(self) -> Dict[str, Any]:
        """
        Get list of all available markets (index families).
        
        Endpoint: GET /index/cc/v1/markets
        
        Returns:
            Dict with available markets
        """
        cache_key = "markets_list"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        url = f"{self.data_api_base}/index/cc/v1/markets"
        headers = {"Authorization": f"Bearer {self.coindesk_api_key}"}
        
        logger.info("📊 Fetching available markets...")
        result = self._make_request("GET", url, headers=headers)
        
        if "error" not in result:
            self._set_cached_data(cache_key, result)
            logger.info(f"✅ Retrieved {len(result.get('markets', []))} markets")
        
        return result
    
    def get_market_instruments(self, market: str = None, 
                              instrument_status: str = "ACTIVE") -> Dict[str, Any]:
        """
        Get list of instruments (symbols) in a market.
        
        Endpoint: GET /index/cc/v1/markets/instruments
        
        Args:
            market: Market identifier (e.g., 'sda' for Standard Digital Assets)
            instrument_status: Filter by status (ACTIVE, DEPRECATED, etc.)
            
        Returns:
            Dict with available instruments
        """
        market = market or self.default_market
        cache_key = f"instruments_{market}_{instrument_status}"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        url = f"{self.data_api_base}/index/cc/v1/markets/instruments"
        headers = {"Authorization": f"Bearer {self.coindesk_api_key}"}
        params = {
            "market": market,
            "instrument_status": instrument_status
        }
        
        logger.info(f"📋 Fetching instruments for market {market}...")
        result = self._make_request("GET", url, params=params, headers=headers)
        
        if "error" not in result:
            self._set_cached_data(cache_key, result)
            logger.info(f"✅ Retrieved {len(result.get('instruments', []))} instruments")
        
        return result
    
    # ==================== HISTORICAL CANDLE DATA ====================
    
    def get_historical_candles(self, instrument: str, market: str = None,
                              timeframe: TimeframeType = TimeframeType.HOUR,
                              limit: int = 500, start_time: str = None,
                              end_time: str = None) -> Dict[str, Any]:
        """
        Get historical OHLCV candle data.
        
        Args:
            instrument: Instrument symbol (e.g., 'XBX-USD', 'ETX-USD')
            market: Market identifier (default: sda)
            timeframe: Timeframe (minute, hour, day)
            limit: Number of candles to retrieve (max varies by timeframe)
            start_time: ISO format start time (optional)
            end_time: ISO format end time (optional)
            
        Returns:
            Dict with OHLCV candle data
        """
        market = market or self.default_market
        timeframe_str = timeframe.value
        
        # Map timeframe to endpoint
        endpoint_map = {
            TimeframeType.MINUTE: "minutes",
            TimeframeType.HOUR: "hours",
            TimeframeType.DAY: "days"
        }
        endpoint = endpoint_map[timeframe]
        
        cache_key = f"candles_{instrument}_{timeframe_str}_{limit}"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        url = f"{self.data_api_base}/index/cc/v1/historical/{endpoint}"
        headers = {"Authorization": f"Bearer {self.coindesk_api_key}"}
        params = {
            "market": market,
            "instrument": instrument,
            "limit": limit,
            "groups": "OHLC,VOLUME"
        }
        
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time
        
        logger.info(f"📈 Fetching {timeframe_str} candles for {instrument}...")
        result = self._make_request("GET", url, params=params, headers=headers)
        
        if "error" not in result:
            self._set_cached_data(cache_key, result)
            candle_count = len(result.get('candles', []))
            logger.info(f"✅ Retrieved {candle_count} {timeframe_str} candles")
        
        return result
    
    def get_minute_candles(self, instrument: str, market: str = None,
                          limit: int = 2000) -> Dict[str, Any]:
        """Get minute-level OHLCV candles."""
        return self.get_historical_candles(instrument, market, TimeframeType.MINUTE, limit)
    
    def get_hourly_candles(self, instrument: str, market: str = None,
                          limit: int = 500) -> Dict[str, Any]:
        """Get hourly OHLCV candles."""
        return self.get_historical_candles(instrument, market, TimeframeType.HOUR, limit)
    
    def get_daily_candles(self, instrument: str, market: str = None,
                         limit: int = 2000) -> Dict[str, Any]:
        """Get daily OHLCV candles."""
        return self.get_historical_candles(instrument, market, TimeframeType.DAY, limit)
    
    # ==================== REAL-TIME DATA ====================
    
    def get_latest_tick(self, instrument: str, market: str = None) -> Dict[str, Any]:
        """
        Get latest price tick for an instrument.
        Note: For real-time updates, use WebSocket instead.
        
        Args:
            instrument: Instrument symbol
            market: Market identifier
            
        Returns:
            Latest price data
        """
        market = market or self.default_market
        
        # Use latest candle as proxy for current price
        result = self.get_historical_candles(
            instrument=instrument,
            market=market,
            timeframe=TimeframeType.MINUTE,
            limit=1
        )
        
        if "error" not in result and "candles" in result and len(result["candles"]) > 0:
            latest_candle = result["candles"][0]
            return {
                "instrument": instrument,
                "market": market,
                "timestamp": latest_candle.get("timestamp"),
                "open": latest_candle.get("open"),
                "high": latest_candle.get("high"),
                "low": latest_candle.get("low"),
                "close": latest_candle.get("close"),
                "volume": latest_candle.get("volume")
            }
        
        return result
    
    # ==================== NEWS DATA ====================
    
    def get_news_articles(self, keywords: Optional[List[str]] = None,
                         limit: int = 20, sort_by: str = "publishedAt") -> Dict[str, Any]:
        """
        Get latest news articles about cryptocurrency.
        
        Args:
            keywords: List of keywords to search (e.g., ['bitcoin', 'ethereum'])
            limit: Number of articles to retrieve
            sort_by: Sort order (publishedAt, relevancy, popularity)
            
        Returns:
            Dict with news articles
        """
        if not self.newsapi_key:
            logger.warning("⚠️ NewsAPI key not configured")
            return {"error": "NewsAPI key not configured"}
        
        keywords = keywords or ['cryptocurrency', 'bitcoin', 'ethereum']
        query = ' OR '.join(keywords)
        
        cache_key = f"news_{query}_{limit}"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        url = f"{self.news_api_base}/everything"
        params = {
            "q": query,
            "pageSize": limit,
            "sortBy": sort_by,
            "apiKey": self.newsapi_key,
            "language": "en"
        }
        
        logger.info(f"📰 Fetching news articles for: {query}...")
        result = self._make_request("GET", url, params=params)
        
        if "error" not in result:
            self._set_cached_data(cache_key, result)
            article_count = len(result.get('articles', []))
            logger.info(f"✅ Retrieved {article_count} articles")
        
        return result
    
    def get_news_headlines(self, category: str = "business",
                          limit: int = 20) -> Dict[str, Any]:
        """
        Get top headlines related to crypto/finance.
        
        Args:
            category: News category (business, finance, technology)
            limit: Number of headlines
            
        Returns:
            Dict with top headlines
        """
        if not self.newsapi_key:
            logger.warning("⚠️ NewsAPI key not configured")
            return {"error": "NewsAPI key not configured"}
        
        cache_key = f"headlines_{category}_{limit}"
        cached = self._get_cached_data(cache_key)
        if cached:
            return cached
        
        url = f"{self.news_api_base}/top-headlines"
        params = {
            "category": category,
            "pageSize": limit,
            "apiKey": self.newsapi_key,
            "country": "us"
        }
        
        logger.info(f"📰 Fetching headlines for category: {category}...")
        result = self._make_request("GET", url, params=params)
        
        if "error" not in result:
            self._set_cached_data(cache_key, result)
            headline_count = len(result.get('articles', []))
            logger.info(f"✅ Retrieved {headline_count} headlines")
        
        return result
    
    # ==================== DATA CONVERSION & ANALYSIS ====================
    
    def candles_to_dataframe(self, candles: List[Dict]) -> pd.DataFrame:
        """
        Convert candle data to pandas DataFrame for technical analysis.
        
        Args:
            candles: List of candle dictionaries
            
        Returns:
            DataFrame with OHLCV data
        """
        if not candles:
            logger.warning("⚠️ No candles to convert")
            return pd.DataFrame()
        
        df = pd.DataFrame(candles)
        
        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        logger.info(f"✅ Converted {len(df)} candles to DataFrame")
        return df
    
    def get_price_changes(self, candles: List[Dict]) -> Dict[str, float]:
        """
        Calculate price changes from candle data.
        
        Args:
            candles: List of candle dictionaries
            
        Returns:
            Dict with price change metrics
        """
        if not candles or len(candles) < 2:
            return {"error": "Insufficient candle data"}
        
        df = self.candles_to_dataframe(candles)
        if df.empty:
            return {"error": "Could not convert candles"}
        
        first_close = df['close'].iloc[-1]  # Oldest close
        last_close = df['close'].iloc[0]    # Newest close
        
        price_change = last_close - first_close
        price_change_pct = (price_change / first_close) * 100 if first_close != 0 else 0
        
        return {
            "first_close": float(first_close),
            "last_close": float(last_close),
            "price_change": float(price_change),
            "price_change_pct": float(price_change_pct),
            "high": float(df['high'].max()),
            "low": float(df['low'].min()),
            "avg_volume": float(df['volume'].mean())
        }
    
    # ==================== COMPREHENSIVE DATA FETCH ====================
    
    def fetch_all_market_data(self, instrument: str = "XBX-USD",
                             market: str = None) -> Dict[str, Any]:
        """
        Fetch comprehensive market data for an instrument.
        Combines market info, historical candles, and latest prices.
        
        Args:
            instrument: Instrument symbol
            market: Market identifier
            
        Returns:
            Comprehensive market data package
        """
        market = market or self.default_market
        logger.info(f"🔄 Fetching comprehensive data for {instrument}...")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "instrument": instrument,
            "market": market,
            "markets": self.get_available_markets(),
            "instruments": self.get_market_instruments(market),
            "latest_tick": self.get_latest_tick(instrument, market),
            "hourly_candles": self.get_hourly_candles(instrument, market, limit=24),
            "daily_candles": self.get_daily_candles(instrument, market, limit=30),
            "news": self.get_news_articles([instrument.split('-')[0].lower()], limit=10)
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_items": len(self.cache),
            "cache_size_mb": sum(
                len(str(v[0])) / (1024 * 1024) 
                for v in self.cache.values()
            ),
            "cache_duration_seconds": self.cache_duration
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        logger.info("🧹 Cache cleared")


# Convenience function for quick usage
def create_coindesk_service() -> CoinDeskService:
    """Create and return a CoinDesk service instance."""
    return CoinDeskService()
