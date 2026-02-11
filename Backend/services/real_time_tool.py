import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import pandas as pd

# Import indicator engine
try:
    from .indicator_engine import IndicatorEngine
    from .common_models import Candle
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    from indicator_engine import IndicatorEngine
    from common_models import Candle


class RealTimeTool:
    """Fetch real-time crypto market data from Binance public APIs."""

    def __init__(self, base_url: str = "https://api.binance.com/api/v3", timeout: int = 10) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.indicator_engine = IndicatorEngine()

        self.logger = logging.getLogger("RealTimeTool")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def fetch_market_data(
        self,
        symbol: str,
        intervals: Optional[List[str]] = None,
        limit: int = 50,
        compute_indicators: bool = True,
    ) -> Dict[str, Any]:
        """
        Fetch live price, OHLCV data for multiple intervals, and 24h volume.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT").
            intervals: List of kline intervals (default: ["1m", "5m", "15m"]).
            limit: Number of candles per interval.
            compute_indicators: Whether to compute technical indicators (default: True).

        Returns:
            Structured JSON-compatible dict with price, ohlcv, volume, and indicators data.
        """
        if not symbol:
            return {"error": "symbol is required"}

        normalized_symbol = symbol.upper()
        if intervals is None:
            intervals = ["1m", "5m", "15m"]

        self.logger.info("📡 Fetching real-time data for %s", normalized_symbol)

        price_data = self._fetch_price(normalized_symbol)
        volume_data = self._fetch_volume(normalized_symbol)
        ohlcv_data: Dict[str, Any] = {}

        for interval in intervals:
            ohlcv_data[interval] = self._fetch_ohlcv(normalized_symbol, interval, limit)
            
            if compute_indicators:
                ohlcv_data[interval]["indicators"] = self._compute_indicators(
                    ohlcv_data[interval]["candles"], interval
                )

        return {
            "symbol": normalized_symbol,
            "price": price_data,
            "volume": volume_data,
            "ohlcv": ohlcv_data,
            "source": "binance",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def _compute_indicators(self, candles: List[Dict], interval: str) -> Dict:
        """Compute technical indicators for given candles."""
        try:
            # Convert candles to Candle objects
            candle_objects = []
            for c in candles:
                candle_objects.append(Candle(
                    timestamp=datetime.fromisoformat(c["open_time"].replace("Z", "")),
                    open=c["open"],
                    high=c["high"],
                    low=c["low"],
                    close=c["close"],
                    volume=c["volume"]
                ))
            
            # Convert to DataFrame and compute indicators
            df = self.indicator_engine.candles_to_df(candle_objects)
            df_with_indicators = self.indicator_engine.compute_indicators(df)
            
            # Get latest indicators
            latest_indicators = self.indicator_engine.get_latest_indicator_snapshot(df_with_indicators)
            
            # Convert to dict
            return latest_indicators.dict()
            
        except Exception as e:
            self.logger.error("Failed to compute indicators: %s", e)
            return {"error": str(e)}

    def _fetch_price(self, symbol: str) -> Dict[str, Any]:
        url = f"{self.base_url}/ticker/price"
        params = {"symbol": symbol}
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return {
                "value": float(data["price"]),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        except requests.RequestException as exc:
            self.logger.error("Price request failed for %s: %s", symbol, exc)
            return {"error": f"price request failed: {exc}"}

    def _fetch_volume(self, symbol: str) -> Dict[str, Any]:
        url = f"{self.base_url}/ticker/24hr"
        params = {"symbol": symbol}
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            return {
                "base_24h": float(data.get("volume", 0.0)),
                "quote_24h": float(data.get("quoteVolume", 0.0)),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        except requests.RequestException as exc:
            self.logger.error("24h volume request failed for %s: %s", symbol, exc)
            return {"error": f"volume request failed: {exc}"}

    def _fetch_ohlcv(self, symbol: str, interval: str, limit: int) -> Dict[str, Any]:
        url = f"{self.base_url}/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        try:
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            self.logger.error("Klines request failed for %s (%s): %s", symbol, interval, exc)
            return {"error": f"ohlcv request failed: {exc}"}

        candles = [
            {
                "open_time": datetime.utcfromtimestamp(entry[0] / 1000).isoformat() + "Z",
                "open": float(entry[1]),
                "high": float(entry[2]),
                "low": float(entry[3]),
                "close": float(entry[4]),
                "volume": float(entry[5]),
                "close_time": datetime.utcfromtimestamp(entry[6] / 1000).isoformat() + "Z",
                "quote_volume": float(entry[7]),
                "trades": int(entry[8]),
            }
            for entry in data
        ]
        return {
            "interval": interval,
            "limit": limit,
            "candles": candles,
        }
