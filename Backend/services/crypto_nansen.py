#!/usr/bin/env python3
"""
Nansen API client for crypto on-chain analytics - UPDATED & CONNECTED
- Ensures BTC/ETH symbols are mapped to actual ERC-20 proxy contracts (WBTC/WETH)
- Avoids "addr=None, chain=None" by adding proper symbol → token → chain resolution
- Keeps request bodies aligned with Nansen v1 API expectations
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import os

from services.common_models import Candle, OnchainSignals


class SmartMoneyTrade:
    """Represents a smart money trader with PnL data"""
    def __init__(self, address: str, label: str, pnl_realized: float, pnl_unrealized: float, total_pnl: float, is_profitable: bool):
        self.address = address
        self.label = label
        self.pnl_realized = pnl_realized
        self.pnl_unrealized = pnl_unrealized
        self.total_pnl = total_pnl
        self.is_profitable = is_profitable


class WhaleHolder:
    """Represents a whale holder with balance and change data"""
    def __init__(self, address: str, label: str, balance_usd: float, balance_change_24h: float,
                 balance_change_7d: float, balance_change_30d: float, ownership_percentage: float, is_accumulating: bool):
        self.address = address
        self.label = label
        self.balance_usd = balance_usd
        self.balance_change_24h = balance_change_24h
        self.balance_change_7d = balance_change_7d
        self.balance_change_30d = balance_change_30d
        self.ownership_percentage = ownership_percentage
        self.is_accumulating = is_accumulating


class NansenClient:
    """Client for Nansen API v1 on-chain analytics"""

    def __init__(self):
        # Load API key from config file
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'api_keys.json')
        with open(config_path) as f:
            config = json.load(f)

        self.api_key = config.get('nansen_api_key', '')
        if not self.api_key:
            raise ValueError("Nansen API key not found in config file (key: 'nansen_api_key')")

        # Updated to v1 API base URL
        self.base_url = "https://api.nansen.ai/api/v1"
        self.headers = {
            "apiKey": self.api_key,
            "Content-Type": "application/json",
        }

        # Cache for API responses (5-minute cache)
        self.cache: Dict[str, Any] = {}
        self.cache_duration = 300  # 5 minutes

        # Setup logging
        self.logger = logging.getLogger('NansenClient')
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Token identifier mappings - ERC-20 tokens on Ethereum (all lowercase)
        # NOTE: We explicitly include WBTC/WETH and then alias BTC/ETH → WBTC/WETH
        self.token_addresses: Dict[str, str] = {
            # Stablecoins & majors
            'USDC': '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',
            'USDT': '0xdac17f958d2ee523a2206206994597c13d831ec7',
            'WBTC': '0x2260fac5e5542a773aa44fbcfedf7c193bc2c599',
            # Wrapped ETH (canonical WETH9)
            'WETH': '0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2',
            # Other blue chips
            'LINK': '0x514910771af9ca656af840dff83e8264ecf986ca',
            'UNI': '0x1f9840a85d5af5bf1d1762f925bdaddc4201f984',
            'AAVE': '0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9',
            'PEPE': '0x6982508145454ce325ddbe47a25d4ec3d2311933',
        }

        # Chain mappings
        # For BTC/ETH symbols we will alias them to their ERC-20 proxies:
        #   BTC → WBTC on Ethereum
        #   ETH → WETH on Ethereum (chain name: 'ethereum')
        self.chain_map: Dict[str, str] = {
            'USDC': 'ethereum',
            'USDT': 'ethereum',
            'WBTC': 'ethereum',
            'WETH': 'ethereum',
            'LINK': 'ethereum',
            'UNI': 'ethereum',
            'AAVE': 'ethereum',
            'PEPE': 'ethereum',
            'ETH': 'ethereum',  # native symbol → ethereum L1
            'BTC': 'ethereum',  # treat BTC as WBTC on ethereum for flows
            'SOL': 'solana',
            'MATIC': 'polygon',
            'AVAX': 'avalanche',
            'BNB': 'bnb',
        }

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
    # Symbol → token / chain resolution
    # ---------------------------------------------------------------------

    def get_token_address(self, symbol: str) -> Optional[str]:
        """Resolve a trading symbol (BTC/ETH/USDC/...) to an ERC-20 token address.

        BTC → WBTC
        ETH → WETH
        Others → direct lookup
        """
        symbol = symbol.upper()

        if symbol == 'BTC':
            return self.token_addresses.get('WBTC')
        if symbol == 'ETH':
            return self.token_addresses.get('WETH')

        return self.token_addresses.get(symbol)

    def get_chain_for_symbol(self, symbol: str) -> Optional[str]:
        """Resolve which L1 chain to query for a given symbol.

        BTC → same chain as WBTC (ethereum)
        ETH → ethereum
        """
        symbol = symbol.upper()

        if symbol == 'BTC':
            # Follow the same chain as WBTC
            return self.chain_map.get('WBTC', 'ethereum')
        if symbol == 'ETH':
            return self.chain_map.get('ETH', 'ethereum')

        return self.chain_map.get(symbol)

    # ---------------------------------------------------------------------
    # Low-level Nansen endpoints
    # ---------------------------------------------------------------------
    def get_smart_money_netflows(self, chains: List[str], token_address: Optional[str] = None) -> Dict[str, Any]:
        cache_key = f"smart_money_netflows_{'-'.join(chains)}_{token_address or 'all'}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached
     
        try:
            endpoint = f"{self.base_url}/smart-money/netflow"
         
        # Build the request body similar to the working example
            body: Dict[str, Any] = {
                "chains": chains,
                "pagination": {
                    "page": 1,
                    "per_page": 10
                    },
                "order_by": [
                    {
                        "direction": "DESC",
                        "field": "net_flow_30d_usd"
                    }
                ],
                "filters": {
                    "include_native_tokens": False,
                    "include_stablecoins": False
                }
            }
         
        # Add token_address to filters if provided
            if token_address:
                body["filters"]["token_address"] = [token_address]
         
            # self.logger.info(f"📡 API call: {endpoint}")
            # self.logger.debug(f"Request: {json.dumps(body, indent=2)}")
         
            response = requests.post(
                endpoint,
                headers=self.headers,
                json=body,
                timeout=10,
            )
         
            if response.status_code == 200:
                data = response.json()
                self._set_cached(cache_key, data)
                # self.logger.debug(f"Smart money netflows API response: {json.dumps(data, indent=2)}")
                self.logger.info(f"✅ Smart money netflows: {len(data.get('data', []))} tokens")
                return data

            self.logger.error(f"❌ API error {response.status_code}: {response.text}")
            return {}
         
        except Exception as e:
            self.logger.error(f"❌ Error in get_smart_money_netflows: {e}")
            return {}
    
    

    def get_token_flow_intelligence(self, chain: str, token_address: str) -> Dict[str, Any]:
        """
        Get summary of token flows across Smart Money, exchanges, etc.
        Endpoint: POST /api/v1/tgm/flow-intelligence
        Credits: 1
        """
        cache_key = f"flow_intelligence_{chain}_{token_address}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            endpoint = f"{self.base_url}/tgm/flow-intelligence"

            # Nansen docs: chain + token_address + timeframe
            body = {
                "chain": chain,
                "token_address": token_address,
            }

            response = requests.post(
                endpoint,
                headers=self.headers,
                json=body,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self._set_cached(cache_key, data)
                # self.logger.debug(f"Flow intelligence API response: {json.dumps(data, indent=2)}")
                self.logger.info("✅ Flow intelligence retrieved")
                return data

            self.logger.error(f"❌ API error {response.status_code}: {response.text}")
            return {}

        except Exception as e:
            self.logger.error(f"❌ Error: {e}")
            return {}

    def get_pnl_leaderboard(self, chain: str, token_address: str) -> Dict[str, Any]:
        """Get list of addresses and their total realised and unrealised PnL.

        Endpoint: POST /api/v1/tgm/pnl-leaderboard
        Credits: 5
        """
        cache_key = f"pnl_leaderboard_{chain}_{token_address}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            endpoint = f"{self.base_url}/tgm/pnl-leaderboard"
            current_datetime = datetime.now()
            today = current_datetime.strftime("%Y-%m-%d")
            past_30day = (current_datetime - timedelta(days=30)).strftime("%Y-%m-%d")

            body = {
                "chain": chain,
                "token_address": token_address,
                "date":{"from":past_30day,"to":today}
            }

            response = requests.post(
                endpoint,
                headers=self.headers,
                json=body,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                self._set_cached(cache_key, data)
                # self.logger.debug(f"PnL leaderboard API response: {json.dumps(data, indent=2)}")
                self.logger.info(f"✅ PnL leaderboard: {len(data.get('data', []))} traders")
                return data

            self.logger.error(f"❌ API error {response.status_code}: {response.text}")
            return {}

        except Exception as e:
            self.logger.error(f"❌ Error in get_pnl_leaderboard: {e}")
            return {}

    def get_token_holders(self, chain: str, token_address: str) -> Dict[str, Any]:
        """
        Get balance of top addresses, smart money, exchanges, etc.
        Endpoint: POST /api/v1/tgm/holders
        Credits: 5
        """
        cache_key = f"token_holders_{chain}_{token_address}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            endpoint = f"{self.base_url}/tgm/holders"

            # Minimal body per docs: chain + token_address
            body = {
                "chain": chain,
                "token_address": token_address,
            }

            response = requests.post(
                endpoint,
                headers=self.headers,
                json=body,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self._set_cached(cache_key, data)
                # self.logger.debug(f"Token holders API response: {json.dumps(data, indent=2)}")
                self.logger.info(f"✅ Token holders: {len(data.get('data', []))} addresses")
                return data

            self.logger.error(f"❌ API error {response.status_code}: {response.text}")
            return {}

        except Exception as e:
            self.logger.error(f"❌ Error: {e}")
            return {}

    def get_token_flows(self, chain: str, token_address: str) -> Dict[str, Any]:
        """Get total inflow/outflow from smart money, exchanges, etc.

        Endpoint: POST /api/v1/tgm/flows
        Credits: 1
        """
        cache_key = f"token_flows_{chain}_{token_address}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            endpoint = f"{self.base_url}/tgm/flows"
            current_datetime = datetime.now()
            today = current_datetime.strftime("%Y-%m-%d")
            past_30day = (current_datetime - timedelta(days=30)).strftime("%Y-%m-%d")
            body = {
                "chain": chain,
                "token_address": token_address,
                "date": {
                    "from": past_30day,
                    "to": today
                },
            }

            response = requests.post(
                endpoint,
                headers=self.headers,
                json=body,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                self._set_cached(cache_key, data)
                # self.logger.debug(f"Token flows API response: {json.dumps(data, indent=2)}")
                self.logger.info("✅ Token flows retrieved")
                return data

            self.logger.error(f"❌ API error {response.status_code}: {response.text}")
            return {}

        except Exception as e:
            self.logger.error(f"❌ Error in get_token_flows: {e}")
            return {}

    # ---------------------------------------------------------------------
    # High-level aggregation into OnchainSignals
    # ---------------------------------------------------------------------
    def get_comprehensive_onchain_data(self, symbol: str) -> OnchainSignals:
        """Get all on-chain signals combined into a standardized format with comprehensive metrics."""
        try:
            token_address = self.get_token_address(symbol)
            chain = self.get_chain_for_symbol(symbol)

            # self.logger.info(f"🔍 On-chain data for {symbol}: addr={token_address}, chain={chain}")

            if not token_address or not chain:
                # self.logger.warning(f"⚠️  No mapping for {symbol}, returning default OnchainSignals")
                return OnchainSignals()

        # Fetch all data (with error handling)
            netflows = self._safe_fetch(lambda: self.get_smart_money_netflows([chain], token_address), "netflows")
            flows = self._safe_fetch(lambda: self.get_token_flows(chain, token_address), "flows")
            flow_intel = self._safe_fetch(lambda: self.get_token_flow_intelligence(chain, token_address), "flow_intel")
            pnl_leaderboard = self._safe_fetch(lambda: self.get_pnl_leaderboard(chain, token_address), "pnl")
            holders = self._safe_fetch(lambda: self.get_token_holders(chain, token_address), "holders")

        # Track data quality
            data_sources = [netflows, flows, pnl_leaderboard, holders]
            data_quality = sum(1 for d in data_sources if d.get('data')) / len(data_sources)

        # ============================================================
        # 1. SMART MONEY NETFLOWS - Extract comprehensive metrics
        # ============================================================
            smart_money_inflow = 0.0
            smart_money_netflow_24h = 0.0
            top_sm_tokens = []
        
            if isinstance(netflows.get('data'), list):
                for item in netflows['data'][:10]:  # Top 10 tokens
                    net_24h = float(item.get('net_flow_24h_usd', 0) or 0)
                    smart_money_netflow_24h += net_24h
                
                    top_sm_tokens.append({
                        'token_symbol': item.get('token_symbol', 'UNKNOWN'),
                        'token_address': item.get('token_address', ''),
                        'net_flow_24h': net_24h,
                        'net_flow_7d': float(item.get('net_flow_7d_usd', 0) or 0),
                        'inflow_24h': float(item.get('inflow_24h_usd', 0) or 0),
                        'outflow_24h': float(item.get('outflow_24h_usd', 0) or 0),
                        })
            
            # Calculate momentum (7d trend vs 24h)
                flow_7d = sum(t.get('net_flow_7d', 0) for t in top_sm_tokens) / 7
                smart_money_momentum = smart_money_netflow_24h / max(abs(flow_7d), 1.0) if flow_7d != 0 else 0.0

        # ============================================================
        # 2. TOKEN FLOWS - Exchange and total metrics
        # ============================================================
            exchange_inflow = 0.0
            exchange_outflow = 0.0
            total_inflow = 0.0
            total_outflow = 0.0
        
            if flows.get('data'):
                flow_data = flows['data']
                if isinstance(flow_data, list) and len(flow_data) > 0:
                    flow_data = flow_data[0]
            
                exchange_inflow = float(flow_data.get('exchange_inflow', 0) or 0)
                exchange_outflow = float(flow_data.get('exchange_outflow', 0) or 0)
                total_inflow = float(flow_data.get('total_inflow', 0) or 0)
                total_outflow = float(flow_data.get('total_outflow', 0) or 0)

            net_exchange_flow = exchange_inflow - exchange_outflow
            exchange_flow_ratio = exchange_inflow / max(exchange_outflow, 1.0) if exchange_outflow > 0 else 0.0

        # ============================================================
        # 3. PNL LEADERBOARD - Smart trader analysis
        # ============================================================
            whale_tx_count = 0
            whale_volume_24h = 0.0
            total_pnl_realized = 0.0
            total_pnl_unrealized = 0.0
            profitable_traders = 0
            smart_traders = []
        
            if isinstance(pnl_leaderboard.get('data'), list):
                pnl_data = pnl_leaderboard['data']
                whale_tx_count = len(pnl_data)
            
                for entry in pnl_data[:20]:  # Top 20 traders
                    pnl_r = float(entry.get('pnl_usd_realised', 0) or 0)
                    pnl_u = float(entry.get('pnl_usd_unrealised', 0) or 0)
                    total_pnl = pnl_r + pnl_u
                
                    total_pnl_realized += pnl_r
                    total_pnl_unrealized += pnl_u
                    whale_volume_24h += abs(pnl_r)  # Use realized PnL as proxy for volume
                
                    if total_pnl > 0:
                        profitable_traders += 1
                
                    smart_traders.append(SmartMoneyTrade(
                        address=entry.get('address', ''),
                        label=entry.get('address_label', 'Unknown'),
                        pnl_realized=pnl_r,
                        pnl_unrealized=pnl_u,
                        total_pnl=total_pnl,
                        is_profitable=total_pnl > 0
                        ))
            
                profitable_pct = (profitable_traders / whale_tx_count * 100) if whale_tx_count > 0 else 0.0
                avg_trader_pnl = (total_pnl_realized + total_pnl_unrealized) / whale_tx_count if whale_tx_count > 0 else 0.0

        # ============================================================
        # 4. HOLDERS - Whale accumulation & distribution patterns
        # ============================================================
            holder_concentration = 0.0
            total_holdings_usd = 0.0
            top_10_holdings_usd = 0.0
            smart_money_holdings = 0.0
            exchange_holdings = 0.0
            defi_protocol_holdings = 0.0
            whale_holders = []
            accumulating_whales = 0
            top_10_change_24h = 0.0
            top_10_change_7d = 0.0
        
            if isinstance(holders.get('data'), list) and holders['data']:
                holders_data = holders['data']
            
            # Process top holders
                for i, holder in enumerate(holders_data[:50]):  # Top 50 holders
                    balance_usd = float(holder.get('value_usd', 0) or 0)
                    balance_change_24h = float(holder.get('balance_change_24h', 0) or 0)
                    balance_change_7d = float(holder.get('balance_change_7d', 0) or 0)
                    balance_change_30d = float(holder.get('balance_change_30d', 0) or 0)
                    ownership_pct = float(holder.get('ownership_percentage', 0) or 0)
                    label = holder.get('address_label', '').lower()
                
                    total_holdings_usd += balance_usd
                
                # Top 10 tracking
                    if i < 10:
                        top_10_holdings_usd += balance_usd
                        top_10_change_24h += balance_change_24h * balance_usd  # Weighted change
                        top_10_change_7d += balance_change_7d * balance_usd
                
                    # Categorize holdings
                    if 'smart' in label or 'fund' in label or 'trader' in label:
                        smart_money_holdings += balance_usd
                    elif 'exchange' in label or 'binance' in label or 'coinbase' in label:
                        exchange_holdings += balance_usd
                    elif 'aave' in label or 'compound' in label or 'maker' in label or 'uniswap' in label:
                        defi_protocol_holdings += balance_usd
                
                # Track accumulation (7d positive change)
                    is_accumulating = balance_change_7d > 0
                    if is_accumulating and i < 20:  # Only count top 20
                        accumulating_whales += 1
                
                # Store whale details (top 20)
                    if i < 20:
                        whale_holders.append(WhaleHolder(
                            address=holder.get('address', ''),
                            label=holder.get('address_label', 'Unknown'),
                            balance_usd=balance_usd,
                            balance_change_24h=balance_change_24h,
                            balance_change_7d=balance_change_7d,
                            balance_change_30d=balance_change_30d,
                            ownership_percentage=ownership_pct,
                            is_accumulating=is_accumulating
                        ))
            
            # Calculate metrics
                if total_holdings_usd > 0:
                    holder_concentration = min(top_10_holdings_usd / total_holdings_usd, 1.0)
                    top_10_change_24h = top_10_change_24h / top_10_holdings_usd if top_10_holdings_usd > 0 else 0.0
                    top_10_change_7d = top_10_change_7d / top_10_holdings_usd if top_10_holdings_usd > 0 else 0.0
            
                whale_accumulation_score = accumulating_whales / min(20, len(holders_data)) if holders_data else 0.0

        # ============================================================
        # 5. ADVANCED SENTIMENT SIGNALS
        # ============================================================
        
        # Multi-factor sentiment (0-1 scale)
            sentiment_factors = []
        
        # Factor 1: Net exchange flow (negative = bullish)
            if abs(net_exchange_flow) > 1000:
                ex_sentiment = max(0, min(1, 0.5 - (net_exchange_flow / 100_000_000)))
                sentiment_factors.append(('exchange_flow', ex_sentiment, 0.25))
        
        # Factor 2: Smart money netflow
            if abs(smart_money_netflow_24h) > 1000:
                sm_sentiment = max(0, min(1, 0.5 + (smart_money_netflow_24h / 50_000_000)))
                sentiment_factors.append(('smart_money', sm_sentiment, 0.30))
        
        # Factor 3: Whale accumulation
            if whale_holders:
                whale_sentiment = whale_accumulation_score
                sentiment_factors.append(('whale_accumulation', whale_sentiment, 0.25))
        
        # Factor 4: Profitability of traders
            if whale_tx_count > 0:
                profit_sentiment = profitable_pct / 100
                sentiment_factors.append(('trader_profit', profit_sentiment, 0.20))
        
        # Weighted average sentiment
            if sentiment_factors:
                onchain_sentiment = sum(s * w for _, s, w in sentiment_factors) / sum(w for _, _, w in sentiment_factors)
            else:
                onchain_sentiment = 0.5
        
        # Accumulation signal (categorical)
            accumulation_signal = "neutral"
            if onchain_sentiment >= 0.7 and whale_accumulation_score > 0.6:
                accumulation_signal = "strong_buy"
            elif onchain_sentiment >= 0.6:
                accumulation_signal = "buy"
            elif onchain_sentiment <= 0.3 and whale_accumulation_score < 0.4:
                accumulation_signal = "strong_sell"
            elif onchain_sentiment <= 0.4:
                accumulation_signal = "sell"
        
        # Smart money confidence (combines holdings + recent flows)
            sm_holdings_ratio = smart_money_holdings / max(total_holdings_usd, 1.0) if total_holdings_usd > 0 else 0.0
            sm_flow_signal = max(0, min(1, 0.5 + (smart_money_netflow_24h / 20_000_000)))
            smart_money_confidence = (sm_holdings_ratio * 0.6) + (sm_flow_signal * 0.4)

        # ============================================================
        # 6. LOGGING & RETURN
        # ============================================================

            return OnchainSignals(
            # Smart Money
                smart_money_inflow=smart_money_netflow_24h,
                smart_money_netflow_24h=smart_money_netflow_24h,
                smart_money_momentum=smart_money_momentum,
                top_smart_money_tokens=top_sm_tokens,
            
            # Exchange
                exchange_inflow=exchange_inflow,
                exchange_outflow=exchange_outflow,
                net_exchange_flow=net_exchange_flow,
                exchange_flow_ratio=exchange_flow_ratio,
            
            # Totals
                total_inflow=total_inflow,
                total_outflow=total_outflow,
            
            # Whales
                whale_tx_count=whale_tx_count,
                whale_volume_24h=whale_volume_24h,
                whale_holders=whale_holders,
                whale_accumulation_score=whale_accumulation_score,
            
            # Holders
                holder_concentration=holder_concentration,
                total_holdings_usd=total_holdings_usd,
                smart_money_holdings=smart_money_holdings,
                exchange_holdings=exchange_holdings,
                defi_protocol_holdings=defi_protocol_holdings,
                top_10_balance_change_24h=top_10_change_24h,
                top_10_balance_change_7d=top_10_change_7d,
            
            # PnL
                total_pnl_realized=total_pnl_realized,
                total_pnl_unrealized=total_pnl_unrealized,
                profitable_traders_count=profitable_traders,
                profitable_traders_percentage=profitable_pct,
                smart_traders=smart_traders,
                avg_trader_pnl=avg_trader_pnl,
            
            # Sentiment
                onchain_sentiment=onchain_sentiment,
                accumulation_signal=accumulation_signal,
                smart_money_confidence=smart_money_confidence,
            
            # Meta
                recent_whale_alerts=min(whale_tx_count // 5, 10),
                avg_daily_volume=abs(exchange_inflow + exchange_outflow) / 2 if (exchange_inflow or exchange_outflow) else 0.0,
                data_quality_score=data_quality,
            )

        except Exception as e:
            self.logger.error(f"❌ Error in get_comprehensive_onchain_data: {e}", exc_info=True)
            return OnchainSignals()

    def _safe_fetch(self, fetch_func, name: str) -> Dict[str, Any]:
        """Safely fetch data with error handling"""
        try:
            return fetch_func()
        except Exception as e:
            self.logger.warning(f"{name} fetch failed: {e}")
            return {}
    


class CryptoDataProvider:
    """Combined provider for crypto price data + Nansen on-chain analytics"""

    def __init__(self, price_api_key: Optional[str]):
        self.nansen_client = NansenClient()
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
        """Proxy into NansenClient for TechnicalAgent."""
        return self.nansen_client.get_comprehensive_onchain_data(symbol)


if __name__ == "__main__":
    try:
        client = NansenClient()
        signals_usdc = client.get_comprehensive_onchain_data('USDC')
        signals_btc = client.get_comprehensive_onchain_data('BTC')
        signals_eth = client.get_comprehensive_onchain_data('ETH')
    except Exception as e:
        pass
