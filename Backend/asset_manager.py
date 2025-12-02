#!/usr/bin/env python3
"""
Asset Management System for ML Agent

Handles new asset requests intelligently, balancing performance with flexibility.
Provides tiered asset support with resource management and graceful degradation.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

from Agents.ml_agent import MLAgent
from services.data_manager import DataManager

class AssetRegistry:
    """Registry of supported assets with tiered classification."""

    def __init__(self):
        # Tier 1: Always available, pre-trained
        self.pre_trained = ['BTC', 'ETH', 'SOL']

        # Tier 2: Can be trained quickly (sufficient data, liquid)
        self.quick_train = [
            'BNB', 'ADA', 'LINK', 'DOT', 'AVAX', 'MATIC',
            'ALGO', 'VET', 'ICP', 'FIL', 'TRX', 'ETC'
        ]

        # Tier 3: Major stocks (Nifty 50 companies)
        self.stock_assets = [
            # Top 15 Nifty 50 companies as requested
            'RELIANCE', 'HDFCBANK', 'INFY', 'ICICIBANK', 'TCS',
            'KOTAKBANK', 'HINDUNILVR', 'ITC', 'BAJFINANCE', 'BHARTIARTL',
            'MARUTI', 'ASIANPAINT', 'SBIN', 'NTPC', 'TITAN',
            # Additional Nifty companies
            'COALINDIA', 'LT', 'TATASTEEL', 'UPL', 'WIPRO',
            # Nifty indices
            'NIFTY',  # Nifty 50 index
            # International stocks
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'
        ]

        # Tier 4: Unsupported (insufficient volatility or data)
        self.unsupported = [
            'USDT', 'USDC', 'BUSD', 'DAI',  # Stablecoins
            'SHIB', 'DOGE', 'PEPE'  # High volatility, low liquidity
        ]

        # All supported assets
        self.all_supported = (self.pre_trained + self.quick_train +
                            self.stock_assets + self.unsupported)

    def check_asset_status(self, symbol: str) -> str:
        """Check the support status of an asset."""
        symbol = symbol.upper()

        if symbol in self.pre_trained:
            return 'pre_trained'
        elif symbol in self.quick_train:
            return 'quick_train'
        elif symbol in self.stock_assets:
            return 'stock_asset'
        elif symbol in self.unsupported:
            return 'unsupported'
        else:
            return 'unknown'

    def get_asset_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about an asset."""
        status = self.check_asset_status(symbol)

        info = {
            'symbol': symbol.upper(),
            'status': status,
            'supported': status != 'unknown' and status != 'unsupported'
        }

        if status == 'pre_trained':
            info.update({
                'description': 'Always available, instant predictions',
                'estimated_time': '< 1 second',
                'storage_impact': 'None (already trained)'
            })
        elif status == 'quick_train':
            info.update({
                'description': 'Can be trained on-demand',
                'estimated_time': '2-3 minutes',
                'storage_impact': '~700KB'
            })
        elif status == 'stock_asset':
            info.update({
                'description': 'Stock asset (requires broker API)',
                'estimated_time': '2-4 minutes',
                'storage_impact': '~700KB'
            })
        elif status == 'unsupported':
            info.update({
                'description': 'Not supported for ML training',
                'reason': self._get_unsupported_reason(symbol)
            })
        else:
            info.update({
                'description': 'Unknown asset',
                'reason': 'Asset not in registry - may not have sufficient data'
            })

        return info

    def _get_unsupported_reason(self, symbol: str) -> str:
        """Get reason why an asset is unsupported."""
        symbol = symbol.upper()

        if symbol in ['USDT', 'USDC', 'BUSD', 'DAI']:
            return 'Stablecoin with insufficient price volatility for ML training'
        elif symbol in ['SHIB', 'DOGE', 'PEPE']:
            return 'High volatility, low liquidity, insufficient reliable data'
        else:
            return 'Asset characteristics not suitable for reliable ML predictions'

class ResourceManager:
    """Manages system resources for training operations."""

    def __init__(self):
        self.active_training = set()
        self.storage_limit = 50 * 1024 * 1024  # 50MB limit
        self.training_timeout = 300  # 5 minutes max per asset
        self.max_concurrent_training = 2  # Max 2 assets training simultaneously

        # Cache for resource checks
        self._last_storage_check = 0
        self._storage_usage = 0

    def can_train_asset(self, symbol: str) -> Tuple[bool, str]:
        """Check if an asset can be trained given current resources."""
        symbol = symbol.upper()

        # Check if already training
        if symbol in self.active_training:
            return False, "Training already in progress for this asset"

        # Check concurrent training limit
        if len(self.active_training) >= self.max_concurrent_training:
            return False, f"Too many concurrent trainings (max {self.max_concurrent_training})"

        # Check storage space
        current_usage = self._get_model_storage_usage()
        estimated_new_usage = current_usage + (700 * 1024)  # 700KB per asset

        if estimated_new_usage > self.storage_limit:
            return False, f"Storage limit exceeded ({current_usage/1024:.0f}KB + 700KB > {self.storage_limit/1024:.0f}KB)"

        # Check system resources (basic check)
        if not self._has_sufficient_resources():
            return False, "Insufficient system resources (CPU/memory)"

        return True, "OK"

    def start_training(self, symbol: str) -> None:
        """Mark training as started."""
        self.active_training.add(symbol.upper())

    def end_training(self, symbol: str) -> None:
        """Mark training as completed."""
        self.active_training.discard(symbol.upper())

    def get_active_training(self) -> List[str]:
        """Get list of assets currently being trained."""
        return list(self.active_training)

    def _get_model_storage_usage(self) -> int:
        """Get current model storage usage in bytes."""
        # Cache storage checks for 30 seconds
        now = time.time()
        if now - self._last_storage_check < 30:
            return self._storage_usage

        models_dir = os.path.join(os.path.dirname(__file__), 'data', 'models')
        if not os.path.exists(models_dir):
            return 0

        total_size = 0
        for filename in os.listdir(models_dir):
            if filename.endswith(('.h5', '.pkl')):
                filepath = os.path.join(models_dir, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except OSError:
                    pass  # File might be in use or deleted

        self._storage_usage = total_size
        self._last_storage_check = now
        return total_size

    def _has_sufficient_resources(self) -> bool:
        """Basic check for sufficient system resources."""
        try:
            # Simple CPU check (if psutil available)
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Allow training if CPU < 80% and memory > 20% available
            return cpu_percent < 80 and memory.available > (memory.total * 0.20)
        except ImportError:
            # If psutil not available, assume resources are OK
            return True

class SmartMLAgent:
    """
    Intelligent ML Agent that handles new assets gracefully.

    Provides tiered support: pre-trained (instant), quick-train (on-demand),
    and unsupported assets with clear feedback.
    """

    def __init__(self, asset: str, auto_train: bool = False,
                 max_training_time: int = 300, enable_fallback: bool = True):
        """
        Initialize Smart ML Agent.

        Args:
            asset: Asset symbol (e.g., 'BTC', 'ETH')
            auto_train: Whether to automatically train unsupported assets
            max_training_time: Maximum training time in seconds
            enable_fallback: Whether to provide fallback signals
        """
        self.asset = asset.upper()
        self.auto_train = auto_train
        self.max_training_time = max_training_time
        self.enable_fallback = enable_fallback

        # Setup logging first
        self.logger = logging.getLogger(f'SmartMLAgent_{self.asset}')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Initialize components
        self.registry = AssetRegistry()
        self.resource_manager = ResourceManager()
        self.data_manager = DataManager()

        # Determine asset status
        self.status = self.registry.check_asset_status(self.asset)
        self.asset_info = self.registry.get_asset_info(self.asset)

        # Initialize ML agent
        self.agent = None
        self._initialize_agent()

    def _initialize_agent(self) -> None:
        """Initialize the appropriate ML agent based on asset status."""
        if self.status == 'pre_trained':
            # Load existing model
            try:
                self.agent = MLAgent(asset=self.asset.lower())
                self.logger.info(f"Loaded pre-trained model for {self.asset}")
            except Exception as e:
                self.logger.error(f"Failed to load pre-trained model for {self.asset}: {e}")
                self.agent = None

        elif self.status in ['quick_train', 'stock_asset'] and self.auto_train:
            # Try to train on-demand
            if self._can_train_asset():
                success = self._train_asset_on_demand()
                if not success:
                    self.logger.warning(f"Failed to train model for {self.asset}")
            else:
                self.logger.info(f"Cannot train {self.asset} due to resource constraints")

        # For unsupported/unknown assets, agent remains None

    def _can_train_asset(self) -> bool:
        """Check if asset can be trained."""
        can_train, reason = self.resource_manager.can_train_asset(self.asset)
        if not can_train:
            self.logger.warning(f"Cannot train {self.asset}: {reason}")
        return can_train

    def _train_asset_on_demand(self) -> bool:
        """Train model for asset on-demand."""
        try:
            self.logger.info(f"Starting on-demand training for {self.asset}")
            self.resource_manager.start_training(self.asset)

            # Quick training config for on-demand
            quick_config = {
                'lstm_epochs': 20,  # Reduced from 50
                'xgb_estimators': 50,  # Reduced from 100
                'data_days': 365  # Reduced from 730
            }

            # Create temporary agent for training
            temp_agent = MLAgent(asset=self.asset.lower())

            # Fetch and prepare data
            self.logger.info(f"Fetching data for {self.asset}...")
            data = self.data_manager.get_training_data(self.asset, days=quick_config['data_days'])

            if data is None or len(data) < 100:
                self.logger.error(f"Insufficient data for {self.asset}")
                return False

            # Train models
            self.logger.info(f"Training LSTM model for {self.asset}...")
            lstm_result = temp_agent.train_lstm(data,
                                               epochs=quick_config['lstm_epochs'],
                                               verbose=0)

            if 'error' in lstm_result:
                self.logger.error(f"LSTM training failed for {self.asset}: {lstm_result['error']}")
                return False

            self.logger.info(f"Training XGBoost model for {self.asset}...")
            xgb_result = temp_agent.train_xgb(data, lookback=30)

            if 'error' in xgb_result:
                self.logger.error(f"XGBoost training failed for {self.asset}: {xgb_result['error']}")
                return False

            # Training successful
            self.agent = temp_agent
            self.logger.info(f"Successfully trained model for {self.asset}")
            return True

        except Exception as e:
            self.logger.error(f"Training failed for {self.asset}: {e}")
            return False
        finally:
            self.resource_manager.end_training(self.asset)

    def get_signal(self, data) -> Dict[str, Any]:
        """
        Get trading signal for the asset.

        Args:
            data: OHLCV DataFrame with technical indicators

        Returns:
            dict: Signal information
        """
        if self.agent is not None:
            # Use trained model
            return self.agent.get_ml_signal(self.asset, data)
        else:
            # Provide fallback response
            return self._get_fallback_signal()

    def _get_fallback_signal(self) -> Dict[str, Any]:
        """Provide fallback signal when no model is available."""
        base_response = {
            'ticker': self.asset,
            'signal': 'neutral',
            'confidence': 0.0,
            'fallback': True
        }

        if self.status == 'unsupported':
            base_response.update({
                'error': 'Asset not supported for ML predictions',
                'reason': self.asset_info.get('reason', 'Unknown'),
                'can_train': False
            })
        elif self.status == 'unknown':
            base_response.update({
                'error': 'Unknown asset - not in registry',
                'reason': 'Asset may not have sufficient historical data',
                'can_train': False
            })
        elif self.status in ['quick_train', 'stock_asset']:
            can_train, reason = self.resource_manager.can_train_asset(self.asset)
            base_response.update({
                'error': 'Model not trained',
                'reason': reason if not can_train else 'Training not requested',
                'can_train': can_train,
                'estimated_time': self.asset_info.get('estimated_time', 'Unknown'),
                'storage_impact': self.asset_info.get('storage_impact', 'Unknown')
            })
        else:
            base_response.update({
                'error': 'Model loading failed',
                'can_train': True
            })

        return base_response

    def get_asset_info(self) -> Dict[str, Any]:
        """Get information about the asset."""
        return self.asset_info

    def is_available(self) -> bool:
        """Check if the asset has a working model."""
        return self.agent is not None

    def get_training_status(self) -> Dict[str, Any]:
        """Get training status for the asset."""
        is_training = self.asset in self.resource_manager.get_active_training()

        return {
            'asset': self.asset,
            'status': self.status,
            'is_available': self.is_available(),
            'is_training': is_training,
            'can_train': self._can_train_asset(),
            'info': self.asset_info
        }

# Convenience functions for easy usage

def get_ml_signal(asset: str, data, auto_train: bool = False) -> Dict[str, Any]:
    """
    Convenience function to get ML signal for any asset.

    Args:
        asset: Asset symbol
        data: OHLCV DataFrame
        auto_train: Whether to train unsupported assets

    Returns:
        dict: Signal information
    """
    agent = SmartMLAgent(asset, auto_train=auto_train)
    return agent.get_signal(data)

def check_asset_availability(asset: str) -> Dict[str, Any]:
    """Check if an asset is available for ML predictions."""
    agent = SmartMLAgent(asset, auto_train=False)
    return agent.get_training_status()

# Example usage
if __name__ == "__main__":
    # Test pre-trained asset
    print("=== Testing BTC (pre-trained) ===")
    btc_agent = SmartMLAgent('BTC')
    print(f"BTC available: {btc_agent.is_available()}")
    print(f"BTC status: {btc_agent.get_training_status()}")

    # Test quick-train asset
    print("\n=== Testing ADA (quick-train) ===")
    ada_agent = SmartMLAgent('ADA', auto_train=False)
    print(f"ADA available: {ada_agent.is_available()}")
    print(f"ADA status: {ada_agent.get_training_status()}")

    # Test unsupported asset
    print("\n=== Testing USDT (unsupported) ===")
    usdt_agent = SmartMLAgent('USDT')
    print(f"USDT available: {usdt_agent.is_available()}")
    print(f"USDT status: {usdt_agent.get_training_status()}")

    # Test unknown asset
    print("\n=== Testing UNKNOWN (unknown) ===")
    unknown_agent = SmartMLAgent('UNKNOWN')
    print(f"UNKNOWN available: {unknown_agent.is_available()}")
    print(f"UNKNOWN status: {unknown_agent.get_training_status()}")