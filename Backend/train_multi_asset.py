#!/usr/bin/env python3
"""
Multi-Asset ML Model Training Script

This script trains separate ML models for multiple cryptocurrencies.
Each asset gets its own specialized LSTM and XGBoost models.
"""

import sys
import os
import json
import logging
import time
from datetime import datetime

# Add Backend to path
sys.path.append(os.path.dirname(__file__))

from services.data_manager import DataManager
from Agents.ml_agent import MLAgent

def setup_logging():
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('multi_asset_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('MultiAsset_Training')

def print_header(title):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f" 🚀 {title}")
    print("="*70)

def print_section(title):
    """Print formatted section header."""
    print(f"\n--- {title} ---")

def validate_environment():
    """Validate that all required components are available."""
    print_section("Environment Validation")

    # Check API keys
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'api_keys.json')
    try:
        with open(config_path) as f:
            keys = json.load(f)

        binance_key = keys.get('binance_api_key', '')
        if not binance_key or binance_key == 'YOUR_BINANCE_API_KEY_HERE':
            print("❌ Binance API key not configured")
            return False
        else:
            print("✅ Binance API key configured")
    except Exception as e:
        print(f"❌ Error reading API keys: {e}")
        return False

    # Check if required packages can be imported
    required_packages = ['tensorflow', 'xgboost', 'pandas', 'numpy', 'sklearn', 'talib']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not available")
            return False

    print("✅ Environment validation passed")
    return True

def train_asset_model(symbol, logger):
    """Train ML models for a specific asset."""
    print_header(f"Training Models for {symbol}")

    try:
        manager = DataManager()

        # Fetch asset data
        print(f"📡 Fetching {symbol} historical data...")
        start_time = time.time()
        data = manager.get_training_data(symbol, days=730, use_cache=False)
        fetch_time = time.time() - start_time

        if data is None or data.empty:
            print(f"❌ Failed to fetch data for {symbol}")
            return None

        print("✅ Data fetched successfully")
        print(f"   📊 Records: {len(data)}")
        print(f"   📅 Date range: {data.index.min()} to {data.index.max()}")
        print(f"   💰 Price range: ${data['close'].min():.4f} - ${data['close'].max():.4f}")
        print(f"   📊 Avg price: ${data['close'].mean():.2f}")
        print(f"   📈 Latest price: ${data['close'].iloc[-1]:.4f}")
        print(f"   ⏱️  Fetch time: {fetch_time:.2f}s")

        # Validate data
        validation = manager.validate_data(data)
        if not validation['valid']:
            print(f"⚠️  Data validation issues: {validation['errors']}")

        # Initialize asset-specific ML agent
        agent = MLAgent(asset=symbol)

        # Train LSTM model
        print(f"\n🧠 Training LSTM model for {symbol}...")
        lstm_start = time.time()
        lstm_result = agent.train_lstm(data, lookback=60, epochs=50, batch_size=32)
        lstm_time = time.time() - lstm_start

        if 'error' in lstm_result:
            print(f"❌ LSTM training failed: {lstm_result['error']}")
            return None

        print("✅ LSTM training completed")
        print(f"   📉 Train Loss: {lstm_result['train_loss']:.4f}")
        print(f"   📊 Val Loss: {lstm_result['val_loss']:.4f}")
        print(f"   ⏱️  Training time: {lstm_time:.2f}s")
        print(f"   📈 Epochs trained: {lstm_result['epochs_trained']}")

        # Train XGBoost model
        print(f"\n🌳 Training XGBoost model for {symbol}...")
        xgb_start = time.time()
        xgb_result = agent.train_xgb(data, lookback=30)
        xgb_time = time.time() - xgb_start

        if 'error' in xgb_result:
            print(f"❌ XGBoost training failed: {xgb_result['error']}")
            return None

        print("✅ XGBoost training completed")
        print(f"   📊 Train RMSE: {xgb_result['train_rmse']:.4f}")
        print(f"   📈 Test RMSE: {xgb_result['test_rmse']:.4f}")
        print(f"   ⏱️  Training time: {xgb_time:.2f}s")

        # Test predictions
        print(f"\n🔮 Testing {symbol} predictions...")
        signal = agent.get_ml_signal(symbol, data, future_steps=5)

        if 'error' in signal:
            print(f"❌ Signal generation failed: {signal['error']}")
        else:
            print("✅ Signal generated successfully")
            print(f"   🎯 Current price: ${signal['current_price']:.4f}")
            print(f"   🔮 Predicted price: ${signal['predicted_price']:.4f}")
            print(f"   🚀 Signal: {signal['signal'].upper()}")
            print(f"   📊 Confidence: {signal['confidence']:.2f}")
            print(f"   🤖 Models used: {signal['models_used']}")

        total_time = time.time() - start_time
        print(f"   ⏱️  Total time: {total_time:.2f}s")

        logger.info(f"Completed training for {symbol}: LSTM_loss={lstm_result['train_loss']:.4f}, XGB_RMSE={xgb_result['train_rmse']:.4f}")

        return {
            'symbol': symbol,
            'data_records': len(data),
            'lstm_result': lstm_result,
            'xgb_result': xgb_result,
            'signal': signal if 'error' not in signal else None,
            'total_time': total_time
        }

    except Exception as e:
        print(f"❌ Error training {symbol}: {e}")
        logger.error(f"Training failed for {symbol}: {e}")
        return None

def validate_trained_models(assets):
    """Validate that all models were created successfully."""
    print_section("Model Validation")

    models_dir = os.path.join(os.path.dirname(__file__), 'data', 'models')
    expected_files = []

    for asset in assets:
        asset_lower = asset.lower()
        expected_files.extend([
            f'lstm_{asset_lower}.h5',
            f'xgb_{asset_lower}.pkl',
            f'scaler_{asset_lower}.pkl'
        ])

    all_present = True
    for filename in expected_files:
        filepath = os.path.join(models_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"✅ {filename}: {size:.1f} KB")
        else:
            print(f"❌ Missing: {filename}")
            all_present = False

    if all_present:
        print("✅ All model files present")
    else:
        print("⚠️  Some model files missing")

    return all_present

def main():
    """Main training execution for multiple assets."""
    print_header("Multi-Asset ML Model Training")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger = setup_logging()

    # Assets to train
    target_assets = ['ETH', 'SOL', 'USDT']

    try:
            # Step 1: Environment validation
        if not validate_environment():
            print("❌ Environment validation failed. Please check configuration.")
            return False

            # Step 2: Train models for each asset
        results = {}
        successful_training = 0

        for symbol in target_assets:
            result = train_asset_model(symbol, logger)
        
            if result:
                results[symbol] = result
                successful_training += 1
            else:
                results[symbol] = {'error': 'Training failed'}

            # Step 3: Validate all models
        validate_trained_models(target_assets)

            # Step 4: Summary report
        print_header("📊 Multi-Asset Training Summary")

        print(f"🎯 Target Assets: {len(target_assets)}")
        print(f"✅ Successful Training: {successful_training}")
        print(f"❌ Failed Training: {len(target_assets) - successful_training}")

        print("\n📈 Detailed Results:")
        for symbol, result in results.items():
            if 'error' in result:
                print(f"❌ {symbol}: Failed - {result['error']}")
            else:
                lstm_loss = result['lstm_result']['train_loss']
                xgb_rmse = result['xgb_result']['train_rmse']
                signal = result['signal']['signal'] if result['signal'] else 'N/A'
                print(f"✅ {symbol}: LSTM Loss={lstm_loss:.4f}, XGB RMSE={xgb_rmse:.4f}, Signal={signal}")

        print("\n💾 Model Files Saved:")
        for symbol in target_assets:
            symbol_lower = symbol.lower()
            print(f"   • lstm_{symbol_lower}.h5, xgb_{symbol_lower}.pkl, scaler_{symbol_lower}.pkl")

        print("\n🚀 Next Steps:")
        print("1. Test real-time predictions for each asset")
        print("2. Compare model performance across assets")
        print("3. Add more assets (BNB, ADA, etc.)")
        print("4. Implement model performance monitoring")
        print("5. Integrate with trading system")

        success = successful_training == len(target_assets)
        if success:
            logger.info(f"Successfully trained models for {successful_training} assets")
        else:
            logger.warning(f"Training completed with {successful_training}/{len(target_assets)} successes")

        return success

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        logger.warning("Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during training: {e}")
        logger.exception("Unexpected error during multi-asset training")
        return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    print(f"\nExiting with code: {exit_code}")

sys.exit(exit_code)