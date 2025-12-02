#!/usr/bin/env python3
"""
Bitcoin ML Model Training Script

This script trains the ML Agent models on Bitcoin historical data.
It fetches data from Binance, trains LSTM and XGBoost models, and validates performance.
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
            logging.FileHandler('btc_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('BTC_Training')

def print_header(title):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

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
    required_packages = ['tensorflow', 'xgboost', 'pandas', 'numpy', 'sklearn']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not available")
            return False

    print("✅ Environment validation passed")
    return True

def fetch_btc_data(logger):
    """Fetch Bitcoin historical data."""
    print_section("Fetching Bitcoin Data")

    try:
        manager = DataManager()

        print("📡 Fetching 2 years of BTC historical data from Binance...")
        start_time = time.time()

        # Fetch 2 years of daily data (force fresh processing)
        btc_data = manager.get_training_data('BTC', days=730, use_cache=False)

        fetch_time = time.time() - start_time

        if btc_data is None or btc_data.empty:
            print("❌ Failed to fetch BTC data")
            return None

        print("✅ Successfully fetched BTC data")
        print(f"   📊 Records: {len(btc_data)}")
        print(f"   📅 Date range: {btc_data.index.min()} to {btc_data.index.max()}")
        print(f"   💰 Price range: ${btc_data['close'].min():.0f} - ${btc_data['close'].max():.0f}")
        print(f"   📊 Avg price: ${btc_data['close'].mean():.2f}")
        print(f"   📈 Latest price: ${btc_data['close'].iloc[-1]:.2f}")

        # Validate data quality
        validation = manager.validate_data(btc_data)
        if validation['valid']:
            print("✅ Data validation passed")
        else:
            print(f"⚠️  Data validation issues: {validation['errors']}")

        logger.info(f"Fetched {len(btc_data)} BTC records in {fetch_time:.2f}s")
        return btc_data

    except Exception as e:
        print(f"❌ Error fetching BTC data: {e}")
        logger.error(f"Data fetching failed: {e}")
        return None

def train_lstm_model(btc_data, logger):
    """Train LSTM model on BTC data."""
    print_section("Training LSTM Model")

    try:
        agent = MLAgent()

        print("🧠 Training LSTM model...")
        print("   Architecture: 2xLSTM(50) + Dropout + Dense layers")
        print("   Lookback: 60 days")
        print("   Epochs: 50 (with early stopping)")

        start_time = time.time()
        lstm_result = agent.train_lstm(btc_data, lookback=60, epochs=50, batch_size=32)
        train_time = time.time() - start_time

        if 'error' in lstm_result:
            print(f"❌ LSTM training failed: {lstm_result['error']}")
            return None

        print("✅ LSTM training completed")
        print(f"   📉 Train Loss: {lstm_result['train_loss']:.4f}")
        print(f"   📊 Val Loss: {lstm_result['val_loss']:.4f}")
        print(f"   ⏱️  Training time: {train_time:.2f}s")
        print(f"   📈 Epochs trained: {lstm_result['epochs_trained']}")

        logger.info(f"LSTM training completed: train_loss={lstm_result['train_loss']:.4f}, val_loss={lstm_result['val_loss']:.4f}")
        return lstm_result

    except Exception as e:
        print(f"❌ Error training LSTM: {e}")
        logger.error(f"LSTM training failed: {e}")
        return None

def train_xgb_model(btc_data, logger):
    """Train XGBoost model on BTC data."""
    print_section("Training XGBoost Model")

    try:
        agent = MLAgent()

        print("🌳 Training XGBoost model...")
        print("   Estimators: 100")
        print("   Max depth: 6")
        print("   Learning rate: 0.1")

        start_time = time.time()
        xgb_result = agent.train_xgb(btc_data, lookback=30)
        train_time = time.time() - start_time

        if 'error' in xgb_result:
            print(f"❌ XGBoost training failed: {xgb_result['error']}")
            return None

        print("✅ XGBoost training completed")
        print(f"   📊 Train RMSE: {xgb_result['train_rmse']:.4f}")
        print(f"   📈 Test RMSE: {xgb_result['test_rmse']:.4f}")
        print(f"   ⏱️  Training time: {train_time:.2f}s")
        logger.info(f"XGBoost training completed: train_rmse={xgb_result['train_rmse']:.4f}, test_rmse={xgb_result['test_rmse']:.4f}")
        return xgb_result

    except Exception as e:
        print(f"❌ Error training XGBoost: {e}")
        logger.error(f"XGBoost training failed: {e}")
        return None

def test_model_predictions(btc_data, logger):
    """Test trained models with predictions."""
    print_section("Testing Model Predictions")

    try:
        agent = MLAgent()

        # Test signal generation
        print("🔮 Generating ML trading signal...")
        signal = agent.get_ml_signal('BTC', btc_data, future_steps=5)

        if 'error' in signal:
            print(f"❌ Signal generation failed: {signal['error']}")
            return None

        print("✅ Signal generated successfully")
        print(f"   🎯 Current price: ${signal['current_price']:.2f}")
        print(f"   🔮 Predicted price: ${signal['predicted_price']:.2f}")
        print(f"   📊 LSTM prediction: ${signal.get('lstm_prediction', 'N/A')}")
        print(f"   🌳 XGBoost prediction: ${signal.get('xgb_prediction', 'N/A')}")
        print(f"   🚀 Signal: {signal['signal'].upper()}")
        print(f"   📊 Confidence: {signal['confidence']:.2f}")
        print(f"   🤖 Models used: {signal['models_used']}")

        # Calculate prediction accuracy metrics
        current_price = signal['current_price']
        predicted_price = signal['predicted_price']
        price_diff = predicted_price - current_price
        percent_diff = (price_diff / current_price) * 100

        print(f"   💰 Price difference: ${price_diff:.2f}")
        print(f"   📈 Percent difference: {percent_diff:.2f}%")

        if abs(percent_diff) < 2:
            print("   📊 Signal: NEUTRAL (within 2% range)")
        elif percent_diff > 2:
            print("   📈 Signal: BULLISH (predicted >2% higher)")
        else:
            print("   📉 Signal: BEARISH (predicted >2% lower)")

        logger.info(f"Signal generated: {signal['signal']} with {signal['confidence']:.2f} confidence")
        return signal

    except Exception as e:
        print(f"❌ Error testing predictions: {e}")
        logger.error(f"Prediction testing failed: {e}")
        return None

def validate_model_files():
    """Validate that model files were created."""
    print_section("Model File Validation")

    models_dir = os.path.join(os.path.dirname(__file__), 'data', 'models')
    required_files = ['lstm_model.h5', 'xgb_model.pkl', 'scaler.pkl']

    all_present = True
    for filename in required_files:
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
    """Main training execution."""
    print_header("🚀 Bitcoin ML Model Training")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger = setup_logging()

    try:
        # Step 1: Environment validation
        if not validate_environment():
            print("❌ Environment validation failed. Please check configuration.")
            return False

        # Step 2: Fetch BTC data
        btc_data = fetch_btc_data(logger)
        if btc_data is None:
            print("❌ Data fetching failed. Cannot proceed with training.")
            return False

        # Step 3: Train LSTM model
        lstm_result = train_lstm_model(btc_data, logger)
        if lstm_result is None:
            print("❌ LSTM training failed.")
            return False

        # Step 4: Train XGBoost model
        xgb_result = train_xgb_model(btc_data, logger)
        if xgb_result is None:
            print("❌ XGBoost training failed.")
            return False

        # Step 5: Test predictions
        signal = test_model_predictions(btc_data, logger)
        if signal is None:
            print("⚠️  Prediction testing failed, but models were trained.")

        # Step 6: Validate model files
        validate_model_files()

        # Summary
        print_header("📊 Training Summary")
        print("✅ Bitcoin ML models trained successfully!")
        print(f"   📊 Training data: {len(btc_data)} records")
        print(f"   🧠 LSTM: Loss = {lstm_result['train_loss']:.4f}")
        print(f"   🌳 XGBoost: RMSE = {xgb_result['train_rmse']:.4f}")
        if signal:
            print(f"   🚀 Signal: {signal['signal'].upper()} ({signal['confidence']:.2f} confidence)")

        print("\nNext steps:")
        print("1. Models are saved in Backend/data/models/")
        print("2. Use ML Agent for real-time BTC predictions")
        print("3. Monitor model performance over time")
        print("4. Retrain periodically with new data")

        logger.info("Bitcoin model training completed successfully")
        return True

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        logger.warning("Training interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during training: {e}")
        logger.exception("Unexpected error during training")
        return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    print(f"\nExiting with code: {exit_code}")

sys.exit(exit_code)