#!/usr/bin/env python3
"""
Data Preparation Example for ML Agent

This script demonstrates how to prepare historical data for training the ML Agent
with BTC, ETH, NIFTY, and other assets.
"""

import sys
import os
import pandas as pd
import logging

# Add Backend to path
sys.path.append(os.path.dirname(__file__))

from services.data_manager import DataManager
from Agents.ml_agent import MLAgent

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_preparation.log'),
            logging.StreamHandler()
        ]
    )

def prepare_crypto_data():
    """Prepare cryptocurrency data."""
    print("\n=== Preparing Cryptocurrency Data ===")

    manager = DataManager()
    ml_agent = MLAgent()

    # List of cryptocurrencies to prepare
    crypto_symbols = ['BTC', 'ETH', 'BNB', 'ADA']

    for symbol in crypto_symbols:
        print(f"\n--- Processing {symbol} ---")

        try:
            # Get training data (fetches, processes, and caches)
            data = manager.get_training_data(symbol, days=365)

            if data is None:
                print(f"❌ Failed to get data for {symbol}")
                continue

            print(f"✅ Got {len(data)} records for {symbol}")
            print(f"   Date range: {data.index.min()} to {data.index.max()}")
            print(f"   Features: {list(data.columns)}")

            # Validate data
            validation = manager.validate_data(data)
            if validation['valid']:
                print("✅ Data validation passed")
            else:
                print(f"⚠️  Data validation issues: {validation['errors']}")

            # Optional: Train ML models
            print(f"   Training LSTM model for {symbol}...")
            lstm_result = ml_agent.train_lstm(data)
            if 'error' not in lstm_result:
                print(f"   LSTM - Train Loss: {lstm_result['train_loss']:.4f}, Val Loss: {lstm_result['val_loss']:.4f}")
            else:
                print(f"   LSTM training failed: {lstm_result['error']}")

            print(f"   Training XGBoost model for {symbol}...")
            xgb_result = ml_agent.train_xgb(data)
            if 'error' not in xgb_result:
                print(f"   XGBoost - Train RMSE: {xgb_result['train_rmse']:.4f}, Test RMSE: {xgb_result['test_rmse']:.4f}")
            else:
                print(f"   XGBoost training failed: {xgb_result['error']}")

        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")

def prepare_stock_data():
    """Prepare stock market data."""
    print("\n=== Preparing Stock Market Data ===")

    manager = DataManager()

    # List of stocks to prepare
    stock_symbols = ['NIFTY', 'RELIANCE', 'TCS', 'INFY']

    for symbol in stock_symbols:
        print(f"\n--- Processing {symbol} ---")

        try:
            # Get training data
            data = manager.get_training_data(symbol, days=365)

            if data is None:
                print(f"❌ Failed to get data for {symbol} (check Kite authentication)")
                continue

            print(f"✅ Got {len(data)} records for {symbol}")
            print(f"   Date range: {data.index.min()} to {data.index.max()}")
            print(f"   Features: {list(data.columns)}")

            # Validate data
            validation = manager.validate_data(data)
            if validation['valid']:
                print("✅ Data validation passed")
            else:
                print(f"⚠️  Data validation issues: {validation['errors']}")

        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")

def test_ml_predictions():
    """Test ML predictions on prepared data."""
    print("\n=== Testing ML Predictions ===")

    manager = DataManager()
    ml_agent = MLAgent()

    test_symbols = ['BTC', 'ETH']

    for symbol in test_symbols:
        print(f"\n--- Testing {symbol} Predictions ---")

        try:
            # Load prepared data
            data = manager.load_data(symbol)

            if data is None:
                print(f"❌ No prepared data found for {symbol}")
                continue

            # Get current price
            current_price = manager.get_current_price(symbol)
            if current_price:
                print(f"   Current Price: ${current_price:.2f}")
            else:
                print("   Current price unavailable")

            # Get ML signal
            signal = ml_agent.get_ml_signal(symbol, data)

            if 'error' in signal:
                print(f"❌ ML prediction failed: {signal['error']}")
            else:
                print("✅ ML Signal generated:")
                print(f"   Predicted Price: ${signal['predicted_price']:.2f}")
                print(f"   Signal: {signal['signal'].upper()}")
                print(f"   Confidence: {signal['confidence']:.2f}")
                print(f"   Models Used: {signal['models_used']}")

        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")

def main():
    """Main execution function."""
    print("🚀 ML Agent Data Preparation Example")
    print("=" * 50)

    setup_logging()

    try:
        # Prepare cryptocurrency data
        prepare_crypto_data()

        # Prepare stock data (requires Kite authentication)
        prepare_stock_data()

        # Test ML predictions
        test_ml_predictions()

        print("\n" + "=" * 50)
        print("✅ Data preparation completed!")
        print("\nNext steps:")
        print("1. Review the generated models in Backend/data/models/")
        print("2. Check data files in Backend/data/")
        print("3. Run the ML agent with real-time data")
        print("4. Integrate with the multi-agent trading system")

    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logging.exception("Unexpected error in main execution")


if __name__ == "__main__":
    main()