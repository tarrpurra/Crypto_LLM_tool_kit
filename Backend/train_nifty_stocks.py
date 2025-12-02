#!/usr/bin/env python3
"""
Nifty 50 Stock ML Model Training Script

Trains ML models for top 15 Nifty 50 companies using Yahoo Finance data
and comprehensive technical indicators for price prediction.
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
            logging.FileHandler('nifty_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('Nifty_Training')

def print_header(title):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)

def print_section(title):
    """Print formatted section header."""
    print(f"\n--- {title} ---")

def get_nifty_50_companies():
    """Get the list of top 15 Nifty 50 companies to train."""
    return [
        'RELIANCE',    # Reliance Industries Ltd.
        'HDFCBANK',    # HDFC Bank
        'INFY',        # Infosys Ltd.
        'ICICIBANK',   # ICICI Bank Ltd.
        'TCS',         # Tata Consultancy Services Ltd.
        'KOTAKBANK',   # Kotak Mahindra Bank Ltd.
        'HINDUNILVR',  # Hindustan Unilever Ltd.
        'ITC',         # ITC Ltd.
        'BAJFINANCE',  # Bajaj Finance Ltd.
        'BHARTIARTL',  # Bharti Airtel Ltd.
        'MARUTI',      # Maruti Suzuki India Ltd.
        'ASIANPAINT',  # Asian Paints Ltd.
        'SBIN',        # State Bank of India
        'NTPC',        # NTPC Ltd.
        'TITAN'        # Titan Company Ltd.
    ]

def validate_environment():
    """Validate that all required components are available."""
    print_section("Environment Validation")

    # Check required packages
    required_packages = ['yfinance', 'talib', 'pandas', 'tensorflow', 'xgboost']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not available - install with: pip install {package}")
            return False

    print("✅ Environment validation passed")
    return True

def fetch_stock_data(symbol, logger):
    """Fetch and prepare data for a single stock."""
    try:
        manager = DataManager()

        print(f"📡 Fetching 3 years of {symbol} data from Yahoo Finance...")
        start_time = time.time()

        # Fetch 3 years of daily data for better technical indicator calculation
        stock_data = manager.get_training_data(symbol, days=1095)

        fetch_time = time.time() - start_time

        if stock_data is None or stock_data.empty:
            print(f"❌ Failed to fetch data for {symbol}")
            return None

        print("✅ Successfully fetched stock data")
        print(f"   📊 Records: {len(stock_data)}")
        print(f"   📅 Date range: {stock_data.index.min()} to {stock_data.index.max()}")
        print(f"   💰 Price range: ₹{stock_data['close'].min():.2f} - ₹{stock_data['close'].max():.2f}")
        print(f"   📈 Latest price: ₹{stock_data['close'].iloc[-1]:.2f}")
        print(f"   📊 Features: {len(stock_data.columns)} columns")
        print(f"   ⏱️  Fetch time: {fetch_time:.2f}s")

        # Validate data quality
        validation = manager.validate_data(stock_data)
        if validation['valid']:
            print("✅ Data validation passed")
        else:
            print(f"⚠️  Data validation issues: {validation['errors']}")

        logger.info(f"Fetched {len(stock_data)} records for {symbol} in {fetch_time:.2f}s")
        return stock_data

    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")
        logger.error(f"Data fetching failed for {symbol}: {e}")
        return None

def train_stock_model(symbol, stock_data, logger):
    """Train ML models for a single stock."""
    try:
        agent = MLAgent(asset=symbol.lower())

        print(f"🧠 Training LSTM model for {symbol}...")
        print("   Architecture: 2xLSTM(50) + Dropout + Dense layers")
        print("   Lookback: 30 days, Epochs: 30 (optimized for stocks)")

        start_time = time.time()
        lstm_result = agent.train_lstm(stock_data, lookback=30, epochs=30, batch_size=32)
        lstm_time = time.time() - start_time

        if 'error' in lstm_result:
            print(f"❌ LSTM training failed: {lstm_result['error']}")
            return None

        print("✅ LSTM training completed")
        print(f"   📉 Train Loss: {lstm_result['train_loss']:.4f}")
        print(f"   📊 Val Loss: {lstm_result['val_loss']:.4f}")
        print(f"   ⏱️  Training time: {lstm_time:.2f}s")

        print(f"🌳 Training XGBoost model for {symbol}...")
        print("   Features: 44 technical indicators")
        print("   Estimators: 100, Max depth: 6")

        xgb_start = time.time()
        xgb_result = agent.train_xgb(stock_data, lookback=30)
        xgb_time = time.time() - xgb_start

        if 'error' in xgb_result:
            print(f"❌ XGBoost training failed: {xgb_result['error']}")
            return None

        print("✅ XGBoost training completed")
        print(f"   📊 Train RMSE: {xgb_result['train_rmse']:.4f}")
        print(f"   📈 Test RMSE: {xgb_result['test_rmse']:.4f}")
        print(f"   ⏱️  Training time: {xgb_time:.2f}s")

        total_time = lstm_time + xgb_time
        print(f"   ⏱️  Total training time: {total_time:.2f}s")

        result = {
            'symbol': symbol,
            'lstm_result': lstm_result,
            'xgb_result': xgb_result,
            'training_time': total_time,
            'data_points': len(stock_data)
        }

        logger.info(f"Training completed for {symbol}: LSTM_loss={lstm_result['train_loss']:.4f}, XGB_RMSE={xgb_result['train_rmse']:.4f}")
        return result

    except Exception as e:
        print(f"❌ Error training models for {symbol}: {e}")
        logger.error(f"Model training failed for {symbol}: {e}")
        return None

def test_stock_predictions(symbol, stock_data, logger):
    """Test trained models with predictions."""
    try:
        agent = MLAgent(asset=symbol.lower())

        # Get current price
        manager = DataManager()
        current_price = manager.get_current_price(symbol)

        # Generate ML signal
        signal = agent.get_ml_signal(symbol, stock_data)

        if 'error' in signal:
            print(f"❌ Signal generation failed: {signal['error']}")
            return None

        print("✅ Signal generated successfully")
        print(f"   🎯 Current price: ₹{signal['current_price']:.2f}")
        print(f"   🔮 Predicted price: ₹{signal['predicted_price']:.2f}")
        print(f"   🚀 Signal: {signal['signal'].upper()}")
        print(f"   📊 Confidence: {signal['confidence']:.2f}")
        print(f"   🤖 Models used: {signal['models_used']}")

        # Calculate prediction metrics
        actual_price = signal['current_price']
        predicted_price = signal['predicted_price']
        price_diff = predicted_price - actual_price
        percent_diff = (price_diff / actual_price) * 100

        print(f"   💰 Price difference: ₹{price_diff:.2f}")
        print(f"   📈 Percent change: {percent_diff:.2f}%")

        if abs(percent_diff) < 2:
            print("   📊 Signal: NEUTRAL (within 2% range)")
        elif percent_diff > 2:
            print("   📈 Signal: BULLISH (predicted >2% higher)")
        else:
            print("   📉 Signal: BEARISH (predicted >2% lower)")

        logger.info(f"Signal generated for {symbol}: {signal['signal']} with {signal['confidence']:.2f} confidence")
        return signal

    except Exception as e:
        print(f"❌ Error testing predictions for {symbol}: {e}")
        logger.error(f"Prediction testing failed for {symbol}: {e}")
        return None

def validate_model_files(stocks_trained):
    """Validate that model files were created for all trained stocks."""
    print_section("Model File Validation")

    models_dir = os.path.join(os.path.dirname(__file__), 'data', 'models')
    expected_files = 0
    found_files = 0

    for symbol in stocks_trained:
        symbol_lower = symbol.lower()
        required_files = [
            f'lstm_{symbol_lower}.h5',
            f'xgb_{symbol_lower}.pkl',
            f'scaler_{symbol_lower}.pkl'
        ]

        expected_files += 3

        for filename in required_files:
            filepath = os.path.join(models_dir, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath) / 1024  # KB
                print(f"✅ Found: {filename} ({size:.1f} KB)")
                found_files += 1
            else:
                print(f"❌ Missing: {filename}")

    success_rate = found_files / expected_files if expected_files > 0 else 0
    print(f"\n📊 Model files: {found_files}/{expected_files} ({success_rate:.1%})")

    if success_rate >= 0.9:  # 90% success rate
        print("✅ Model file validation passed")
    else:
        print("⚠️  Some model files missing")

    return success_rate >= 0.9

def main():
    """Main training execution for Nifty 50 stocks."""
    print_header("🚀 Nifty 50 Stock ML Model Training")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Training top 15 Nifty 50 companies with comprehensive technical indicators")

    logger = setup_logging()

    try:
        # Step 1: Environment validation
        if not validate_environment():
            print("❌ Environment validation failed. Please install required packages.")
            return False

        # Step 2: Get list of companies to train
        nifty_companies = get_nifty_50_companies()
        print(f"\n🎯 Target Companies: {len(nifty_companies)}")
        for i, company in enumerate(nifty_companies, 1):
            print("2d")

        # Step 3: Train models for each company
        training_results = []
        successful_training = 0

        for i, symbol in enumerate(nifty_companies, 1):
            print_header(f"Training {symbol} ({i}/{len(nifty_companies)})")

            try:
                # Fetch data
                stock_data = fetch_stock_data(symbol, logger)
                if stock_data is None:
                    print(f"⚠️  Skipping {symbol} due to data issues")
                    continue

                # Train models
                training_result = train_stock_model(symbol, stock_data, logger)
                if training_result is None:
                    print(f"⚠️  Skipping {symbol} due to training issues")
                    continue

                # Test predictions
                signal = test_stock_predictions(symbol, stock_data, logger)
                if signal:
                    training_result['signal'] = signal

                training_results.append(training_result)
                successful_training += 1

                print(f"✅ Successfully trained {symbol}")

                # Small delay between trainings to be respectful to Yahoo Finance
                if i < len(nifty_companies):
                    print("⏳ Cooling down for 2 seconds...")
                    time.sleep(2)

            except Exception as e:
                print(f"❌ Unexpected error training {symbol}: {e}")
                logger.error(f"Unexpected error for {symbol}: {e}")
                continue

        # Step 4: Validate model files
        stocks_trained = [r['symbol'] for r in training_results]
        validate_model_files(stocks_trained)

        # Step 5: Generate summary report
        print_header("📊 Nifty 50 Training Summary")

        print(f"🎯 Target Companies: {len(nifty_companies)}")
        print(f"✅ Successful Training: {successful_training}")
        print(f"❌ Failed Training: {len(nifty_companies) - successful_training}")
        print(".1f")

        # Performance summary
        if training_results:
            print("\n📈 Performance Summary:")
            lstm_losses = [r['lstm_result']['train_loss'] for r in training_results]
            xgb_rmses = [r['xgb_result']['train_rmse'] for r in training_results]

            print(f"   🧠 Avg LSTM Loss: {sum(lstm_losses)/len(lstm_losses):.4f}")
            print(f"   🌳 Avg XGB RMSE: {sum(xgb_rmses)/len(xgb_rmses):.4f}")
            print(f"   📊 Best LSTM Loss: {min(lstm_losses):.4f}")
            print(f"   🏆 Best XGB RMSE: {min(xgb_rmses):.4f}")

            # Signal distribution
            signals = [r.get('signal', {}).get('signal', 'unknown') for r in training_results if 'signal' in r]
            if signals:
                bullish = signals.count('bullish')
                bearish = signals.count('bearish')
                neutral = signals.count('neutral')
                print("\n🚀 Signal Distribution:")
                print(f"   📈 Bullish: {bullish}")
                print(f"   📉 Bearish: {bearish}")
                print(f"   📊 Neutral: {neutral}")

        # Step 6: Save detailed results
        results_file = os.path.join(os.path.dirname(__file__), 'nifty_training_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'training_date': datetime.now().isoformat(),
                'companies_targeted': len(nifty_companies),
                'companies_trained': successful_training,
                'results': training_results
            }, f, indent=2, default=str)

        print(f"\n💾 Detailed results saved to: {results_file}")

        print("\n🎉 Next Steps:")
        print("1. Models are saved in Backend/data/models/")
        print("2. Use SmartMLAgent for real-time stock predictions")
        print("3. Monitor model performance and retrain periodically")
        print("4. Add more Nifty 50 companies as needed")
        print("5. Integrate with trading system for automated signals")

        logger.info(f"Nifty 50 training completed: {successful_training}/{len(nifty_companies)} successful")

        success = successful_training > 0
        return success

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