import os
import json
import logging
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Try to import XGBoost, but make it optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError as e:
    XGBOOST_AVAILABLE = False
    logging.warning(f"XGBoost not available: {e}. XGBoost models will not be functional.")
    xgb = None

# Try to import TensorFlow, but make it optional
try:
    import tensorflow as tf
    if hasattr(tf, 'config'):
        tf.config.run_functions_eagerly(True)
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except (ImportError, AttributeError) as e:
    TENSORFLOW_AVAILABLE = False
    logging.warning(f"TensorFlow not available: {e}. LSTM models will not be functional.")
    
import joblib

# Try to import DataManager
try:
    from services.data_manager import DataManager
except ImportError:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from services.data_manager import DataManager

class MLAgent:
    def __init__(self, asset='default'):
        # Set up logging first
        self.logger = logging.getLogger(f'MLAgent_{asset}')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.asset = asset.lower()
        self.data_manager = DataManager()
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        # Asset-specific model paths
        self.lstm_model_path = os.path.join(self.models_dir, f'lstm_{self.asset}.h5')
        self.xgb_model_path = os.path.join(self.models_dir, f'xgb_{self.asset}.pkl')
        self.scaler_path = os.path.join(self.models_dir, f'scaler_{self.asset}.pkl')

        # Initialize models
        self.lstm_model = None
        self.xgb_model = None
        self.scaler = MinMaxScaler()

        # Load existing models if available
        self.load_models()

    def load_models(self):
        """Load pre-trained models if they exist."""
        try:
            if os.path.exists(self.lstm_model_path):
                if not TENSORFLOW_AVAILABLE:
                    self.logger.warning("TensorFlow/Keras not available; cannot load LSTM model")
                else:
                    # Load model with compile=False to avoid optimizer issues
                    self.lstm_model = load_model(self.lstm_model_path, compile=False)
                    # Recompile with a fresh optimizer
                    from tensorflow.keras.optimizers import Adam
                    self.lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
                    self.logger.info("Loaded LSTM model")
            if os.path.exists(self.xgb_model_path):
                if not XGBOOST_AVAILABLE:
                    self.logger.warning("XGBoost not available; cannot load XGBoost model")
                else:
                    self.xgb_model = joblib.load(self.xgb_model_path)
                    self.logger.info("Loaded XGBoost model")
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                self.logger.info("Loaded scaler")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

    def save_models(self):
        """Save trained models."""
        try:
            if self.lstm_model:
                self.lstm_model.save(self.lstm_model_path)
                self.logger.info("Saved LSTM model")
            if self.xgb_model:
                joblib.dump(self.xgb_model, self.xgb_model_path)
                self.logger.info("Saved XGBoost model")
            joblib.dump(self.scaler, self.scaler_path)
            self.logger.info("Saved scaler")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")

    def prepare_lstm_data(self, data, lookback=60, fit_scaler=True):
        """Prepare data for LSTM training/prediction with proper scaling."""
        if not isinstance(data, pd.DataFrame) or 'close' not in data.columns:
            raise ValueError("Data must be DataFrame with 'close' column")

        prices = data['close'].values.reshape(-1, 1)

        # Only fit scaler on training data to prevent data leakage
        if fit_scaler:
            scaled_prices = self.scaler.fit_transform(prices)
        else:
            scaled_prices = self.scaler.transform(prices)

        X, y = [], []
        for i in range(lookback, len(scaled_prices)):
            X.append(scaled_prices[i-lookback:i, 0])
            y.append(scaled_prices[i, 0])

        X = np.array(X)
        y = np.array(y)
        
        # Check if we have valid data before reshaping
        if len(X) > 0 and len(X.shape) > 1:
            X = X.reshape((X.shape[0], X.shape[1], 1))

        return X, y

    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture with better metrics."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras not available. Cannot build LSTM model.")
            
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        # Use a fresh optimizer instance with better metrics
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.metrics import MeanAbsoluteError, RootMeanSquaredError
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=[MeanAbsoluteError(), RootMeanSquaredError()]
        )
        return model

    def train_lstm(self, data, lookback=60, epochs=50, batch_size=32, verbose=1):
        """Train LSTM model on historical data with chronological split."""
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning("TensorFlow/Keras not available. Skipping LSTM training.")
            return None
            
        try:
            # Split data chronologically first (80% train, 20% test)
            split_idx = int(len(data) * 0.8)
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]

            # Prepare training data (fits scaler on training data only)
            X_train, y_train = self.prepare_lstm_data(train_data, lookback, fit_scaler=True)

            # Prepare test data (uses fitted scaler - no data leakage)
            X_test, y_test = self.prepare_lstm_data(test_data, lookback, fit_scaler=False)
            
            # Check if we have valid data for training
            if len(X_train) == 0 or len(X_test) == 0:
                raise ValueError(f"Insufficient data for training. Need at least {lookback + 1} records")

            # Always build a fresh model to avoid optimizer conflicts
            self.lstm_model = self.build_lstm_model((X_train.shape[1], 1))

            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            history = self.lstm_model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                shuffle=False,  # CRITICAL: No shuffling within epochs for time series
                verbose=verbose
            )

            self.save_models()
            self.logger.info("LSTM training completed")

            # Safely extract loss values
            train_loss = float(history.history['loss'][-1])
            val_loss = float(history.history['val_loss'][-1])

            return {
                "model": "LSTM",
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epochs_trained": len(history.history['loss'])
            }

        except Exception as e:
            self.logger.error(f"LSTM training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    def walk_forward_validation(self, data, lookback=60, n_splits=5):
        """
        Perform walk-forward validation for time series.

        Args:
            data: Historical data
            lookback: Lookback window
            n_splits: Number of validation splits

        Returns:
            dict: Validation metrics
        """
        try:
            n_samples = len(data)
            split_size = n_samples // (n_splits + 1)

            lstm_scores = []
            xgb_scores = []

            for i in range(n_splits):
                train_end = split_size * (i + 1)
                test_end = split_size * (i + 2)

                if test_end > n_samples:
                    test_end = n_samples

                train_data = data.iloc[:train_end]
                test_data = data.iloc[train_end:test_end]

                if len(train_data) < lookback or len(test_data) < lookback:
                    continue

                # Train on this fold
                self.train_lstm(train_data, lookback=lookback, epochs=20, verbose=0)
                self.train_xgb(train_data, lookback=lookback//2)

                # Test on next period
                signal = self.get_ml_signal("VALIDATION", test_data)
                if 'error' not in signal:
                    # Calculate prediction accuracy
                    actual_price = signal['current_price']
                    predicted_price = signal['predicted_price']
                    accuracy = 1 - abs(predicted_price - actual_price) / actual_price

                    if signal['lstm_prediction']:
                        lstm_scores.append(accuracy)
                    if signal['xgb_prediction']:
                        xgb_scores.append(accuracy)

            return {
                'lstm_avg_accuracy': np.mean(lstm_scores) if lstm_scores else None,
                'xgb_avg_accuracy': np.mean(xgb_scores) if xgb_scores else None,
                'validation_splits': len(lstm_scores)
            }

        except Exception as e:
            self.logger.error(f"Walk-forward validation failed: {e}")
            return {"error": str(e)}

    def predict_lstm(self, data, lookback=60, future_steps=1):
        """Predict future prices using LSTM."""
        try:
            if not self.lstm_model:
                return {"error": "LSTM model not trained"}

            if not isinstance(data, pd.DataFrame) or 'close' not in data.columns:
                return {"error": "Data must be DataFrame with 'close' column"}

            if len(data) < lookback:
                return {"error": f"Insufficient data for prediction: need {lookback} records"}

            if not hasattr(self.scaler, 'scale_'):
                return {"error": "Scaler not fitted"}

            prices = data['close'].values[-lookback:].reshape(-1, 1)
            scaled_prices = self.scaler.transform(prices)
            X_pred = scaled_prices.reshape((1, lookback, 1))

            predictions = []
            for _ in range(future_steps):
                # Get prediction
                pred_output = self.lstm_model.predict(X_pred, verbose=0)
                
                # Handle both eager and graph execution
                if isinstance(pred_output, tf.Tensor):
                    pred = float(pred_output[0][0].numpy())
                else:
                    pred = float(pred_output[0][0])
                
                predictions.append(pred)
                
                # Update input for next prediction
                X_pred = np.roll(X_pred, -1, axis=1)
                X_pred[0, -1, 0] = pred

            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.scaler.inverse_transform(predictions).flatten()

            return predictions.tolist()

        except Exception as e:
            self.logger.error(f"LSTM prediction failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    def prepare_xgb_data(self, data, lookback=30):
        """Prepare data for XGBoost training with comprehensive features."""
        if not isinstance(data, pd.DataFrame) or 'close' not in data.columns:
            raise ValueError("Data must be DataFrame with 'close' column")

        df = data.copy()

        # Use all available technical indicators as features
        # Basic price features
        features = ['open', 'high', 'low', 'close']

        # Return features
        if 'returns' in df.columns:
            features.extend(['returns', 'log_returns'])

        # Moving averages
        ma_features = [col for col in df.columns if col.startswith('ma_') or col.startswith('ema_')]
        features.extend(ma_features)

        # MACD features
        macd_features = [col for col in df.columns if 'macd' in col]
        features.extend(macd_features)

        # Momentum indicators
        momentum_features = ['rsi', 'stoch_k', 'stoch_d', 'williams_r', 'cci']
        features.extend([f for f in momentum_features if f in df.columns])

        # Trend indicators
        trend_features = [col for col in df.columns if 'adx' in col]
        features.extend(trend_features)

        # Volatility indicators
        volatility_features = [col for col in df.columns if 'volatility' in col or 'atr' in col or 'bb_' in col]
        features.extend(volatility_features)

        # Momentum features
        momentum_cols = [col for col in df.columns if 'momentum' in col or 'roc' in col]
        features.extend(momentum_cols)

        # Volume features (if available)
        volume_features = [col for col in df.columns if 'volume' in col or 'obv' in col or 'cmf' in col]
        features.extend(volume_features)

        # Price channel features
        channel_features = [col for col in df.columns if 'high_' in col or 'low_' in col]
        features.extend(channel_features)

        # Ensure we have enough data and drop NaN values
        df = df.dropna()

        if len(df) < lookback:
            raise ValueError(f"Insufficient data: need at least {lookback} records, got {len(df)}")

        # Select available features
        available_features = [f for f in features if f in df.columns]

        if len(available_features) == 0:
            # Fallback to basic features
            available_features = ['close']

        X = df[available_features]
        y = df['close'].shift(-1).dropna()  # Predict next day's close

        # Align X and y
        X = X.iloc[:-1]

        self.logger.info(f"XGBoost features: {available_features}")
        return X, y

    def train_xgb(self, data, lookback=30):
        """Train XGBoost model with chronological split."""
        try:
            X, y = self.prepare_xgb_data(data, lookback)

            # CRITICAL: Chronological split for time series (no shuffling)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            if not XGBOOST_AVAILABLE:
                self.logger.warning("XGBoost not available, skipping XGBoost training")
                return None

            self.xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )

            self.xgb_model.fit(X_train, y_train)

            # Predictions and metrics
            train_pred = self.xgb_model.predict(X_train)
            test_pred = self.xgb_model.predict(X_test)

            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

            self.save_models()
            self.logger.info("XGBoost training completed")

            return {
                "model": "XGBoost",
                "train_rmse": float(train_rmse),
                "test_rmse": float(test_rmse)
            }

        except Exception as e:
            self.logger.error(f"XGBoost training failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    def predict_xgb(self, data):
        """Predict next price using XGBoost."""
        try:
            if not self.xgb_model:
                return {"error": "XGBoost model not trained"}

            X, _ = self.prepare_xgb_data(data)
            latest_features = X.iloc[-1:].values
            prediction = self.xgb_model.predict(latest_features)[0]

            return float(prediction)

        except Exception as e:
            self.logger.error(f"XGBoost prediction failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {"error": str(e)}

    def get_ml_signal(self, symbol, data, future_steps=5):
        """
        Generate ML-based trading signal.

        Args:
            symbol: Trading symbol
            data: DataFrame with historical price data (columns: close, high, low, volume, etc.)
            future_steps: Number of future steps to predict (for LSTM)

        Returns:
            dict: Signal with predictions and trading recommendation
        """
        try:
            current_price = float(data['close'].iloc[-1])

            # Get LSTM predictions
            lstm_preds = self.predict_lstm(data, future_steps=future_steps)
            if isinstance(lstm_preds, dict) and 'error' in lstm_preds:
                lstm_pred = None
            else:
                lstm_pred = float(np.mean(lstm_preds)) if lstm_preds else None

            # Get XGBoost prediction
            xgb_pred = self.predict_xgb(data)
            if isinstance(xgb_pred, dict) and 'error' in xgb_pred:
                xgb_pred = None

            # Combine predictions
            predictions = [p for p in [lstm_pred, xgb_pred] if p is not None]
            if not predictions:
                return {
                    "ticker": symbol,
                    "error": "No ML models available for prediction"
                }

            avg_prediction = float(np.mean(predictions))
            confidence = 1 - abs(avg_prediction - current_price) / current_price  # Simple confidence

            # Determine signal
            if avg_prediction > current_price * 1.02:  # 2% threshold
                action = "bullish"
            elif avg_prediction < current_price * 0.98:
                action = "bearish"
            else:
                action = "neutral"

            return {
                "ticker": symbol,
                "current_price": current_price,
                "predicted_price": avg_prediction,
                "lstm_prediction": lstm_pred,
                "xgb_prediction": xgb_pred,
                "signal": action,
                "confidence": float(min(confidence, 1.0)),
                "models_used": len(predictions)
            }

        except Exception as e:
            self.logger.error(f"ML signal generation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                "ticker": symbol,
                "error": str(e)
            }


    def fetch_train_predict(self, symbol, days=365, interval='1d', lookback=60, future_steps=5,
                            train_if_missing=True, force_retrain=False, lstm_epochs=30,
                            batch_size=32, xgb_lookback=30, add_features=True):
        '''Fetch data, (optionally) train models, and return ML predictions.'''
        start_time = time.time()
        response = {
            "ticker": symbol,
            "asset": self.asset,
            "status": "failed",
            "data": {},
            "model_status": {
                "lstm": {"available": TENSORFLOW_AVAILABLE, "trained": self.lstm_model is not None},
                "xgb": {"available": XGBOOST_AVAILABLE, "trained": self.xgb_model is not None}
            },
            "training": {},
            "predictions": {},
            "errors": []
        }

        if not symbol:
            response["errors"].append("symbol is required")
            return response

        try:
            raw_data = self.data_manager.fetch_historical_data(symbol, days=days, interval=interval)
            if raw_data is None or raw_data.empty:
                response["errors"].append("No historical data available")
                return response

            data = self.data_manager.prepare_ml_data(raw_data, add_features=add_features)
            if data is None or data.empty:
                response["errors"].append("Failed to prepare ML data")
                return response

            response["data"] = {
                "records": len(data),
                "start": str(data.index.min()),
                "end": str(data.index.max()),
                "interval": interval,
                "columns": list(data.columns)
            }

            # Train models if needed
            if train_if_missing or force_retrain:
                if force_retrain or self.lstm_model is None:
                    response["training"]["lstm"] = self.train_lstm(
                        data,
                        lookback=lookback,
                        epochs=lstm_epochs,
                        batch_size=batch_size,
                        verbose=0
                    )
                if force_retrain or self.xgb_model is None:
                    response["training"]["xgb"] = self.train_xgb(data, lookback=xgb_lookback)

            # Update model status after training
            response["model_status"]["lstm"]["trained"] = self.lstm_model is not None
            response["model_status"]["xgb"]["trained"] = self.xgb_model is not None

            # Predict
            response["predictions"] = self.get_ml_signal(symbol, data, future_steps=future_steps)

            # Collect errors
            if isinstance(response["predictions"], dict) and "error" in response["predictions"]:
                response["errors"].append(response["predictions"]["error"])
            for model_name, result in response["training"].items():
                if isinstance(result, dict) and "error" in result:
                    response["errors"].append(f"{model_name}: {result['error']}")

            if response["errors"]:
                response["status"] = "partial" if response["predictions"] else "failed"
            else:
                response["status"] = "success"

            response["elapsed_ms"] = int((time.time() - start_time) * 1000)
            return response

        except Exception as e:
            response["errors"].append(str(e))
            response["elapsed_ms"] = int((time.time() - start_time) * 1000)
            return response

    def get_tool_manifest(self):
        '''Return tool metadata for registry or dynamic discovery.'''
        return {
            "name": "MLAgent",
            "version": "1.1.0",
            "description": "ML price prediction tool (LSTM, XGBoost) with data fetch + training pipeline",
            "capabilities": [
                "price_prediction",
                "model_training",
                "signal_generation",
                "data_fetching"
            ],
            "methods": {
                "fetch_train_predict": "Fetch data, train models if needed, and return predictions",
                "get_ml_signal": "Return predictions using already-prepared data"
            },
            "dependencies": ["DataManager"],
            "crypto_config": {
                "supported_exchanges": ["binance", "coinbase"],
                "default_pair": "BTC/USDT",
                "models": ["lstm", "xgboost"]
            }
        }

    def invoke(self, method: str, **kwargs):
        '''Invoke a method by name with arguments.'''
        if not hasattr(self, method):
            return {
                "status": "failed",
                "error": f"Method '{method}' not found",
                "available_methods": list(self.get_tool_manifest()["methods"].keys())
            }
        try:
            func = getattr(self, method)
            return func(**kwargs)
        except TypeError as e:
            return {"status": "failed", "error": f"Invalid arguments: {str(e)}"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

# Example usage
if __name__ == "__main__":
    # Mock data for testing
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    prices = np.random.randn(200).cumsum() + 100
    data = pd.DataFrame({'close': prices}, index=dates)

    agent = MLAgent()

    # Train models (in real usage, this would be done separately)
    print("Training LSTM model...")
    lstm_result = agent.train_lstm(data, epochs=10)
    print(json.dumps(lstm_result, indent=2))
    
    print("\nTraining XGBoost model...")
    xgb_result = agent.train_xgb(data)
    print(json.dumps(xgb_result, indent=2))

    # Get signal
    print("\nGenerating ML signal...")
    signal = agent.get_ml_signal("TEST", data)
    print(json.dumps(signal, indent=2))