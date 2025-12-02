# Data Preparation Guide for ML Agent

## Overview
This guide explains how to prepare historical price data for training and inference with the ML Agent. The agent requires OHLCV (Open, High, Low, Close, Volume) data in a structured pandas DataFrame format.

## Data Requirements

### Required Columns
- **timestamp**: DateTime index (UTC timezone recommended)
- **open**: Opening price
- **high**: Highest price
- **low**: Lowest price
- **close**: Closing price (required for ML models)
- **volume**: Trading volume (optional but recommended)

### Data Sources

#### Cryptocurrency Data
- **CoinAPI**: Professional crypto data API
  - URL: https://www.coinapi.io/
  - Supports: BTC, ETH, and 300+ cryptocurrencies
  - Granularity: 1m, 1h, 1d intervals
- **Binance API**: Free tier available
  - URL: https://binance-docs.github.io/apidocs/
  - Supports: Major crypto pairs
  - Rate limits: 1200 requests per minute

#### Stock Data
- **Zerodha KiteConnect**: Indian stock market
  - URL: https://kite.trade/
  - Supports: NIFTY, individual stocks
  - Historical data: 1m, 1d intervals
- **AngelOne SmartAPI**: Alternative Indian broker
  - URL: https://smartapis.angelone.in/
  - Similar coverage to Zerodha

## Data Preparation Pipeline

### 1. API Key Configuration
Update `Backend/configs/api_keys.json`:
```json
{
  "newsapi_key": "your_newsapi_key",
  "gemini_api_key": "your_gemini_key",
  "coinapi_key": "your_coinapi_key",
  "binance_api_key": "your_binance_key",
  "zerodha_api_key": "your_zerodha_key",
  "zerodha_secret": "your_zerodha_secret",
  "angelone_api_key": "your_angelone_key"
}
```

### 2. Data Fetching
Use the appropriate service based on asset type:

```python
from Backend.services.crypto_service import CryptoService
from Backend.services.kite_service import KiteService

# For crypto
crypto_service = CryptoService()
btc_data = crypto_service.get_historical_data('BTC', days=365, interval='1d')

# For stocks
kite_service = KiteService()
nifty_data = kite_service.get_historical_data('NIFTY', days=365, interval='1d')
```

### 3. Data Cleaning
- Remove missing values
- Handle outliers
- Ensure chronological order
- Convert to proper data types

### 4. Feature Engineering
The ML Agent automatically adds:
- **Returns**: Daily percentage changes
- **Moving Averages**: 5-day and 20-day SMAs
- **Volatility**: 20-day rolling standard deviation
- **Technical Indicators**: RSI, MACD (future enhancement)

### 5. Data Validation
- Check for sufficient data points (minimum 100)
- Validate price ranges (no negative prices)
- Ensure volume data consistency
- Verify timestamp continuity

## Expected Data Format

### Pandas DataFrame Structure
```python
import pandas as pd

# Example data structure
data = pd.DataFrame({
    'open': [45000.0, 45100.0, 44900.0, ...],
    'high': [45500.0, 45300.0, 45200.0, ...],
    'low': [44800.0, 44900.0, 44700.0, ...],
    'close': [45200.0, 45000.0, 45100.0, ...],
    'volume': [25000000, 22000000, 23000000, ...]
}, index=pd.date_range('2024-01-01', periods=len(data), freq='D'))
```

### Data Quality Checks
```python
# Basic validation
assert 'close' in data.columns, "Close price column required"
assert len(data) >= 100, "Minimum 100 data points required"
assert (data['close'] > 0).all(), "All close prices must be positive"
assert data.index.is_monotonic_increasing, "Data must be chronologically sorted"
```

## Integration with ML Agent

### Training Phase
```python
from Backend.Agents.ml_agent import MLAgent

agent = MLAgent()

# Train models
agent.train_lstm(data)
agent.train_xgb(data)
```

### Inference Phase
```python
# Get predictions
signal = agent.get_ml_signal("BTC", data)
print(signal)
```

## Sample Data Sources

### Free Alternatives (for testing)
- **Yahoo Finance**: `yfinance` library
- **Alpha Vantage**: Free API with limits
- **CryptoCompare**: Free crypto data

### Implementation Example
```python
import yfinance as yf

# Fetch sample data (for testing only)
def get_sample_data(symbol, days=365):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=f"{days}d")
    return data[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)
```

## Best Practices

### Data Volume
- **Training**: Minimum 1-2 years of daily data
- **Validation**: 20% of data for model validation
- **Testing**: Recent data not seen during training

### Frequency
- **Daily data**: Best for long-term predictions
- **Hourly data**: For short-term models
- **Minute data**: High frequency, requires more processing

### Asset-Specific Considerations
- **Crypto**: 24/7 trading, high volatility
- **Stocks**: Market hours, lower volatility
- **Indices (NIFTY)**: Broad market representation

### Data Storage
- Save processed data in `Backend/data/`
- Use parquet format for efficient storage
- Cache fetched data to avoid repeated API calls

## Troubleshooting

### Common Issues
1. **API Rate Limits**: Implement caching and request throttling
2. **Missing Data**: Use forward/backward fill for gaps
3. **Outliers**: Apply statistical filters or manual review
4. **Timezone Issues**: Standardize to UTC

### Performance Optimization
- Cache historical data locally
- Use vectorized pandas operations
- Implement parallel data fetching for multiple assets

## Next Steps
1. Obtain API keys for your preferred data sources
2. Implement the data services (`crypto_service.py`, `kite_service.py`)
3. Create data manager for unified access
4. Test with sample data before production use
5. Set up automated data pipelines for model retraining