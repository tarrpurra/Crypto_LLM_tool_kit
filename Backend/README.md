# Trading Agent - Backend

## Overview

This backend system provides a comprehensive trading platform focused on cryptocurrency trading. It integrates multiple agents to analyze market data, generate trading signals, and manage risk.

## Features

- **Multi-Agent Architecture**: News, Technical, ML, Risk, and User agents working together
- **Cryptocurrency Focus**: Specialized tools for crypto trading with on-chain data support
- **Risk Management**: Comprehensive risk assessment and position sizing
- **Machine Learning**: LSTM and XGBoost models for price prediction
- **Technical Analysis**: 50+ technical indicators using TA-Lib
- **Parallel Processing**: Concurrent data fetching for improved performance

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                            Trading System (main.py)                           │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                            UserAgent (user_agent.py)                          │
│  - Fetches user data, portfolio, and account state from database              │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                            NewsAgent (news_agent.py)                          │
│  - Fetches cryptocurrency news from NewsAPI                                   │
│  - Analyzes sentiment using FinBERT                                           │
│  - Summarizes with Gemini AI                                                  │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                        TechnicalAgent (technical_agent.py)                    │
│  - Computes 50+ technical indicators using TA-Lib                             │
│  - Analyzes market conditions (trend, volatility, volume)                     │
│  - Provides risk management levels (stop loss, take profit)                   │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                              MLAgent (ml_agent.py)                            │
│  - LSTM model for time series prediction                                      │
│  - XGBoost model for feature-based prediction                                 │
│  - Walk-forward validation for robust backtesting                             │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                             RiskAgent (risk_agent.py)                         │
│  - Position sizing based on risk tolerance                                    │
│  - Exposure limits (symbol, market, daily loss)                               │
│  - Volatility and liquidity scaling                                           │
│  - Confidence-based position scaling                                          │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                        ThinkingAgent (Core/Thinking_agent.py)                 │
│  - LLM-based decision making                                                  │
│  - Weighs inputs from all agents                                              │
│  - Generates final trading recommendations                                    │
└───────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Data Collection**: Agents fetch data from various sources (NewsAPI, Binance, CoinAPI)
2. **Data Processing**: DataManager cleans, validates, and enriches data with technical indicators
3. **Signal Generation**: Agents generate trading signals based on their analysis
4. **Risk Assessment**: RiskAgent evaluates trade proposals and determines position sizes
5. **Decision Making**: ThinkingAgent combines all inputs to generate final recommendations

## Key Components

### Agents

- **UserAgent**: Manages user data, portfolio, and account state
- **NewsAgent**: Fetches and analyzes cryptocurrency news and sentiment
- **TechnicalAgent**: Performs comprehensive technical analysis
- **MLAgent**: Provides machine learning-based price predictions
- **RiskAgent**: Manages risk and position sizing

### Services

- **DataManager**: Handles data fetching, processing, and storage
- **CryptoService**: Interface to cryptocurrency data providers
- **CryptoNansen**: On-chain data analysis (when available)

### Core

- **ThinkingAgent**: LLM-based decision engine
- **IndicatorEngine**: Technical indicator calculations
- **ScoringEngine**: Signal strength scoring

## Setup

### Requirements

- Python 3.9+
- Required packages in `requirements.txt`
- API keys for NewsAPI, Gemini, and cryptocurrency data providers

### Installation

```bash
cd Backend
pip install -r requirements.txt
```

### Configuration

Create a `configs/api_keys.json` file with your API keys:

```json
{
  "newsapi_key": "your_newsapi_key",
  "gemini_api_key": "your_gemini_key",
  "binance_api_key": "your_binance_key",
  "coinapi_key": "your_coinapi_key"
}
```

## Usage

### Running the Trading System

```bash
python main.py
```

### Example Workflow

```python
from main import TradingSystem

# Initialize the system
trading_system = TradingSystem(user_id=1, api_key="your_api_key")

# Run trading workflow for BTC
recommendation = trading_system.run_trading_workflow(symbol="BTC", asset_type="crypto")

print(json.dumps(recommendation, indent=2))
```

## Testing

Run unit tests to verify functionality:

```bash
cd Backend/Testing
python run_unit_tests.py
```

## Data Sources

### Cryptocurrency Data

- **Binance API**: Primary source for OHLCV data
- **CoinAPI**: Backup source for cryptocurrency data
- **Nansen API**: On-chain data (when available)

### News Data

- **NewsAPI**: Cryptocurrency news aggregation
- **FinBERT**: Sentiment analysis model
- **Gemini AI**: News summarization and analysis

## Risk Management

The system implements comprehensive risk management:

- **Position Sizing**: Based on account equity and risk tolerance
- **Exposure Limits**: Per-symbol and per-market limits
- **Daily Loss Limits**: Circuit breaker for excessive losses
- **Volatility Scaling**: Adjusts position sizes based on market volatility
- **Liquidity Scaling**: Reduces positions for illiquid assets
- **Confidence Scaling**: Adjusts positions based on signal confidence

## Machine Learning

### Models

- **LSTM**: Time series prediction using historical price data
- **XGBoost**: Feature-based prediction using technical indicators

### Training

Models are trained using walk-forward validation to ensure robustness:

```python
agent = MLAgent(asset="BTC")
result = agent.train_lstm(data, epochs=50)
```

### Prediction

```python
signal = agent.get_ml_signal("BTC", data)
print(f"Signal: {signal['signal']}, Confidence: {signal['confidence']}")
```

## Technical Indicators

The system computes over 50 technical indicators:

- **Trend Indicators**: SMA, EMA, MACD, ADX
- **Momentum Indicators**: RSI, Stochastic, Williams %R, CCI
- **Volatility Indicators**: ATR, Bollinger Bands
- **Volume Indicators**: OBV, CMF
- **Price Patterns**: Various candlestick patterns

## Performance Optimization

- **Caching**: Data caching to reduce API calls
- **Parallel Processing**: Concurrent data fetching using asyncio
- **Efficient Data Storage**: Parquet format for processed data
- **Model Optimization**: Early stopping and model validation

## Future Enhancements

- **Additional Data Sources**: Integrate more cryptocurrency exchanges
- **Enhanced On-Chain Analysis**: Deeper integration with Nansen and Glassnode
- **Portfolio Optimization**: Advanced portfolio management strategies
- **Backtesting Framework**: Comprehensive historical testing
- **Real-time Alerts**: Trading signal notifications

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Write tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please contact the development team.
