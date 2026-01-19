"""
Test helper functions and utilities
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def get_test_data_path(*paths) -> str:
    """Get path to test data directory"""
    return os.path.join(os.path.dirname(__file__), '..', 'data', *paths)


def load_golden_dataset(filename: str) -> Any:
    """Load a golden dataset file"""
    file_path = get_test_data_path('golden_datasets', filename)
    
    if filename.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    elif filename.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {filename}")


def generate_mock_candles(symbol: str, start_date: str, periods: int = 100) -> pd.DataFrame:
    """Generate mock OHLCV candle data for testing"""
    
    # Convert start date to datetime
    start_dt = pd.to_datetime(start_date)
    
    # Generate date range
    date_range = pd.date_range(start=start_dt, periods=periods, freq='1H')
    
    # Generate mock price data with some trends and volatility
    base_price = 50000 if symbol == 'BTC' else 3000 if symbol == 'ETH' else 200
    
    # Create price series with trends
    prices = []
    current_price = base_price
    
    for i in range(periods):
        # Add some trends
        trend = 1.0 + (i / periods) * 0.1  # Gentle uptrend
        
        # Add volatility
        volatility = 1.0 + np.random.normal(0, 0.02)
        
        current_price = current_price * trend * volatility
        prices.append(current_price)
    
    # Create OHLCV data
    data = {
        'timestamp': date_range,
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [int(np.random.normal(1000, 200)) for _ in range(periods)]
    }
    
    return pd.DataFrame(data)


def generate_mock_news(symbol: str, count: int = 10) -> List[Dict]:
    """Generate mock news data for testing"""
    
    news_items = []
    
    for i in range(count):
        # Generate random timestamps
        timestamp = (datetime.now() - timedelta(days=i)).isoformat()
        
        # Generate mock news content
        headlines = [
            f"{symbol} reaches new all-time high amid market optimism",
            f"Regulatory concerns impact {symbol} price movement",
            f"Institutional adoption drives {symbol} demand",
            f"Technical analysis suggests {symbol} may face resistance",
            f"Market volatility affects {symbol} trading volume",
            f"Experts predict bullish trend for {symbol}",
            f"{symbol} price consolidates after recent rally",
            f"New partnership announced for {symbol} ecosystem",
            f"Analysts debate {symbol} future price targets",
            f"Market sentiment turns positive for {symbol}"
        ]
        
        headline = headlines[i % len(headlines)]
        
        # Generate random sentiment scores
        sentiment = np.random.uniform(-1, 1)
        confidence = np.random.uniform(0.7, 1.0)
        
        news_item = {
            'symbol': symbol,
            'timestamp': timestamp,
            'headline': headline,
            'content': f"This is a mock news article about {symbol}. {headline}. Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            'sentiment': float(sentiment),
            'confidence': float(confidence),
            'source': f"MockNews_{i}",
            'url': f"https://mocknews.com/{symbol}_{i}"
        }
        
        news_items.append(news_item)
    
    return news_items


def create_test_config() -> Dict:
    """Create a test configuration"""
    return {
        'test_mode': True,
        'debug': True,
        'api_keys': {
            'nansen_api_key': 'mock_key_for_testing',
            'binance_api_key': 'mock_binance_key',
            'coinbase_api_key': 'mock_coinbase_key'
        },
        'risk_limits': {
            'max_drawdown': 0.2,  # 20%
            'max_exposure': 0.5,  # 50%
            'daily_loss_limit': 0.05  # 5%
        }
    }


def validate_agent_output(output: Dict, expected_schema: Dict) -> bool:
    """Validate agent output against expected schema"""
    
    try:
        # Check required fields
        for field, field_type in expected_schema.items():
            if field not in output:
                print(f"Missing required field: {field}")
                return False
            
            if not isinstance(output[field], field_type):
                print(f"Field {field} has wrong type. Expected {field_type}, got {type(output[field])}")
                return False
        
        # Check confidence ranges
        if 'confidence' in output and not (0 <= output['confidence'] <= 100):
            print(f"Confidence out of range: {output['confidence']}")
            return False
            
        return True
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False


def compare_decisions(actual: Dict, expected: Dict, tolerance: float = 0.01) -> Dict:
    """Compare actual vs expected decisions with tolerance"""
    
    comparison = {
        'match': True,
        'differences': []
    }
    
    for key in expected:
        if key not in actual:
            comparison['match'] = False
            comparison['differences'].append(f"Missing key: {key}")
            continue
        
        if isinstance(expected[key], (int, float)):
            if abs(actual[key] - expected[key]) > tolerance:
                comparison['match'] = False
                comparison['differences'].append(f"{key}: expected {expected[key]}, got {actual[key]}")
        elif actual[key] != expected[key]:
            comparison['match'] = False
            comparison['differences'].append(f"{key}: expected {expected[key]}, got {actual[key]}")
    
    return comparison