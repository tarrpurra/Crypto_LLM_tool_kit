"""
Unit tests for Technical Agent
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Agents.technical_agent import TechnicalAgent
from services.common_models import TechnicalIndicators, StrengthScores, TechnicalSignal, RiskManagement, MarketCondition


class TestTechnicalAgent(unittest.TestCase):
    """Test Technical Agent functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock the logging to reduce noise during tests
        with patch('Agents.technical_agent.logging'):
            self.agent = TechnicalAgent()
        
    def test_initialization(self):
        """Test that TechnicalAgent initializes correctly"""
        self.assertIsNotNone(self.agent)
        self.assertIsNotNone(self.agent.crypto_provider)
        self.assertIsNotNone(self.agent.indicator_engine)
        self.assertIsNotNone(self.agent.scoring_engine)

    def test_decide_action_logic(self):
        """Test the decision logic for different strength scores"""
        
        test_cases = [
            # (strength, expected_bias, expected_signal, expected_confidence_range)
            (80, "bullish", "LONG", (0.8, 1.0)),
            (75, "bullish", "LONG", (0.8, 1.0)),
            (65, "bullish", "HOLD", (0.6, 0.6)),
            (50, "neutral", "HOLD", (0.5, 0.5)),
            (35, "bearish", "HOLD", (0.6, 0.6)),
            (20, "bearish", "SHORT", (0.8, 1.0)),
            (15, "bearish", "SHORT", (0.8, 1.0)),
        ]
        
        for strength, expected_bias, expected_signal, expected_conf_range in test_cases:
            with self.subTest(strength=strength):
                bias, signal, confidence = self.agent._decide_action(strength, "crypto")
                
                self.assertEqual(bias, expected_bias)
                self.assertEqual(signal, expected_signal)
                self.assertTrue(expected_conf_range[0] <= confidence <= expected_conf_range[1])

    def test_risk_management_calculation(self):
        """Test risk management calculation logic"""
        
        # Create mock DataFrame
        import pandas as pd
        data = {
            'close': [50000.0, 50100.0, 50200.0, 50300.0, 50400.0],
            'volume': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
        }
        df = pd.DataFrame(data)
        
        # Test bullish scenario
        risk_mgmt = self.agent._calculate_risk_management(
            current_price=50000.0,
            atr=1000.0,  # 1% ATR
            bias="bullish",
            df=df
        )
        
        self.assertIsNotNone(risk_mgmt)
        self.assertIsNotNone(risk_mgmt.stoploss)
        self.assertIsNotNone(risk_mgmt.target1)
        self.assertIsNotNone(risk_mgmt.target2)
        self.assertIsNotNone(risk_mgmt.rr_ratio)
        
        # Verify stoploss is below current price for bullish
        self.assertLess(risk_mgmt.stoploss, 50000.0)
        
        # Verify targets are above current price for bullish
        self.assertGreater(risk_mgmt.target1, 50000.0)
        self.assertGreater(risk_mgmt.target2, 50000.0)
        
        # Test bearish scenario
        risk_mgmt_bear = self.agent._calculate_risk_management(
            current_price=50000.0,
            atr=1000.0,
            bias="bearish",
            df=df
        )
        
        # Verify stoploss is above current price for bearish
        self.assertGreater(risk_mgmt_bear.stoploss, 50000.0)
        
        # Verify targets are below current price for bearish
        self.assertLess(risk_mgmt_bear.target1, 50000.0)
        self.assertLess(risk_mgmt_bear.target2, 50000.0)

    def test_market_condition_assessment(self):
        """Test market condition assessment logic"""
        
        # Create mock DataFrame with bullish trend
        import pandas as pd
        import numpy as np
        
        # Create data with uptrend
        prices = [50000.0 + i * 200 for i in range(100)]
        volumes = [1000.0 + i * 50 for i in range(100)]
        
        df = pd.DataFrame({
            'close': prices,
            'volume': volumes
        })
        
        condition = self.agent._assess_market_condition(df)
        
        self.assertIsInstance(condition, MarketCondition)
        self.assertIn(condition.trend, ["bullish", "bearish", "sideways"])
        self.assertIn(condition.volatility, ["low", "medium", "high"])
        self.assertIn(condition.volume, ["low", "normal", "high"])
        self.assertIn(condition.overall, ["bullish", "bearish", "neutral"])

    def test_error_signal_creation(self):
        """Test that error signals are created correctly"""
        
        error_signal = self.agent._create_error_signal("BTC", "crypto", "Test error")
        
        self.assertIsInstance(error_signal, TechnicalSignal)
        self.assertEqual(error_signal.symbol, "BTC")
        self.assertEqual(error_signal.asset_type, "crypto")
        self.assertEqual(error_signal.signal, "HOLD")
        self.assertEqual(error_signal.bias, "neutral")
        self.assertEqual(error_signal.confidence, 0.0)
        self.assertFalse(error_signal.indicators_calculated)
        self.assertFalse(error_signal.onchain_data_available)

    @patch('Agents.technical_agent.TechnicalAgent._get_price_data')
    def test_get_signal_with_api_failure(self, mock_get_price_data):
        """Test signal generation when API fails - should return error signal"""
        
        # Mock the data retrieval to fail (return empty list)
        mock_get_price_data.return_value = []  # Simulate API failure
        
        # This should return an error signal, not fake data
        try:
            signal = self.agent.get_signal('BTC', 'crypto', lookback=50)
            
            # Should be an error signal
            self.assertIsInstance(signal, TechnicalSignal)
            self.assertEqual(signal.symbol, 'BTC')
            self.assertEqual(signal.asset_type, 'crypto')
            self.assertEqual(signal.signal, 'HOLD')  # Error signals should be HOLD
            self.assertEqual(signal.confidence, 0.0)  # Error signals should have 0 confidence
            self.assertFalse(signal.indicators_calculated)  # Should indicate failure
            
        except Exception as e:
            self.fail(f"get_signal with API failure failed: {e}")

    @patch('Agents.technical_agent.CryptoDataProvider.get_candles')
    def test_error_handling_for_api_failure(self, mock_get_candles):
        """Test that API failures return proper errors instead of fake data"""
        
        # Mock the crypto provider to fail
        mock_get_candles.return_value = []  # Simulate API failure
        
        # Test that crypto provider failure returns empty list (not mock data)
        candles = self.agent._get_price_data('BTC', 'crypto', '1d', 10)
        self.assertEqual(candles, [])  # Should return empty list, not mock data
        
        # Test equity provider not implemented (should also return empty)
        candles_equity = self.agent._get_price_data('AAPL', 'equity', '1d', 10)
        self.assertEqual(candles_equity, [])  # Should return empty list, not mock data


if __name__ == '__main__':
    unittest.main()