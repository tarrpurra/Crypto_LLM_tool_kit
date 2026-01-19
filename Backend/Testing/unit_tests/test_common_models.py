"""
Unit tests for common models and data structures
"""

import sys
import os
import unittest
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from services.common_models import TechnicalIndicators, StrengthScores, OnchainSignals, RiskManagement


class TestCommonModels(unittest.TestCase):
    """Test common data models used across the system"""

    def test_technical_indicators_creation(self):
        """Test TechnicalIndicators model creation and validation"""
        
        # Test valid creation
        indicators = TechnicalIndicators(
            rsi=65.0,
            macd=123.45,
            macd_signal=98.76,
            ema20=50000.0,
            ema50=48000.0,
            ema200=45000.0
        )
        
        self.assertEqual(indicators.rsi, 65.0)
        self.assertEqual(indicators.macd, 123.45)
        self.assertEqual(indicators.ema20, 50000.0)
        
        # Test with None values (should be allowed)
        indicators_none = TechnicalIndicators(rsi=None, macd=None)
        self.assertIsNone(indicators_none.rsi)
        self.assertIsNone(indicators_none.macd)

    def test_strength_scores_creation(self):
        """Test StrengthScores model creation and validation"""
        
        # Test valid creation
        scores = StrengthScores(
            technical_strength=75.0,
            onchain_score=60.0,
            onchain_accumulation_score=8.0,
            onchain_whale_score=7.0,
            final_strength=80.0
        )
        
        self.assertEqual(scores.technical_strength, 75.0)
        self.assertEqual(scores.onchain_score, 60.0)
        self.assertEqual(scores.final_strength, 80.0)
        
        # Test score ranges
        self.assertTrue(0 <= scores.technical_strength <= 100)
        self.assertTrue(0 <= scores.final_strength <= 100)

    def test_onchain_signals_creation(self):
        """Test OnchainSignals model creation and validation"""
        
        # Test valid creation
        onchain = OnchainSignals(
            smart_money_inflow=1000000,
            whale_tx_count=25,
            holder_concentration=0.45,
            exchange_net_flow=-500000
        )
        
        self.assertEqual(onchain.smart_money_inflow, 1000000)
        self.assertEqual(onchain.whale_tx_count, 25)
        self.assertEqual(onchain.holder_concentration, 0.45)
        
        # Test with zero/negative values
        onchain_zero = OnchainSignals(
            smart_money_inflow=0,
            whale_tx_count=0,
            holder_concentration=0.0
        )
        self.assertEqual(onchain_zero.smart_money_inflow, 0)
        self.assertEqual(onchain_zero.whale_tx_count, 0)

    def test_risk_management_creation(self):
        """Test RiskManagement model creation and validation"""
        
        # Test valid creation
        risk_mgmt = RiskManagement(
            stoploss=48000.0,
            target1=52000.0,
            target2=55000.0,
            rr_ratio=2.5
        )
        
        self.assertEqual(risk_mgmt.stoploss, 48000.0)
        self.assertEqual(risk_mgmt.target1, 52000.0)
        self.assertEqual(risk_mgmt.rr_ratio, 2.5)
        
        # Test with None values (optional fields)
        risk_none = RiskManagement(stoploss=None, target1=None)
        self.assertIsNone(risk_none.stoploss)
        self.assertIsNone(risk_none.target1)

    def test_model_serialization(self):
        """Test that models can be serialized to dict"""
        
        indicators = TechnicalIndicators(rsi=70.0, ema20=51000.0)
        scores = StrengthScores(technical_strength=80.0, final_strength=85.0)
        
        # Test serialization (using model_dump for Pydantic V2)
        indicators_dict = indicators.model_dump()
        scores_dict = scores.model_dump()
        
        self.assertEqual(indicators_dict['rsi'], 70.0)
        self.assertEqual(scores_dict['technical_strength'], 80.0)
        
        # Test that serialization includes all fields
        self.assertIn('rsi', indicators_dict)
        self.assertIn('technical_strength', scores_dict)

    def test_model_equality(self):
        """Test model equality comparison"""
        
        indicators1 = TechnicalIndicators(rsi=65.0, macd=123.45)
        indicators2 = TechnicalIndicators(rsi=65.0, macd=123.45)
        indicators3 = TechnicalIndicators(rsi=70.0, macd=123.45)
        
        self.assertEqual(indicators1, indicators2)
        self.assertNotEqual(indicators1, indicators3)


if __name__ == '__main__':
    unittest.main()