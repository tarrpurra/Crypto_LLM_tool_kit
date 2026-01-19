#!/usr/bin/env python3
"""
Simple test to check if imports work
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from services.common_models import TechnicalIndicators, StrengthScores, OnchainSignals
    print("✅ Successfully imported TechnicalIndicators, StrengthScores, OnchainSignals")

    # Test creating instances
    indicators = TechnicalIndicators(rsi=65.0, ema20=50000.0)
    print(f"✅ Created TechnicalIndicators: RSI={indicators.rsi}")

    scores = StrengthScores(technical_strength=75.0, final_strength=80.0)
    print(f"✅ Created StrengthScores: final_strength={scores.final_strength}")

    onchain = OnchainSignals(smart_money_inflow=1000000, whale_tx_count=25)
    print(f"✅ Created OnchainSignals: smart_money_inflow=${onchain.smart_money_inflow:,.0f}")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")