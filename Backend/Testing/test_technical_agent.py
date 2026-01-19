#!/usr/bin/env python3
"""
Test script for Technical Agent
Demonstrates integration with Nansen API and signal generation
"""

import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from Agents.technical_agent import TechnicalAgent
import json


def test_technical_agent():
    """Test the Technical Agent with sample data"""

    print("Testing Technical Agent")
    print("=" * 50)

    # Initialize agent (loads Nansen API key from config)
    agent = TechnicalAgent()

    # Test assets
    test_assets = [
        ('BTC', 'crypto'),
        ('ETH', 'crypto'),

    ]

    results = {}

    for symbol, asset_type in test_assets:
        print(f"\nAnalyzing {symbol} ({asset_type})")
        print("-" * 30)

        try:
            # Get technical signal
            signal = agent.get_signal(symbol, asset_type, lookback=100)

            # Display results
            print(f"Signal: {signal.signal}")
            print(f"Bias: {signal.bias}")
            print(f"Confidence: {signal.confidence:.2f}")
            print(f"Current Price: ${signal.current_price:.2f}")
            print(f"Final Strength: {signal.strength_scores.final_strength:.1f}/100")

            # Technical breakdown
            scores = signal.strength_scores
            print(f"Technical Strength: {scores.technical_strength:.1f}")
            print(f"On-chain Score: {scores.onchain_score:.1f}")
            print(f"Accumulation Score: {scores.onchain_accumulation_score:.1f}/10")
            print(f"Whale Score: {scores.onchain_whale_score:.1f}/10")

            # Risk management
            if signal.risk_management.stoploss:
                print(f"Stop Loss: ${signal.risk_management.stoploss:.2f}")
            if signal.risk_management.target1:
                print(f"Target 1: ${signal.risk_management.target1:.2f}")
            if signal.risk_management.rr_ratio:
                print(f"Risk-Reward: {signal.risk_management.rr_ratio:.2f}")

            # Indicators
            ind = signal.technical_indicators
            print(f"RSI: {ind.rsi:.1f}")
            print(f"MACD: {ind.macd:.2f} (Signal: {ind.macd_signal:.2f})")
            print(f"EMA20: ${ind.ema20:.2f}")

            # Market condition
            condition = agent.get_market_condition(symbol, asset_type)
            print(f"Market: {condition.overall} (trend: {condition.trend}, vol: {condition.volatility})")

            # Store result
            results[symbol] = {
                'signal': signal.signal,
                'confidence': signal.confidence,
                'strength': signal.strength_scores.final_strength,
                'bias': signal.bias
            }

        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            results[symbol] = {'error': str(e)}

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    for symbol, result in results.items():
        if 'error' not in result:
            print(f"{symbol}: {result['signal']} ({result['confidence']:.2f} confidence, {result['strength']:.1f} strength)")
        else:
            print(f"{symbol}: ERROR - {result['error']}")

    return results


def test_with_nansen():
    """Test with Nansen API (requires API key)"""

    print("\nTesting with Nansen API")
    print("=" * 50)

    # Check for API key
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'api_keys.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

        nansen_key = config.get('nansen_api_key')
        if nansen_key:
            print("Nansen API key found")

            # Initialize with real API (loads from config)
            agent = TechnicalAgent()

            # Test BTC with real on-chain data
            signal = agent.get_signal('BTC', 'crypto', lookback=30)

            print(f"BTC Signal: {signal.signal}")
            print(f"Confidence: {signal.confidence:.2f}")
            print(f"On-chain Available: {signal.onchain_data_available}")

            if signal.onchain_signals:
                onchain = signal.onchain_signals
                print(f"Smart Money Inflow: ${onchain.smart_money_inflow:,.0f}")
                print(f"Whale TX Count: {onchain.whale_tx_count}")
                print(f"Holder Concentration: {onchain.holder_concentration:.2f}")

        else:
            print("⚠️  Nansen API key not found in config")
    else:
        print("⚠️  Config file not found")


if __name__ == "__main__":
    # Run basic tests
    results = test_technical_agent()

    # Test with Nansen if available
    test_with_nansen()

    print("\nTechnical Agent testing complete!")
    print("\nNext steps:")
    print("1. Add Nansen API key to Backend/configs/api_keys.json")
    print("2. Implement real price feed (Binance/Coinbase API)")
    print("3. Integrate with MultiAgentController")
    print("4. Add backtesting framework")