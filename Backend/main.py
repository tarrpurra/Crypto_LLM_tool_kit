#!/usr/bin/env python3
"""
Main Trading System with Tool Registry Integration and TUI Interface

Orchestrates the complete trading workflow using the Tool Registry system
and provides a Text-Based User Interface for interactive trading.
"""

import json
import logging
import asyncio
import sys
import os
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor

# Set TensorFlow environment variables to suppress warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add Backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import agents
from Agents.user_agent import UserAgent
from Agents.news_agent import NewsAgent
from Agents.risk_agent import RiskAgent, RiskConfig
from Agents.technical_agent import TechnicalAgent
from Agents.ml_agent import MLAgent
from Agents.tool_registry_agent import ToolRegistryAgent
from Core.Thinking_agent import VibeTraderThinker

# Import tool registry
from services.tool_registry import ToolRegistry, ToolMetadata, ToolType, ToolStatus
from services.data_manager import DataManager

# Set up logging with colors
import colorlog

# Create a color formatter
formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'purple',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
        'SUCCESS': 'green',
    },
    secondary_log_colors={},
    style='%'
)

# Configure logging
handler = colorlog.StreamHandler()
handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger('Main')

class TradingSystem:
    """
    Main trading system with tool registry integration.
    
    Orchestrates all agents and provides comprehensive trading workflow management.
    """
    
    def __init__(self, user_id: int, api_key: str):
        """Initialize all agents and the trading system with tool registry integration."""
        self.user_id = user_id
        self.api_key = api_key
        
        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        self.registry_agent = ToolRegistryAgent()
        self.data_manager = DataManager()
        
        # Register all agents with the tool registry
        self._initialize_tool_registry()
        
        # Initialize agents
        self.user_agent = UserAgent(user_id)
        self.news_agent = NewsAgent()
        self.technical_agent = TechnicalAgent()
        self.ml_agent = MLAgent()
        self.risk_agent = RiskAgent(RiskConfig())
        self.thinker = VibeTraderThinker(api_key)

        logger.info("✅ Trading system initialized with all agents and tool registry")
    
    def _initialize_tool_registry(self) -> None:
        """Initialize and register all tools with the tool registry."""
        logger.info("🛠️  Initializing Tool Registry...")
        
        # Register UserAgent
        self.registry_agent.register_agent(
            agent_name="UserAgent",
            agent_type=ToolType.PORTFOLIO,
            version="1.0.0",
            description="Manages user data, portfolio, and account state",
            capabilities=["user_data", "portfolio_management", "account_state"],
            dependencies=[],
            crypto_config={
                "supported_exchanges": ["binance", "coinbase", "kraken"],
                "default_pair": "BTC/USDT",
                "portfolio_tracking": True,
                "multi_wallet_support": True
            }
        )
        
        # Register NewsAgent
        self.registry_agent.register_agent(
            agent_name="NewsAgent",
            agent_type=ToolType.NEWS,
            version="1.0.0",
            description="Fetches and analyzes cryptocurrency news and sentiment",
            capabilities=["news_fetching", "sentiment_analysis", "crypto_data"],
            dependencies=[],
            crypto_config={
                "supported_exchanges": ["binance", "coinbase"],
                "default_pair": "BTC/USDT",
                "data_sources": ["newsapi", "twitter", "reddit"],
                "sentiment_thresholds": {
                    "bullish": None,
                    "bearish": None,
                    "neutral_min": None,
                    "neutral_max": None
                }
            }
        )
        
        # Register TechnicalAgent
        self.registry_agent.register_agent(
            agent_name="TechnicalAgent",
            agent_type=ToolType.TECHNICAL,
            version="1.0.0",
            description="Performs comprehensive technical analysis with 50+ indicators",
            capabilities=["technical_analysis", "market_condition", "risk_management"],
            dependencies=["NewsAgent"],
            crypto_config={
                "supported_exchanges": ["binance", "coinbase", "kraken"],
                "default_pair": "BTC/USDT",
                "indicators": ["rsi", "macd", "bollinger_bands", "atr"],
                "timeframes": ["1m", "1h", "1d", "1w"]
            }
        )
        
        # Register MLAgent
        self.registry_agent.register_agent(
            agent_name="MLAgent",
            agent_type=ToolType.ML,
            version="1.0.0",
            description="Machine learning models for price prediction (LSTM, XGBoost)",
            capabilities=["price_prediction", "signal_generation", "backtesting"],
            dependencies=["TechnicalAgent"],
            crypto_config={
                "supported_exchanges": ["binance", "coinbase"],
                "default_pair": "BTC/USDT",
                "models": ["lstm", "xgboost"],
                "training_periods": ["30d", "90d", "180d", "365d"]
            }
        )
        
        # Register RiskAgent
        self.registry_agent.register_agent(
            agent_name="RiskAgent",
            agent_type=ToolType.RISK,
            version="1.0.0",
            description="Comprehensive risk management and position sizing",
            capabilities=["risk_assessment", "position_sizing", "exposure_management"],
            dependencies=["MLAgent", "UserAgent"],
            crypto_config={
                "supported_exchanges": ["binance", "coinbase", "kraken"],
                "default_pair": "BTC/USDT",
                "risk_parameters": {
                    "max_risk_per_trade": 0.01,
                    "max_exposure": 0.15,
                    "volatility_scaling": True
                }
            }
        )
        
        # Register ThinkingAgent
        self.registry_agent.register_agent(
            agent_name="ThinkingAgent",
            agent_type=ToolType.OTHER,
            version="1.0.0",
            description="LLM-based decision making and recommendation engine",
            capabilities=["decision_making", "recommendation", "multi_agent_coordination"],
            dependencies=["NewsAgent", "TechnicalAgent", "MLAgent", "RiskAgent", "UserAgent"],
            crypto_config={
                "supported_exchanges": ["binance", "coinbase"],
                "default_pair": "BTC/USDT",
                "llm_model": "openrouter/mistralai/mixtral-8x7b-instruct",
                "decision_factors": ["news", "technical", "ml", "risk", "portfolio"]
            }
        )
        
        logger.info("🔧 Tool Registry initialized with all agents registered")
        logger.info(f"   Registered tools: {len(self.registry_agent.registry.discover_tools())}")
    
    def fetch_user_data(self) -> Dict[str, Any]:
        """Fetch user data from the UserAgent with registry integration."""
        # Log tool invocation
        start_time = time.time()
        
        try:
            user_data = self.user_agent.fetch_user_data()
            
            # Log successful invocation
            response_time = time.time() - start_time
            self.tool_registry.log_tool_invocation(
                tool_name="UserAgent",
                success=True,
                response_time=response_time,
                actor="trading_system"
            )
            
            logger.info(f"📊 Fetched user data for user {self.user_id} ({response_time:.3f}s)")
            return user_data
            
        except Exception as e:
            # Log failed invocation
            response_time = time.time() - start_time
            self.tool_registry.log_tool_invocation(
                tool_name="UserAgent",
                success=False,
                response_time=response_time,
                error_message=str(e),
                actor="trading_system"
            )
            
            logger.error(f"❌ Error fetching user data: {e}")
            raise
    
    def fetch_news_data(self, symbol: str, days: int = 1) -> Dict[str, Any]:
        """Fetch news data from the NewsAgent with registry integration."""
        # Log tool invocation
        start_time = time.time()
        
        try:
            news_data = self.news_agent.get_news_signal(symbol, days)
            
            # Log successful invocation
            response_time = time.time() - start_time
            self.tool_registry.log_tool_invocation(
                tool_name="NewsAgent",
                success=True,
                response_time=response_time,
                actor="trading_system"
            )
            
            logger.info(f"📰 Fetched news data for {symbol} ({response_time:.3f}s)")
            return news_data
            
        except Exception as e:
            # Log failed invocation
            response_time = time.time() - start_time
            self.tool_registry.log_tool_invocation(
                tool_name="NewsAgent",
                success=False,
                response_time=response_time,
                error_message=str(e),
                actor="trading_system"
            )
            
            logger.error(f"❌ Error fetching news data for {symbol}: {e}")
            raise
    
    def fetch_technical_data(self, symbol: str, asset_type: str = 'crypto') -> Dict[str, Any]:
        """Fetch technical analysis data from the TechnicalAgent with registry integration."""
        # Log tool invocation
        start_time = time.time()
        
        try:
            technical_signal = self.technical_agent.get_signal(symbol, asset_type)
            
            # Log successful invocation
            response_time = time.time() - start_time
            self.tool_registry.log_tool_invocation(
                tool_name="TechnicalAgent",
                success=True,
                response_time=response_time,
                actor="trading_system"
            )
            
            logger.info(f"📈 Fetched technical data for {symbol} ({response_time:.3f}s)")
            return technical_signal.__dict__
            
        except Exception as e:
            # Log failed invocation
            response_time = time.time() - start_time
            self.tool_registry.log_tool_invocation(
                tool_name="TechnicalAgent",
                success=False,
                response_time=response_time,
                error_message=str(e),
                actor="trading_system"
            )
            
            logger.error(f"❌ Error fetching technical data for {symbol}: {e}")
            raise
    
    def fetch_ml_data(self, symbol: str, data: Any) -> Dict[str, Any]:
        """Fetch ML-based predictions from the MLAgent with registry integration."""
        # Log tool invocation
        start_time = time.time()
        
        try:
            ml_signal = self.ml_agent.get_ml_signal(symbol, data)
            
            # Log successful invocation
            response_time = time.time() - start_time
            self.tool_registry.log_tool_invocation(
                tool_name="MLAgent",
                success=True,
                response_time=response_time,
                actor="trading_system"
            )
            
            logger.info(f"🤖 Fetched ML data for {symbol} ({response_time:.3f}s)")
            return ml_signal
            
        except Exception as e:
            # Log failed invocation
            response_time = time.time() - start_time
            self.tool_registry.log_tool_invocation(
                tool_name="MLAgent",
                success=False,
                response_time=response_time,
                error_message=str(e),
                actor="trading_system"
            )
            
            logger.error(f"❌ Error fetching ML data for {symbol}: {e}")
            return {
                "ticker": symbol,
                "error": str(e)
            }
    
    def evaluate_risk(self, trade_request: Dict[str, Any], account_state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate risk using the RiskAgent with registry integration."""
        # Log tool invocation
        start_time = time.time()
        
        try:
            # Convert dictionaries to dataclasses
            from Backend.Agents.risk_agent import TradeRequest as RiskTradeRequest, AccountState as RiskAccountState
            
            risk_trade_request = RiskTradeRequest(
                symbol=trade_request.get('symbol', ''),
                market=trade_request.get('market', 'crypto'),
                side=trade_request.get('side', 'BUY'),
                entry_price=trade_request.get('entry_price', 0.0),
                stop_loss=trade_request.get('stop_loss', 0.0),
                take_profit=trade_request.get('take_profit', None),
                signal_confidence=trade_request.get('signal_confidence', 0.7)
            )

            risk_account_state = RiskAccountState(
                equity_value=account_state.get('equity_value', 0.0),
                cash_available=account_state.get('cash_available', 0.0),
                open_positions=account_state.get('open_positions', []),
                day_pnl=account_state.get('day_pnl', 0.0)
            )

            risk_decision = self.risk_agent.evaluate(risk_trade_request, risk_account_state)
            
            # Log successful invocation
            response_time = time.time() - start_time
            self.tool_registry.log_tool_invocation(
                tool_name="RiskAgent",
                success=True,
                response_time=response_time,
                actor="trading_system"
            )
            
            logger.info(f"🛡️  Evaluated risk for trade request ({response_time:.3f}s)")
            return risk_decision.__dict__
            
        except Exception as e:
            # Log failed invocation
            response_time = time.time() - start_time
            self.tool_registry.log_tool_invocation(
                tool_name="RiskAgent",
                success=False,
                response_time=response_time,
                error_message=str(e),
                actor="trading_system"
            )
            
            logger.error(f"❌ Error evaluating risk: {e}")
            raise
    
    def generate_recommendation(self, user_data: Dict[str, Any], news_data: Dict[str, Any], 
                                 technical_data: Dict[str, Any], ml_data: Dict[str, Any],
                                 risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a trading recommendation using the ThinkingAgent with registry integration."""
        # Log tool invocation
        start_time = time.time()
        
        try:
            # Prepare the prompt for the ThinkingAgent
            prompt = self._prepare_thinking_prompt(user_data, news_data, technical_data, ml_data, risk_data)

            # Generate recommendation
            recommendation = self.thinker.run(prompt)
            
            # Log successful invocation
            response_time = time.time() - start_time
            self.tool_registry.log_tool_invocation(
                tool_name="ThinkingAgent",
                success=True,
                response_time=response_time,
                actor="trading_system"
            )
            
            logger.info(f"🧠 Generated trading recommendation ({response_time:.3f}s)")
            return recommendation
            
        except Exception as e:
            # Log failed invocation
            response_time = time.time() - start_time
            self.tool_registry.log_tool_invocation(
                tool_name="ThinkingAgent",
                success=False,
                response_time=response_time,
                error_message=str(e),
                actor="trading_system"
            )
            
            logger.error(f"❌ Error generating recommendation: {e}")
            raise
    
    def _prepare_thinking_prompt(self, user_data: Dict[str, Any], news_data: Dict[str, Any],
                                  technical_data: Dict[str, Any], ml_data: Dict[str, Any],
                                  risk_data: Dict[str, Any]) -> str:
        """Prepare a structured prompt for the ThinkingAgent."""
        prompt = """
        User Data:
        {user_data}

        News Data:
        {news_data}

        Technical Data:
        {technical_data}

        ML Data:
        {ml_data}

        Risk Data:
        {risk_data}

        Based on the above data, provide a comprehensive trading recommendation.
        """
        return prompt.format(
            user_data=json.dumps(user_data, indent=2),
            news_data=json.dumps(news_data, indent=2),
            technical_data=json.dumps(technical_data, indent=2),
            ml_data=json.dumps(ml_data, indent=2),
            risk_data=json.dumps(risk_data, indent=2)
        )
    
    async def fetch_data_parallel(self, symbol: str, asset_type: str = 'crypto') -> Dict[str, Any]:
        """Fetch data from all agents in parallel using asyncio."""
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=4)
        
        # Define tasks to run in parallel
        tasks = {
            'user_data': loop.run_in_executor(executor, self.fetch_user_data),
            'news_data': loop.run_in_executor(executor, lambda: self.fetch_news_data(symbol)),
            'technical_data': loop.run_in_executor(executor, lambda: self.fetch_technical_data(symbol, asset_type)),
        }
        
        # Run tasks concurrently
        results = await asyncio.gather(*tasks.values())
        
        # Map results back to their keys
        data = {}
        for key, result in zip(tasks.keys(), results):
            data[key] = result
        
        return data
    
    def run_trading_workflow(self, symbol: str, asset_type: str = 'crypto') -> Dict[str, Any]:
        """Run the complete trading workflow for a given symbol using parallel execution."""
        logger.info(f"🚀 Starting trading workflow for {symbol}")
        
        # Validate tool chain before execution
        tool_chain = ["NewsAgent", "TechnicalAgent", "MLAgent", "RiskAgent", "ThinkingAgent"]
        chain_valid, chain_msg = self.tool_registry.validate_tool_chain(tool_chain)
        
        if not chain_valid:
            logger.error(f"❌ Invalid tool chain: {chain_msg}")
            return {
                "error": f"Tool chain validation failed: {chain_msg}",
                "symbol": symbol
            }
        
        logger.info(f"✅ Tool chain validated: {' -> '.join(tool_chain)}")

        try:
            # Step 1: Fetch data from all agents in parallel
            data = asyncio.run(self.fetch_data_parallel(symbol, asset_type))
            user_data = data['user_data']
            news_data = data['news_data']
            technical_data = data['technical_data']

            # Step 2: Fetch ML data (real data from MLAgent)
            # Fetch historical data for ML analysis
            historical_data = self.data_manager.fetch_historical_data(symbol, days=365, asset_type='crypto')
            
            if historical_data is not None and not historical_data.empty:
                ml_data = self.fetch_ml_data(symbol, historical_data)
            else:
                logger.warning(f"⚠️ No historical data available for {symbol}, using mock ML data")
                ml_data = {
                    "ticker": symbol,
                    "current_price": technical_data.get('current_price', 0.0),
                    "predicted_price": technical_data.get('current_price', 0.0) * 1.05,  # Mock prediction
                    "signal": "bullish",
                    "confidence": 0.85
                }

            # Step 3: Evaluate risk (real trade request based on ML signal)
            trade_request = {
                "symbol": symbol,
                "market": asset_type,
                "side": ml_data.get('signal', 'BUY'),
                "entry_price": technical_data.get('current_price', 0.0),
                "stop_loss": technical_data.get('current_price', 0.0) * 0.95,  # 5% stop loss
                "take_profit": technical_data.get('current_price', 0.0) * 1.10,  # 10% take profit
                "signal_confidence": ml_data.get('confidence', 0.7)
            }
            risk_data = self.evaluate_risk(trade_request, user_data.get('account_state', {}))

            # Step 4: Generate recommendation
            recommendation = self.generate_recommendation(user_data, news_data, technical_data, ml_data, risk_data)

            # Step 5: Add registry metrics to recommendation
            recommendation['_registry_metrics'] = {
                "tool_chain": tool_chain,
                "registry_health": self.tool_registry.check_system_health(),
                "timestamp": datetime.now().isoformat()
            }

            logger.info(f"✅ Completed trading workflow for {symbol}")
            
            # Save registry state after successful workflow
            self.data_manager._save_registry()
            
            return recommendation

        except Exception as e:
            logger.error(f"❌ Error in trading workflow for {symbol}: {e}")
            
            # Update tool statuses on failure
            self.tool_registry.update_tool_status("ThinkingAgent", ToolStatus.FAILED, str(e))
            
            return {
                "error": str(e),
                "symbol": symbol,
                "registry_status": self.tool_registry.check_system_health()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status including all agents and registry."""
        return {
            "system": "TradingSystem",
            "user_id": self.user_id,
            "agents": self.registry_agent.get_all_agents(),
            "registry_health": self.tool_registry.check_system_health(),
            "workflow_metrics": self.registry_agent.get_workflow_metrics(),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_agent_status(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific agent."""
        return self.registry_agent.get_agent_status(agent_name)


class TradingTUI:
    """
    Text-Based User Interface for the Trading System.
    
    Provides interactive command-line interface for trading operations.
    """
    
    def __init__(self, trading_system: TradingSystem):
        """Initialize the TUI with a trading system."""
        self.system = trading_system
        self.running = False
    
    def start(self) -> None:
        """Start the TUI main loop."""
        self.running = True
        
        print("\n" + "="*60)
        print("🚀 TRADING SYSTEM TUI")
        print("="*60)
        print("Type 'help' for available commands, 'exit' to quit\n")
        
        while self.running:
            try:
                command = input("> ").strip()
                self._process_command(command)
            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                self.running = False
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _process_command(self, command: str) -> None:
        """Process user commands."""
        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower() if parts else ""
        args = parts[1:]
        
        if cmd == "help":
            self._show_help()
        
        elif cmd == "exit" or cmd == "quit":
            self.running = False
            print("👋 Goodbye!")
        
        elif cmd == "status":
            self._show_status()
        
        elif cmd == "agents":
            self._show_agents()
        
        elif cmd == "trade":
            if len(args) >= 1:
                self._run_trade_workflow(args[0])
            else:
                print("❌ Usage: trade <symbol> [asset_type]")
        
        elif cmd == "registry":
            self._show_registry_info()
        
        elif cmd == "health":
            self._show_health()
        
        elif cmd == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n" + "="*60)
            print("🚀 TRADING SYSTEM TUI")
            print("="*60)
        
        else:
            print(f"❌ Unknown command: {cmd}")
    
    def _show_help(self) -> None:
        """Show help information."""
        print("\n📚 Available Commands:")
        print("  help              - Show this help message")
        print("  exit/quit         - Exit the TUI")
        print("  status            - Show system status")
        print("  agents            - List all registered agents")
        print("  trade <symbol>    - Run trading workflow for symbol (e.g., trade BTC)")
        print("  registry          - Show registry information")
        print("  health            - Show system health metrics")
        print("  clear             - Clear the screen")
        print()
    
    def _show_status(self) -> None:
        """Show system status."""
        status = self.system.get_system_status()
        
        print("\n📊 SYSTEM STATUS")
        print("-" * 40)
        print(f"User ID: {status['user_id']}")
        print(f"Timestamp: {status['timestamp']}")
        print(f"\nRegistry Health: {status['registry_health']['registry_status']}")
        print(f"  Healthy Tools: {status['registry_health']['healthy_tools']}/{status['registry_health']['tool_count']}")
        print(f"\nWorkflow Metrics:")
        print(f"  Total Agents: {status['workflow_metrics']['total_agents']}")
        print(f"  Healthy Agents: {status['workflow_metrics']['healthy_agents']}")
        print(f"  Compliance Level: {status['workflow_metrics']['compliance']['compliance_level']}")
        print()
    
    def _show_agents(self) -> None:
        """Show all registered agents."""
        agents = self.system.registry_agent.get_all_agents()
        
        print("\n🤖 REGISTERED AGENTS")
        print("-" * 40)
        print(f"Total: {len(agents)} agents\n")
        
        for agent in agents:
            status_icon = "✅" if agent["status"] == "HEALTHY" else "⚠️"
            print(f"{status_icon} {agent['name']} ({agent['type']})")
            print(f"   Version: {agent['version']}")
            print(f"   Status: {agent['status']}")
            print(f"   Capabilities: {', '.join(agent['capabilities'])}")
            if agent['dependencies']:
                print(f"   Dependencies: {', '.join(agent['dependencies'])}")
            print()
    
    def _run_trade_workflow(self, symbol: str) -> None:
        """Run trading workflow for a symbol."""
        asset_type = 'crypto'
        
        print(f"\n🔍 Running trading workflow for {symbol}...")
        
        try:
            result = self.system.run_trading_workflow(symbol, asset_type)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"\n📊 RECOMMENDATION FOR {symbol}")
                print("-" * 40)
                print(f"Recommendation: {result['recommendation']}")
                print(f"Confidence: {result['confidence']}%")
                print(f"\nKey Drivers:")
                for driver in result['key_drivers']:
                    print(f"  • {driver['source_agent']}: {driver['summary']} (weight: {driver['weight']})")
                
                print(f"\nRisk Plan:")
                print(f"  Position Size: {result['risk_plan']['position_size_pct']}%")
                print(f"  Entry: {result['risk_plan']['entry']['type']}")
                print(f"  Stop Loss: {result['risk_plan']['stop_loss']['type']} at {result['risk_plan']['stop_loss']['value']}")
                print(f"  Take Profit: {result['risk_plan']['take_profit']['type']} at {result['risk_plan']['take_profit']['value']}")
                print(f"  Time Horizon: {result['risk_plan']['time_horizon']}")
                
                if 'invalidations' in result['risk_plan'] and result['risk_plan']['invalidations']:
                    print(f"\nInvalidations:")
                    for inv in result['risk_plan']['invalidations']:
                        print(f"  • {inv}")
                
                if 'notes' in result and result['notes']:
                    print(f"\nNotes:")
                    for note in result['notes']:
                        print(f"  • {note}")
                
                if '_registry_metrics' in result:
                    print(f"\nRegistry Metrics:")
                    print(f"  Tool Chain: {' → '.join(result['_registry_metrics']['tool_chain'])}")
                    print(f"  Registry Status: {result['_registry_metrics']['registry_health']['registry_status']}")
        
        except Exception as e:
            print(f"❌ Error running workflow: {e}")
    
    def _show_registry_info(self) -> None:
        """Show registry information."""
        health = self.system.tool_registry.check_system_health()
        metrics = self.system.registry_agent.get_workflow_metrics()
        
        print("\n📋 REGISTRY INFORMATION")
        print("-" * 40)
        print(f"Registry Status: {health['registry_status']}")
        print(f"Total Tools: {health['tool_count']}")
        print(f"Healthy Tools: {health['healthy_tools']}")
        print(f"Degraded Tools: {health['degraded_tools']}")
        print(f"Failed Tools: {health['failed_tools']}")
        print(f"\nWorkflow Metrics:")
        print(f"  Total Agents: {metrics['total_agents']}")
        print(f"  Healthy Agents: {metrics['healthy_agents']}")
        print(f"  Compliance Level: {metrics['compliance']['compliance_level']}")
        print(f"  Compliance Rate: {metrics['compliance']['compliance_rate']:.1%}")
        print()
    
    def _show_health(self) -> None:
        """Show detailed system health."""
        health_report = self.system.registry_agent.monitor_agent_health()
        
        print("\n🏥 SYSTEM HEALTH REPORT")
        print("-" * 40)
        print(f"Generated: {health_report['timestamp']}")
        print(f"\nSummary:")
        print(f"  Total Agents: {health_report['summary']['total_agents']}")
        print(f"  Healthy: {health_report['summary']['healthy']}")
        print(f"  Degraded: {health_report['summary']['degraded']}")
        print(f"  Failed: {health_report['summary']['failed']}")
        print(f"  Health Rate: {health_report['summary']['health_rate']:.1%}")
        
        print(f"\nAgent Health:")
        for agent_name, agent_health in health_report['agents'].items():
            status_icon = "✅" if agent_health['healthy'] else "❌"
            print(f"  {status_icon} {agent_name}: {agent_health['status']}")
        print()


def main():
    """Main entry point for the trading system."""
    # Load API key from configuration
    config_path = 'configs/api_keys.json'
    
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        # Try different key names
        api_key = config.get('openrouter_key') or config.get('DATA_BASE') or config.get('api_key')
        
        if not api_key:
            logger.warning("⚠️ No API key found in configuration. Using placeholder.")
            api_key = "placeholder_api_key"
    
    except Exception as e:
        logger.error(f"❌ Error loading API key: {e}")
        api_key = "placeholder_api_key"
    
    # Initialize the trading system
    trading_system = TradingSystem(user_id=1, api_key=api_key)
    
    # Start the TUI
    tui = TradingTUI(trading_system)
    tui.start()


if __name__ == "__main__":
    main()
