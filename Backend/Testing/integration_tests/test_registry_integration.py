#!/usr/bin/env python3
"""
Integration Tests for Tool Registry System

Tests the integration of the Tool Registry with the main trading system,
including workflow validation, dependency management, and real-world scenarios.
"""

import unittest
import sys
import os
import time
from datetime import datetime

# Add Backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import TradingSystem
from services.tool_registry import ToolRegistry, ToolMetadata, ToolType, ToolStatus
from services.data_manager import DataManager


class TestRegistryIntegration(unittest.TestCase):
    """Integration tests for Tool Registry with Trading System."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use a test API key
        self.test_api_key = "test_api_key_12345"
        self.test_user_id = 1
        
        # Initialize trading system (which initializes registry)
        self.trading_system = TradingSystem(self.test_user_id, self.test_api_key)
    
    def test_trading_system_initialization(self):
        """Test that trading system properly initializes the tool registry."""
        # Check that registry is initialized
        self.assertIsNotNone(self.trading_system.tool_registry)
        self.assertIsInstance(self.trading_system.tool_registry, ToolRegistry)
        
        # Check that all expected tools are registered
        expected_tools = ["UserAgent", "NewsAgent", "TechnicalAgent", "MLAgent", "RiskAgent", "ThinkingAgent"]
        
        for tool_name in expected_tools:
            tool = self.trading_system.tool_registry.get_tool(tool_name)
            self.assertIsNotNone(tool, f"Tool {tool_name} should be registered")
            self.assertEqual(tool.status, ToolStatus.HEALTHY)
    
    def test_tool_invocation_logging(self):
        """Test that tool invocations are properly logged."""
        # Get initial invocation counts
        initial_user_invocations = self.trading_system.tool_registry._tools["UserAgent"].invocation_count
        initial_news_invocations = self.trading_system.tool_registry._tools["NewsAgent"].invocation_count
        
        # Fetch data (which should log invocations)
        user_data = self.trading_system.fetch_user_data()
        news_data = self.trading_system.fetch_news_data("BTC")
        
        # Check that invocations were logged
        user_invocations = self.trading_system.tool_registry._tools["UserAgent"].invocation_count
        news_invocations = self.trading_system.tool_registry._tools["NewsAgent"].invocation_count
        
        self.assertEqual(user_invocations, initial_user_invocations + 1)
        self.assertEqual(news_invocations, initial_news_invocations + 1)
        
        # Verify metrics are updated
        user_metrics = self.trading_system.tool_registry.get_tool_metrics("UserAgent")
        self.assertIsNotNone(user_metrics)
        self.assertGreater(user_metrics["invocation_count"], 0)
        self.assertGreater(user_metrics["last_invocation"], 0)
    
    def test_workflow_validation(self):
        """Test that the trading workflow validates tool chains."""
        # Test with valid tool chain
        result = self.trading_system.run_trading_workflow("BTC", "crypto")
        
        # Should have registry metrics in the result
        self.assertIn("_registry_metrics", result)
        registry_metrics = result["_registry_metrics"]
        
        # Verify tool chain is recorded
        self.assertIn("tool_chain", registry_metrics)
        self.assertGreater(len(registry_metrics["tool_chain"]), 0)
        
        # Verify system health is included
        self.assertIn("registry_health", registry_metrics)
    
    def test_registry_persistence(self):
        """Test that registry state persists across sessions."""
        # Get current registry state
        original_tools = self.trading_system.tool_registry.discover_tools()
        original_count = len(original_tools)
        
        # Save registry
        self.trading_system.data_manager._save_registry()
        
        # Create new registry instance
        new_registry = ToolRegistry()
        new_data_manager = DataManager()
        
        # Load the saved registry
        new_data_manager._load_registry()
        
        # Check that tools are restored
        restored_tools = new_registry.discover_tools()
        self.assertEqual(len(restored_tools), original_count)
        
        # Check that tool metadata is preserved
        for tool in restored_tools:
            self.assertIsNotNone(tool.name)
            self.assertIsNotNone(tool.version)
            self.assertIsInstance(tool.status, ToolStatus)
    
    def test_dependency_validation_in_workflow(self):
        """Test that workflow respects tool dependencies."""
        # Manually break a dependency to test validation
        # First, get the current state
        original_status = self.trading_system.tool_registry._tools["MLAgent"].status
        
        # Set MLAgent to failed status
        self.trading_system.tool_registry.update_tool_status(
            "MLAgent", 
            ToolStatus.FAILED, 
            "Test failure"
        )
        
        # Try to run workflow - should handle gracefully
        result = self.trading_system.run_trading_workflow("BTC", "crypto")
        
        # Restore original status
        self.trading_system.tool_registry.update_tool_status(
            "MLAgent", 
            original_status, 
            "Restored"
        )
        
        # The workflow should still complete (with mock data) but log the issue
        self.assertIn("recommendation", result)
    
    def test_registry_metrics_in_workflow(self):
        """Test that registry metrics are properly collected during workflow."""
        # Run workflow
        result = self.trading_system.run_trading_workflow("BTC", "crypto")
        
        # Verify registry metrics are included
        self.assertIn("_registry_metrics", result)
        metrics = result["_registry_metrics"]
        
        # Check metrics structure
        self.assertIn("tool_chain", metrics)
        self.assertIn("registry_health", metrics)
        self.assertIn("timestamp", metrics)
        
        # Verify health metrics
        health = metrics["registry_health"]
        self.assertIn("registry_status", health)
        self.assertIn("tool_count", health)
        self.assertIn("healthy_tools", health)
    
    def test_error_handling_and_recovery(self):
        """Test that the system handles errors gracefully and maintains registry integrity."""
        # Get initial tool count
        initial_tools = len(self.trading_system.tool_registry.discover_tools())
        
        # Run workflow with potentially problematic symbol
        result = self.trading_system.run_trading_workflow("INVALID_SYMBOL", "crypto")
        
        # Should handle error gracefully
        self.assertIn("error", result)
        
        # Registry should still be intact
        final_tools = len(self.trading_system.tool_registry.discover_tools())
        self.assertEqual(initial_tools, final_tools)
        
        # All tools should still be healthy
        healthy_tools = self.trading_system.tool_registry.discover_tools(status=ToolStatus.HEALTHY)
        self.assertEqual(len(healthy_tools), initial_tools)
    
    def test_concurrent_workflow_execution(self):
        """Test concurrent execution of trading workflows."""
        import threading
        
        results = []
        errors = []
        
        def run_workflow(symbol):
            try:
                result = self.trading_system.run_trading_workflow(symbol, "crypto")
                results.append((symbol, result))
            except Exception as e:
                errors.append((symbol, str(e)))
        
        # Run multiple workflows concurrently
        symbols = ["BTC", "ETH", "SOL"]
        threads = []
        
        for symbol in symbols:
            thread = threading.Thread(target=lambda s=symbol: run_workflow(s))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(len(results), 3)
        self.assertEqual(len(errors), 0)
        
        # Verify all results have registry metrics
        for symbol, result in results:
            self.assertIn("_registry_metrics", result)
            self.assertIn("recommendation", result)
    
    def test_registry_backup_and_restore(self):
        """Test registry backup and restore functionality."""
        # Create backup
        backup = self.trading_system.data_manager.get_registry_backup()
        
        # Verify backup structure
        self.assertIn("tools", backup)
        self.assertIn("dependencies", backup)
        self.assertIn("audit_log", backup)
        self.assertIn("metrics", backup)
        
        # Verify backup content
        self.assertGreater(len(backup["tools"]), 0)
        self.assertGreater(len(backup["dependencies"]), 0)
        
        # Restore from backup
        success, message = self.trading_system.data_manager.restore_registry_backup(backup)
        
        self.assertTrue(success)
        
        # Verify tools are restored
        restored_tools = self.trading_system.tool_registry.discover_tools()
        self.assertEqual(len(restored_tools), len(backup["tools"]))
    
    def test_compliance_validation_in_workflow(self):
        """Test that workflow validates tool compliance."""
        # All tools should be compliant after initialization
        tools = self.trading_system.tool_registry.discover_tools()
        
        for tool in tools:
            compliant, message = self.trading_system.tool_registry.validate_crypto_compliance(tool.name)
            self.assertTrue(compliant, f"Tool {tool.name} should be compliant: {message}")
        
        # Run workflow
        result = self.trading_system.run_trading_workflow("BTC", "crypto")
        
        # Should complete successfully with compliant tools
        self.assertIn("recommendation", result)


class TestRegistryPerformance(unittest.TestCase):
    """Performance tests for Tool Registry operations."""
    
    def setUp(self):
        self.registry = ToolRegistry()
        self.trading_system = TradingSystem(1, "test_key")
    
    def test_high_volume_registrations(self):
        """Test performance with many tool registrations."""
        import time
        
        start_time = time.time()
        
        # Register 100 tools
        for i in range(100):
            metadata = ToolMetadata(
                name=f"PerfTool{i}",
                version="1.0.0",
                tool_type=ToolType.TECHNICAL,
                description=f"Performance test tool {i}",
                capabilities=["performance"],
                dependencies=[],
                crypto_config={
                    "supported_exchanges": ["binance"],
                    "default_pair": "BTC/USDT"
                }
            )
            self.registry.register_tool(metadata, "perf_test")
        
        registration_time = time.time() - start_time
        
        print(f"\nHigh Volume Registration Test:")
        print(f"  Registered 100 tools in {registration_time:.3f}s")
        print(f"  Average: {registration_time/100*1000:.1f}ms per tool")
        
        # Should complete in reasonable time
        self.assertLess(registration_time, 1.0)
    
    def test_workflow_performance(self):
        """Test end-to-end workflow performance."""
        import time
        
        # Run workflow multiple times
        iterations = 5
        total_time = 0
        
        for i in range(iterations):
            start_time = time.time()
            result = self.trading_system.run_trading_workflow("BTC", "crypto")
            iteration_time = time.time() - start_time
            total_time += iteration_time
            
            # Verify result
            self.assertIn("recommendation", result)
        
        average_time = total_time / iterations
        
        print(f"\nWorkflow Performance Test:")
        print(f"  {iterations} iterations completed in {total_time:.3f}s")
        print(f"  Average: {average_time:.3f}s per workflow")
        
        # Should complete in reasonable time
        self.assertLess(average_time, 10.0)  # Less than 10 seconds per workflow


class TestRegistryRealWorldScenarios(unittest.TestCase):
    """Real-world scenario tests for Tool Registry."""
    
    def setUp(self):
        self.trading_system = TradingSystem(1, "test_key")
    
    def test_tool_degradation_and_recovery(self):
        """Test handling of tool degradation and recovery."""
        # Get initial healthy tools
        initial_healthy = len(self.trading_system.tool_registry.discover_tools(status=ToolStatus.HEALTHY))
        
        # Degrade a tool
        self.trading_system.tool_registry.update_tool_status(
            "NewsAgent", 
            ToolStatus.DEGRADED, 
            "Simulated network issue"
        )
        
        # Verify degradation
        degraded_tools = self.trading_system.tool_registry.discover_tools(status=ToolStatus.DEGRADED)
        self.assertEqual(len(degraded_tools), 1)
        self.assertEqual(degraded_tools[0].name, "NewsAgent")
        
        # Run workflow - should handle degraded tool gracefully
        result = self.trading_system.run_trading_workflow("BTC", "crypto")
        self.assertIn("recommendation", result)
        
        # Recover the tool
        self.trading_system.tool_registry.update_tool_status(
            "NewsAgent", 
            ToolStatus.HEALTHY, 
            "Network restored"
        )
        
        # Verify recovery
        final_healthy = len(self.trading_system.tool_registry.discover_tools(status=ToolStatus.HEALTHY))
        self.assertEqual(final_healthy, initial_healthy)
    
    def test_configuration_update_workflow(self):
        """Test updating tool configurations during workflow."""
        # Get original config
        original_config = self.trading_system.tool_registry.get_crypto_config("RiskAgent")
        original_risk = original_config["risk_parameters"]["max_risk_per_trade"]
        
        # Update configuration
        success, message = self.trading_system.tool_registry.update_crypto_config(
            "RiskAgent",
            {"risk_parameters": {"max_risk_per_trade": 0.005}},  # More conservative
            "test"
        )
        
        self.assertTrue(success)
        
        # Verify update
        updated_config = self.trading_system.tool_registry.get_crypto_config("RiskAgent")
        self.assertEqual(updated_config["risk_parameters"]["max_risk_per_trade"], 0.005)
        
        # Run workflow with updated config
        result = self.trading_system.run_trading_workflow("BTC", "crypto")
        self.assertIn("recommendation", result)
        
        # Restore original config
        self.trading_system.tool_registry.update_crypto_config(
            "RiskAgent",
            {"risk_parameters": {"max_risk_per_trade": original_risk}},
            "test"
        )
    
    def test_audit_trail_completeness(self):
        """Test that audit trail is complete and accurate."""
        # Get initial audit count
        initial_audit = len(self.trading_system.tool_registry.get_audit_log())
        
        # Perform various operations
        self.trading_system.fetch_user_data()
        self.trading_system.fetch_news_data("BTC")
        self.trading_system.run_trading_workflow("BTC", "crypto")
        
        # Get final audit count
        final_audit = len(self.trading_system.tool_registry.get_audit_log())
        
        # Should have more audit entries
        self.assertGreater(final_audit, initial_audit)
        
        # Verify recent audit entries
        recent_audit = self.trading_system.tool_registry.get_audit_log(limit=10)
        
        # Should contain our operations
        tool_names = [entry.tool_name for entry in recent_audit]
        self.assertIn("UserAgent", tool_names)
        self.assertIn("NewsAgent", tool_names)
        self.assertIn("ThinkingAgent", tool_names)


if __name__ == "__main__":
    # Run integration tests
    unittest.main(verbosity=2)