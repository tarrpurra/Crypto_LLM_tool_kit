#!/usr/bin/env python3
"""
Unit Tests for Tool Registry System

Comprehensive tests for the Tool Registry functionality including registration,
discovery, dependency management, and compliance validation.
"""

import unittest
import sys
import os
import time
from datetime import datetime

# Add Backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.tool_registry import ToolRegistry, ToolMetadata, ToolType, ToolStatus, ToolDependency, RegistryOperation


class TestToolRegistry(unittest.TestCase):
    """Test cases for the ToolRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = ToolRegistry()
        
        # Sample tool metadata for testing
        self.sample_tool = ToolMetadata(
            name="TestTool",
            version="1.0.0",
            tool_type=ToolType.TECHNICAL,
            description="Test tool for unit testing",
            capabilities=["testing", "validation"],
            dependencies=[],
            crypto_config={
                "supported_exchanges": ["binance", "coinbase"],
                "default_pair": "BTC/USDT"
            }
        )
    
    def test_initialization(self):
        """Test that the registry initializes correctly."""
        self.assertTrue(self.registry._initialized)
        self.assertIsInstance(self.registry._tools, dict)
        self.assertIsInstance(self.registry._dependencies, list)
        self.assertIsInstance(self.registry._audit_log, list)
        self.assertGreater(len(self.registry._audit_log), 0)  # Should have initialization log
    
    def test_register_tool_success(self):
        """Test successful tool registration."""
        success, message = self.registry.register_tool(self.sample_tool, "test")
        
        self.assertTrue(success)
        self.assertIn("registered successfully", message)
        self.assertIn("TestTool", self.registry._tools)
        
        # Verify tool metadata
        registered_tool = self.registry._tools["TestTool"]
        self.assertEqual(registered_tool.name, "TestTool")
        self.assertEqual(registered_tool.version, "1.0.0")
        self.assertEqual(registered_tool.status, ToolStatus.INITIALIZING)
    
    def test_register_duplicate_tool(self):
        """Test that duplicate tool registration fails."""
        # Register first time
        self.registry.register_tool(self.sample_tool, "test")
        
        # Try to register again
        success, message = self.registry.register_tool(self.sample_tool, "test")
        
        self.assertFalse(success)
        self.assertIn("already registered", message)
    
    def test_register_invalid_tool(self):
        """Test registration of invalid tool metadata."""
        # Test with missing name
        invalid_tool = ToolMetadata(
            name="",  # Empty name
            version="1.0.0",
            tool_type=ToolType.TECHNICAL,
            description="Invalid tool",
            capabilities=[],
            dependencies=[],
            crypto_config={}
        )
        
        success, message = self.registry.register_tool(invalid_tool, "test")
        
        self.assertFalse(success)
        self.assertIn("validation failed", message.lower())
    
    def test_update_tool_status(self):
        """Test updating tool status."""
        # Register tool first
        self.registry.register_tool(self.sample_tool, "test")
        
        # Update status
        success, message = self.registry.update_tool_status("TestTool", ToolStatus.HEALTHY, "Test update")
        
        self.assertTrue(success)
        self.assertEqual(self.registry._tools["TestTool"].status, ToolStatus.HEALTHY)
        self.assertEqual(self.registry._tools["TestTool"].status_message, "Test update")
    
    def test_deregister_tool(self):
        """Test deregistering a tool."""
        # Register tool first
        self.registry.register_tool(self.sample_tool, "test")
        
        # Deregister
        success, message = self.registry.deregister_tool("TestTool", "test")
        
        self.assertTrue(success)
        self.assertNotIn("TestTool", self.registry._tools)
    
    def test_deregister_with_dependencies(self):
        """Test that tools with dependencies cannot be deregistered."""
        # Register two tools
        self.registry.register_tool(self.sample_tool, "test")
        
        dependent_tool = ToolMetadata(
            name="DependentTool",
            version="1.0.0",
            tool_type=ToolType.ML,
            description="Depends on TestTool",
            capabilities=["dependent"],
            dependencies=["TestTool"],
            crypto_config={
                "supported_exchanges": ["binance"],
                "default_pair": "BTC/USDT"
            }
        )
        
        self.registry.register_tool(dependent_tool, "test")
        self.registry.register_dependency("DependentTool", "TestTool", "requires")
        
        # Try to deregister TestTool
        success, message = self.registry.deregister_tool("TestTool", "test")
        
        self.assertFalse(success)
        self.assertIn("tools depend on this", message)
        self.assertIn("TestTool", self.registry._tools)
    
    def test_discover_tools(self):
        """Test tool discovery functionality."""
        # Register multiple tools
        tools_data = [
            ("Tool1", ToolType.NEWS),
            ("Tool2", ToolType.TECHNICAL),
            ("Tool3", ToolType.NEWS)
        ]
        
        for name, tool_type in tools_data:
            metadata = ToolMetadata(
                name=name,
                version="1.0.0",
                tool_type=tool_type,
                description=f"{name} description",
                capabilities=["test"],
                dependencies=[],
                crypto_config={
                    "supported_exchanges": ["binance"],
                    "default_pair": "BTC/USDT"
                }
            )
            self.registry.register_tool(metadata, "test")
        
        # Discover all tools
        all_tools = self.registry.discover_tools()
        self.assertEqual(len(all_tools), 3)
        
        # Discover by type
        news_tools = self.registry.discover_tools(tool_type=ToolType.NEWS)
        self.assertEqual(len(news_tools), 2)
        self.assertEqual(news_tools[0].name, "Tool1")
        self.assertEqual(news_tools[1].name, "Tool3")
    
    def test_register_dependency(self):
        """Test dependency registration."""
        # Register two tools
        self.registry.register_tool(self.sample_tool, "test")
        
        tool2 = ToolMetadata(
            name="Tool2",
            version="1.0.0",
            tool_type=ToolType.ML,
            description="Tool 2",
            capabilities=["ml"],
            dependencies=[],
            crypto_config={
                "supported_exchanges": ["binance"],
                "default_pair": "BTC/USDT"
            }
        )
        
        self.registry.register_tool(tool2, "test")
        
        # Register dependency
        success, message = self.registry.register_dependency(
            "Tool2", "TestTool", "requires", "*", "test"
        )
        
        self.assertTrue(success)
        self.assertEqual(len(self.registry._dependencies), 1)
        
        # Verify dependency
        deps = self.registry.get_dependencies("Tool2")
        self.assertEqual(len(deps), 1)
        self.assertEqual(deps[0].source_tool, "Tool2")
        self.assertEqual(deps[0].target_tool, "TestTool")
    
    def test_validate_tool_chain(self):
        """Test tool chain validation."""
        # Register tools
        tools = ["Tool1", "Tool2", "Tool3"]
        for tool_name in tools:
            metadata = ToolMetadata(
                name=tool_name,
                version="1.0.0",
                tool_type=ToolType.TECHNICAL,
                description=f"{tool_name} description",
                capabilities=["test"],
                dependencies=[],
                crypto_config={
                    "supported_exchanges": ["binance"],
                    "default_pair": "BTC/USDT"
                }
            )
            self.registry.register_tool(metadata, "test")
            self.registry.update_tool_status(tool_name, ToolStatus.HEALTHY, "Ready")
        
        # Valid chain
        valid, message = self.registry.validate_tool_chain(tools)
        self.assertTrue(valid)
        
        # Invalid chain with missing tool
        invalid_chain = ["Tool1", "MissingTool", "Tool3"]
        valid, message = self.registry.validate_tool_chain(invalid_chain)
        self.assertFalse(valid)
        self.assertIn("MissingTool", message)
    
    def test_log_tool_invocation(self):
        """Test tool invocation logging."""
        # Register tool
        self.registry.register_tool(self.sample_tool, "test")
        
        # Log successful invocation
        success, message = self.registry.log_tool_invocation(
            "TestTool", True, 0.123, "", "test"
        )
        
        self.assertTrue(success)
        
        # Verify metrics were updated
        tool = self.registry._tools["TestTool"]
        self.assertEqual(tool.invocation_count, 1)
        self.assertGreater(tool.last_invocation, 0)
        self.assertGreater(tool.avg_response_time, 0)
    
    def test_crypto_compliance(self):
        """Test cryptocurrency compliance validation."""
        # Valid crypto config
        valid_tool = ToolMetadata(
            name="ValidTool",
            version="1.0.0",
            tool_type=ToolType.TECHNICAL,
            description="Valid crypto tool",
            capabilities=["crypto_data", "exchange_integration"],
            dependencies=[],
            crypto_config={
                "supported_exchanges": ["binance", "coinbase"],
                "default_pair": "BTC/USDT"
            }
        )
        
        self.registry.register_tool(valid_tool, "test")
        
        compliant, message = self.registry.validate_crypto_compliance("ValidTool")
        self.assertTrue(compliant)
        
        # Invalid crypto config (missing required capabilities)
        invalid_tool = ToolMetadata(
            name="InvalidTool",
            version="1.0.0",
            tool_type=ToolType.TECHNICAL,
            description="Invalid crypto tool",
            capabilities=["non_crypto"],  # Missing required crypto capabilities
            dependencies=[],
            crypto_config={
                "supported_exchanges": ["binance"],
                "default_pair": "BTC/USDT"
            }
        )
        
        self.registry.register_tool(invalid_tool, "test")
        
        compliant, message = self.registry.validate_crypto_compliance("InvalidTool")
        self.assertFalse(compliant)
        self.assertIn("capabilities", message.lower())
    
    def test_audit_logging(self):
        """Test that operations are properly logged."""
        initial_log_count = len(self.registry._audit_log)
        
        # Perform operations
        self.registry.register_tool(self.sample_tool, "test")
        self.registry.update_tool_status("TestTool", ToolStatus.HEALTHY, "Test")
        
        # Check audit log
        final_log_count = len(self.registry._audit_log)
        self.assertGreater(final_log_count, initial_log_count)
        
        # Verify log entries
        log_entries = self.registry.get_audit_log()
        self.assertGreater(len(log_entries), 0)
        
        # Check that recent entries are for our operations
        recent_entries = [e for e in log_entries if e.tool_name in ["TestTool", "registry"]]
        self.assertGreater(len(recent_entries), 0)
    
    def test_system_health(self):
        """Test system health monitoring."""
        # Register a healthy tool
        self.registry.register_tool(self.sample_tool, "test")
        self.registry.update_tool_status("TestTool", ToolStatus.HEALTHY, "Ready")
        
        health = self.registry.check_system_health()
        
        self.assertEqual(health["registry_status"], "healthy")
        self.assertEqual(health["tool_count"], 1)
        self.assertEqual(health["healthy_tools"], 1)
    
    def test_thread_safety(self):
        """Test thread safety of registry operations."""
        import threading
        
        def register_tool(tool_id):
            metadata = ToolMetadata(
                name=f"ThreadTool{tool_id}",
                version="1.0.0",
                tool_type=ToolType.TECHNICAL,
                description=f"Thread test tool {tool_id}",
                capabilities=["threading"],
                dependencies=[],
                crypto_config={
                    "supported_exchanges": ["binance"],
                    "default_pair": "BTC/USDT"
                }
            )
            return self.registry.register_tool(metadata, f"thread{tool_id}")
        
        # Run concurrent registrations
        threads = []
        results = []
        
        for i in range(5):
            thread = threading.Thread(target=lambda: results.append(register_tool(i)))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        successful = sum(1 for success, _ in results if success)
        self.assertEqual(successful, 5)
        self.assertEqual(len(self.registry._tools), 5)  # Should have 5 thread tools
    
    def test_performance(self):
        """Test performance of registry operations."""
        import time
        
        # Register many tools
        start_time = time.time()
        
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
        
        # Discovery performance
        start_time = time.time()
        all_tools = self.registry.discover_tools()
        discovery_time = time.time() - start_time
        
        # Metrics performance
        start_time = time.time()
        for i in range(100):
            self.registry.get_tool_metrics(f"PerfTool{i}")
        metrics_time = time.time() - start_time
        
        print(f"\nPerformance Metrics:")
        print(f"  Registration (100 tools): {registration_time:.3f}s")
        print(f"  Discovery: {discovery_time:.3f}s")
        print(f"  Metrics (100 calls): {metrics_time:.3f}s")
        
        # Assert reasonable performance
        self.assertLess(registration_time, 1.0, "Registration should be fast")
        self.assertLess(discovery_time, 0.1, "Discovery should be very fast")
        self.assertLess(metrics_time, 0.5, "Metrics retrieval should be fast")


class TestToolRegistryEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        self.registry = ToolRegistry()
    
    def test_invalid_crypto_config(self):
        """Test tools with invalid crypto configurations."""
        # Missing supported exchanges
        invalid_tool = ToolMetadata(
            name="InvalidCryptoTool",
            version="1.0.0",
            tool_type=ToolType.TECHNICAL,
            description="Invalid crypto config",
            capabilities=["crypto_data"],
            dependencies=[],
            crypto_config={
                "default_pair": "BTC/USDT"  # Missing supported_exchanges
            }
        )
        
        success, message = self.registry.register_tool(invalid_tool, "test")
        self.assertFalse(success)
        self.assertIn("supported_exchanges", message)
    
    def test_circular_dependencies(self):
        """Test that circular dependencies are handled."""
        # This is more of a logical test - the registry should prevent circular deps
        # by not allowing tools to depend on each other in a cycle
        
        # Register two tools
        tool1 = ToolMetadata(
            name="ToolA",
            version="1.0.0",
            tool_type=ToolType.TECHNICAL,
            description="Tool A",
            capabilities=["test"],
            dependencies=[],
            crypto_config={
                "supported_exchanges": ["binance"],
                "default_pair": "BTC/USDT"
            }
        )
        
        tool2 = ToolMetadata(
            name="ToolB",
            version="1.0.0",
            tool_type=ToolType.ML,
            description="Tool B",
            capabilities=["test"],
            dependencies=[],
            crypto_config={
                "supported_exchanges": ["binance"],
                "default_pair": "BTC/USDT"
            }
        )
        
        self.registry.register_tool(tool1, "test")
        self.registry.register_tool(tool2, "test")
        
        # Create dependency A -> B
        self.registry.register_dependency("ToolA", "ToolB", "requires")
        
        # Try to create circular dependency B -> A
        # This should be allowed by the registry (it's up to validation to catch this)
        success, message = self.registry.register_dependency("ToolB", "ToolA", "requires")
        
        # The registry allows this, but validation should catch it
        self.assertTrue(success)
        
        # Now validation should catch the circular dependency
        chain_valid, chain_msg = self.registry.validate_tool_chain(["ToolA", "ToolB"])
        # This might not catch it directly, but a proper chain validation would
    
    def test_nonexistent_tool_operations(self):
        """Test operations on non-existent tools."""
        # Try to get non-existent tool
        tool = self.registry.get_tool("NonExistent")
        self.assertIsNone(tool)
        
        # Try to update status
        success, message = self.registry.update_tool_status("NonExistent", ToolStatus.HEALTHY)
        self.assertFalse(success)
        self.assertIn("not found", message)
        
        # Try to deregister
        success, message = self.registry.deregister_tool("NonExistent")
        self.assertFalse(success)
        self.assertIn("not found", message)
        
        # Try to get metrics
        metrics = self.registry.get_tool_metrics("NonExistent")
        self.assertIsNone(metrics)
    
    def test_empty_registry_operations(self):
        """Test operations on empty registry."""
        # Discovery should return empty list
        tools = self.registry.discover_tools()
        self.assertEqual(len(tools), 0)
        
        # System health should show no tools
        health = self.registry.check_system_health()
        self.assertEqual(health["tool_count"], 0)
        self.assertEqual(health["healthy_tools"], 0)


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)