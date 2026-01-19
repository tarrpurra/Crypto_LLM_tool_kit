#!/usr/bin/env python3
"""
Tool Registry System for Trading Agent Platform

Centralized management, registration, and orchestration of all tools and agents
with cryptocurrency-specific functionality.
"""

import logging
import time
import json
import threading
from typing import Dict, List, Optional, Any, Type, Tuple
from dataclasses import dataclass, asdict
from enum import Enum, auto
import hashlib
import uuid

# Set up logging
logger = logging.getLogger('ToolRegistry')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ToolType(Enum):
    """Enumeration of tool types in the trading system."""
    SENTIMENT = auto()
    NEWS = auto()
    TECHNICAL = auto()
    ML = auto()
    RISK = auto()
    PORTFOLIO = auto()
    DATA = auto()
    OTHER = auto()


class ToolStatus(Enum):
    """Enumeration of tool lifecycle states."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    HEALTHY = auto()
    DEGRADED = auto()
    FAILED = auto()
    TERMINATED = auto()


@dataclass
class ToolMetadata:
    """Metadata for registered tools."""
    name: str
    version: str
    tool_type: ToolType
    description: str
    capabilities: List[str]
    dependencies: List[str]
    crypto_config: Dict[str, Any]
    created_at: float
    updated_at: float
    status: ToolStatus = ToolStatus.UNINITIALIZED
    status_message: str = ""
    last_health_check: float = 0.0
    invocation_count: int = 0
    last_invocation: float = 0.0
    avg_response_time: float = 0.0


@dataclass
class RegistryOperation:
    """Audit record for registry operations."""
    operation_type: str
    tool_name: str
    timestamp: float
    actor: str
    details: Dict[str, Any]
    success: bool
    error_message: str = ""


@dataclass
class ToolDependency:
    """Represents a dependency between tools."""
    source_tool: str
    target_tool: str
    dependency_type: str  # "requires", "optional", "conflicts"
    version_constraint: str = "*"


class ToolRegistry:
    """
    Centralized tool registry for the Trading Agent platform.
    
    Provides dynamic registration, discovery, configuration management,
    lifecycle tracking, and audit capabilities for all tools and agents.
    """
    
    def __init__(self):
        """Initialize the tool registry."""
        self._tools: Dict[str, ToolMetadata] = {}
        self._dependencies: List[ToolDependency] = []
        self._audit_log: List[RegistryOperation] = []
        self._configurations: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._initialized = False
        
        # Initialize with built-in tool types
        self._initialize_registry()
        
    def _initialize_registry(self) -> None:
        """Initialize the registry with default configurations."""
        with self._lock:
            if not self._initialized:
                # Set default configurations for cryptocurrency tools
                self._configurations['default'] = {
                    'api_timeout': 30,
                    'max_retries': 3,
                    'rate_limit': 100,
                    'cache_ttl': 300,
                    'crypto_specific': {
                        'supported_exchanges': ['binance', 'coinbase', 'kraken'],
                        'default_pair': 'BTC/USDT',
                        'min_volume_threshold': 1000000,
                        'sentiment_thresholds': {
                            'bullish': 0.7,
                            'bearish': 0.3,
                            'neutral_min': 0.4,
                            'neutral_max': 0.6
                        }
                    }
                }
                
                self._initialized = True
                logger.info("✅ Tool Registry initialized")
                
                # Log initialization
                self._log_operation(
                    operation_type="initialize",
                    tool_name="registry",
                    actor="system",
                    details={"status": "success"},
                    success=True
                )
    
    def _generate_tool_id(self, tool_name: str) -> str:
        """Generate a unique identifier for a tool."""
        return hashlib.md5(f"{tool_name}_{time.time()}_{uuid.uuid4()}".encode()).hexdigest()
    
    def _validate_tool_metadata(self, metadata: ToolMetadata) -> Tuple[bool, str]:
        """Validate tool metadata before registration."""
        if not metadata.name or not isinstance(metadata.name, str):
            return False, "Tool name must be a non-empty string"
        
        if not metadata.version or not isinstance(metadata.version, str):
            return False, "Tool version must be a non-empty string"
        
        if not isinstance(metadata.capabilities, list):
            return False, "Capabilities must be a list"
        
        if not isinstance(metadata.dependencies, list):
            return False, "Dependencies must be a list"
        
        # Validate cryptocurrency-specific configuration
        if 'crypto_config' in metadata.__dict__:
            if not isinstance(metadata.crypto_config, dict):
                return False, "Crypto configuration must be a dictionary"
            
            # Check for required crypto-specific fields
            required_fields = ['supported_exchanges', 'default_pair']
            for field in required_fields:
                if field not in metadata.crypto_config:
                    return False, f"Missing required crypto config field: {field}"
        
        return True, "Validation successful"
    
    def _validate_crypto_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate cryptocurrency-specific configuration."""
        if not isinstance(config, dict):
            return False, "Configuration must be a dictionary"
        
        # Check required fields
        required_fields = ['supported_exchanges', 'default_pair']
        for field in required_fields:
            if field not in config:
                return False, f"Missing required field: {field}"
        
        # Validate exchanges
        if not isinstance(config['supported_exchanges'], list) or len(config['supported_exchanges']) == 0:
            return False, "Supported exchanges must be a non-empty list"
        
        # Validate default pair format (e.g., BTC/USDT)
        if not isinstance(config['default_pair'], str) or '/' not in config['default_pair']:
            return False, "Default pair must be in format like 'BTC/USDT'"
        
        return True, "Crypto configuration valid"
    
    def _log_operation(self, operation_type: str, tool_name: str, actor: str,
                      details: Dict[str, Any], success: bool, error_message: str = "") -> None:
        """Log a registry operation for audit purposes."""
        operation = RegistryOperation(
            operation_type=operation_type,
            tool_name=tool_name,
            timestamp=time.time(),
            actor=actor,
            details=details,
            success=success,
            error_message=error_message
        )
        
        with self._lock:
            self._audit_log.append(operation)
            
        # Log to file or external system in production
        logger.info(f"📝 Registry operation: {operation_type} {tool_name} by {actor} - {'✅' if success else '❌'}")
    
    def _check_dependencies(self, tool_name: str) -> Tuple[bool, str]:
        """Check if all dependencies for a tool are satisfied."""
        with self._lock:
            if tool_name not in self._tools:
                return False, f"Tool {tool_name} not found in registry"
            
            tool_metadata = self._tools[tool_name]
            missing_deps = []
            
            for dep in tool_metadata.dependencies:
                if dep not in self._tools:
                    missing_deps.append(dep)
            
            if missing_deps:
                return False, f"Missing dependencies: {', '.join(missing_deps)}"
            
            return True, "All dependencies satisfied"
    
    def register_tool(self, tool_metadata: ToolMetadata, actor: str = "system") -> Tuple[bool, str]:
        """
        Register a new tool with the registry.
        
        Args:
            tool_metadata: Metadata for the tool to register
            actor: Entity performing the registration
            
        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            # Validate metadata
            valid, validation_msg = self._validate_tool_metadata(tool_metadata)
            if not valid:
                self._log_operation(
                    operation_type="register",
                    tool_name=tool_metadata.name,
                    actor=actor,
                    details={"validation_error": validation_msg},
                    success=False,
                    error_message=validation_msg
                )
                return False, f"Validation failed: {validation_msg}"
            
            # Check for duplicate registration
            if tool_metadata.name in self._tools:
                error_msg = f"Tool {tool_metadata.name} already registered"
                self._log_operation(
                    operation_type="register",
                    tool_name=tool_metadata.name,
                    actor=actor,
                    details={"status": "duplicate"},
                    success=False,
                    error_message=error_msg
                )
                return False, error_msg
            
            # Set timestamps
            now = time.time()
            tool_metadata.created_at = now
            tool_metadata.updated_at = now
            tool_metadata.status = ToolStatus.INITIALIZING
            tool_metadata.status_message = "Tool registration initiated"
            
            # Register the tool
            self._tools[tool_metadata.name] = tool_metadata
            
            # Log successful registration
            self._log_operation(
                operation_type="register",
                tool_name=tool_metadata.name,
                actor=actor,
                details={
                    "version": tool_metadata.version,
                    "type": tool_metadata.tool_type.name,
                    "capabilities": tool_metadata.capabilities
                },
                success=True
            )
            
            logger.info(f"🔧 Registered tool: {tool_metadata.name} v{tool_metadata.version}")
            
            return True, f"Tool {tool_metadata.name} registered successfully"
    
    def update_tool_status(self, tool_name: str, status: ToolStatus,
                          status_message: str = "", actor: str = "system") -> Tuple[bool, str]:
        """
        Update the status of a registered tool.
        
        Args:
            tool_name: Name of the tool to update
            status: New status
            status_message: Optional status message
            actor: Entity performing the update
            
        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if tool_name not in self._tools:
                error_msg = f"Tool {tool_name} not found"
                self._log_operation(
                    operation_type="update_status",
                    tool_name=tool_name,
                    actor=actor,
                    details={"target_status": status.name},
                    success=False,
                    error_message=error_msg
                )
                return False, error_msg
            
            # Update status
            self._tools[tool_name].status = status
            self._tools[tool_name].status_message = status_message
            self._tools[tool_name].updated_at = time.time()
            
            # Log status update
            self._log_operation(
                operation_type="update_status",
                tool_name=tool_name,
                actor=actor,
                details={
                    "old_status": status.name,  # Note: This would need tracking
                    "new_status": status.name,
                    "message": status_message
                },
                success=True
            )
            
            logger.info(f"🔄 Updated {tool_name} status to {status.name}: {status_message}")
            
            return True, f"Status updated to {status.name}"
    
    def deregister_tool(self, tool_name: str, actor: str = "system") -> Tuple[bool, str]:
        """
        Deregister a tool from the registry.
        
        Args:
            tool_name: Name of the tool to deregister
            actor: Entity performing the deregistration
            
        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if tool_name not in self._tools:
                error_msg = f"Tool {tool_name} not found"
                self._log_operation(
                    operation_type="deregister",
                    tool_name=tool_name,
                    actor=actor,
                    details={},
                    success=False,
                    error_message=error_msg
                )
                return False, error_msg
            
            # Check for dependent tools
            dependent_tools = self.get_dependent_tools(tool_name)
            if dependent_tools:
                error_msg = f"Cannot deregister: tools depend on this: {', '.join(dependent_tools)}"
                self._log_operation(
                    operation_type="deregister",
                    tool_name=tool_name,
                    actor=actor,
                    details={"dependent_tools": dependent_tools},
                    success=False,
                    error_message=error_msg
                )
                return False, error_msg
            
            # Deregister the tool
            tool_metadata = self._tools.pop(tool_name)
            
            # Log deregistration
            self._log_operation(
                operation_type="deregister",
                tool_name=tool_name,
                actor=actor,
                details={
                    "version": tool_metadata.version,
                    "type": tool_metadata.tool_type.name
                },
                success=True
            )
            
            logger.info(f"🗑️  Deregistered tool: {tool_name}")
            
            return True, f"Tool {tool_name} deregistered successfully"
    
    def discover_tools(self, tool_type: Optional[ToolType] = None,
                      status: Optional[ToolStatus] = None) -> List[ToolMetadata]:
        """
        Discover tools in the registry with optional filters.
        
        Args:
            tool_type: Optional filter by tool type
            status: Optional filter by status
            
        Returns:
            List of matching ToolMetadata objects
        """
        with self._lock:
            tools = list(self._tools.values())
            
            # Apply filters
            if tool_type:
                tools = [t for t in tools if t.tool_type == tool_type]
            
            if status:
                tools = [t for t in tools if t.status == status]
            
            return tools
    
    def get_tool(self, tool_name: str) -> Optional[ToolMetadata]:
        """
        Get metadata for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ToolMetadata if found, None otherwise
        """
        with self._lock:
            return self._tools.get(tool_name)
    
    def register_dependency(self, source_tool: str, target_tool: str,
                           dependency_type: str = "requires",
                           version_constraint: str = "*",
                           actor: str = "system") -> Tuple[bool, str]:
        """
        Register a dependency between tools.
        
        Args:
            source_tool: Tool that depends on another
            target_tool: Tool being depended on
            dependency_type: Type of dependency (requires, optional, conflicts)
            version_constraint: Version constraint for the dependency
            actor: Entity performing the registration
            
        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            # Validate tools exist
            if source_tool not in self._tools:
                error_msg = f"Source tool {source_tool} not found"
                self._log_operation(
                    operation_type="register_dependency",
                    tool_name=f"{source_tool}->{target_tool}",
                    actor=actor,
                    details={"type": dependency_type},
                    success=False,
                    error_message=error_msg
                )
                return False, error_msg
            
            if target_tool not in self._tools:
                error_msg = f"Target tool {target_tool} not found"
                self._log_operation(
                    operation_type="register_dependency",
                    tool_name=f"{source_tool}->{target_tool}",
                    actor=actor,
                    details={"type": dependency_type},
                    success=False,
                    error_message=error_msg
                )
                return False, error_msg
            
            # Check for duplicate dependency
            for dep in self._dependencies:
                if (dep.source_tool == source_tool and 
                    dep.target_tool == target_tool and
                    dep.dependency_type == dependency_type):
                    error_msg = "Dependency already registered"
                    self._log_operation(
                        operation_type="register_dependency",
                        tool_name=f"{source_tool}->{target_tool}",
                        actor=actor,
                        details={"type": dependency_type},
                        success=False,
                        error_message=error_msg
                    )
                    return False, error_msg
            
            # Register dependency
            dependency = ToolDependency(
                source_tool=source_tool,
                target_tool=target_tool,
                dependency_type=dependency_type,
                version_constraint=version_constraint
            )
            self._dependencies.append(dependency)
            
            # Update source tool's dependency list
            self._tools[source_tool].dependencies.append(target_tool)
            
            # Log dependency registration
            self._log_operation(
                operation_type="register_dependency",
                tool_name=f"{source_tool}->{target_tool}",
                actor=actor,
                details={
                    "type": dependency_type,
                    "version_constraint": version_constraint
                },
                success=True
            )
            
            logger.info(f"🔗 Registered dependency: {source_tool} {dependency_type} {target_tool}")
            
            return True, "Dependency registered successfully"
    
    def get_dependent_tools(self, tool_name: str) -> List[str]:
        """
        Get tools that depend on a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            List of tool names that depend on the specified tool
        """
        with self._lock:
            return [dep.source_tool for dep in self._dependencies 
                   if dep.target_tool == tool_name]
    
    def get_dependencies(self, tool_name: str) -> List[ToolDependency]:
        """
        Get dependencies for a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            List of ToolDependency objects
        """
        with self._lock:
            return [dep for dep in self._dependencies if dep.source_tool == tool_name]
    
    def validate_tool_chain(self, tool_chain: List[str]) -> Tuple[bool, str]:
        """
        Validate that a chain of tools can be executed based on dependencies.
        
        Args:
            tool_chain: List of tool names in execution order
            
        Returns:
            Tuple of (valid, message)
        """
        with self._lock:
            # Check all tools exist
            missing_tools = [t for t in tool_chain if t not in self._tools]
            if missing_tools:
                return False, f"Missing tools: {', '.join(missing_tools)}"
            
            # Check dependencies are satisfied for each tool
            for tool_name in tool_chain:
                tool = self._tools[tool_name]
                
                # Check if tool is healthy
                if tool.status != ToolStatus.HEALTHY:
                    return False, f"Tool {tool_name} is not healthy (status: {tool.status.name})"
                
                # Check dependencies
                for dep_name in tool.dependencies:
                    if dep_name not in tool_chain:
                        return False, f"Tool {tool_name} depends on {dep_name} which is not in the chain"
                    
                    # Check dependency appears before the tool that needs it
                    dep_index = tool_chain.index(dep_name)
                    tool_index = tool_chain.index(tool_name)
                    if dep_index > tool_index:
                        return False, f"Dependency {dep_name} must appear before {tool_name} in chain"
            
            return True, "Tool chain is valid"
    
    def log_tool_invocation(self, tool_name: str, success: bool,
                           response_time: float = 0.0, 
                           error_message: str = "",
                           actor: str = "system") -> Tuple[bool, str]:
        """
        Log a tool invocation for metrics and monitoring.
        
        Args:
            tool_name: Name of the tool invoked
            success: Whether the invocation was successful
            response_time: Response time in seconds
            error_message: Error message if applicable
            actor: Entity performing the invocation
            
        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if tool_name not in self._tools:
                error_msg = f"Tool {tool_name} not found"
                self._log_operation(
                    operation_type="log_invocation",
                    tool_name=tool_name,
                    actor=actor,
                    details={"success": success, "response_time": response_time},
                    success=False,
                    error_message=error_msg
                )
                return False, error_msg
            
            # Update tool metrics
            tool = self._tools[tool_name]
            tool.invocation_count += 1
            tool.last_invocation = time.time()
            
            # Update average response time (moving average)
            if tool.avg_response_time == 0:
                tool.avg_response_time = response_time
            else:
                alpha = 0.1  # Weight for new measurement
                tool.avg_response_time = (
                    alpha * response_time + 
                    (1 - alpha) * tool.avg_response_time
                )
            
            # Log invocation
            self._log_operation(
                operation_type="invoke",
                tool_name=tool_name,
                actor=actor,
                details={
                    "success": success,
                    "response_time": response_time,
                    "error": error_message if not success else None
                },
                success=True
            )
            
            logger.debug(f"📊 Tool invocation: {tool_name} - {'✅' if success else '❌'} {response_time:.3f}s")
            
            return True, "Invocation logged successfully"
    
    def get_tool_metrics(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary of metrics or None if tool not found
        """
        with self._lock:
            if tool_name not in self._tools:
                return None
            
            tool = self._tools[tool_name]
            return {
                "invocation_count": tool.invocation_count,
                "last_invocation": tool.last_invocation,
                "avg_response_time": tool.avg_response_time,
                "status": tool.status.name,
                "status_message": tool.status_message,
                "last_health_check": tool.last_health_check
            }
    
    def get_registry_metrics(self) -> Dict[str, Any]:
        """
        Get overall registry metrics.
        
        Returns:
            Dictionary of registry metrics
        """
        with self._lock:
            healthy_tools = sum(1 for t in self._tools.values() if t.status == ToolStatus.HEALTHY)
            total_tools = len(self._tools)
            
            return {
                "total_tools": total_tools,
                "healthy_tools": healthy_tools,
                "degraded_tools": sum(1 for t in self._tools.values() if t.status == ToolStatus.DEGRADED),
                "failed_tools": sum(1 for t in self._tools.values() if t.status == ToolStatus.FAILED),
                "total_dependencies": len(self._dependencies),
                "total_audit_entries": len(self._audit_log),
                "uptime": time.time() - min(t.created_at for t in self._tools.values()) if self._tools else 0
            }
    
    def get_audit_log(self, limit: int = 100) -> List[RegistryOperation]:
        """
        Get recent audit log entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of RegistryOperation objects
        """
        with self._lock:
            return self._audit_log[-limit:]
    
    def export_audit_log(self, filepath: str = "registry_audit.json") -> Tuple[bool, str]:
        """
        Export audit log to a file.
        
        Args:
            filepath: Path to export to
            
        Returns:
            Tuple of (success, message)
        """
        try:
            with self._lock:
                audit_data = [asdict(op) for op in self._audit_log]
            
            with open(filepath, 'w') as f:
                json.dump(audit_data, f, indent=2)
            
            logger.info(f"📤 Exported audit log to {filepath}")
            return True, f"Audit log exported to {filepath}"
            
        except Exception as e:
            error_msg = f"Failed to export audit log: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
    
    def validate_crypto_compliance(self, tool_name: str) -> Tuple[bool, str]:
        """
        Validate that a tool meets cryptocurrency-specific compliance requirements.
        
        Args:
            tool_name: Name of the tool to validate
            
        Returns:
            Tuple of (compliant, message)
        """
        with self._lock:
            if tool_name not in self._tools:
                return False, f"Tool {tool_name} not found"
            
            tool = self._tools[tool_name]
            
            # Check crypto configuration exists
            if not hasattr(tool, 'crypto_config') or not tool.crypto_config:
                return False, "Missing cryptocurrency configuration"
            
            # Validate crypto config
            valid, msg = self._validate_crypto_config(tool.crypto_config)
            if not valid:
                return False, f"Invalid crypto config: {msg}"
            
            # Check required crypto-specific capabilities
            required_caps = ['crypto_data', 'exchange_integration']
            missing_caps = [cap for cap in required_caps if cap not in tool.capabilities]
            
            if missing_caps:
                return False, f"Missing required crypto capabilities: {', '.join(missing_caps)}"
            
            # Check supported exchanges
            if 'supported_exchanges' not in tool.crypto_config:
                return False, "No supported exchanges specified"
            
            if not tool.crypto_config['supported_exchanges']:
                return False, "No exchanges listed in supported_exchanges"
            
            return True, "Tool is crypto-compliant"
    
    def get_crypto_config(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get cryptocurrency-specific configuration for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Crypto configuration dictionary or None
        """
        with self._lock:
            tool = self._tools.get(tool_name)
            return tool.crypto_config if tool and hasattr(tool, 'crypto_config') else None
    
    def update_crypto_config(self, tool_name: str, config_update: Dict[str, Any],
                            actor: str = "system") -> Tuple[bool, str]:
        """
        Update cryptocurrency-specific configuration for a tool.
        
        Args:
            tool_name: Name of the tool
            config_update: Dictionary of configuration updates
            actor: Entity performing the update
            
        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            if tool_name not in self._tools:
                error_msg = f"Tool {tool_name} not found"
                self._log_operation(
                    operation_type="update_crypto_config",
                    tool_name=tool_name,
                    actor=actor,
                    details=config_update,
                    success=False,
                    error_message=error_msg
                )
                return False, error_msg
            
            # Validate the update
            updated_config = {**self._tools[tool_name].crypto_config, **config_update}
            valid, msg = self._validate_crypto_config(updated_config)
            
            if not valid:
                self._log_operation(
                    operation_type="update_crypto_config",
                    tool_name=tool_name,
                    actor=actor,
                    details=config_update,
                    success=False,
                    error_message=f"Validation failed: {msg}"
                )
                return False, f"Validation failed: {msg}"
            
            # Apply the update
            self._tools[tool_name].crypto_config = updated_config
            self._tools[tool_name].updated_at = time.time()
            
            # Log the update
            self._log_operation(
                operation_type="update_crypto_config",
                tool_name=tool_name,
                actor=actor,
                details={"updated_fields": list(config_update.keys())},
                success=True
            )
            
            logger.info(f"🔧 Updated crypto config for {tool_name}")
            
            return True, "Crypto configuration updated successfully"
    
    def get_tool_health(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get health status and metrics for a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Dictionary of health information or None
        """
        with self._lock:
            if tool_name not in self._tools:
                return None
            
            tool = self._tools[tool_name]
            
            # Perform health check (update last_health_check timestamp)
            tool.last_health_check = time.time()
            
            return {
                "status": tool.status.name,
                "status_message": tool.status_message,
                "last_health_check": tool.last_health_check,
                "uptime": tool.last_health_check - tool.created_at,
                "invocation_count": tool.invocation_count,
                "avg_response_time": tool.avg_response_time,
                "dependencies_healthy": self._check_dependencies(tool_name)[0]
            }
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Check overall system health.
        
        Returns:
            Dictionary of system health metrics
        """
        with self._lock:
            tool_health = {}
            for name, tool in self._tools.items():
                health = self.get_tool_health(name)
                if health:
                    tool_health[name] = health
            
            return {
                "registry_status": "healthy" if self._initialized else "uninitialized",
                "tool_count": len(self._tools),
                "healthy_tools": sum(1 for h in tool_health.values() if h["status"] == "HEALTHY"),
                "tool_health": tool_health,
                "timestamp": time.time()
            }


# Example usage and testing
if __name__ == "__main__":
    # Initialize the registry
    registry = ToolRegistry()
    
    # Example: Register a sentiment tool
    sentiment_metadata = ToolMetadata(
        name="SentimentTool",
        version="1.0.0",
        tool_type=ToolType.SENTIMENT,
        description="Analyzes cryptocurrency sentiment from news and social media",
        capabilities=["sentiment_analysis", "news_processing", "crypto_data"],
        dependencies=[],
        crypto_config={
            "supported_exchanges": ["binance", "coinbase"],
            "default_pair": "BTC/USDT",
            "sentiment_thresholds": {
                "bullish": 0.75,
                "bearish": 0.25,
                "neutral_min": 0.3,
                "neutral_max": 0.7
            },
            "data_sources": ["newsapi", "twitter", "reddit"]
        },
        created_at=time.time(),
        updated_at=time.time()
    )
    
    # Register the tool
    success, message = registry.register_tool(sentiment_metadata)
    print(f"Registration: {success}, {message}")
    
    # Update tool status
    success, message = registry.update_tool_status("SentimentTool", ToolStatus.HEALTHY, "Tool initialized successfully")
    print(f"Status update: {success}, {message}")
    
    # Discover tools
    sentiment_tools = registry.discover_tools(tool_type=ToolType.SENTIMENT)
    print(f"Found {len(sentiment_tools)} sentiment tools")
    
    # Get tool info
    tool_info = registry.get_tool("SentimentTool")
    if tool_info:
        print(f"Tool: {tool_info.name} v{tool_info.version}")
        print(f"Status: {tool_info.status.name}")
        print(f"Capabilities: {tool_info.capabilities}")
    
    # Validate crypto compliance
    compliant, msg = registry.validate_crypto_compliance("SentimentTool")
    print(f"Crypto compliance: {compliant}, {msg}")
    
    # Get registry metrics
    metrics = registry.get_registry_metrics()
    print(f"Registry metrics: {metrics}")
    
    # Check system health
    health = registry.check_system_health()
    print(f"System health: {health['registry_status']}")
    
    # Export audit log
    success, message = registry.export_audit_log("test_registry_audit.json")
    print(f"Audit export: {success}, {message}")