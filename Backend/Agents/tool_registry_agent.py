#!/usr/bin/env python3
"""
Tool Registry Agent

Agent interface for the Tool Registry System, providing agent-specific
functionality and integration with the trading workflow.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict

from services.tool_registry import ToolRegistry, ToolMetadata, ToolType, ToolStatus
from services.data_manager import DataManager

# Set up logging
logger = logging.getLogger('ToolRegistryAgent')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class ToolRegistryAgent:
    """
    Agent interface for the Tool Registry System.
    
    Provides agent-specific functionality for tool management, discovery,
    and integration with the trading workflow.
    """
    
    def __init__(self):
        """Initialize the Tool Registry Agent."""
        self.registry = ToolRegistry()
        self.data_manager = DataManager()
        
        logger.info("✅ Tool Registry Agent initialized")
    
    def register_agent(self, agent_name: str, agent_type: ToolType,
                      version: str = "1.0.0",
                      description: str = "",
                      capabilities: List[str] = None,
                      dependencies: List[str] = None,
                      crypto_config: Dict[str, Any] = None) -> Tuple[bool, str]:
        """
        Register an agent with the tool registry.
        
        Args:
            agent_name: Name of the agent
            agent_type: Type of agent (ToolType enum)
            version: Agent version
            description: Agent description
            capabilities: List of capabilities
            dependencies: List of dependencies
            crypto_config: Cryptocurrency-specific configuration
            
        Returns:
            Tuple of (success, message)
        """
        # Set defaults
        if capabilities is None:
            capabilities = []
        if dependencies is None:
            dependencies = []
        if crypto_config is None:
            crypto_config = {
                "supported_exchanges": ["binance", "coinbase"],
                "default_pair": "BTC/USDT"
            }
        
        # Create metadata
        metadata = ToolMetadata(
            name=agent_name,
            version=version,
            tool_type=agent_type,
            description=description,
            capabilities=capabilities,
            dependencies=dependencies,
            crypto_config=crypto_config,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # Register with registry
        success, message = self.registry.register_tool(metadata, "agent")
        
        if success:
            # Update status to HEALTHY for agents
            self.registry.update_tool_status(agent_name, ToolStatus.HEALTHY, "Agent initialized")
            logger.info(f"🔧 Registered agent: {agent_name}")
        
        return success, message
    
    def get_agent_status(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the status and health of an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dictionary of agent status or None if not found
        """
        tool = self.registry.get_tool(agent_name)
        
        if tool:
            return {
                "name": tool.name,
                "status": tool.status.name,
                "status_message": tool.status_message,
                "version": tool.version,
                "type": tool.tool_type.name,
                "invocation_count": tool.invocation_count,
                "last_invocation": tool.last_invocation,
                "avg_response_time": tool.avg_response_time
            }
        
        return None
    
    def get_all_agents(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered agents.
        
        Returns:
            List of agent information dictionaries
        """
        tools = self.registry.discover_tools()
        
        agents = []
        for tool in tools:
            agents.append({
                "name": tool.name,
                "type": tool.tool_type.name,
                "status": tool.status.name,
                "version": tool.version,
                "capabilities": tool.capabilities,
                "dependencies": tool.dependencies
            })
        
        return agents
    
    def get_agent_by_type(self, agent_type: ToolType) -> List[Dict[str, Any]]:
        """
        Get agents by type.
        
        Args:
            agent_type: Type of agent to filter by
            
        Returns:
            List of matching agents
        """
        tools = self.registry.discover_tools(tool_type=agent_type)
        
        agents = []
        for tool in tools:
            agents.append({
                "name": tool.name,
                "status": tool.status.name,
                "version": tool.version
            })
        
        return agents
    
    def log_agent_invocation(self, agent_name: str, success: bool,
                            response_time: float = 0.0,
                            error_message: str = "") -> Tuple[bool, str]:
        """
        Log an agent invocation for metrics and monitoring.
        
        Args:
            agent_name: Name of the agent
            success: Whether the invocation was successful
            response_time: Response time in seconds
            error_message: Error message if applicable
            
        Returns:
            Tuple of (success, message)
        """
        return self.registry.log_tool_invocation(
            agent_name, success, response_time, error_message, "agent"
        )
    
    def get_agent_metrics(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Get performance metrics for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dictionary of metrics or None if agent not found
        """
        return self.registry.get_tool_metrics(agent_name)
    
    def get_registry_health(self) -> Dict[str, Any]:
        """
        Get overall registry health metrics.
        
        Returns:
            Dictionary of registry health metrics
        """
        return self.registry.check_system_health()
    
    def validate_agent_chain(self, agent_chain: List[str]) -> Tuple[bool, str]:
        """
        Validate that a chain of agents can be executed based on dependencies.
        
        Args:
            agent_chain: List of agent names in execution order
            
        Returns:
            Tuple of (valid, message)
        """
        return self.registry.validate_tool_chain(agent_chain)
    
    def get_agent_dependencies(self, agent_name: str) -> List[Dict[str, Any]]:
        """
        Get dependencies for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of dependency information
        """
        dependencies = self.registry.get_dependencies(agent_name)
        
        result = []
        for dep in dependencies:
            result.append({
                "source": dep.source_tool,
                "target": dep.target_tool,
                "type": dep.dependency_type,
                "version_constraint": dep.version_constraint
            })
        
        return result
    
    def get_dependent_agents(self, agent_name: str) -> List[str]:
        """
        Get agents that depend on a specific agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            List of agent names that depend on the specified agent
        """
        return self.registry.get_dependent_tools(agent_name)
    
    def update_agent_status(self, agent_name: str, status: ToolStatus,
                           message: str = "") -> Tuple[bool, str]:
        """
        Update the status of an agent.
        
        Args:
            agent_name: Name of the agent
            status: New status (ToolStatus enum)
            message: Status message
            
        Returns:
            Tuple of (success, message)
        """
        return self.registry.update_tool_status(agent_name, status, message, "agent")
    
    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """
        Get cryptocurrency-specific configuration for an agent.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Crypto configuration dictionary or None
        """
        return self.registry.get_crypto_config(agent_name)
    
    def update_agent_config(self, agent_name: str,
                           config_update: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Update cryptocurrency-specific configuration for an agent.
        
        Args:
            agent_name: Name of the agent
            config_update: Dictionary of configuration updates
            
        Returns:
            Tuple of (success, message)
        """
        return self.registry.update_crypto_config(agent_name, config_update, "agent")
    
    def validate_agent_compliance(self, agent_name: str) -> Tuple[bool, str]:
        """
        Validate that an agent meets cryptocurrency-specific compliance requirements.
        
        Args:
            agent_name: Name of the agent to validate
            
        Returns:
            Tuple of (compliant, message)
        """
        return self.registry.validate_crypto_compliance(agent_name)
    
    def get_audit_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent audit log entries.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of audit log entries
        """
        operations = self.registry.get_audit_log(limit)
        
        return [asdict(op) for op in operations]
    
    def export_audit_log(self, filepath: str = "agent_audit_log.json") -> Tuple[bool, str]:
        """
        Export audit log to a file.
        
        Args:
            filepath: Path to export to
            
        Returns:
            Tuple of (success, message)
        """
        return self.registry.export_audit_log(filepath)
    
    def get_workflow_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for the complete trading workflow.
        
        Returns:
            Dictionary of workflow metrics
        """
        # Get all agents
        agents = self.get_all_agents()
        
        # Calculate metrics
        total_agents = len(agents)
        healthy_agents = sum(1 for a in agents if a["status"] == "HEALTHY")
        
        # Get registry metrics
        registry_metrics = self.get_registry_health()
        
        return {
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "agent_types": {
                "NEWS": len(self.get_agent_by_type(ToolType.NEWS)),
                "TECHNICAL": len(self.get_agent_by_type(ToolType.TECHNICAL)),
                "ML": len(self.get_agent_by_type(ToolType.ML)),
                "RISK": len(self.get_agent_by_type(ToolType.RISK)),
                "PORTFOLIO": len(self.get_agent_by_type(ToolType.PORTFOLIO)),
                "OTHER": len(self.get_agent_by_type(ToolType.OTHER))
            },
            "registry_health": registry_metrics,
            "compliance": self._calculate_overall_compliance()
        }
    
    def _calculate_overall_compliance(self) -> Dict[str, Any]:
        """Calculate overall compliance metrics."""
        agents = self.get_all_agents()
        
        compliant_count = 0
        non_compliant_count = 0
        
        for agent in agents:
            compliant, _ = self.validate_agent_compliance(agent["name"])
            if compliant:
                compliant_count += 1
            else:
                non_compliant_count += 1
        
        compliance_rate = compliant_count / len(agents) if agents else 0
        
        return {
            "compliant_agents": compliant_count,
            "non_compliant_agents": non_compliant_count,
            "compliance_rate": compliance_rate,
            "compliance_level": self._determine_compliance_level(compliance_rate)
        }
    
    def _determine_compliance_level(self, compliance_rate: float) -> str:
        """Determine compliance level based on rate."""
        if compliance_rate == 1.0:
            return "EXCELLENT"
        elif compliance_rate >= 0.9:
            return "GOOD"
        elif compliance_rate >= 0.7:
            return "FAIR"
        elif compliance_rate >= 0.5:
            return "POOR"
        else:
            return "CRITICAL"
    
    def get_agent_workflow_recommendation(self) -> Dict[str, Any]:
        """
        Get recommended agent workflow based on current registry state.
        
        Returns:
            Dictionary with recommended workflow
        """
        # Standard workflow
        standard_workflow = [
            "NewsAgent",
            "TechnicalAgent", 
            "MLAgent",
            "RiskAgent",
            "ThinkingAgent"
        ]
        
        # Validate standard workflow
        valid, message = self.validate_agent_chain(standard_workflow)
        
        if valid:
            return {
                "recommended_workflow": standard_workflow,
                "valid": True,
                "message": "Standard workflow is valid",
                "alternatives": []
            }
        else:
            # Find alternative workflow
            alternatives = self._find_alternative_workflows()
            
            return {
                "recommended_workflow": [],
                "valid": False,
                "message": f"Standard workflow invalid: {message}",
                "alternatives": alternatives
            }
    
    def _find_alternative_workflows(self) -> List[Dict[str, Any]]:
        """Find alternative workflows when standard workflow is invalid."""
        # This would be more sophisticated in a real implementation
        # For now, return empty list
        return []
    
    def monitor_agent_health(self) -> Dict[str, Any]:
        """
        Monitor and report on agent health status.
        
        Returns:
            Dictionary of health monitoring results
        """
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "agents": {},
            "summary": {}
        }
        
        agents = self.get_all_agents()
        
        for agent in agents:
            agent_name = agent["name"]
            status = agent["status"]
            
            health_report["agents"][agent_name] = {
                "status": status,
                "healthy": status == "HEALTHY",
                "metrics": self.get_agent_metrics(agent_name)
            }
        
        # Calculate summary
        total = len(agents)
        healthy = sum(1 for a in agents if a["status"] == "HEALTHY")
        degraded = sum(1 for a in agents if a["status"] == "DEGRADED")
        failed = sum(1 for a in agents if a["status"] == "FAILED")
        
        health_report["summary"] = {
            "total_agents": total,
            "healthy": healthy,
            "degraded": degraded,
            "failed": failed,
            "health_rate": healthy / total if total > 0 else 0
        }
        
        return health_report


# Example usage
if __name__ == "__main__":
    # Initialize the agent
    registry_agent = ToolRegistryAgent()
    
    # Example: Register an agent
    success, message = registry_agent.register_agent(
        agent_name="TestAgent",
        agent_type=ToolType.TECHNICAL,
        version="1.0.0",
        description="Test agent for demonstration",
        capabilities=["testing", "demonstration"],
        dependencies=[],
        crypto_config={
            "supported_exchanges": ["binance", "coinbase"],
            "default_pair": "BTC/USDT"
        }
    )
    
    print(f"Agent registration: {success}, {message}")
    
    # Get all agents
    agents = registry_agent.get_all_agents()
    print(f"\nRegistered agents: {len(agents)}")
    for agent in agents:
        print(f"  - {agent['name']} ({agent['type']}): {agent['status']}")
    
    # Get agent status
    status = registry_agent.get_agent_status("TestAgent")
    if status:
        print(f"\nTestAgent status: {status['status']}")
    
    # Get workflow metrics
    metrics = registry_agent.get_workflow_metrics()
    print(f"\nWorkflow metrics:")
    print(f"  Total agents: {metrics['total_agents']}")
    print(f"  Healthy agents: {metrics['healthy_agents']}")
    print(f"  Compliance level: {metrics['compliance']['compliance_level']}")
    
    # Get registry health
    health = registry_agent.get_registry_health()
    print(f"\nRegistry health: {health['registry_status']}")
    print(f"  Healthy tools: {health['healthy_tools']}/{health['tool_count']}")