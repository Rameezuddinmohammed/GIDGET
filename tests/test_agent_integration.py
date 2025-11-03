"""Integration tests for the complete agent system."""

import asyncio
import pytest
from datetime import datetime
from uuid import uuid4

from src.code_intelligence.agents import AgentOrchestrator, AgentState, BaseAgent
from src.code_intelligence.agents.base import AgentConfig
from src.code_intelligence.agents.orchestrator import OrchestrationConfig
from src.code_intelligence.agents.communication import (
    AgentCommunicationProtocol, ConflictResolver, StateValidator
)
from src.code_intelligence.agents.tools import GitTool, FileSystemTool


class TestAgent(BaseAgent):
    """Test agent for integration testing."""
    
    def __init__(self, name: str, test_findings: list = None):
        config = AgentConfig(name=name, description=f"Test {name} agent")
        super().__init__(config)
        self.test_findings = test_findings or []
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute test agent logic."""
        self._log_execution_start(state)
        
        # Validate state
        if not self._validate_state(state):
            state.add_error("Invalid state received", self.config.name)
            return state
            
        # Add test findings
        for finding_data in self.test_findings:
            finding = self._create_finding(**finding_data)
            state.add_finding(self.config.name, finding)
            
        self._log_execution_end(state, success=True)
        return state


@pytest.mark.asyncio
class TestCompleteAgentSystem:
    """Test the complete agent system integration."""
    
    async def test_full_system_integration(self):
        """Test complete system with all components."""
        # Setup orchestrator
        config = OrchestrationConfig(
            max_execution_time_seconds=60,
            graceful_degradation=True,
            enable_parallel_execution=False  # Sequential for testing
        )
        orchestrator = AgentOrchestrator(config)
        
        # Setup communication protocol
        comm_protocol = AgentCommunicationProtocol()
        conflict_resolver = ConflictResolver()
        state_validator = StateValidator()
        
        # Create test agents
        query_agent = TestAgent("query_parser", [
            {
                "finding_type": "query_analysis",
                "content": "Parsed query: 'Show me code evolution patterns'",
                "confidence": 0.95,
                "metadata": {"intent": "code_evolution", "scope": "repository"}
            }
        ])
        
        history_agent = TestAgent("git_historian", [
            {
                "finding_type": "git_history",
                "content": "Found 15 commits with significant changes",
                "confidence": 0.88,
                "metadata": {"commit_count": 15, "time_span": "3 months"}
            }
        ])
        
        code_agent = TestAgent("code_analyzer", [
            {
                "finding_type": "code_analysis",
                "content": "Identified 5 major refactoring patterns",
                "confidence": 0.92,
                "metadata": {"patterns": ["extract_method", "move_class", "rename"]}
            }
        ])
        
        synthesis_agent = TestAgent("synthesizer", [
            {
                "finding_type": "synthesis",
                "content": "Code evolution shows consistent refactoring towards modularity",
                "confidence": 0.90,
                "metadata": {"trend": "increasing_modularity", "confidence_trend": "high"}
            }
        ])
        
        # Register agents
        orchestrator.register_agent("orchestrator", query_agent)
        orchestrator.register_agent("historian", history_agent)
        orchestrator.register_agent("analyst", code_agent)
        orchestrator.register_agent("synthesizer", synthesis_agent)
        
        # Execute query
        result = await orchestrator.execute_query(
            "Show me code evolution patterns in this repository",
            "/test/repository"
        )
        
        # Convert result if needed
        if isinstance(result, dict):
            result = AgentState(**result)
            
        # Validate results
        assert isinstance(result, AgentState)
        assert not result.has_errors()
        
        # Check findings
        all_findings = result.get_all_findings()
        assert len(all_findings) >= 3  # At least 3 agents should execute
        
        # Validate state consistency
        validation_issues = await state_validator.validate_state(result)
        assert len(validation_issues) == 0, f"State validation issues: {validation_issues}"
        
        # Check overall confidence
        avg_confidence = sum(f.confidence for f in all_findings) / len(all_findings)
        assert avg_confidence > 0.85
        
        # Verify session tracking
        assert result.session_id
        assert result.created_at <= result.updated_at
        
    async def test_agent_communication_flow(self):
        """Test agent communication and coordination."""
        comm_protocol = AgentCommunicationProtocol()
        
        # Setup dependencies
        from src.code_intelligence.agents.communication import AgentDependency
        
        deps = [
            AgentDependency(
                dependent="code_analyzer",
                dependency="git_historian", 
                dependency_type="data"
            ),
            AgentDependency(
                dependent="synthesizer",
                dependency="code_analyzer",
                dependency_type="analysis"
            )
        ]
        
        for dep in deps:
            comm_protocol.register_dependency(dep)
            
        # Calculate execution order
        agents = ["git_historian", "code_analyzer", "synthesizer"]
        order = comm_protocol.calculate_execution_order(agents)
        
        # Verify correct ordering
        assert order.index("git_historian") < order.index("code_analyzer")
        assert order.index("code_analyzer") < order.index("synthesizer")
        
        # Test message passing
        from src.code_intelligence.agents.communication import AgentMessage, MessageType, MessagePriority
        
        message = AgentMessage(
            sender="git_historian",
            recipient="code_analyzer",
            message_type=MessageType.NOTIFICATION,
            priority=MessagePriority.HIGH,
            content={"commits_analyzed": 15, "patterns_found": ["refactoring", "feature_addition"]}
        )
        
        await comm_protocol.send_message(message)
        received = await comm_protocol.receive_messages("code_analyzer")
        
        assert len(received) == 1
        assert received[0].content["commits_analyzed"] == 15
        
    async def test_conflict_resolution_workflow(self):
        """Test conflict detection and resolution."""
        conflict_resolver = ConflictResolver()
        
        # Create state with conflicting findings
        state = AgentState(session_id=str(uuid4()))
        
        from src.code_intelligence.agents.state import AgentFinding, Citation
        
        # Add conflicting findings
        finding1 = AgentFinding(
            agent_name="agent_a",
            finding_type="performance_impact",
            content="Performance significantly improved after refactoring",
            confidence=0.85,
            citations=[
                Citation(
                    file_path="src/main.py",
                    line_number=42,
                    description="Optimized algorithm implementation"
                )
            ]
        )
        
        finding2 = AgentFinding(
            agent_name="agent_b", 
            finding_type="performance_impact",
            content="Performance decreased due to additional complexity",
            confidence=0.80,
            citations=[
                Citation(
                    file_path="src/utils.py",
                    line_number=15,
                    description="Added validation overhead"
                )
            ]
        )
        
        state.add_finding("agent_a", finding1)
        state.add_finding("agent_b", finding2)
        
        # Detect conflicts
        conflicts = await conflict_resolver.detect_conflicts(state)
        assert len(conflicts) > 0
        
        # Resolve conflict
        conflict_id, findings = conflicts[0]
        resolution = await conflict_resolver.resolve_conflict(
            conflict_id, findings, "confidence_weighted"
        )
        
        assert resolution.resolved_finding is not None
        assert resolution.agents_involved == ["agent_a", "agent_b"]
        assert resolution.resolution_strategy == "confidence_weighted"
        
    async def test_error_handling_and_recovery(self):
        """Test error handling and graceful degradation."""
        config = OrchestrationConfig(
            max_execution_time_seconds=30,
            graceful_degradation=True,
            agent_timeout_seconds=5
        )
        orchestrator = AgentOrchestrator(config)
        
        # Create agents with one that fails
        class FailingAgent(BaseAgent):
            def __init__(self):
                super().__init__(AgentConfig(name="failing_agent", description="Fails on purpose"))
                
            async def execute(self, state: AgentState) -> AgentState:
                raise Exception("Simulated agent failure")
                
        class WorkingAgent(BaseAgent):
            def __init__(self, name: str):
                super().__init__(AgentConfig(name=name, description=f"Working {name}"))
                
            async def execute(self, state: AgentState) -> AgentState:
                finding = self._create_finding(
                    finding_type="success",
                    content=f"Agent {self.config.name} executed successfully",
                    confidence=0.9
                )
                state.add_finding(self.config.name, finding)
                return state
        
        # Register agents
        orchestrator.register_agent("orchestrator", WorkingAgent("orchestrator"))
        orchestrator.register_agent("failing_agent", FailingAgent())
        orchestrator.register_agent("analyst", WorkingAgent("analyst"))
        orchestrator.register_agent("synthesizer", WorkingAgent("synthesizer"))
        orchestrator.register_agent("verifier", WorkingAgent("verifier"))
        
        # Execute query
        result = await orchestrator.execute_query("Test query", "/test/repo")
        
        # Convert result if needed
        if isinstance(result, dict):
            result = AgentState(**result)
            
        # Should have errors but still some results due to graceful degradation
        assert result.has_errors()
        assert len(result.get_all_findings()) > 0  # Some agents succeeded
        
        # Check that error is properly recorded
        error_messages = [error.lower() for error in result.errors]
        assert any("failing_agent" in error for error in error_messages)
        
    async def test_performance_monitoring(self):
        """Test agent performance monitoring."""
        from src.code_intelligence.agents.base import agent_monitor
        
        # Clear previous data
        agent_monitor.execution_times.clear()
        agent_monitor.success_rates.clear()
        agent_monitor.error_counts.clear()
        
        # Record some executions
        agent_monitor.record_execution("test_agent", 0.5, True)
        agent_monitor.record_execution("test_agent", 0.7, True)
        agent_monitor.record_execution("test_agent", 1.2, False)
        
        # Check metrics
        avg_time = agent_monitor.get_average_execution_time("test_agent")
        success_rate = agent_monitor.get_success_rate("test_agent")
        error_count = agent_monitor.get_error_count("test_agent")
        
        assert abs(avg_time - 0.8) < 0.1  # (0.5 + 0.7 + 1.2) / 3
        assert abs(success_rate - 0.667) < 0.01  # 2/3
        assert error_count == 1
        
        # Get health summary
        health = agent_monitor.get_health_summary()
        assert "test_agent" in health
        assert health["test_agent"]["total_executions"] == 3
        
    async def test_tool_integration(self):
        """Test agent tool integration."""
        # Test FileSystem tool
        fs_tool = FileSystemTool()
        
        # Create a temporary test file
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write("def test_function():\n    return 'Hello, World!'\n")
            temp_file = f.name
            
        try:
            # Test file reading
            result = await fs_tool.execute("read_file", file_path=temp_file)
            
            assert result["file_path"] == temp_file
            assert "test_function" in result["content"]
            assert result["line_count"] == 2
            
            # Test file info
            info = await fs_tool.execute("get_file_info", file_path=temp_file)
            
            assert info["file_path"] == temp_file
            assert info["is_file"] == True
            assert info["extension"] == ".py"
            
        finally:
            # Clean up
            os.unlink(temp_file)
            
    async def test_state_serialization(self):
        """Test state serialization and deserialization."""
        # Create a complex state
        original_state = AgentState(
            session_id=str(uuid4()),
            query={"original": "test query", "parsed": {"intent": "analysis"}},
            repository={"path": "/test", "commit": "abc123"}
        )
        
        from src.code_intelligence.agents.state import AgentFinding, Citation
        
        finding = AgentFinding(
            agent_name="test_agent",
            finding_type="test",
            content="Test finding",
            confidence=0.85,
            citations=[
                Citation(
                    file_path="test.py",
                    line_number=10,
                    description="Test citation"
                )
            ],
            metadata={"key": "value"}
        )
        
        original_state.add_finding("test_agent", finding)
        original_state.add_error("Test error", "test_agent")
        original_state.add_warning("Test warning")
        
        # Serialize to dict (simulating LangGraph behavior)
        state_dict = original_state.model_dump()
        
        # Deserialize back to AgentState
        restored_state = AgentState(**state_dict)
        
        # Verify all data is preserved
        assert restored_state.session_id == original_state.session_id
        assert restored_state.query == original_state.query
        assert restored_state.repository == original_state.repository
        assert len(restored_state.get_all_findings()) == 1
        assert len(restored_state.errors) == 1
        assert len(restored_state.warnings) == 1
        
        # Verify finding details
        restored_finding = restored_state.get_all_findings()[0]
        assert restored_finding.agent_name == "test_agent"
        assert restored_finding.confidence == 0.85
        assert len(restored_finding.citations) == 1
        assert restored_finding.citations[0].file_path == "test.py"


if __name__ == "__main__":
    # Run integration tests
    asyncio.run(TestCompleteAgentSystem().test_full_system_integration())