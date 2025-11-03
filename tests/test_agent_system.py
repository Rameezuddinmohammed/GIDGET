"""Tests for the multi-agent system."""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from src.code_intelligence.agents.state import (
    AgentState, AgentFinding, Citation, QueryScope, ParsedQuery
)
from src.code_intelligence.agents.base import BaseAgent, AgentConfig, LLMConfig
from src.code_intelligence.agents.orchestrator import AgentOrchestrator, OrchestrationConfig
from src.code_intelligence.agents.communication import (
    AgentCommunicationProtocol, ConflictResolver, StateValidator,
    AgentMessage, MessageType, MessagePriority, AgentDependency
)


class MockAgent(BaseAgent):
    """Mock agent for testing."""
    
    def __init__(self, name: str, findings: list = None, should_fail: bool = False):
        config = AgentConfig(name=name, description=f"Mock {name} agent")
        super().__init__(config)
        self.findings = findings or []
        self.should_fail = should_fail
        self.execution_count = 0
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute mock agent logic."""
        self.execution_count += 1
        
        if self.should_fail:
            raise Exception(f"Mock failure from {self.config.name}")
            
        # Add mock findings
        for finding_data in self.findings:
            finding = self._create_finding(**finding_data)
            state.add_finding(self.config.name, finding)
            
        return state


@pytest.fixture
def agent_state():
    """Create a test agent state."""
    return AgentState(
        session_id=str(uuid4()),
        query={
            "original": "How did the Calculator class change?",
            "parsed": {
                "intent": "code_evolution",
                "entities": ["Calculator"],
                "scope": "class"
            }
        },
        repository={
            "path": "/test/repo",
            "current_commit": "abc123"
        }
    )


@pytest.fixture
def orchestrator():
    """Create a test orchestrator."""
    config = OrchestrationConfig(
        max_execution_time_seconds=30,
        agent_timeout_seconds=5,
        graceful_degradation=True
    )
    return AgentOrchestrator(config)


@pytest.fixture
def communication_protocol():
    """Create a test communication protocol."""
    return AgentCommunicationProtocol()


@pytest.fixture
def conflict_resolver():
    """Create a test conflict resolver."""
    return ConflictResolver()


@pytest.fixture
def state_validator():
    """Create a test state validator."""
    return StateValidator()


class TestAgentState:
    """Test agent state management."""
    
    def test_state_initialization(self):
        """Test state initialization."""
        session_id = str(uuid4())
        state = AgentState(session_id=session_id)
        
        assert state.session_id == session_id
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.updated_at, datetime)
        assert state.query == {}
        assert state.repository == {}
        assert state.agent_results == {}
        assert state.errors == []
        assert state.warnings == []
        
    def test_add_finding(self, agent_state):
        """Test adding findings to state."""
        finding = AgentFinding(
            agent_name="test_agent",
            finding_type="code_change",
            content="Function was modified",
            confidence=0.9
        )
        
        agent_state.add_finding("test_agent", finding)
        
        assert len(agent_state.agent_results["test_agent"]) == 1
        assert agent_state.agent_results["test_agent"][0] == finding
        
    def test_update_progress(self, agent_state):
        """Test progress updates."""
        agent_state.update_progress("test_agent", "analyzing", "processing")
        
        assert agent_state.progress["current_agent"] == "test_agent"
        assert agent_state.progress["current_step"] == "analyzing"
        assert agent_state.progress["status"] == "processing"
        
    def test_error_handling(self, agent_state):
        """Test error and warning handling."""
        agent_state.add_error("Test error", "test_agent")
        agent_state.add_warning("Test warning", "test_agent")
        
        assert len(agent_state.errors) == 1
        assert len(agent_state.warnings) == 1
        assert "[test_agent]" in agent_state.errors[0]
        assert "[test_agent]" in agent_state.warnings[0]
        assert agent_state.has_errors()
        assert agent_state.has_warnings()
        
    def test_get_findings(self, agent_state):
        """Test finding retrieval methods."""
        finding1 = AgentFinding(
            agent_name="agent1",
            finding_type="test",
            content="Finding 1",
            confidence=0.8
        )
        finding2 = AgentFinding(
            agent_name="agent2", 
            finding_type="test",
            content="Finding 2",
            confidence=0.9
        )
        
        agent_state.add_finding("agent1", finding1)
        agent_state.add_finding("agent2", finding2)
        
        assert len(agent_state.get_findings_by_agent("agent1")) == 1
        assert len(agent_state.get_findings_by_agent("agent2")) == 1
        assert len(agent_state.get_all_findings()) == 2


class TestBaseAgent:
    """Test base agent functionality."""
    
    def test_agent_initialization(self):
        """Test agent initialization."""
        config = AgentConfig(
            name="test_agent",
            description="Test agent",
            max_retries=3
        )
        
        agent = MockAgent("test_agent")
        assert agent.config.name == "test_agent"
        assert agent.tools == {}
        
    @pytest.mark.asyncio
    async def test_agent_execution(self, agent_state):
        """Test agent execution."""
        findings = [
            {
                "finding_type": "code_change",
                "content": "Test finding",
                "confidence": 0.8
            }
        ]
        
        agent = MockAgent("test_agent", findings=findings)
        result_state = await agent.execute(agent_state)
        
        assert agent.execution_count == 1
        assert len(result_state.get_findings_by_agent("test_agent")) == 1
        
    @pytest.mark.asyncio
    async def test_agent_failure_handling(self, agent_state):
        """Test agent failure handling."""
        agent = MockAgent("test_agent", should_fail=True)
        
        with pytest.raises(Exception, match="Mock failure"):
            await agent.execute(agent_state)
            
    def test_finding_creation(self):
        """Test finding creation utilities."""
        agent = MockAgent("test_agent")
        
        finding = agent._create_finding(
            finding_type="test",
            content="Test content",
            confidence=0.9
        )
        
        assert finding.agent_name == "test_agent"
        assert finding.finding_type == "test"
        assert finding.content == "Test content"
        assert finding.confidence == 0.9
        
    def test_citation_creation(self):
        """Test citation creation utilities."""
        agent = MockAgent("test_agent")
        
        citation = agent._create_citation(
            file_path="test.py",
            description="Test citation",
            line_number=42
        )
        
        assert citation.file_path == "test.py"
        assert citation.description == "Test citation"
        assert citation.line_number == 42


class TestAgentOrchestrator:
    """Test agent orchestration."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        config = OrchestrationConfig(max_execution_time_seconds=60)
        orchestrator = AgentOrchestrator(config)
        
        assert orchestrator.config.max_execution_time_seconds == 60
        assert orchestrator.agents == {}
        assert orchestrator.graph is not None
        
    def test_agent_registration(self, orchestrator):
        """Test agent registration."""
        agent = MockAgent("test_agent")
        orchestrator.register_agent("test_agent", agent)
        
        assert "test_agent" in orchestrator.agents
        assert orchestrator.agents["test_agent"] == agent
        
    @pytest.mark.asyncio
    async def test_query_execution_success(self, orchestrator):
        """Test successful query execution."""
        # Register mock agents
        orchestrator.register_agent("orchestrator", MockAgent("orchestrator"))
        orchestrator.register_agent("analyst", MockAgent("analyst"))
        orchestrator.register_agent("synthesizer", MockAgent("synthesizer"))
        orchestrator.register_agent("verifier", MockAgent("verifier"))
        
        result = await orchestrator.execute_query(
            "Test query",
            "/test/repo"
        )
        
        # LangGraph returns dict, convert to AgentState for testing
        if isinstance(result, dict):
            result = AgentState(**result)
            
        assert isinstance(result, AgentState)
        assert result.query["original"] == "Test query"
        assert result.repository["path"] == "/test/repo"
        
    @pytest.mark.asyncio
    async def test_query_execution_timeout(self, orchestrator):
        """Test query execution timeout."""
        # Create agent that takes too long
        slow_agent = MockAgent("slow_agent")
        
        async def slow_execute(state):
            await asyncio.sleep(10)  # Longer than timeout
            return state
            
        slow_agent.execute = slow_execute
        orchestrator.register_agent("orchestrator", slow_agent)
        
        # Set very short timeout
        orchestrator.config.max_execution_time_seconds = 1
        
        result = await orchestrator.execute_query("Test query", "/test/repo")
        
        assert result.has_errors()
        assert "timed out" in result.errors[0].lower()
        
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, orchestrator):
        """Test graceful degradation on agent failure."""
        # Register agents with one that fails
        orchestrator.register_agent("orchestrator", MockAgent("orchestrator"))
        orchestrator.register_agent("analyst", MockAgent("analyst", should_fail=True))
        orchestrator.register_agent("synthesizer", MockAgent("synthesizer", findings=[
            {"finding_type": "test", "content": "Partial result", "confidence": 0.7}
        ]))
        orchestrator.register_agent("verifier", MockAgent("verifier"))
        
        result = await orchestrator.execute_query("Test query", "/test/repo")
        
        # LangGraph returns dict, convert to AgentState for testing
        if isinstance(result, dict):
            result = AgentState(**result)
        
        # Should have some results despite failure
        assert len(result.get_all_findings()) > 0
        assert result.has_errors()


class TestAgentCommunication:
    """Test agent communication protocols."""
    
    def test_dependency_registration(self, communication_protocol):
        """Test dependency registration."""
        dep = AgentDependency(
            dependent="agent_b",
            dependency="agent_a",
            dependency_type="data"
        )
        
        communication_protocol.register_dependency(dep)
        
        assert len(communication_protocol.dependencies) == 1
        assert communication_protocol.dependencies[0] == dep
        
    def test_execution_order_calculation(self, communication_protocol):
        """Test execution order calculation."""
        # Register dependencies: C -> B -> A
        deps = [
            AgentDependency(dependent="agent_b", dependency="agent_a", dependency_type="data"),
            AgentDependency(dependent="agent_c", dependency="agent_b", dependency_type="data")
        ]
        
        for dep in deps:
            communication_protocol.register_dependency(dep)
            
        agents = ["agent_a", "agent_b", "agent_c"]
        order = communication_protocol.calculate_execution_order(agents)
        
        assert order.index("agent_a") < order.index("agent_b")
        assert order.index("agent_b") < order.index("agent_c")
        
    @pytest.mark.asyncio
    async def test_message_sending_receiving(self, communication_protocol):
        """Test message sending and receiving."""
        message = AgentMessage(
            sender="agent_a",
            recipient="agent_b",
            message_type=MessageType.REQUEST,
            priority=MessagePriority.HIGH,
            content={"test": "data"}
        )
        
        await communication_protocol.send_message(message)
        messages = await communication_protocol.receive_messages("agent_b")
        
        assert len(messages) == 1
        assert messages[0].sender == "agent_a"
        assert messages[0].content["test"] == "data"
        
    @pytest.mark.asyncio
    async def test_message_priority_ordering(self, communication_protocol):
        """Test message priority ordering."""
        messages = [
            AgentMessage(
                sender="sender", recipient="receiver",
                message_type=MessageType.REQUEST,
                priority=MessagePriority.LOW,
                content={"order": 3}
            ),
            AgentMessage(
                sender="sender", recipient="receiver",
                message_type=MessageType.REQUEST,
                priority=MessagePriority.CRITICAL,
                content={"order": 1}
            ),
            AgentMessage(
                sender="sender", recipient="receiver",
                message_type=MessageType.REQUEST,
                priority=MessagePriority.HIGH,
                content={"order": 2}
            )
        ]
        
        for msg in messages:
            await communication_protocol.send_message(msg)
            
        received = await communication_protocol.receive_messages("receiver")
        
        assert len(received) == 3
        assert received[0].content["order"] == 1  # Critical first
        assert received[1].content["order"] == 2  # High second
        assert received[2].content["order"] == 3  # Low last


class TestConflictResolution:
    """Test conflict resolution."""
    
    @pytest.mark.asyncio
    async def test_conflict_detection(self, conflict_resolver, agent_state):
        """Test conflict detection."""
        # Add contradictory findings
        finding1 = AgentFinding(
            agent_name="agent1",
            finding_type="performance",
            content="Performance increased significantly",
            confidence=0.8
        )
        finding2 = AgentFinding(
            agent_name="agent2",
            finding_type="performance", 
            content="Performance decreased after changes",
            confidence=0.9
        )
        
        agent_state.add_finding("agent1", finding1)
        agent_state.add_finding("agent2", finding2)
        
        conflicts = await conflict_resolver.detect_conflicts(agent_state)
        
        assert len(conflicts) == 1
        conflict_id, findings = conflicts[0]
        assert len(findings) == 2
        
    @pytest.mark.asyncio
    async def test_confidence_weighted_resolution(self, conflict_resolver):
        """Test confidence-weighted conflict resolution."""
        findings = [
            AgentFinding(
                agent_name="agent1",
                finding_type="test",
                content="Finding with low confidence",
                confidence=0.6
            ),
            AgentFinding(
                agent_name="agent2",
                finding_type="test",
                content="Finding with high confidence",
                confidence=0.9
            )
        ]
        
        resolution = await conflict_resolver.resolve_conflict(
            "test_conflict", findings, "confidence_weighted"
        )
        
        assert resolution.resolution_strategy == "confidence_weighted"
        assert resolution.resolution_strategy == "confidence_weighted"
        assert resolution.resolved_finding is not None
        assert resolution.resolved_finding.content.startswith("[RESOLVED]")
        assert resolution.confidence_adjustment < 0  # Penalty applied
        
    @pytest.mark.asyncio
    async def test_evidence_based_resolution(self, conflict_resolver):
        """Test evidence-based conflict resolution."""
        citation1 = Citation(
            file_path="test.py",
            line_number=42,
            description="Evidence 1"
        )
        citation2 = Citation(
            file_path="test.py", 
            line_number=43,
            commit_sha="abc123",
            description="Evidence 2"
        )
        
        findings = [
            AgentFinding(
                agent_name="agent1",
                finding_type="test",
                content="Finding with weak evidence",
                confidence=0.8,
                citations=[citation1]
            ),
            AgentFinding(
                agent_name="agent2",
                finding_type="test",
                content="Finding with strong evidence",
                confidence=0.7,
                citations=[citation1, citation2]
            )
        ]
        
        resolution = await conflict_resolver.resolve_conflict(
            "test_conflict", findings, "evidence_based"
        )
        
        assert resolution.resolution_strategy == "evidence_based"
        assert resolution.resolved_finding is not None
        assert resolution.resolved_finding.content.startswith("[EVIDENCE-BASED]")


class TestStateValidation:
    """Test state validation."""
    
    @pytest.mark.asyncio
    async def test_valid_state(self, state_validator, agent_state):
        """Test validation of valid state."""
        issues = await state_validator.validate_state(agent_state)
        assert len(issues) == 0
        
    @pytest.mark.asyncio
    async def test_missing_session_id(self, state_validator):
        """Test validation with missing session ID."""
        state = AgentState(session_id="")
        issues = await state_validator.validate_state(state)
        
        assert len(issues) > 0
        assert any("session ID" in issue for issue in issues)
        
    @pytest.mark.asyncio
    async def test_invalid_confidence_scores(self, state_validator, agent_state):
        """Test validation of invalid confidence scores."""
        # Pydantic validation prevents creating invalid findings
        # So we test by manually adding invalid data to the state
        from src.code_intelligence.agents.state import AgentFinding
        
        # Create a finding with valid confidence first
        finding = AgentFinding(
            agent_name="test_agent",
            finding_type="test",
            content="Test finding",
            confidence=0.8
        )
        
        # Manually modify the confidence to invalid value (bypassing Pydantic validation)
        finding.confidence = 1.5
        
        agent_state.add_finding("test_agent", finding)
        issues = await state_validator.validate_state(agent_state)
        
        assert len(issues) > 0
        assert any("confidence score" in issue for issue in issues)
        
    @pytest.mark.asyncio
    async def test_invalid_citations(self, state_validator, agent_state):
        """Test validation of invalid citations."""
        citation = Citation(
            file_path="",  # Invalid - empty path
            line_number=-1,  # Invalid - negative line number
            description="Test citation"
        )
        
        finding = AgentFinding(
            agent_name="test_agent",
            finding_type="test",
            content="Test finding",
            confidence=0.8,
            citations=[citation]
        )
        
        agent_state.add_finding("test_agent", finding)
        issues = await state_validator.validate_state(agent_state)
        
        assert len(issues) >= 2  # Both file_path and line_number issues
        assert any("file_path" in issue for issue in issues)
        assert any("line number" in issue for issue in issues)


class TestIntegration:
    """Integration tests for the complete agent system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        # Setup orchestrator with communication protocol
        config = OrchestrationConfig(
            max_execution_time_seconds=30,
            graceful_degradation=True
        )
        orchestrator = AgentOrchestrator(config)
        
        # Create mock agents with realistic findings
        orchestrator_agent = MockAgent("orchestrator", findings=[
            {
                "finding_type": "query_analysis",
                "content": "Analyzed query about Calculator class changes",
                "confidence": 0.95
            }
        ])
        
        historian_agent = MockAgent("historian", findings=[
            {
                "finding_type": "git_history",
                "content": "Found 3 commits affecting Calculator class",
                "confidence": 0.9
            }
        ])
        
        analyst_agent = MockAgent("analyst", findings=[
            {
                "finding_type": "code_analysis",
                "content": "Calculator class had method signature changes",
                "confidence": 0.85
            }
        ])
        
        synthesizer_agent = MockAgent("synthesizer", findings=[
            {
                "finding_type": "synthesis",
                "content": "Calculator evolved through 3 commits with method changes",
                "confidence": 0.88
            }
        ])
        
        verifier_agent = MockAgent("verifier", findings=[
            {
                "finding_type": "verification",
                "content": "All findings verified against source code",
                "confidence": 0.92
            }
        ])
        
        # Register agents
        orchestrator.register_agent("orchestrator", orchestrator_agent)
        orchestrator.register_agent("historian", historian_agent)
        orchestrator.register_agent("analyst", analyst_agent)
        orchestrator.register_agent("synthesizer", synthesizer_agent)
        orchestrator.register_agent("verifier", verifier_agent)
        
        # Execute query
        result = await orchestrator.execute_query(
            "How did the Calculator class change over time?",
            "/test/repo"
        )
        
        # Verify results - LangGraph returns dict, convert back to AgentState
        if isinstance(result, dict):
            result = AgentState(**result)
        
        assert isinstance(result, AgentState)
        assert not result.has_errors()
        
        all_findings = result.get_all_findings()
        assert len(all_findings) >= 4  # At least 4 agents executed
        
        # Check that key agents executed (historian may be skipped based on query)
        executed_agents = ["orchestrator", "analyst", "synthesizer", "verifier"]
        for agent_name in executed_agents:
            agent_findings = result.get_findings_by_agent(agent_name)
            assert len(agent_findings) == 1
            
        # Verify overall confidence
        avg_confidence = sum(f.confidence for f in all_findings) / len(all_findings)
        assert avg_confidence > 0.8
        
    @pytest.mark.asyncio
    async def test_workflow_with_conflicts(self):
        """Test workflow with conflicting agent findings."""
        orchestrator = AgentOrchestrator()
        conflict_resolver = ConflictResolver()
        
        # Create agents with conflicting findings
        agent1 = MockAgent("agent1", findings=[
            {
                "finding_type": "performance",
                "content": "Performance improved significantly",
                "confidence": 0.8
            }
        ])
        
        agent2 = MockAgent("agent2", findings=[
            {
                "finding_type": "performance",
                "content": "Performance decreased after changes",
                "confidence": 0.9
            }
        ])
        
        orchestrator.register_agent("orchestrator", MockAgent("orchestrator"))
        orchestrator.register_agent("agent1", agent1)
        orchestrator.register_agent("agent2", agent2)
        orchestrator.register_agent("synthesizer", MockAgent("synthesizer"))
        orchestrator.register_agent("verifier", MockAgent("verifier"))
        
        # Execute query
        result = await orchestrator.execute_query("Test query", "/test/repo")
        
        # LangGraph returns dict, convert to AgentState for testing
        if isinstance(result, dict):
            result = AgentState(**result)
        
        # Detect and resolve conflicts
        conflicts = await conflict_resolver.detect_conflicts(result)
        assert len(conflicts) > 0
        
        # Resolve first conflict
        conflict_id, findings = conflicts[0]
        resolution = await conflict_resolver.resolve_conflict(
            conflict_id, findings, "confidence_weighted"
        )
        
        assert resolution.resolved_finding is not None
        assert resolution.agents_involved == ["agent1", "agent2"]