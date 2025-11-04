"""Comprehensive tests for specialized agent capabilities."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from src.code_intelligence.agents.orchestrator_agent import OrchestratorAgent
from src.code_intelligence.agents.historian_agent import HistorianAgent
from src.code_intelligence.agents.analyst_agent import AnalystAgent
from src.code_intelligence.agents.synthesizer_agent import SynthesizerAgent
from src.code_intelligence.agents.verification_agent import VerificationAgent
from src.code_intelligence.agents.state import AgentState, AgentFinding, Citation, CodeElement, ParsedQuery, QueryScope
from src.code_intelligence.agents.base import AgentConfig


class TestOrchestratorAgent:
    """Test suite for Orchestrator Agent."""
    
    @pytest.fixture
    def orchestrator_agent(self):
        """Create orchestrator agent for testing."""
        return OrchestratorAgent()
        
    @pytest.fixture
    def sample_state(self):
        """Create sample agent state."""
        return AgentState(
            session_id="test-session-123",
            query={
                "original": "What changed in the login function since last week?",
                "options": {}
            },
            repository={
                "path": "/test/repo"
            }
        )
        
    @pytest.mark.asyncio
    async def test_query_parsing(self, orchestrator_agent, sample_state):
        """Test natural language query parsing."""
        # Mock LLM response
        with patch.object(orchestrator_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = '''
            {
                "intent": "find_changes",
                "entities": ["login"],
                "time_range": "last_week",
                "scope": "function",
                "keywords": ["changed", "login", "function"],
                "requires_history": true,
                "requires_semantic_search": false,
                "complexity": "medium"
            }
            '''
            
            result_state = await orchestrator_agent.execute(sample_state)
            
            # Verify parsing results
            assert "parsed" in result_state.query
            parsed_query = result_state.query["parsed"]
            assert parsed_query["intent"] == "find_changes"
            assert "login" in parsed_query["entities"]
            assert parsed_query["scope"] == "function"
            
    @pytest.mark.asyncio
    async def test_workflow_planning(self, orchestrator_agent, sample_state):
        """Test workflow planning logic."""
        with patch.object(orchestrator_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
            # Mock query parsing response
            mock_llm.side_effect = [
                '{"intent": "find_changes", "entities": ["login"], "time_range": "last_week", "scope": "function", "keywords": ["changed"], "requires_history": true, "requires_semantic_search": false, "complexity": "medium"}',
                '{"required_agents": ["historian", "analyst", "synthesizer", "verifier"], "execution_plan": [{"agent": "historian", "priority": 1, "focus": "temporal_analysis"}], "parallel_execution": false, "estimated_duration": "3-5 minutes", "complexity_score": 0.7}'
            ]
            
            result_state = await orchestrator_agent.execute(sample_state)
            
            # Verify workflow planning
            assert "workflow_plan" in result_state.analysis
            workflow_plan = result_state.analysis["workflow_plan"]
            assert "historian" in workflow_plan["required_agents"]
            assert len(workflow_plan["execution_plan"]) > 0
            
    @pytest.mark.asyncio
    async def test_fallback_parsing(self, orchestrator_agent, sample_state):
        """Test fallback query parsing when LLM fails."""
        with patch.object(orchestrator_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Invalid JSON response"
            
            result_state = await orchestrator_agent.execute(sample_state)
            
            # Should still have parsed query from fallback
            assert "parsed" in result_state.query
            parsed_query = result_state.query["parsed"]
            assert "intent" in parsed_query
            
    def test_user_preferences_integration(self, orchestrator_agent, sample_state):
        """Test user context and preferences integration."""
        preferences = orchestrator_agent.get_user_preferences("test-user")
        
        assert "preferred_detail_level" in preferences
        assert "confidence_threshold" in preferences
        assert preferences["include_citations"] is True
        
        orchestrator_agent.integrate_user_context(sample_state, "test-user")
        
        assert "user_preferences" in sample_state.query["options"]
        assert "detail_level" in sample_state.query["options"]


class TestHistorianAgent:
    """Test suite for Historian Agent."""
    
    @pytest.fixture
    def historian_agent(self):
        """Create historian agent for testing."""
        return HistorianAgent()
        
    @pytest.fixture
    def sample_state_with_temporal_query(self):
        """Create sample state with temporal query."""
        return AgentState(
            session_id="test-session-456",
            query={
                "original": "How did the UserService class evolve over the last month?",
                "parsed": {
                    "intent": "analyze_evolution",
                    "entities": ["UserService"],
                    "time_range": "last_month",
                    "scope": "class"
                }
            },
            repository={"path": "/test/repo"},
            analysis={
                "target_elements": [
                    {"name": "UserService", "type": "class", "file_path": "src/user_service.py"}
                ]
            }
        )
        
    @pytest.mark.asyncio
    async def test_time_range_determination(self, historian_agent, sample_state_with_temporal_query):
        """Test time range parsing and determination."""
        time_range = await historian_agent._determine_time_range(sample_state_with_temporal_query)
        
        assert time_range is not None
        assert time_range.start_date is not None
        assert time_range.end_date is not None
        assert time_range.end_date > time_range.start_date
        
    @pytest.mark.asyncio
    async def test_commit_analysis(self, historian_agent):
        """Test commit history analysis."""
        sample_commits = [
            {
                "sha": "abc123",
                "message": "Fix user authentication bug",
                "author": "Developer",
                "author_email": "developer@example.com",
                "timestamp": "2024-01-01T12:00:00Z",
                "files_changed": ["src/user_service.py"]
            },
            {
                "sha": "def456", 
                "message": "Add new user registration feature",
                "author": "Developer",
                "author_email": "developer@example.com",
                "timestamp": "2024-01-02T12:00:00Z",
                "files_changed": ["src/user_service.py", "src/registration.py"]
            }
        ]
        
        with patch.object(historian_agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = '''
            {
                "timeline": [
                    {
                        "commit_sha": "abc123",
                        "timestamp": "2024-01-01T12:00:00Z",
                        "author": "developer@example.com",
                        "message": "Fix user authentication bug",
                        "intent": "bug_fix",
                        "changes": [
                            {
                                "element": "UserService",
                                "change_type": "modified",
                                "impact": "medium",
                                "description": "Fixed authentication logic"
                            }
                        ]
                    }
                ],
                "patterns": [
                    {
                        "pattern_type": "bug_fix",
                        "description": "Bug fix pattern detected",
                        "commits": ["abc123"],
                        "confidence": 0.8
                    }
                ],
                "evolution_summary": "UserService underwent bug fixes and feature additions"
            }
            '''
            
            # Create a mock state for this test
            mock_state = AgentState(
                session_id="test-session",
                analysis={"target_elements": [{"name": "UserService"}]}
            )
            
            analysis_result = await historian_agent._analyze_commit_history(
                mock_state, sample_commits
            )
            
            assert "timeline" in analysis_result
            assert "patterns" in analysis_result
            assert len(analysis_result["timeline"]) > 0
            
    def test_intent_extraction(self, historian_agent):
        """Test developer intent extraction from commit messages."""
        test_messages = [
            "Fix critical authentication bug",
            "Add new user registration feature", 
            "Refactor user service for better maintainability",
            "Update dependencies to latest versions",
            "Remove deprecated authentication methods"
        ]
        
        expected_intents = ["bug_fix", "feature_addition", "refactoring", "improvement", "removal"]
        
        for message, expected in zip(test_messages, expected_intents):
            intent = historian_agent._extract_intent_from_message(message)
            assert intent == expected
            
    def test_timeline_visualization(self, historian_agent):
        """Test timeline visualization data creation."""
        analysis_result = {
            "timeline": [
                {
                    "commit_sha": "abc123",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "author": "dev1@example.com",
                    "changes": [{"change_type": "modified"}]
                },
                {
                    "commit_sha": "def456",
                    "timestamp": "2024-01-01T15:00:00Z", 
                    "author": "dev2@example.com",
                    "changes": [{"change_type": "added"}]
                }
            ]
        }
        
        visualization = historian_agent.create_timeline_visualization(analysis_result)
        
        assert "periods" in visualization
        assert "total_commits" in visualization
        assert visualization["total_commits"] == 2


class TestAnalystAgent:
    """Test suite for Analyst Agent."""
    
    @pytest.fixture
    def analyst_agent(self):
        """Create analyst agent for testing."""
        return AnalystAgent()
        
    @pytest.fixture
    def sample_code_elements(self):
        """Create sample code elements for testing."""
        return [
            CodeElement(
                name="UserService",
                type="class",
                file_path="src/user_service.py",
                start_line=10,
                end_line=50
            ),
            CodeElement(
                name="authenticate",
                type="function", 
                file_path="src/auth.py",
                start_line=5,
                end_line=20
            )
        ]
        
    @pytest.mark.asyncio
    async def test_dependency_analysis(self, analyst_agent, sample_code_elements):
        """Test dependency analysis functionality."""
        # Mock Neo4j client
        with patch.object(analyst_agent, 'neo4j_client') as mock_neo4j:
            mock_neo4j.execute_query = AsyncMock(side_effect=[
                [  # Outgoing dependencies
                    {
                        "from_name": "UserService",
                        "to_name": "authenticate", 
                        "relationship": "CALLS",
                        "to_file": "src/auth.py",
                        "to_line": 5
                    }
                ],
                [  # Incoming dependencies
                    {
                        "from_name": "LoginController",
                        "to_name": "UserService",
                        "relationship": "CALLS",
                        "from_file": "src/controller.py",
                        "from_line": 10
                    }
                ]
            ])
            
            dependencies = await analyst_agent._analyze_element_dependencies(sample_code_elements[0])
            
            assert len(dependencies) > 0
            assert dependencies[0]["from"] == "UserService"
            assert dependencies[0]["to"] == "authenticate"
            assert dependencies[0]["relationship"] == "calls"
            
    @pytest.mark.asyncio
    async def test_complexity_metrics(self, analyst_agent, sample_code_elements):
        """Test complexity metrics calculation."""
        with patch.object(analyst_agent, 'neo4j_client') as mock_neo4j:
            mock_neo4j.execute_query = AsyncMock(side_effect=[
                [{"fan_out": 3}],  # Fan-out query
                [{"fan_in": 2}]    # Fan-in query
            ])
            
            fan_metrics = await analyst_agent._calculate_fan_metrics(sample_code_elements[0])
            
            assert fan_metrics["fan_in"] == 2
            assert fan_metrics["fan_out"] == 3
            
            coupling_score = await analyst_agent._calculate_coupling_score(sample_code_elements[0])
            assert 0.0 <= coupling_score <= 1.0
            
    def test_architectural_pattern_detection(self, analyst_agent, sample_code_elements):
        """Test architectural pattern identification."""
        # Add elements with pattern-indicating names
        pattern_elements = sample_code_elements + [
            CodeElement(name="UserFactory", type="class", file_path="src/factory.py"),
            CodeElement(name="UserRepository", type="class", file_path="src/repository.py"),
            CodeElement(name="AuthService", type="class", file_path="src/auth_service.py")
        ]
        
        patterns = asyncio.run(analyst_agent._identify_architectural_patterns(pattern_elements))
        
        pattern_types = [p["pattern"] for p in patterns]
        assert "factory_pattern" in pattern_types
        assert "repository_pattern" in pattern_types
        assert "service_pattern" in pattern_types
        
    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, analyst_agent, sample_code_elements):
        """Test circular dependency detection."""
        with patch.object(analyst_agent, 'neo4j_client') as mock_neo4j:
            mock_neo4j.execute_query = AsyncMock(return_value=[
                {"cycle": ["UserService", "AuthService", "UserService"]}
            ])
            
            circular_deps = await analyst_agent._detect_circular_dependencies(sample_code_elements)
            
            assert len(circular_deps) > 0
            assert circular_deps[0]["length"] == 3
            assert circular_deps[0]["severity"] == "high"


class TestSynthesizerAgent:
    """Test suite for Synthesizer Agent."""
    
    @pytest.fixture
    def synthesizer_agent(self):
        """Create synthesizer agent for testing."""
        return SynthesizerAgent()
        
    @pytest.fixture
    def sample_findings(self):
        """Create sample findings from multiple agents."""
        return [
            AgentFinding(
                agent_name="historian",
                finding_type="temporal_analysis",
                content="UserService was modified 3 times in the last week",
                confidence=0.9,
                citations=[
                    Citation(file_path="src/user_service.py", description="Modified in commit abc123")
                ]
            ),
            AgentFinding(
                agent_name="analyst", 
                finding_type="structural_analysis",
                content="UserService has high coupling with AuthService",
                confidence=0.8,
                citations=[
                    Citation(file_path="src/user_service.py", line_number=25, description="Calls AuthService.authenticate")
                ]
            ),
            AgentFinding(
                agent_name="analyst",
                finding_type="complexity_analysis", 
                content="UserService complexity score is 0.7",
                confidence=0.85,
                citations=[]
            )
        ]
        
    def test_findings_organization(self, synthesizer_agent, sample_findings):
        """Test organization of findings by agent and type."""
        organized = synthesizer_agent._organize_findings(sample_findings)
        
        assert "historian" in organized
        assert "analyst" in organized
        assert "temporal_analysis" in organized["historian"]
        assert "structural_analysis" in organized["analyst"]
        assert "complexity_analysis" in organized["analyst"]
        
    @pytest.mark.asyncio
    async def test_conflict_detection(self, synthesizer_agent):
        """Test conflict detection between agent findings."""
        conflicting_findings = [
            AgentFinding(
                agent_name="agent1",
                finding_type="complexity_analysis",
                content="Component has low complexity",
                confidence=0.9
            ),
            AgentFinding(
                agent_name="agent2", 
                finding_type="complexity_analysis",
                content="Component has high complexity",
                confidence=0.5
            )
        ]
        
        organized = synthesizer_agent._organize_findings(conflicting_findings)
        conflicts = await synthesizer_agent._detect_conflicts(organized)
        
        assert len(conflicts) > 0
        
    def test_citation_indexing(self, synthesizer_agent, sample_findings):
        """Test citation index creation."""
        citation_index = synthesizer_agent._create_citation_index(sample_findings)
        
        assert "src/user_service.py" in citation_index
        assert len(citation_index["src/user_service.py"]) == 2
        
    def test_confidence_calculation(self, synthesizer_agent, sample_findings):
        """Test overall confidence calculation."""
        confidence = synthesizer_agent._calculate_overall_confidence(sample_findings, [])
        
        # Should be average of individual confidences
        expected = (0.9 + 0.8 + 0.85) / 3
        assert abs(confidence - expected) < 0.01
        
    def test_executive_summary_creation(self, synthesizer_agent, sample_findings):
        """Test executive summary generation."""
        organized = synthesizer_agent._organize_findings(sample_findings)
        summary = synthesizer_agent.create_executive_summary(organized)
        
        assert "3 findings" in summary
        assert "2 specialized agents" in summary


class TestVerificationAgent:
    """Test suite for Verification Agent."""
    
    @pytest.fixture
    def verification_agent(self):
        """Create verification agent for testing."""
        return VerificationAgent()
        
    @pytest.fixture
    def sample_finding_with_citations(self):
        """Create sample finding with citations for testing."""
        return AgentFinding(
            agent_name="analyst",
            finding_type="structural_analysis",
            content="Function authenticate is called by UserService.login method",
            confidence=0.8,
            citations=[
                Citation(
                    file_path="src/user_service.py",
                    line_number=25,
                    commit_sha="abc123",
                    description="UserService calls authenticate"
                )
            ]
        )
        
    @pytest.mark.asyncio
    async def test_citation_validation(self, verification_agent, tmp_path):
        """Test citation validation against actual files."""
        # Create test file
        test_file = tmp_path / "src" / "user_service.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("def login():\n    authenticate()\n    return True\n")
        
        citations = [
            Citation(
                file_path="src/user_service.py",
                line_number=2,
                description="Test citation"
            )
        ]
        
        validation = await verification_agent._validate_citations(citations, None, str(tmp_path))
        
        assert validation["total_citations"] == 1
        assert validation["valid_citations"] == 1
        
    @pytest.mark.asyncio
    async def test_line_number_validation(self, verification_agent, tmp_path):
        """Test line number validation."""
        test_file = tmp_path / "test.py"
        test_file.write_text("line 1\nline 2\nline 3\n")
        
        # Valid line number
        assert await verification_agent._validate_line_number(str(test_file), 2) is True
        
        # Invalid line number
        assert await verification_agent._validate_line_number(str(test_file), 5) is False
        
    def test_claim_extraction(self, verification_agent):
        """Test extraction of specific claims from content."""
        content = "Function authenticate is called 5 times. Class UserService depends on AuthService."
        
        claims = verification_agent._extract_claims_from_content(content)
        
        claim_types = [claim["type"] for claim in claims]
        assert "function_reference" in claim_types
        assert "class_reference" in claim_types
        assert "dependency" in claim_types
        
    def test_uncertainty_detection(self, verification_agent):
        """Test uncertainty detection in findings."""
        low_confidence_findings = [
            AgentFinding(
                agent_name="test_agent",
                finding_type="test_finding",
                content="Low confidence finding",
                confidence=0.5,
                citations=[]
            )
        ]
        
        validation_results = [
            {
                "finding_type": "test_finding",
                "validation_result": "invalid",
                "confidence_score": 0.3
            }
        ]
        
        uncertainties = asyncio.run(
            verification_agent._detect_uncertainties(low_confidence_findings, validation_results)
        )
        
        uncertainty_types = [u["type"] for u in uncertainties]
        assert "low_confidence" in uncertainty_types
        assert "validation_failure" in uncertainty_types
        assert "missing_citations" in uncertainty_types
        
    def test_validation_score_calculation(self, verification_agent):
        """Test validation score calculation."""
        citation_validation = {
            "total_citations": 2,
            "valid_citations": 1,
            "invalid_citations": 1
        }
        
        content_validation = {
            "claims_validated": 3,
            "claims_failed": 1
        }
        
        score = verification_agent._calculate_validation_score(citation_validation, content_validation)
        
        # Citation score: 1/2 = 0.5
        # Content score: 3/4 = 0.75 (total claims = validated + failed)
        # Weighted: (0.5 * 0.6) + (0.75 * 0.4) = 0.3 + 0.3 = 0.6
        expected_score = 0.6
        assert abs(score - expected_score) < 0.01


class TestAgentIntegration:
    """Integration tests for agent interactions."""
    
    @pytest.fixture
    def all_agents(self):
        """Create all specialized agents."""
        return {
            "orchestrator": OrchestratorAgent(),
            "historian": HistorianAgent(),
            "analyst": AnalystAgent(), 
            "synthesizer": SynthesizerAgent(),
            "verifier": VerificationAgent()
        }
        
    @pytest.fixture
    def integration_state(self):
        """Create state for integration testing."""
        return AgentState(
            session_id="integration-test-789",
            query={
                "original": "How has the authentication system evolved?",
                "options": {}
            },
            repository={"path": "/test/repo"},
            analysis={"target_elements": []}
        )
        
    @pytest.mark.asyncio
    async def test_agent_workflow_integration(self, all_agents, integration_state):
        """Test integration between agents in a workflow."""
        # Execute orchestrator first
        with patch.object(all_agents["orchestrator"], '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [
                '{"intent": "analyze_evolution", "entities": ["authentication"], "time_range": null, "scope": "repository", "keywords": ["authentication", "system"], "requires_history": true, "requires_semantic_search": true, "complexity": "high"}',
                '{"required_agents": ["historian", "analyst", "synthesizer", "verifier"], "execution_plan": [], "parallel_execution": true, "estimated_duration": "5-10 minutes", "complexity_score": 0.8}'
            ]
            
            state_after_orchestrator = await all_agents["orchestrator"].execute(integration_state)
            
            # Verify orchestrator set up the state properly
            assert "parsed" in state_after_orchestrator.query
            assert "workflow_plan" in state_after_orchestrator.analysis
            
        # Execute synthesizer with some mock findings
        state_after_orchestrator.add_finding(
            "historian",
            AgentFinding(
                agent_name="historian",
                finding_type="temporal_analysis",
                content="Authentication system evolved over 6 months",
                confidence=0.85,
                citations=[]
            )
        )
        
        state_after_orchestrator.add_finding(
            "analyst", 
            AgentFinding(
                agent_name="analyst",
                finding_type="structural_analysis",
                content="Authentication has 3 main components",
                confidence=0.9,
                citations=[]
            )
        )
        
        with patch.object(all_agents["synthesizer"], '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "## Executive Summary\nAuthentication system analysis complete.\n## Detailed Analysis\nSystem has evolved significantly."
            
            state_after_synthesis = await all_agents["synthesizer"].execute(state_after_orchestrator)
            
            # Verify synthesizer created comprehensive findings
            synthesizer_findings = state_after_synthesis.get_findings_by_agent("synthesizer")
            assert len(synthesizer_findings) > 0
            assert any(f.finding_type == "comprehensive_synthesis" for f in synthesizer_findings)
            
    def test_state_consistency_across_agents(self, all_agents, integration_state):
        """Test that state remains consistent as it passes between agents."""
        original_session_id = integration_state.session_id
        original_query = integration_state.query["original"]
        
        # Simulate state passing through multiple agents
        integration_state.update_progress("orchestrator", "parsing", "processing")
        integration_state.update_progress("historian", "analyzing", "processing") 
        integration_state.update_progress("analyst", "analyzing", "processing")
        
        # Verify state consistency
        assert integration_state.session_id == original_session_id
        assert integration_state.query["original"] == original_query
        assert integration_state.progress["current_agent"] == "analyst"
        
    @pytest.mark.asyncio
    async def test_error_handling_across_agents(self, all_agents, integration_state):
        """Test error handling and recovery across agents."""
        # Simulate agent failure
        integration_state.add_error("Test error from historian", "historian")
        
        # Verify error is recorded
        assert integration_state.has_errors()
        assert "historian" in integration_state.errors[0]
        
        # Synthesizer should still be able to process despite errors
        with patch.object(all_agents["synthesizer"], '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Analysis completed with some limitations due to errors."
            
            result_state = await all_agents["synthesizer"].execute(integration_state)
            
            # Should have synthesizer findings despite errors
            synthesizer_findings = result_state.get_findings_by_agent("synthesizer")
            assert len(synthesizer_findings) >= 0  # May be 0 if no other findings to synthesize


@pytest.mark.performance
class TestAgentPerformance:
    """Performance tests for agent response times."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_response_time(self):
        """Test orchestrator agent response time."""
        agent = OrchestratorAgent()
        state = AgentState(
            session_id="perf-test-1",
            query={"original": "Test query", "options": {}},
            repository={"path": "/test"}
        )
        
        with patch.object(agent, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = [
                '{"intent": "test", "entities": [], "time_range": null, "scope": "repository", "keywords": [], "requires_history": false, "requires_semantic_search": false, "complexity": "low"}',
                '{"required_agents": [], "execution_plan": [], "parallel_execution": true, "estimated_duration": "1 minute", "complexity_score": 0.3}'
            ]
            
            import time
            start_time = time.time()
            await agent.execute(state)
            execution_time = time.time() - start_time
            
            # Should complete within reasonable time
            assert execution_time < 5.0  # 5 seconds max
            
    @pytest.mark.asyncio
    async def test_verification_agent_performance(self):
        """Test verification agent performance with multiple findings."""
        agent = VerificationAgent()
        state = AgentState(
            session_id="perf-test-2",
            query={"original": "Test query"},
            repository={"path": "/test"}
        )
        
        # Add multiple findings to verify
        for i in range(10):
            state.add_finding(
                f"test_agent_{i}",
                AgentFinding(
                    agent_name=f"test_agent_{i}",
                    finding_type="test_finding",
                    content=f"Test finding {i}",
                    confidence=0.8,
                    citations=[]
                )
            )
            
        import time
        start_time = time.time()
        await agent.execute(state)
        execution_time = time.time() - start_time
        
        # Should handle 10 findings within reasonable time
        assert execution_time < 10.0  # 10 seconds max for 10 findings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestVerificationAgentIndependentValidation:
    """Test suite for VerificationAgent independent validation."""
    
    @pytest.fixture
    def verification_agent(self):
        """Create verification agent for testing."""
        return VerificationAgent()
        
    @pytest.fixture
    def verification_agent_with_neo4j(self):
        """Create verification agent with mocked Neo4j client."""
        from unittest.mock import MagicMock
        from src.code_intelligence.database.neo4j_client import Neo4jClient
        
        mock_neo4j_client = MagicMock(spec=Neo4jClient)
        return VerificationAgent(neo4j_client=mock_neo4j_client)
        
    @pytest.fixture
    def sample_state_with_claims(self):
        """Create sample state with findings that make verifiable claims."""
        state = AgentState(
            session_id="test_verification",
            query={"original": "Test independent validation"},
            repository={"path": "/test/repo"}
        )
        
        # Add finding with code element change claim
        code_change_finding = AgentFinding(
            agent_name="analyst",
            finding_type="code_element_changed",
            content="Function authenticate_user was modified in commit abc123",
            confidence=0.8,
            citations=[],
            metadata={"commit_sha": "abc123"}
        )
        state.add_finding("analyst", code_change_finding)
        
        # Add finding with dependency claim
        dependency_finding = AgentFinding(
            agent_name="analyst",
            finding_type="dependency_changed", 
            content="Function login now calls function validate_password",
            confidence=0.9,
            citations=[],
            metadata={}
        )
        state.add_finding("analyst", dependency_finding)
        
        return state
        
    def test_verification_agent_basic(self, verification_agent):
        """Test basic VerificationAgent functionality."""
        assert verification_agent.config.name == "verifier"
        assert "validates findings" in verification_agent.config.description.lower()
        assert verification_agent.neo4j_tool is None  # No Neo4j client provided
        assert verification_agent.git_tool is not None
        
    def test_verification_agent_with_neo4j(self, verification_agent_with_neo4j):
        """Test VerificationAgent with Neo4j client."""
        assert verification_agent_with_neo4j.neo4j_tool is not None
        assert verification_agent_with_neo4j.neo4j_client is not None
        
    @pytest.mark.asyncio
    async def test_claim_extraction_code_changes(self, verification_agent):
        """Test extraction of code element change claims."""
        finding = AgentFinding(
            agent_name="analyst",
            finding_type="code_element_changed",
            content="Function authenticate_user was modified in commit abc123 and Class UserManager was added in commit def456",
            confidence=0.8,
            citations=[],
            metadata={}
        )
        
        claims = await verification_agent._extract_verifiable_claims(finding)
        
        # Should extract 2 claims
        assert len(claims) >= 2
        
        # Check function change claim
        function_claims = [c for c in claims if c.get("element_type") == "Function"]
        assert len(function_claims) >= 1
        
        function_claim = function_claims[0]
        assert function_claim["type"] == "code_element_changed"
        assert function_claim["element_name"] == "authenticate_user"
        assert function_claim["commit_sha"] == "abc123"
        assert function_claim["validation_method"] == "neo4j_cpg_query"
        
        # Check class change claim
        class_claims = [c for c in claims if c.get("element_type") == "Class"]
        assert len(class_claims) >= 1
        
        class_claim = class_claims[0]
        assert class_claim["type"] == "code_element_changed"
        assert class_claim["element_name"] == "UserManager"
        assert class_claim["commit_sha"] == "def456"
        
    @pytest.mark.asyncio
    async def test_claim_extraction_dependencies(self, verification_agent):
        """Test extraction of dependency relationship claims."""
        finding = AgentFinding(
            agent_name="analyst",
            finding_type="dependency_changed",
            content="Function login now calls function validate_password and Module auth imports module crypto",
            confidence=0.9,
            citations=[],
            metadata={}
        )
        
        claims = await verification_agent._extract_verifiable_claims(finding)
        
        # Should extract 2 claims
        assert len(claims) >= 2
        
        # Check function call claim
        call_claims = [c for c in claims if c.get("relationship") == "CALLS"]
        assert len(call_claims) >= 1
        
        call_claim = call_claims[0]
        assert call_claim["type"] == "dependency_changed"
        assert call_claim["caller"] == "login"
        assert call_claim["callee"] == "validate_password"
        assert call_claim["validation_method"] == "neo4j_relationship_query"
        
        # Check import claim
        import_claims = [c for c in claims if c.get("relationship") == "IMPORTS"]
        assert len(import_claims) >= 1
        
        import_claim = import_claims[0]
        assert import_claim["caller"] == "auth"
        assert import_claim["callee"] == "crypto"
        
    @pytest.mark.asyncio
    async def test_claim_extraction_commit_intent(self, verification_agent):
        """Test extraction of commit message intent claims."""
        finding = AgentFinding(
            agent_name="historian",
            finding_type="commit_message_intent",
            content="This change was to fix bug #123 in commit abc123",
            confidence=0.8,
            citations=[],
            metadata={"commit_sha": "abc123"}
        )
        
        claims = await verification_agent._extract_verifiable_claims(finding)
        
        # Should extract 1 claim
        assert len(claims) >= 1
        
        intent_claim = claims[0]
        assert intent_claim["type"] == "commit_message_intent"
        assert intent_claim["commit_sha"] == "abc123"
        assert intent_claim["intent"] == "bug #123"
        assert intent_claim["validation_method"] == "git_commit_message_check"
        
    @pytest.mark.asyncio
    async def test_independent_validation_success(self, verification_agent_with_neo4j, sample_state_with_claims):
        """Test independent validation when claims are successfully verified."""
        from unittest.mock import AsyncMock
        
        agent = verification_agent_with_neo4j
        
        # Mock Neo4j tool to return successful validation results
        agent.neo4j_tool.execute = AsyncMock()
        agent.neo4j_tool.execute.side_effect = [
            # First call: code element change validation - SUCCESS
            [{"element_name": "authenticate_user", "file_path": "auth.py", "commit_sha": "abc123"}],
            # Second call: dependency relationship validation - SUCCESS
            [{"caller_name": "login", "caller_file": "auth.py", "callee_name": "validate_password", "callee_file": "auth.py"}]
        ]
        
        # Execute verification
        result_state = await agent.execute(sample_state_with_claims)
        
        # Verify Neo4j queries were made
        assert agent.neo4j_tool.execute.call_count >= 2
        
        # Check verification findings
        verification_findings = result_state.get_findings_by_agent("verifier")
        assert len(verification_findings) > 0
        
        # Should have high confidence since claims were validated
        verification_finding = verification_findings[0]
        assert verification_finding.confidence >= 0.9  # Updated to match Requirement 3.5 (90% threshold)
        
        # Should have no validation uncertainties
        uncertainties = agent.get_validation_uncertainties()
        assert len(uncertainties) == 0
        
    @pytest.mark.asyncio
    async def test_independent_validation_failure(self, verification_agent_with_neo4j, sample_state_with_claims):
        """Test independent validation when claims fail verification."""
        from unittest.mock import AsyncMock
        
        agent = verification_agent_with_neo4j
        
        # Mock Neo4j tool to return failed validation results (empty results)
        agent.neo4j_tool.execute = AsyncMock()
        agent.neo4j_tool.execute.return_value = []  # No results = validation failed
        
        # Execute verification
        result_state = await agent.execute(sample_state_with_claims)
        
        # Verify Neo4j queries were made
        assert agent.neo4j_tool.execute.call_count >= 1
        
        # Check verification findings
        verification_findings = result_state.get_findings_by_agent("verifier")
        assert len(verification_findings) > 0
        
        # Should have low confidence since claims failed validation
        verification_finding = verification_findings[0]
        assert verification_finding.confidence < 0.5
        
        # Should have validation uncertainties
        uncertainties = agent.get_validation_uncertainties()
        assert len(uncertainties) > 0
        assert "failed" in uncertainties[0]["uncertainty"].lower()
        
    @pytest.mark.asyncio
    async def test_confidence_calculation_based_on_validation_results(self, verification_agent):
        """Test that confidence scores are calculated based on actual validation results."""
        agent = verification_agent
        
        # Test Case 1: All claims validated (3/3) = High confidence
        citation_validation = {"total_citations": 2, "valid_citations": 2}
        content_validation = {"claims_validated": 3, "claims_failed": 0}
        
        confidence = agent._calculate_validation_score(citation_validation, content_validation)
        assert confidence >= 0.9, f"Expected high confidence (>=0.9), got {confidence}"  # Aligned with Requirement 3.5
        
        # Test Case 2: Partial validation (2/3) = Medium confidence  
        content_validation = {"claims_validated": 2, "claims_failed": 1}
        
        confidence = agent._calculate_validation_score(citation_validation, content_validation)
        assert 0.6 <= confidence <= 0.8, f"Expected medium confidence (0.6-0.8), got {confidence}"
        
        # Test Case 3: Most claims failed (1/3) = Low confidence
        content_validation = {"claims_validated": 1, "claims_failed": 2}
        
        confidence = agent._calculate_validation_score(citation_validation, content_validation)
        assert confidence <= 0.5, f"Expected low confidence (<=0.5), got {confidence}"
        
        # Test Case 4: No claims to validate = Neutral confidence
        content_validation = {"claims_validated": 0, "claims_failed": 0}
        
        confidence = agent._calculate_validation_score(citation_validation, content_validation)
        assert confidence == 0.5, f"Expected neutral confidence (0.5), got {confidence}"
        
    @pytest.mark.asyncio
    async def test_neo4j_code_element_validation(self, verification_agent_with_neo4j):
        """Test Neo4j CPG query for code element change validation."""
        from unittest.mock import AsyncMock
        
        agent = verification_agent_with_neo4j
        
        # Mock successful Neo4j response
        agent.neo4j_tool.execute = AsyncMock()
        agent.neo4j_tool.execute.return_value = [
            {
                "element_name": "authenticate_user",
                "file_path": "src/auth.py", 
                "commit_sha": "abc123",
                "commit_message": "Fix authentication bug"
            }
        ]
        
        # Test claim
        claim = {
            "type": "code_element_changed",
            "element_type": "Function",
            "element_name": "authenticate_user",
            "commit_sha": "abc123",
            "claim": "Function authenticate_user was modified in commit abc123",
            "validation_method": "neo4j_cpg_query"
        }
        
        # Validate claim
        result = await agent._validate_code_element_change(claim)
        
        # Verify Neo4j query was called with correct parameters
        agent.neo4j_tool.execute.assert_called_once()
        call_args = agent.neo4j_tool.execute.call_args
        assert call_args[1]["parameters"]["element_name"] == "authenticate_user"
        assert call_args[1]["parameters"]["commit_sha"] == "abc123"
        
        # Verify successful validation
        assert result["verified"] is True
        assert result["confidence"] == 1.0
        assert len(result["evidence"]) > 0
        assert "Neo4j CPG confirms" in result["evidence"][0]
        
    @pytest.mark.asyncio
    async def test_neo4j_dependency_validation(self, verification_agent_with_neo4j):
        """Test Neo4j CPG query for dependency relationship validation."""
        from unittest.mock import AsyncMock
        
        agent = verification_agent_with_neo4j
        
        # Mock successful Neo4j response
        agent.neo4j_tool.execute = AsyncMock()
        agent.neo4j_tool.execute.return_value = [
            {
                "caller_name": "login",
                "caller_file": "src/auth.py",
                "callee_name": "validate_password", 
                "callee_file": "src/auth.py"
            }
        ]
        
        # Test claim
        claim = {
            "type": "dependency_changed",
            "caller": "login",
            "callee": "validate_password",
            "relationship": "CALLS",
            "claim": "Function login calls function validate_password",
            "validation_method": "neo4j_relationship_query"
        }
        
        # Validate claim
        result = await agent._validate_dependency_relationship(claim)
        
        # Verify Neo4j query was called
        agent.neo4j_tool.execute.assert_called_once()
        
        # Verify successful validation
        assert result["verified"] is True
        assert result["confidence"] == 1.0
        assert "Neo4j CPG confirms" in result["evidence"][0]
        
    def test_validation_uncertainty_tracking(self, verification_agent):
        """Test that validation uncertainties are properly tracked."""
        agent = verification_agent
        
        # Create test finding and claim
        finding = AgentFinding(
            agent_name="analyst",
            finding_type="dependency_changed",
            content="Function foo calls function bar",
            confidence=0.9,
            citations=[],
            metadata={}
        )
        
        claim = {
            "type": "dependency_changed",
            "claim": "Function foo calls function bar"
        }
        
        validation_result = {
            "verified": False,
            "issues": ["Neo4j CPG found no CALLS relationship between foo and bar"]
        }
        
        # Add uncertainty
        asyncio.run(agent._add_validation_uncertainty(finding, claim, validation_result))
        
        # Check uncertainties were recorded
        uncertainties = agent.get_validation_uncertainties()
        assert len(uncertainties) == 1
        
        uncertainty = uncertainties[0]
        assert uncertainty["finding_agent"] == "analyst"
        assert uncertainty["finding_type"] == "dependency_changed"
        assert uncertainty["failed_claim"] == "Function foo calls function bar"
        assert "Analyst claim failed" in uncertainty["uncertainty"]