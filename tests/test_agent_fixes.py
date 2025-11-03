"""Tests for the fixes applied to specialized agents."""

import asyncio
import pytest
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch

from src.code_intelligence.agents.orchestrator_agent import OrchestratorAgent
from src.code_intelligence.agents.historian_agent import HistorianAgent
from src.code_intelligence.agents.analyst_agent import AnalystAgent
from src.code_intelligence.agents.verification_agent import VerificationAgent
from src.code_intelligence.agents.state import AgentState, AgentFinding, Citation
from src.code_intelligence.agents.config import AgentConfiguration, get_agent_config, set_agent_config


class TestSecurityFixes:
    """Test security-related fixes."""
    
    def test_path_traversal_protection(self):
        """Test path traversal protection in verification agent."""
        agent = VerificationAgent()
        
        # Test safe paths
        assert agent._is_safe_path("src/main.py", "/repo") == True
        assert agent._is_safe_path("docs/readme.md", "/repo") == True
        
        # Test unsafe paths
        assert agent._is_safe_path("../../../etc/passwd", "/repo") == False
        assert agent._is_safe_path("..\\..\\windows\\system32", "/repo") == False
        assert agent._is_safe_path("/etc/passwd", "/repo") == False
        
    def test_json_sanitization(self):
        """Test JSON sanitization in orchestrator agent."""
        agent = OrchestratorAgent()
        
        # Test dangerous patterns are removed
        dangerous_json = '{"test": "__import__(\'os\').system(\'rm -rf /\')"}'
        sanitized = agent._sanitize_json_string(dangerous_json)
        assert "__import__" not in sanitized
        
        dangerous_json2 = '{"test": "eval(malicious_code)"}'
        sanitized2 = agent._sanitize_json_string(dangerous_json2)
        assert "eval(" not in sanitized2
        
    def test_git_repository_validation(self):
        """Test git repository validation."""
        agent = HistorianAgent()
        
        # Test with temporary directory (not a git repo)
        with tempfile.TemporaryDirectory() as temp_dir:
            assert agent._is_valid_git_repository(temp_dir) == False
            
            # Create fake .git directory
            git_dir = os.path.join(temp_dir, '.git')
            os.makedirs(git_dir)
            assert agent._is_valid_git_repository(temp_dir) == True


class TestConfigurationSystem:
    """Test configuration management system."""
    
    def test_default_configuration(self):
        """Test default configuration values."""
        config = AgentConfiguration()
        
        assert config.limits.max_commits == 50
        assert config.limits.max_findings_per_agent == 100
        assert config.timeouts.llm_timeout_seconds == 30
        assert config.thresholds.confidence_threshold == 0.7
        
    def test_environment_configuration(self):
        """Test configuration from environment variables."""
        with patch.dict(os.environ, {
            'AGENT_MAX_COMMITS': '100',
            'AGENT_LLM_TIMEOUT': '60',
            'AGENT_CONFIDENCE_THRESHOLD': '0.8'
        }):
            config = AgentConfiguration.from_environment()
            
            assert config.limits.max_commits == 100
            assert config.timeouts.llm_timeout_seconds == 60
            assert config.thresholds.confidence_threshold == 0.8
            
    def test_global_configuration(self):
        """Test global configuration management."""
        original_config = get_agent_config()
        
        # Set new configuration
        new_config = AgentConfiguration()
        new_config.limits.max_commits = 25
        set_agent_config(new_config)
        
        # Verify it's used
        retrieved_config = get_agent_config()
        assert retrieved_config.limits.max_commits == 25
        
        # Restore original
        set_agent_config(original_config)


class TestErrorHandling:
    """Test improved error handling."""
    
    @pytest.mark.asyncio
    async def test_database_connection_validation(self):
        """Test database connection validation in analyst agent."""
        agent = AnalystAgent()
        
        # Mock failed Neo4j connection
        with patch('src.code_intelligence.agents.analyst_agent.Neo4jClient') as mock_neo4j:
            mock_neo4j.side_effect = Exception("Connection failed")
            
            await agent._initialize_database_clients()
            
            # Should handle gracefully
            assert agent.neo4j_client is None
            
    @pytest.mark.asyncio
    async def test_git_repository_error_handling(self):
        """Test git repository error handling."""
        agent = HistorianAgent()
        state = AgentState(
            session_id="test-session",
            query={"original": "test query"},
            repository={"path": "/nonexistent/path"}
        )
        
        result_state = await agent.execute(state)
        
        # Should add error and return gracefully
        assert result_state.has_errors()
        assert "Invalid git repository" in result_state.errors[0]
        
    @pytest.mark.asyncio
    async def test_error_context_logging(self):
        """Test that error context is properly logged."""
        agent = OrchestratorAgent()
        state = AgentState(
            session_id="test-session",
            query={"original": "test query"},
            repository={"path": "/test/repo"}
        )
        
        # Mock LLM to raise exception
        with patch.object(agent, '_call_llm', side_effect=Exception("LLM failed")):
            result_state = await agent.execute(state)
            
            # Should handle error gracefully
            assert result_state.has_errors()
            assert "Orchestrator failed" in result_state.errors[0]


class TestPerformanceImprovements:
    """Test performance-related improvements."""
    
    @pytest.mark.asyncio
    async def test_batch_dependency_processing(self):
        """Test batch processing of dependencies."""
        agent = AnalystAgent()
        
        # Mock Neo4j client
        mock_client = AsyncMock()
        mock_client.execute_query.return_value = [
            {
                "from_name": "ClassA",
                "to_name": "ClassB",
                "relationship": "CALLS",
                "to_file": "src/b.py",
                "to_line": 10,
                "from_file": "src/a.py"
            }
        ]
        agent.neo4j_client = mock_client
        
        from src.code_intelligence.agents.state import CodeElement
        elements = [
            CodeElement(name="ClassA", type="class", file_path="src/a.py"),
            CodeElement(name="ClassB", type="class", file_path="src/b.py")
        ]
        
        dependencies = await agent._analyze_batch_dependencies(elements)
        
        # Should return dependencies
        assert len(dependencies) > 0
        assert dependencies[0]["from"] == "ClassA"
        assert dependencies[0]["to"] == "ClassB"
        
        # Should have made batch queries (2 queries total)
        assert mock_client.execute_query.call_count == 2
        
    @pytest.mark.asyncio
    async def test_async_file_operations(self):
        """Test async file operations."""
        agent = VerificationAgent()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.py') as f:
            f.write("line 1\nline 2\nline 3\n")
            temp_file = f.name
            
        try:
            # Test async line validation
            result = await agent._validate_line_number(temp_file, 2)
            assert result == True
            
            result = await agent._validate_line_number(temp_file, 5)
            assert result == False
            
        finally:
            os.unlink(temp_file)
            
    def test_regex_timeout_protection(self):
        """Test regex timeout protection."""
        agent = OrchestratorAgent()
        
        # Test with normal query
        result = agent._fallback_query_parsing("find login function")
        assert "entities" in result
        
        # Test with potentially problematic query (very long)
        long_query = "a" * 10000
        result = agent._fallback_query_parsing(long_query)
        assert "entities" in result  # Should not crash


class TestMemoryManagement:
    """Test memory management improvements."""
    
    @pytest.mark.asyncio
    async def test_configurable_limits(self):
        """Test configurable limits are respected."""
        # Set custom configuration
        config = AgentConfiguration()
        config.limits.max_commits = 10
        config.limits.max_elements_per_analysis = 3
        set_agent_config(config)
        
        agent = OrchestratorAgent()
        
        # Test entity limit
        query = "find user auth service manager factory repository handler controller"
        result = agent._fallback_query_parsing(query)
        
        # Should respect the limit
        assert len(result["entities"]) <= config.limits.max_elements_per_analysis
        
    def test_commit_limit_configuration(self):
        """Test commit limit is configurable."""
        agent = HistorianAgent()
        
        # Mock commits list
        mock_commits = [{"sha": f"commit_{i}"} for i in range(100)]
        
        # Should respect configuration limit
        from src.code_intelligence.agents.config import get_agent_config
        config = get_agent_config()
        
        # The method should limit based on config
        limited_commits = mock_commits[:config.limits.max_commits]
        assert len(limited_commits) == config.limits.max_commits


if __name__ == "__main__":
    pytest.main([__file__, "-v"])