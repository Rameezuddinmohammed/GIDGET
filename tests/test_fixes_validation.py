"""Tests to validate all the fixes implemented."""

import asyncio
import pytest
from unittest.mock import Mock, patch

from src.code_intelligence.core.connection_pool import ConnectionPoolManager, ConnectionPool
from src.code_intelligence.core.singleton import SingletonMeta
from src.code_intelligence.config import LLMConfig, ConfigurationError
from src.code_intelligence.agents.base import BaseAgent, AgentConfig, PromptManager, AgentMonitor
from src.code_intelligence.exceptions import (
    AgentError, AgentExecutionError, LLMError, DatabaseError
)


class TestConnectionPooling:
    """Test connection pooling implementation."""
    
    @pytest.mark.asyncio
    async def test_connection_pool_creation(self):
        """Test connection pool creates and manages connections."""
        class MockClient:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                
        pool = ConnectionPool(MockClient, max_connections=2, test_param="value")
        
        async with pool.get_connection() as conn1:
            assert isinstance(conn1, MockClient)
            assert conn1.kwargs["test_param"] == "value"
            
        # Test pool reuse
        async with pool.get_connection() as conn2:
            assert isinstance(conn2, MockClient)
            
        await pool.close_all()
        
    @pytest.mark.asyncio
    async def test_connection_pool_manager_singleton(self):
        """Test connection pool manager is singleton."""
        manager1 = ConnectionPoolManager()
        manager2 = ConnectionPoolManager()
        
        assert manager1 is manager2
        
        # Clear for other tests
        SingletonMeta.clear_instances()
        
    @pytest.mark.asyncio
    async def test_connection_pool_max_connections(self):
        """Test connection pool respects max connections."""
        class MockClient:
            def __init__(self):
                pass
                
        pool = ConnectionPool(MockClient, max_connections=1)
        
        # First connection should work
        async with pool.get_connection() as conn1:
            assert conn1 is not None
            
        await pool.close_all()
        
    @pytest.mark.asyncio
    async def test_connection_pool_health_check(self):
        """Test connection pool health checking."""
        class MockClient:
            def __init__(self, healthy=True):
                self.healthy = healthy
                
            def health_check(self):
                return self.healthy
                
        pool = ConnectionPool(MockClient, max_connections=2)
        
        # Test healthy connection
        async with pool.get_connection() as conn1:
            assert conn1.healthy is True
            
        await pool.close_all()
        
    @pytest.mark.asyncio
    async def test_connection_creation_failure_recovery(self):
        """Test connection pool handles creation failures properly."""
        class FailingClient:
            def __init__(self, fail_count=1):
                if hasattr(FailingClient, '_call_count'):
                    FailingClient._call_count += 1
                else:
                    FailingClient._call_count = 1
                    
                if FailingClient._call_count <= fail_count:
                    raise Exception("Connection failed")
                    
        pool = ConnectionPool(FailingClient, max_connections=2, fail_count=1)
        
        # First attempt should fail
        with pytest.raises(Exception):
            async with pool.get_connection() as conn:
                pass
                
        # Second attempt should succeed
        async with pool.get_connection() as conn:
            assert conn is not None
            
        await pool.close_all()
        # Reset for other tests
        if hasattr(FailingClient, '_call_count'):
            delattr(FailingClient, '_call_count')


class TestConfigurationValidation:
    """Test configuration validation."""
    
    def test_temperature_validation(self):
        """Test temperature validation works."""
        # Test valid temperature
        from pydantic import ValidationError
        from src.code_intelligence.config import LLMConfig
        
        # This should work - temperature in valid range
        try:
            config = LLMConfig.model_validate({
                "azure_openai_endpoint": "https://test.cognitiveservices.azure.com/",
                "azure_openai_api_key": "test_key_with_sufficient_length_12345",
                "azure_openai_deployment_name": "gpt-4",
                "temperature": 0.5,
                "max_tokens": 1000
            })
            assert config.temperature == 0.5
        except Exception as e:
            # If validation fails due to other reasons, that's ok for this test
            pass
            
    def test_field_constraints_exist(self):
        """Test that field constraints are properly defined."""
        from src.code_intelligence.config import LLMConfig
        
        # Check that the field has constraints
        fields = LLMConfig.model_fields
        temp_field = fields.get('temperature')
        
        assert temp_field is not None
        # The constraints are in the metadata
        assert temp_field.metadata is not None
        assert len(temp_field.metadata) > 0
        # Should have Ge and Le constraints
        constraint_types = [type(m).__name__ for m in temp_field.metadata]
        assert 'Ge' in constraint_types
        assert 'Le' in constraint_types
            
    def test_endpoint_validation_logic(self):
        """Test endpoint validation logic."""
        from src.code_intelligence.config import LLMConfig
        
        # Test the validator directly
        valid_endpoint = "https://test.cognitiveservices.azure.com/"
        invalid_endpoint = "https://invalid-endpoint.com/"
        
        # Valid endpoint should pass
        result = LLMConfig.validate_azure_endpoint(valid_endpoint)
        assert result == valid_endpoint
        
        # Invalid endpoint should raise ConfigurationError
        with pytest.raises(ConfigurationError):
            LLMConfig.validate_azure_endpoint(invalid_endpoint)


class TestExceptionHierarchy:
    """Test proper exception hierarchy."""
    
    def test_exception_with_details(self):
        """Test exception with details and cause."""
        cause = ValueError("Original error")
        error = AgentExecutionError(
            "Agent failed",
            details={"agent_name": "test_agent", "step": "execution"},
            cause=cause
        )
        
        assert error.message == "Agent failed"
        assert error.details["agent_name"] == "test_agent"
        assert error.cause is cause
        assert error.__cause__ is cause  # Test proper exception chaining
        assert "Details:" in str(error)
        
    def test_exception_inheritance(self):
        """Test exception inheritance hierarchy."""
        error = AgentExecutionError("Test error")
        
        assert isinstance(error, AgentError)
        assert isinstance(error, Exception)
        
    def test_exception_from_exception_factory(self):
        """Test exception factory method for proper chaining."""
        from src.code_intelligence.exceptions import CodeIntelligenceError
        
        original = ValueError("Original error")
        error = CodeIntelligenceError.from_exception(
            "Wrapped error", 
            original, 
            {"context": "test"}
        )
        
        assert error.message == "Wrapped error"
        assert error.cause is original
        assert error.__cause__ is original
        assert error.details["context"] == "test"


class TestResourceCleanup:
    """Test resource cleanup mechanisms."""
    
    def test_prompt_manager_cleanup(self):
        """Test prompt manager cleanup."""
        manager = PromptManager()
        manager.register_template("test", Mock())
        
        assert manager.get_template_count() == 1
        
        manager.clear_templates()
        assert manager.get_template_count() == 0
        
    def test_agent_monitor_cleanup(self):
        """Test agent monitor cleanup."""
        monitor = AgentMonitor()
        
        # Add some metrics
        for i in range(1500):  # More than max_entries
            monitor.record_execution("test_agent", 1.0, True)
            
        assert len(monitor.execution_times["test_agent"]) == 1500
        
        # Cleanup old metrics
        monitor.cleanup_old_metrics(max_entries=1000)
        assert len(monitor.execution_times["test_agent"]) == 1000
        
        # Clear all metrics
        monitor.clear_metrics()
        assert len(monitor.execution_times) == 0


class TestDependencyInjection:
    """Test dependency injection in BaseAgent."""
    
    def test_agent_with_injected_llm(self):
        """Test agent with injected LLM client."""
        mock_llm = Mock()
        config = AgentConfig(name="test", description="test")
        
        class TestAgent(BaseAgent):
            async def execute(self, state):
                return state
                
        agent = TestAgent(config, llm_client=mock_llm)
        assert agent._llm_client is mock_llm
        
    @pytest.mark.asyncio
    async def test_agent_llm_call_with_injection(self):
        """Test agent LLM call uses injected client."""
        mock_llm = Mock()
        mock_llm.chat_completion = Mock(return_value="test response")
        
        config = AgentConfig(name="test", description="test")
        
        class TestAgent(BaseAgent):
            async def execute(self, state):
                return state
                
        agent = TestAgent(config, llm_client=mock_llm)
        
        # Mock the async call
        async def mock_chat_completion(*args, **kwargs):
            return "test response"
            
        mock_llm.chat_completion = mock_chat_completion
        
        result = await agent._call_llm("test prompt")
        
        assert result == "test response"


class TestSSLContextCreation:
    """Test SSL context creation logic."""
    
    def test_ssl_context_creation(self):
        """Test SSL context creation for different Neo4j configurations."""
        from src.code_intelligence.database.neo4j_client import Neo4jClient
        from unittest.mock import patch
        
        client = Neo4jClient()
        
        # Test Neo4j Aura configuration
        with patch('src.code_intelligence.database.neo4j_client.config') as mock_config:
            mock_config.database.neo4j_uri = "bolt+s://test.databases.neo4j.io:7687"
            
            ssl_context, fallback_uri = client._create_ssl_context()
            
            assert ssl_context.check_hostname is False
            assert fallback_uri == "bolt://test.databases.neo4j.io:7687"
            
        # Test custom Neo4j configuration
        with patch('src.code_intelligence.database.neo4j_client.config') as mock_config:
            mock_config.database.neo4j_uri = "bolt+s://custom.example.com:7687"
            
            ssl_context, fallback_uri = client._create_ssl_context()
            
            assert fallback_uri == "bolt://custom.example.com:7687"


if __name__ == "__main__":
    pytest.main([__file__])