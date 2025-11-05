"""Tests for the intelligent caching system."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.code_intelligence.caching.cache_manager import CacheManager, cache_manager
from src.code_intelligence.caching.invalidation_service import CacheInvalidationService, invalidation_service
from src.code_intelligence.agents.developer_query_orchestrator import DeveloperQueryOrchestrator
from src.code_intelligence.agents.state import AgentState
from src.code_intelligence.agents.base import AgentConfig


class TestCacheManager:
    """Test cache manager functionality."""
    
    @pytest.fixture
    def cache_mgr(self):
        """Create a fresh cache manager for testing."""
        return CacheManager()
    
    def test_generate_query_hash(self, cache_mgr):
        """Test query hash generation."""
        query = "Find all functions that call getUserData"
        repo_id = "test-repo"
        options = {"max_commits": 100}
        
        hash1 = cache_mgr._generate_query_hash(query, repo_id, options)
        hash2 = cache_mgr._generate_query_hash(query, repo_id, options)
        
        # Same inputs should generate same hash
        assert hash1 == hash2
        assert len(hash1) == 16  # Should be 16 character hash
        
        # Different inputs should generate different hashes
        hash3 = cache_mgr._generate_query_hash("Different query", repo_id, options)
        assert hash1 != hash3
    
    def test_calculate_ttl(self, cache_mgr):
        """Test TTL calculation logic."""
        base_query = "test query"
        result_data = {"findings": []}
        
        # High confidence should get longer TTL
        high_conf_ttl = cache_mgr._calculate_ttl(0.95, base_query, result_data)
        low_conf_ttl = cache_mgr._calculate_ttl(0.6, base_query, result_data)
        assert high_conf_ttl > low_conf_ttl
        
        # Historical queries should get longer TTL
        history_ttl = cache_mgr._calculate_ttl(0.8, "when was this function added", result_data)
        regular_ttl = cache_mgr._calculate_ttl(0.8, "what does this function do", result_data)
        assert history_ttl > regular_ttl
        
        # Performance queries should get shorter TTL
        perf_ttl = cache_mgr._calculate_ttl(0.8, "why is this slow", result_data)
        assert perf_ttl < regular_ttl
    
    @pytest.mark.asyncio
    async def test_should_cache_result(self, cache_mgr):
        """Test result caching criteria."""
        # High confidence with findings should be cached
        good_result = {"findings": [{"content": "test"}]}
        assert await cache_mgr.should_cache_result(0.8, good_result)
        
        # Low confidence should not be cached
        assert not await cache_mgr.should_cache_result(0.6, good_result)
        
        # Empty results should not be cached
        empty_result = {"findings": []}
        assert not await cache_mgr.should_cache_result(0.8, empty_result)
        
        # Error results should not be cached
        error_result = {"findings": [{"content": "test"}], "errors": ["error"]}
        assert not await cache_mgr.should_cache_result(0.8, error_result)
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, cache_mgr):
        """Test cache statistics tracking."""
        stats = await cache_mgr.get_cache_stats()
        
        assert "hits" in stats
        assert "misses" in stats
        assert "stores" in stats
        assert "hit_rate" in stats
        assert stats["hit_rate"] == 0.0  # No requests yet


class TestCacheInvalidationService:
    """Test cache invalidation service."""
    
    @pytest.fixture
    def invalidation_svc(self):
        """Create a fresh invalidation service for testing."""
        return CacheInvalidationService()
    
    @pytest.mark.asyncio
    async def test_handle_repository_update(self, invalidation_svc):
        """Test repository update handling."""
        repo_id = "test-repo"
        
        await invalidation_svc.handle_repository_update(
            repository_id=repo_id,
            update_type="commit_ingested",
            details={"commit_sha": "abc123"}
        )
        
        # Should have queued an invalidation event
        assert len(invalidation_svc.pending_invalidations) == 1
        event = invalidation_svc.pending_invalidations[0]
        assert event.repository_id == repo_id
        assert event.event_type == "commit_ingested"
    
    @pytest.mark.asyncio
    async def test_handle_commit_ingestion(self, invalidation_svc):
        """Test commit ingestion handling."""
        repo_id = "test-repo"
        commit_sha = "abc123"
        modified_files = ["src/main.py", "src/utils.py"]
        commit_message = "Fix bug in getUserData"
        
        await invalidation_svc.handle_commit_ingestion(
            repository_id=repo_id,
            commit_sha=commit_sha,
            modified_files=modified_files,
            commit_message=commit_message
        )
        
        # Should have created an invalidation event
        assert len(invalidation_svc.pending_invalidations) == 1
        event = invalidation_svc.pending_invalidations[0]
        assert event.details["commit_sha"] == commit_sha
        assert event.details["modified_files"] == modified_files
    
    @pytest.mark.asyncio
    async def test_invalidation_stats(self, invalidation_svc):
        """Test invalidation statistics."""
        stats = await invalidation_svc.get_invalidation_stats()
        
        assert "pending_invalidations" in stats
        assert "total_invalidations" in stats
        assert "cache_hit_rate" in stats
        assert stats["pending_invalidations"] == 0


class TestDeveloperQueryOrchestratorCaching:
    """Test caching integration in DeveloperQueryOrchestrator."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        config = AgentConfig(
            name="test_orchestrator",
            description="Test orchestrator"
        )
        return DeveloperQueryOrchestrator(config)
    
    @pytest.fixture
    def sample_state(self):
        """Create sample agent state."""
        state = AgentState(
            session_id="test-session",
            query={
                "original": "Find all functions that call getUserData",
                "options": {}
            },
            repository={
                "id": "test-repo",
                "path": "/path/to/repo"
            }
        )
        return state
    
    @pytest.mark.asyncio
    async def test_check_cache_miss(self, orchestrator, sample_state):
        """Test cache check when no cached result exists."""
        with patch('src.code_intelligence.caching.cache_manager.cache_manager.get_cached_result') as mock_get:
            mock_get.return_value = None
            
            result = await orchestrator._check_cache(sample_state)
            assert result is None
            mock_get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_cache_hit(self, orchestrator, sample_state):
        """Test cache check when cached result exists."""
        cached_data = {
            "result_data": {
                "findings": {"analyst": [{"content": "test finding"}]},
                "analysis": {"test": "data"}
            },
            "confidence_score": 0.85,
            "created_at": datetime.now().isoformat()
        }
        
        with patch('src.code_intelligence.caching.cache_manager.cache_manager.get_cached_result') as mock_get:
            mock_get.return_value = cached_data
            
            result = await orchestrator._check_cache(sample_state)
            assert result == cached_data
            mock_get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_apply_cached_result(self, orchestrator, sample_state):
        """Test applying cached result to state."""
        cached_data = {
            "result_data": {
                "findings": {
                    "analyst": [{
                        "finding_type": "code_analysis",
                        "content": "Found function getUserData",
                        "confidence": 0.9,
                        "metadata": {"file": "main.py"},
                        "citations": []
                    }]
                },
                "analysis": {"target_function": "getUserData"},
                "verification": {"confidence": 0.9}
            },
            "confidence_score": 0.9,
            "created_at": datetime.now().isoformat()
        }
        
        result_state = await orchestrator._apply_cached_result(sample_state, cached_data)
        
        # Check that findings were applied
        assert "analyst" in result_state.agent_results
        assert len(result_state.agent_results["analyst"]) == 1
        
        # Check that analysis data was applied
        assert result_state.analysis["target_function"] == "getUserData"
        assert result_state.analysis["from_cache"] is True
        
        # Check that verification data was applied
        assert result_state.verification["confidence"] == 0.9
        assert result_state.verification["cache_confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_store_final_result(self, orchestrator, sample_state):
        """Test storing final result in cache."""
        # Add some findings to the state
        from src.code_intelligence.agents.base import AgentFinding
        
        finding = AgentFinding(
            agent_name="analyst",
            finding_type="code_analysis",
            content="Found getUserData function",
            confidence=0.85,
            metadata={"file": "main.py"}
        )
        sample_state.add_finding("analyst", finding)
        sample_state.verification["overall_confidence"] = 0.85
        
        with patch('src.code_intelligence.caching.cache_manager.cache_manager.should_cache_result') as mock_should:
            with patch('src.code_intelligence.caching.cache_manager.cache_manager.store_result') as mock_store:
                mock_should.return_value = True
                mock_store.return_value = True
                
                result = await orchestrator.store_final_result(sample_state)
                
                assert result is True
                mock_should.assert_called_once()
                mock_store.assert_called_once()
                
                # Check the stored data structure
                call_args = mock_store.call_args
                stored_data = call_args[1]["result_data"]
                assert "findings" in stored_data
                assert "analysis" in stored_data
                assert "verification" in stored_data
    
    @pytest.mark.asyncio
    async def test_execute_with_cache_hit(self, orchestrator, sample_state):
        """Test orchestrator execution with cache hit."""
        cached_data = {
            "result_data": {
                "findings": {"orchestrator": [{"content": "cached result"}]},
                "analysis": {"from_cache": True}
            },
            "confidence_score": 0.9
        }
        
        with patch.object(orchestrator, '_check_cache') as mock_check:
            with patch.object(orchestrator, '_apply_cached_result') as mock_apply:
                with patch('src.code_intelligence.monitoring.agent_monitor.agent_monitor.start_execution') as mock_start:
                    with patch('src.code_intelligence.monitoring.agent_monitor.agent_monitor.record_execution') as mock_record:
                        mock_check.return_value = cached_data
                        mock_apply.return_value = sample_state
                        mock_start.return_value = "exec-123"
                        
                        result = await orchestrator.execute(sample_state)
                        
                        assert result == sample_state
                        mock_check.assert_called_once()
                        mock_apply.assert_called_once_with(sample_state, cached_data)
                        mock_record.assert_called_once()


@pytest.mark.asyncio
async def test_cache_integration_end_to_end():
    """Test end-to-end cache integration."""
    # This test would require more setup with actual database connections
    # For now, we'll test the integration points
    
    state = AgentState(
        session_id="test-session",
        query={"original": "test query"},
        repository={"id": "test-repo"}
    )
    
    # Test that cache manager can be imported and used
    from src.code_intelligence.caching import cache_manager
    stats = await cache_manager.get_cache_stats()
    assert isinstance(stats, dict)
    
    # Test that invalidation service can be imported and used
    from src.code_intelligence.caching import invalidation_service
    inv_stats = await invalidation_service.get_invalidation_stats()
    assert isinstance(inv_stats, dict)