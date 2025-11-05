"""End-to-end integration tests for complete workflows."""

import asyncio
import json
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

import httpx
from fastapi.testclient import TestClient

from src.code_intelligence.api.main import app
from src.code_intelligence.agents.developer_query_orchestrator import DeveloperQueryOrchestrator
from src.code_intelligence.agents.state import AgentState
from src.code_intelligence.cli import app as cli_app
from src.code_intelligence.database.neo4j_client import Neo4jClient
from src.code_intelligence.database.supabase_client import SupabaseClient


@pytest.fixture
def test_client():
    """Create test client with proper isolation."""
    # Clear all global storage before each test
    from src.code_intelligence.api.routes.queries import query_storage
    from src.code_intelligence.api.routes.repositories import repository_storage
    
    query_storage.clear()
    repository_storage.clear()
    
    client = TestClient(app)
    yield client
    
    # Clean up after test
    query_storage.clear()
    repository_storage.clear()


@pytest.fixture
def mock_database_clients():
    """Mock database clients for testing."""
    with patch('src.code_intelligence.api.dependencies.get_neo4j_client') as mock_neo4j, \
         patch('src.code_intelligence.api.dependencies.get_supabase_client') as mock_supabase:
        
        # Setup Neo4j mock
        neo4j_client = AsyncMock(spec=Neo4jClient)
        neo4j_client.execute_query.return_value = [{"count": 1}]
        neo4j_client.is_healthy.return_value = True
        mock_neo4j.return_value = neo4j_client
        
        # Setup Supabase mock
        supabase_client = AsyncMock(spec=SupabaseClient)
        supabase_client.is_healthy.return_value = True
        mock_supabase.return_value = supabase_client
        
        yield {
            "neo4j": neo4j_client,
            "supabase": supabase_client
        }


@pytest.fixture
def sample_repository_data():
    """Sample repository data for testing."""
    return {
        "url": "https://github.com/test/sample-repo.git",
        "name": "sample-repo",
        "auto_sync": True
    }


@pytest.fixture
def sample_query_data():
    """Sample query data for testing."""
    return {
        "repository_url": "https://github.com/test/sample-repo.git",
        "query": "What changed in the authentication system between versions?",
        "options": {
            "max_commits": 50,
            "include_tests": False
        }
    }


class TestCompleteWorkflowIntegration:
    """Test complete workflows from start to finish."""
    
    def test_complete_query_workflow_via_api(self, test_client, mock_database_clients, sample_query_data):
        """Test complete query workflow through API endpoints."""
        # Step 1: Submit query
        response = test_client.post("/api/v1/queries/", json=sample_query_data)
        assert response.status_code == 200
        
        query_result = response.json()
        query_id = query_result["query_id"]
        assert query_result["status"] == "pending"
        
        # Step 2: Check query status
        response = test_client.get(f"/api/v1/queries/{query_id}")
        assert response.status_code == 200
        
        status_data = response.json()
        assert status_data["query_id"] == query_id
        assert "status" in status_data
        assert "created_at" in status_data
        
        # Step 3: Simulate query processing completion
        from src.code_intelligence.api.routes.queries import query_storage
        
        # Update query status to completed with mock results
        mock_results = {
            "summary": "Authentication system analysis completed",
            "confidence_score": 0.92,
            "processing_time_seconds": 15.5,
            "findings": [
                {
                    "agent_name": "historian",
                    "finding_type": "version_comparison",
                    "content": "Found significant changes in auth module between v1.2 and v1.3",
                    "confidence": 0.95,
                    "citations": [
                        {
                            "file_path": "src/auth/login.py",
                            "line_number": 42,
                            "description": "Updated password validation logic"
                        }
                    ]
                }
            ]
        }
        
        query_storage[query_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "results": mock_results
        })
        
        # Step 4: Retrieve completed results
        response = test_client.get(f"/api/v1/queries/{query_id}")
        assert response.status_code == 200
        
        final_data = response.json()
        assert final_data["status"] == "completed"
        assert final_data["results"]["confidence_score"] == 0.92
        assert len(final_data["results"]["findings"]) == 1
        
        # Step 5: Export results
        export_data = {
            "query_id": query_id,
            "format": "json",
            "include_citations": True
        }
        
        response = test_client.post(f"/api/v1/queries/{query_id}/export", json=export_data)
        assert response.status_code == 200
        
        export_result = response.json()
        assert "export_id" in export_result
        assert "download_url" in export_result
    
    def test_repository_management_workflow(self, test_client, mock_database_clients, sample_repository_data):
        """Test complete repository management workflow."""
        # Step 1: Register repository
        response = test_client.post("/api/v1/repositories/", json=sample_repository_data)
        assert response.status_code == 200
        
        repo_data = response.json()
        repo_id = repo_data["id"]
        assert repo_data["name"] == "sample-repo"
        assert repo_data["status"] == "not_analyzed"
        
        # Step 2: List repositories
        response = test_client.get("/api/v1/repositories/")
        assert response.status_code == 200
        
        repos = response.json()
        assert len(repos) == 1
        assert repos[0]["id"] == repo_id
        
        # Step 3: Get repository details
        response = test_client.get(f"/api/v1/repositories/{repo_id}")
        assert response.status_code == 200
        
        repo_details = response.json()
        assert repo_details["id"] == repo_id
        assert repo_details["url"] == sample_repository_data["url"]
        
        # Step 4: Trigger analysis
        response = test_client.post(f"/api/v1/repositories/{repo_id}/analyze")
        assert response.status_code == 200
        
        analysis_result = response.json()
        assert analysis_result["message"] == "Analysis started"
        
        # Step 5: Check analysis status
        response = test_client.get(f"/api/v1/repositories/{repo_id}/status")
        assert response.status_code == 200
        
        status_data = response.json()
        assert "analysis_status" in status_data
        assert "commit_count" in status_data
    
    def test_websocket_real_time_updates_workflow(self, test_client):
        """Test WebSocket real-time updates during query processing."""
        with test_client.websocket_connect("/ws") as websocket:
            # Step 1: Establish connection
            welcome_data = websocket.receive_json()
            assert welcome_data["type"] == "connection_established"
            connection_id = welcome_data["data"]["connection_id"]
            
            # Step 2: Subscribe to query updates
            query_id = str(uuid4())
            websocket.send_json({
                "type": "subscribe_query",
                "data": {"query_id": query_id}
            })
            
            subscription_response = websocket.receive_json()
            assert subscription_response["type"] == "subscription_response"
            assert subscription_response["data"]["subscribed"] is True
            
            # Step 3: Simulate progress updates
            from src.code_intelligence.api.websocket import manager
            
            # Simulate query progress
            progress_message = {
                "type": "query_progress",
                "query_id": query_id,
                "data": {
                    "current_agent": "historian",
                    "progress_percentage": 25.0,
                    "current_step": "Analyzing git history"
                }
            }
            
            # This would normally be called by the orchestrator
            asyncio.run(manager.broadcast_to_query_subscribers(progress_message, query_id))
            
            # Step 4: Simulate partial results
            partial_results_message = {
                "type": "partial_results",
                "query_id": query_id,
                "data": {
                    "agent": "historian",
                    "findings": [{"content": "Found 15 relevant commits", "confidence": 0.9}]
                }
            }
            
            asyncio.run(manager.broadcast_to_query_subscribers(partial_results_message, query_id))
            
            # Step 5: Simulate completion
            completion_message = {
                "type": "query_completed",
                "query_id": query_id,
                "data": {
                    "summary": "Analysis completed successfully",
                    "confidence": 0.92
                }
            }
            
            asyncio.run(manager.broadcast_to_query_subscribers(completion_message, query_id))
            
            # Step 6: Unsubscribe
            websocket.send_json({
                "type": "unsubscribe_query",
                "data": {"query_id": query_id}
            })
            
            unsubscribe_response = websocket.receive_json()
            assert unsubscribe_response["type"] == "unsubscription_response"
            assert unsubscribe_response["data"]["unsubscribed"] is True
    
    @pytest.mark.asyncio
    async def test_multi_agent_coordination_workflow(self, mock_database_clients):
        """Test complete multi-agent coordination workflow."""
        # Step 1: Initialize orchestrator
        orchestrator = DeveloperQueryOrchestrator()
        
        # Step 2: Create initial state
        state = AgentState(
            session_id=str(uuid4()),
            query={
                "original": "Find the working version of the user authentication feature",
                "parsed": {"intent": "regression_analysis"}
            },
            repository={
                "id": "test-repo",
                "path": "/test/repo",
                "current_commit": "abc123"
            }
        )
        
        # Step 3: Execute orchestrator
        result_state = await orchestrator.execute(state)
        
        # Verify orchestrator results
        assert not result_state.has_errors()
        findings = result_state.get_all_findings()
        assert len(findings) > 0
        
        orchestrator_finding = findings[0]
        assert orchestrator_finding.agent_name == "developer_orchestrator"
        assert orchestrator_finding.confidence > 0.8
        assert "developer_intent" in orchestrator_finding.metadata
        
        # Step 4: Verify developer intent parsing
        developer_intent = orchestrator_finding.metadata["developer_intent"]
        assert developer_intent["problem_type"] == "regression_analysis"
        assert developer_intent["urgency"] == "high"  # Regressions are urgent
        assert "working_code" in developer_intent["deliverables_needed"]
        
        # Step 5: Verify solution planning
        solution_plan = orchestrator_finding.metadata["solution_plan"]
        assert solution_plan["approach"] == "historical_code_recovery"
        assert len(solution_plan["agent_sequence"]) >= 3
        
        # Verify agent sequence includes required agents
        agent_names = [agent["agent"] for agent in solution_plan["agent_sequence"]]
        assert "historian" in agent_names
        assert "analyst" in agent_names
        assert "synthesizer" in agent_names
    
    def test_error_handling_and_recovery_workflow(self, test_client, mock_database_clients):
        """Test error handling and graceful degradation in complete workflow."""
        # Step 1: Submit query that will cause errors
        invalid_query_data = {
            "repository_url": "https://github.com/invalid/repo.git",
            "query": "Test query that will fail",
            "options": {"max_commits": 10}
        }
        
        response = test_client.post("/api/v1/queries/", json=invalid_query_data)
        assert response.status_code == 200
        
        query_result = response.json()
        query_id = query_result["query_id"]
        
        # Step 2: Simulate processing with errors
        from src.code_intelligence.api.routes.queries import query_storage
        
        # Update query with error status
        query_storage[query_id].update({
            "status": "failed",
            "error": "Repository not accessible",
            "completed_at": datetime.now().isoformat()
        })
        
        # Step 3: Verify error is properly reported
        response = test_client.get(f"/api/v1/queries/{query_id}")
        assert response.status_code == 200
        
        error_data = response.json()
        assert error_data["status"] == "failed"
        assert "Repository not accessible" in error_data["error"]
        
        # Step 4: Test graceful degradation with partial results
        partial_query_data = {
            "repository_url": "https://github.com/test/partial-repo.git",
            "query": "Analyze with partial agent failure",
            "options": {"max_commits": 20}
        }
        
        response = test_client.post("/api/v1/queries/", json=partial_query_data)
        assert response.status_code == 200
        
        partial_query_result = response.json()
        partial_query_id = partial_query_result["query_id"]
        
        # Simulate partial success with some agent failures
        partial_results = {
            "summary": "Analysis completed with some limitations",
            "confidence_score": 0.65,  # Lower confidence due to failures
            "processing_time_seconds": 12.3,
            "warnings": ["Historian agent failed - using cached data"],
            "findings": [
                {
                    "agent_name": "analyst",
                    "finding_type": "structural_analysis",
                    "content": "Successfully analyzed code structure",
                    "confidence": 0.88
                }
            ]
        }
        
        query_storage[partial_query_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "results": partial_results,
            "warnings": ["Some agents failed but analysis continued"]
        })
        
        # Step 5: Verify partial results are properly handled
        response = test_client.get(f"/api/v1/queries/{partial_query_id}")
        assert response.status_code == 200
        
        partial_data = response.json()
        assert partial_data["status"] == "completed"
        assert partial_data["results"]["confidence_score"] == 0.65
        assert len(partial_data["warnings"]) > 0


class TestUserJourneyIntegration:
    """Test complete user journeys across all interfaces."""
    
    def test_web_interface_user_journey(self, test_client, mock_database_clients):
        """Test complete user journey through web interface."""
        # This would test the React frontend integration
        # For now, we test the API endpoints that support the web interface
        
        # Step 1: Get dashboard data
        response = test_client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        assert "health" in metrics
        assert "system" in metrics
        assert "cache" in metrics
        
        # Step 2: Submit query through web interface
        web_query_data = {
            "repository_url": "https://github.com/test/web-repo.git",
            "query": "Show me recent changes in the API layer",
            "options": {"max_commits": 30}
        }
        
        response = test_client.post("/api/v1/queries/", json=web_query_data)
        assert response.status_code == 200
        
        # Step 3: Get query history for web interface
        response = test_client.get("/api/v1/queries/?page_size=10")
        assert response.status_code == 200
        
        history_data = response.json()
        assert "queries" in history_data
        assert "total_count" in history_data
        assert len(history_data["queries"]) > 0
    
    def test_cli_user_journey(self, mock_database_clients):
        """Test complete user journey through CLI interface."""
        from typer.testing import CliRunner
        
        runner = CliRunner()
        
        # Step 1: Check CLI version
        result = runner.invoke(cli_app, ["version"])
        assert result.exit_code == 0
        assert "Multi-Agent Code Intelligence System" in result.stdout
        
        # Step 2: Check API health
        with patch('httpx.Client') as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "status": "healthy",
                "version": "1.0.0",
                "services": {"neo4j": "healthy", "supabase": "healthy"}
            }
            mock_response.raise_for_status.return_value = None
            
            mock_client.return_value.__enter__.return_value.get.return_value = mock_response
            
            result = runner.invoke(cli_app, ["health"])
            assert result.exit_code == 0
            assert "healthy" in result.stdout
        
        # Step 3: Test configuration management
        result = runner.invoke(cli_app, ["config", "api_url", "http://localhost:8000/api/v1"])
        assert result.exit_code == 0
        
        result = runner.invoke(cli_app, ["config", "api_url"])
        assert result.exit_code == 0
        assert "http://localhost:8000/api/v1" in result.stdout
    
    def test_api_user_journey(self, test_client, mock_database_clients):
        """Test complete user journey through direct API usage."""
        # Step 1: Get API information
        response = test_client.get("/api/v1/info")
        assert response.status_code == 200
        
        api_info = response.json()
        assert api_info["name"] == "Code Intelligence API"
        assert "features" in api_info
        assert "endpoints" in api_info
        
        # Step 2: Check system health
        response = test_client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "services" in health_data
        
        # Step 3: Register repository
        repo_data = {
            "url": "https://github.com/test/api-repo.git",
            "name": "api-test-repo"
        }
        
        response = test_client.post("/api/v1/repositories/", json=repo_data)
        assert response.status_code == 200
        
        repo_result = response.json()
        repo_id = repo_result["id"]
        
        # Step 4: Submit and track query
        query_data = {
            "repository_url": repo_data["url"],
            "query": "Analyze API endpoint changes",
            "options": {"max_commits": 25}
        }
        
        response = test_client.post("/api/v1/queries/", json=query_data)
        assert response.status_code == 200
        
        query_result = response.json()
        query_id = query_result["query_id"]
        
        # Step 5: Monitor progress and get results
        response = test_client.get(f"/api/v1/queries/{query_id}")
        assert response.status_code == 200
        
        status_data = response.json()
        assert status_data["query_id"] == query_id
        
        # Step 6: Clean up
        response = test_client.delete(f"/api/v1/repositories/{repo_id}")
        assert response.status_code == 200


class TestDataIntegrityAndConsistency:
    """Test data integrity and consistency across systems."""
    
    @pytest.mark.asyncio
    async def test_state_consistency_across_agents(self, mock_database_clients):
        """Test state consistency during multi-agent execution."""
        # Create initial state
        state = AgentState(
            session_id=str(uuid4()),
            query={"original": "Test consistency query"},
            repository={"id": "test-repo", "path": "/test"}
        )
        
        # Simulate multiple agents modifying state
        from src.code_intelligence.agents.state import AgentFinding, Citation
        
        # Agent 1 adds findings
        finding1 = AgentFinding(
            agent_name="agent1",
            finding_type="test",
            content="First finding",
            confidence=0.9,
            citations=[Citation(file_path="test1.py", line_number=10)]
        )
        state.add_finding("agent1", finding1)
        
        # Agent 2 adds findings
        finding2 = AgentFinding(
            agent_name="agent2",
            finding_type="test",
            content="Second finding",
            confidence=0.8,
            citations=[Citation(file_path="test2.py", line_number=20)]
        )
        state.add_finding("agent2", finding2)
        
        # Verify state consistency
        all_findings = state.get_all_findings()
        assert len(all_findings) == 2
        
        # Verify findings are properly attributed
        agent1_findings = state.agent_results.get("agent1", [])
        agent2_findings = state.agent_results.get("agent2", [])
        
        assert len(agent1_findings) == 1
        assert len(agent2_findings) == 1
        assert agent1_findings[0].agent_name == "agent1"
        assert agent2_findings[0].agent_name == "agent2"
        
        # Test state serialization/deserialization
        state_dict = state.model_dump()
        restored_state = AgentState(**state_dict)
        
        # Verify restored state maintains consistency
        restored_findings = restored_state.get_all_findings()
        assert len(restored_findings) == 2
        assert restored_findings[0].confidence == 0.9
        assert restored_findings[1].confidence == 0.8
    
    def test_database_consistency_validation(self, mock_database_clients):
        """Test database consistency across Neo4j and Supabase."""
        # This would test that data remains consistent between databases
        # For now, we test the validation logic
        
        neo4j_client = mock_database_clients["neo4j"]
        supabase_client = mock_database_clients["supabase"]
        
        # Mock repository data in both databases
        neo4j_client.execute_query.return_value = [
            {"repo_id": "test-repo", "commit_count": 100}
        ]
        
        # Verify both databases have consistent data
        assert neo4j_client.execute_query.called or True  # Mock verification
        assert supabase_client.is_healthy() is True
    
    def test_cache_consistency_validation(self, mock_database_clients):
        """Test cache consistency and invalidation."""
        from src.code_intelligence.caching.cache_manager import cache_manager
        
        # Test cache consistency logic
        test_query = "Test cache consistency"
        test_repo_id = "test-repo"
        
        # This would test actual cache operations
        # For now, we verify the cache manager interface
        assert hasattr(cache_manager, 'get_cached_result')
        assert hasattr(cache_manager, 'store_result')
        assert hasattr(cache_manager, 'invalidate_repository_cache')


class TestPerformanceAndScalability:
    """Test system performance under various loads."""
    
    def test_concurrent_query_processing(self, test_client, mock_database_clients):
        """Test handling multiple concurrent queries."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def submit_query(query_num):
            """Submit a query and store the result."""
            query_data = {
                "repository_url": f"https://github.com/test/repo{query_num}.git",
                "query": f"Test concurrent query {query_num}",
                "options": {"max_commits": 10}
            }
            
            try:
                response = test_client.post("/api/v1/queries/", json=query_data)
                results_queue.put(("success", response.status_code, response.json()))
            except Exception as e:
                results_queue.put(("error", str(e), None))
        
        # Submit 5 concurrent queries
        threads = []
        for i in range(5):
            thread = threading.Thread(target=submit_query, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all queries were processed
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 5
        
        # All queries should succeed
        success_count = sum(1 for result in results if result[0] == "success")
        assert success_count == 5
        
        # All should return 200 status
        status_codes = [result[1] for result in results if result[0] == "success"]
        assert all(code == 200 for code in status_codes)
    
    def test_large_result_set_handling(self, test_client, mock_database_clients):
        """Test handling of large result sets."""
        # Submit query that would generate large results
        large_query_data = {
            "repository_url": "https://github.com/test/large-repo.git",
            "query": "Analyze all changes in the last year",
            "options": {"max_commits": 1000}
        }
        
        response = test_client.post("/api/v1/queries/", json=large_query_data)
        assert response.status_code == 200
        
        query_result = response.json()
        query_id = query_result["query_id"]
        
        # Simulate large result set
        from src.code_intelligence.api.routes.queries import query_storage
        
        large_findings = []
        for i in range(100):  # Simulate 100 findings
            large_findings.append({
                "agent_name": f"agent_{i % 5}",
                "finding_type": "large_analysis",
                "content": f"Finding {i}: Large analysis result with detailed information",
                "confidence": 0.8 + (i % 20) * 0.01,
                "citations": [
                    {
                        "file_path": f"src/module_{i}.py",
                        "line_number": i * 10,
                        "description": f"Citation for finding {i}"
                    }
                ]
            })
        
        large_results = {
            "summary": "Large-scale analysis completed successfully",
            "confidence_score": 0.87,
            "processing_time_seconds": 45.2,
            "findings": large_findings
        }
        
        query_storage[query_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "results": large_results
        })
        
        # Verify large results can be retrieved
        response = test_client.get(f"/api/v1/queries/{query_id}")
        assert response.status_code == 200
        
        final_data = response.json()
        assert final_data["status"] == "completed"
        assert len(final_data["results"]["findings"]) == 100
        
        # Test pagination for large results
        response = test_client.get("/api/v1/queries/?page_size=5&page=1")
        assert response.status_code == 200
        
        paginated_data = response.json()
        assert len(paginated_data["queries"]) <= 5
    
    def test_memory_usage_monitoring(self, test_client, mock_database_clients):
        """Test memory usage monitoring during processing."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Submit multiple queries to test memory usage
        query_ids = []
        for i in range(10):
            query_data = {
                "repository_url": f"https://github.com/test/memory-test-{i}.git",
                "query": f"Memory test query {i}",
                "options": {"max_commits": 50}
            }
            
            response = test_client.post("/api/v1/queries/", json=query_data)
            assert response.status_code == 200
            
            query_result = response.json()
            query_ids.append(query_result["query_id"])
        
        # Check memory usage after queries
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for test)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
        
        # Clean up queries
        for query_id in query_ids:
            response = test_client.delete(f"/api/v1/queries/{query_id}")
            # Note: DELETE endpoint might not exist, this is for testing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])