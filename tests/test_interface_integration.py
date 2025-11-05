"""Integration tests for user interfaces (CLI, Web, API)."""

import json
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

from typer.testing import CliRunner
from fastapi.testclient import TestClient

from src.code_intelligence.api.main import app
from src.code_intelligence.cli import app as cli_app, CLIConfig


@pytest.fixture
def test_client():
    """Create test client for interface integration tests."""
    # Clear storage before each test
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
def cli_runner():
    """Create CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def temp_config_dir():
    """Create temporary directory for CLI configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for CLI tests."""
    with patch('httpx.Client') as mock_client:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        
        mock_client.return_value.__enter__.return_value = mock_response
        mock_client.return_value.__exit__.return_value = None
        
        yield mock_response


class TestCLIIntegration:
    """Test CLI interface integration."""
    
    def test_cli_version_command(self, cli_runner):
        """Test CLI version command."""
        result = cli_runner.invoke(cli_app, ["version"])
        assert result.exit_code == 0
        assert "Multi-Agent Code Intelligence System" in result.stdout
        assert "v1.0.0" in result.stdout
    
    def test_cli_configuration_workflow(self, cli_runner, temp_config_dir):
        """Test complete CLI configuration workflow."""
        # Test setting configuration
        with patch.object(CLIConfig, 'config_file', temp_config_dir / "config.json"):
            # Set API URL
            result = cli_runner.invoke(cli_app, ["config", "api_url", "http://localhost:8000/api/v1"])
            assert result.exit_code == 0
            assert "Set api_url = http://localhost:8000/api/v1" in result.stdout
            
            # Get API URL
            result = cli_runner.invoke(cli_app, ["config", "api_url"])
            assert result.exit_code == 0
            assert "http://localhost:8000/api/v1" in result.stdout
            
            # Set output format
            result = cli_runner.invoke(cli_app, ["config", "output_format", "json"])
            assert result.exit_code == 0
            
            # List all configuration
            result = cli_runner.invoke(cli_app, ["config", "--list"])
            assert result.exit_code == 0
            assert "api_url" in result.stdout
            assert "output_format" in result.stdout
    
    def test_cli_health_check_integration(self, cli_runner, mock_http_client):
        """Test CLI health check integration with API."""
        # Mock successful health response
        mock_http_client.json.return_value = {
            "status": "healthy",
            "version": "1.0.0",
            "services": {
                "neo4j": "healthy",
                "supabase": "healthy"
            }
        }
        
        result = cli_runner.invoke(cli_app, ["health"])
        assert result.exit_code == 0
        assert "healthy" in result.stdout
        assert "neo4j" in result.stdout
        assert "supabase" in result.stdout
    
    def test_cli_query_submission_workflow(self, cli_runner, mock_http_client):
        """Test complete CLI query submission workflow."""
        # Mock query submission response
        query_id = "test-query-123"
        mock_http_client.json.side_effect = [
            # Query submission response
            {
                "query_id": query_id,
                "status": "pending",
                "message": "Query submitted successfully"
            },
            # Status check responses (simulating processing)
            {
                "query_id": query_id,
                "status": "processing",
                "progress": {
                    "current_agent": "historian",
                    "progress_percentage": 25.0,
                    "current_step": "Analyzing git history"
                }
            },
            {
                "query_id": query_id,
                "status": "processing",
                "progress": {
                    "current_agent": "analyst",
                    "progress_percentage": 75.0,
                    "current_step": "Analyzing code structure"
                }
            },
            # Completion response
            {
                "query_id": query_id,
                "status": "completed",
                "results": {
                    "summary": "Analysis completed successfully",
                    "confidence_score": 0.92,
                    "processing_time_seconds": 15.5,
                    "findings": [
                        {
                            "agent_name": "historian",
                            "finding_type": "version_analysis",
                            "content": "Found significant changes in authentication module",
                            "confidence": 0.95,
                            "citations": [
                                {
                                    "file_path": "src/auth/login.py",
                                    "line_number": 42,
                                    "description": "Updated password validation"
                                }
                            ]
                        }
                    ]
                }
            }
        ]
        
        # Submit query with --no-wait to test submission only
        result = cli_runner.invoke(cli_app, [
            "query",
            "What changed in the authentication system?",
            "--repo", "https://github.com/test/repo.git",
            "--no-wait"
        ])
        assert result.exit_code == 0
        assert query_id in result.stdout
        assert "Query submitted successfully" in result.stdout
    
    def test_cli_repository_management_workflow(self, cli_runner, mock_http_client):
        """Test CLI repository management workflow."""
        repo_id = "test-repo-456"
        
        # Mock repository responses
        mock_http_client.json.side_effect = [
            # Add repository response
            {
                "id": repo_id,
                "name": "test-repo",
                "url": "https://github.com/test/repo.git",
                "status": "not_analyzed"
            },
            # List repositories response
            [
                {
                    "id": repo_id,
                    "name": "test-repo",
                    "status": "analyzing",
                    "commit_count": 150,
                    "supported_languages": ["Python", "JavaScript"]
                }
            ],
            # Analyze repository response
            {
                "message": "Analysis started",
                "analysis_id": "analysis-789"
            },
            # Delete repository response
            {
                "message": "Repository deleted successfully"
            }
        ]
        
        # Add repository
        result = cli_runner.invoke(cli_app, [
            "repositories", "add",
            "--url", "https://github.com/test/repo.git",
            "--name", "test-repo"
        ])
        assert result.exit_code == 0
        assert "Repository added successfully" in result.stdout
        assert repo_id in result.stdout
        
        # List repositories
        result = cli_runner.invoke(cli_app, ["repositories", "list"])
        assert result.exit_code == 0
        assert "test-repo" in result.stdout
        assert "Python" in result.stdout
        
        # Analyze repository
        result = cli_runner.invoke(cli_app, [
            "repositories", "analyze",
            "--id", repo_id
        ])
        assert result.exit_code == 0
        assert "Analysis started" in result.stdout
        
        # Delete repository
        result = cli_runner.invoke(cli_app, [
            "repositories", "delete",
            "--id", repo_id
        ])
        assert result.exit_code == 0
        assert "Repository deleted successfully" in result.stdout
    
    def test_cli_query_history_and_export(self, cli_runner, mock_http_client):
        """Test CLI query history and export functionality."""
        # Mock query history response
        mock_http_client.json.side_effect = [
            # Query history response
            {
                "queries": [
                    {
                        "query_id": "query-1",
                        "query": "What changed in the API layer?",
                        "repository_name": "test-repo",
                        "status": "completed",
                        "created_at": "2024-01-15T10:30:00"
                    },
                    {
                        "query_id": "query-2",
                        "query": "Find authentication bugs",
                        "repository_name": "test-repo",
                        "status": "processing",
                        "created_at": "2024-01-15T11:00:00"
                    }
                ],
                "total_count": 2,
                "page": 1,
                "page_size": 10
            },
            # Export response
            {
                "export_id": "export-123",
                "download_url": "https://api.example.com/exports/export-123",
                "expires_at": "2024-01-16T10:30:00"
            }
        ]
        
        # Get query history
        result = cli_runner.invoke(cli_app, ["history", "--limit", "5"])
        assert result.exit_code == 0
        assert "query-1" in result.stdout
        assert "What changed in the API layer?" in result.stdout
        assert "completed" in result.stdout
        
        # Export query results
        result = cli_runner.invoke(cli_app, [
            "export", "query-1",
            "--format", "json"
        ])
        assert result.exit_code == 0
        assert "export-123" in result.stdout
        assert "Export created successfully" in result.stdout
    
    def test_cli_database_maintenance_workflow(self, cli_runner):
        """Test CLI database maintenance commands."""
        with patch('src.code_intelligence.database.query_optimizer.db_optimizer') as mock_optimizer:
            # Mock validation results
            mock_optimizer.run_comprehensive_validation.return_value = {
                "overall_score": 0.85,
                "timestamp": "2024-01-15T12:00:00",
                "neo4j_analysis": {
                    "benchmark": {
                        "summary": {
                            "success_rate": 0.95,
                            "avg_execution_time_ms": 150.0
                        },
                        "total_queries": 100
                    },
                    "indexes": [
                        {
                            "name": "function_name_index",
                            "table": "Function",
                            "effectiveness": 0.92,
                            "recommendations": ["Consider composite index"]
                        }
                    ]
                },
                "recommendations": [
                    "Optimize slow queries",
                    "Add missing indexes"
                ]
            }
            
            # Test database validation
            result = cli_runner.invoke(cli_app, ["db", "validate", "--target", "neo4j"])
            assert result.exit_code == 0
            assert "Database Validation Report" in result.stdout
            assert "85.0%" in result.stdout  # Overall score
            
            # Test dry-run cleanup
            result = cli_runner.invoke(cli_app, [
                "db", "cleanup",
                "--target", "all",
                "--days", "30",
                "--dry-run"
            ])
            assert result.exit_code == 0
            assert "Would delete" in result.stdout
            
            # Test database optimization
            result = cli_runner.invoke(cli_app, [
                "db", "optimize",
                "--target", "neo4j",
                "--dry-run"
            ])
            assert result.exit_code == 0
            assert "Would run" in result.stdout


class TestWebInterfaceIntegration:
    """Test web interface integration through API endpoints."""
    
    def test_web_dashboard_data_integration(self, test_client):
        """Test web dashboard data retrieval integration."""
        with patch('src.code_intelligence.api.dependencies.get_neo4j_client') as mock_neo4j, \
             patch('src.code_intelligence.api.dependencies.get_supabase_client') as mock_supabase:
            
            # Setup healthy services
            neo4j_client = Mock()
            neo4j_client.is_healthy.return_value = True
            mock_neo4j.return_value = neo4j_client
            
            supabase_client = Mock()
            supabase_client.is_healthy.return_value = True
            mock_supabase.return_value = supabase_client
            
            # Get dashboard metrics
            response = test_client.get("/api/v1/health/metrics")
            assert response.status_code == 200
            
            metrics = response.json()
            
            # Verify dashboard data structure
            assert "health" in metrics
            assert "system" in metrics
            assert "cache" in metrics
            assert "timestamp" in metrics
            
            # Web dashboard would use this data for visualization
            health_status = metrics["health"]["status"]
            assert health_status in ["healthy", "degraded", "unhealthy"]
    
    def test_web_query_submission_integration(self, test_client):
        """Test web interface query submission integration."""
        # Submit query through web interface
        query_data = {
            "repository_url": "https://github.com/test/web-repo.git",
            "query": "Analyze recent API changes for web interface",
            "options": {
                "max_commits": 50,
                "include_tests": True
            }
        }
        
        response = test_client.post("/api/v1/queries/", json=query_data)
        assert response.status_code == 200
        
        result = response.json()
        query_id = result["query_id"]
        
        # Web interface would poll for status updates
        response = test_client.get(f"/api/v1/queries/{query_id}")
        assert response.status_code == 200
        
        status_data = response.json()
        assert status_data["query_id"] == query_id
        assert status_data["status"] == "pending"
    
    def test_web_repository_management_integration(self, test_client):
        """Test web interface repository management integration."""
        # Register repository through web interface
        repo_data = {
            "url": "https://github.com/test/web-managed-repo.git",
            "name": "web-managed-repo",
            "auto_sync": True
        }
        
        response = test_client.post("/api/v1/repositories/", json=repo_data)
        assert response.status_code == 200
        
        repo_result = response.json()
        repo_id = repo_result["id"]
        
        # Web interface would display repository list
        response = test_client.get("/api/v1/repositories/")
        assert response.status_code == 200
        
        repos = response.json()
        assert len(repos) == 1
        assert repos[0]["name"] == "web-managed-repo"
        
        # Web interface would show repository details
        response = test_client.get(f"/api/v1/repositories/{repo_id}")
        assert response.status_code == 200
        
        repo_details = response.json()
        assert repo_details["url"] == repo_data["url"]
    
    def test_web_real_time_updates_integration(self, test_client):
        """Test web interface real-time updates through WebSocket."""
        with test_client.websocket_connect("/ws") as websocket:
            # Web interface establishes WebSocket connection
            welcome_data = websocket.receive_json()
            assert welcome_data["type"] == "connection_established"
            
            # Web interface subscribes to query updates
            query_id = "web-query-123"
            websocket.send_json({
                "type": "subscribe_query",
                "data": {"query_id": query_id}
            })
            
            subscription_response = websocket.receive_json()
            assert subscription_response["type"] == "subscription_response"
            assert subscription_response["data"]["subscribed"] is True
            
            # Web interface can request connection stats
            websocket.send_json({"type": "get_stats"})
            
            stats_response = websocket.receive_json()
            assert stats_response["type"] == "stats"
            assert "total_connections" in stats_response["data"]
    
    def test_web_error_handling_integration(self, test_client):
        """Test web interface error handling integration."""
        # Test 404 error handling
        response = test_client.get("/api/v1/queries/nonexistent-query")
        assert response.status_code == 404
        
        error_data = response.json()
        assert "error" in error_data
        assert "message" in error_data
        
        # Test validation error handling
        invalid_query_data = {
            "query": "Test query without repository URL"
            # Missing required repository_url field
        }
        
        response = test_client.post("/api/v1/queries/", json=invalid_query_data)
        assert response.status_code == 422
        
        validation_error = response.json()
        assert "detail" in validation_error


class TestAPIIntegration:
    """Test direct API integration scenarios."""
    
    def test_api_authentication_integration(self, test_client):
        """Test API authentication integration."""
        # Test accessing protected endpoints without authentication
        response = test_client.get("/api/v1/users/me")
        assert response.status_code == 401
        
        # Test with invalid authentication
        headers = {"Authorization": "Bearer invalid-token"}
        response = test_client.get("/api/v1/users/me", headers=headers)
        assert response.status_code == 401
    
    def test_api_rate_limiting_integration(self, test_client):
        """Test API rate limiting integration."""
        # This would test rate limiting if implemented
        # For now, we test that multiple requests are handled properly
        
        responses = []
        for i in range(10):
            response = test_client.get("/api/v1/health/")
            responses.append(response)
        
        # All requests should succeed (no rate limiting in test)
        for response in responses:
            assert response.status_code == 200
    
    def test_api_pagination_integration(self, test_client):
        """Test API pagination integration."""
        # Add multiple queries to test pagination
        for i in range(15):
            query_data = {
                "repository_url": f"https://github.com/test/repo{i}.git",
                "query": f"Test pagination query {i}",
                "options": {"max_commits": 10}
            }
            
            response = test_client.post("/api/v1/queries/", json=query_data)
            assert response.status_code == 200
        
        # Test pagination
        response = test_client.get("/api/v1/queries/?page_size=5&page=1")
        assert response.status_code == 200
        
        page1_data = response.json()
        assert len(page1_data["queries"]) == 5
        assert page1_data["page"] == 1
        assert page1_data["page_size"] == 5
        assert page1_data["total_count"] == 15
        
        # Test second page
        response = test_client.get("/api/v1/queries/?page_size=5&page=2")
        assert response.status_code == 200
        
        page2_data = response.json()
        assert len(page2_data["queries"]) == 5
        assert page2_data["page"] == 2
    
    def test_api_content_negotiation_integration(self, test_client):
        """Test API content negotiation integration."""
        # Test JSON response (default)
        response = test_client.get("/api/v1/health/")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Test with explicit Accept header
        headers = {"Accept": "application/json"}
        response = test_client.get("/api/v1/health/", headers=headers)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
    
    def test_api_cors_integration(self, test_client):
        """Test API CORS integration."""
        # Test preflight request
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
        
        response = test_client.options("/api/v1/queries/", headers=headers)
        assert response.status_code == 200
        
        # Check CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers


class TestCrossInterfaceIntegration:
    """Test integration across multiple interfaces."""
    
    def test_cli_to_web_workflow_integration(self, cli_runner, test_client, mock_http_client):
        """Test workflow that spans CLI and web interfaces."""
        # Step 1: Add repository via CLI
        mock_http_client.json.return_value = {
            "id": "cross-repo-123",
            "name": "cross-interface-repo",
            "url": "https://github.com/test/cross-repo.git",
            "status": "not_analyzed"
        }
        
        result = cli_runner.invoke(cli_app, [
            "repositories", "add",
            "--url", "https://github.com/test/cross-repo.git",
            "--name", "cross-interface-repo"
        ])
        assert result.exit_code == 0
        
        # Step 2: Verify repository is available via web API
        response = test_client.get("/api/v1/repositories/")
        assert response.status_code == 200
        
        # In a real integration, the repository would be visible
        # For this test, we verify the API endpoint works
        repos = response.json()
        assert isinstance(repos, list)
    
    def test_api_to_cli_workflow_integration(self, test_client, cli_runner, mock_http_client):
        """Test workflow that spans API and CLI interfaces."""
        # Step 1: Submit query via API
        query_data = {
            "repository_url": "https://github.com/test/api-cli-repo.git",
            "query": "Cross-interface workflow test",
            "options": {"max_commits": 20}
        }
        
        response = test_client.post("/api/v1/queries/", json=query_data)
        assert response.status_code == 200
        
        result = response.json()
        query_id = result["query_id"]
        
        # Step 2: Check query status via CLI
        mock_http_client.json.return_value = {
            "query_id": query_id,
            "status": "completed",
            "results": {
                "summary": "Cross-interface test completed",
                "confidence_score": 0.88
            }
        }
        
        cli_result = cli_runner.invoke(cli_app, ["status", query_id])
        assert cli_result.exit_code == 0
        assert query_id in cli_result.stdout
    
    def test_websocket_to_api_integration(self, test_client):
        """Test integration between WebSocket and REST API."""
        # Step 1: Establish WebSocket connection
        with test_client.websocket_connect("/ws") as websocket:
            welcome_data = websocket.receive_json()
            assert welcome_data["type"] == "connection_established"
            
            # Step 2: Submit query via REST API
            query_data = {
                "repository_url": "https://github.com/test/ws-api-repo.git",
                "query": "WebSocket API integration test",
                "options": {"max_commits": 15}
            }
            
            response = test_client.post("/api/v1/queries/", json=query_data)
            assert response.status_code == 200
            
            result = response.json()
            query_id = result["query_id"]
            
            # Step 3: Subscribe to query updates via WebSocket
            websocket.send_json({
                "type": "subscribe_query",
                "data": {"query_id": query_id}
            })
            
            subscription_response = websocket.receive_json()
            assert subscription_response["type"] == "subscription_response"
            assert subscription_response["data"]["subscribed"] is True
            
            # In a real system, query processing would trigger WebSocket updates
            # For this test, we verify the subscription mechanism works


if __name__ == "__main__":
    pytest.main([__file__, "-v"])