"""Tests for API endpoints."""

import pytest
import json
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from src.code_intelligence.api.main import app
from src.code_intelligence.api.models import QueryStatus, RepositoryStatus


@pytest.fixture
def client():
    """Create test client with proper test isolation."""
    # Clear all global storage before each test to ensure isolation
    from src.code_intelligence.api.routes.queries import query_storage
    from src.code_intelligence.api.routes.repositories import repository_storage
    
    query_storage.clear()
    repository_storage.clear()
    
    test_client = TestClient(app)
    yield test_client
    
    # Clean up after test
    query_storage.clear()
    repository_storage.clear()


@pytest.fixture
def mock_supabase():
    """Mock Supabase client."""
    with patch('src.code_intelligence.api.dependencies.get_supabase_client') as mock:
        mock_client = AsyncMock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_neo4j():
    """Mock Neo4j client."""
    with patch('src.code_intelligence.api.dependencies.get_neo4j_client') as mock:
        mock_client = AsyncMock()
        mock.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_orchestrator():
    """Mock orchestrator agent."""
    with patch('src.code_intelligence.api.dependencies.get_orchestrator') as mock:
        mock_agent = Mock()
        mock.return_value = mock_agent
        yield mock_agent


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_basic_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "services" in data
    
    def test_liveness_check(self, client):
        """Test liveness check endpoint."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "alive"
        assert "timestamp" in data
    
    def test_detailed_health_check(self, client, mock_supabase, mock_neo4j):
        """Test detailed health check with service dependencies."""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "timestamp" in data


class TestQueryEndpoints:
    """Test query management endpoints."""
    
    def test_submit_query(self, client, mock_supabase, mock_orchestrator):
        """Test query submission."""
        query_data = {
            "repository_url": "https://github.com/test/repo.git",
            "query": "What changed in the authentication system?",
            "options": {
                "max_commits": 50,
                "include_tests": False
            }
        }
        
        response = client.post("/api/v1/queries/", json=query_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "query_id" in data
        assert data["status"] == QueryStatus.PENDING
        assert data["message"] == "Query submitted successfully"
    
    def test_submit_query_missing_fields(self, client):
        """Test query submission with missing required fields."""
        query_data = {
            "query": "What changed?"
            # Missing repository_url
        }
        
        response = client.post("/api/v1/queries/", json=query_data)
        assert response.status_code == 422  # Validation error
    
    def test_get_query_status_not_found(self, client):
        """Test getting status of non-existent query."""
        response = client.get("/api/v1/queries/nonexistent-id")
        assert response.status_code == 404
    
    def test_get_query_history_empty(self, client):
        """Test getting query history when empty."""
        response = client.get("/api/v1/queries/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["queries"] == []
        assert data["total_count"] == 0
        assert data["page"] == 1
        assert data["page_size"] == 20
    
    def test_cancel_query_not_found(self, client):
        """Test cancelling non-existent query."""
        response = client.delete("/api/v1/queries/nonexistent-id")
        assert response.status_code == 404
    
    def test_export_query_not_found(self, client):
        """Test exporting non-existent query."""
        export_data = {
            "query_id": "nonexistent-id",
            "format": "json",
            "include_citations": True
        }
        
        response = client.post("/api/v1/queries/nonexistent-id/export", json=export_data)
        assert response.status_code == 404


class TestRepositoryEndpoints:
    """Test repository management endpoints."""
    
    def test_register_repository(self, client, mock_supabase):
        """Test repository registration."""
        repo_data = {
            "url": "https://github.com/test/repo.git",
            "name": "test-repo",
            "auto_sync": True
        }
        
        response = client.post("/api/v1/repositories/", json=repo_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert data["name"] == "test-repo"
        assert data["url"] == repo_data["url"]
        assert data["status"] == RepositoryStatus.NOT_ANALYZED
    
    def test_register_repository_missing_url(self, client):
        """Test repository registration with missing URL."""
        repo_data = {
            "name": "test-repo"
            # Missing url
        }
        
        response = client.post("/api/v1/repositories/", json=repo_data)
        assert response.status_code == 422  # Validation error
    
    def test_list_repositories_empty(self, client):
        """Test listing repositories when empty."""
        response = client.get("/api/v1/repositories/")
        assert response.status_code == 200
        
        data = response.json()
        assert data == []
    
    def test_get_repository_not_found(self, client):
        """Test getting non-existent repository."""
        response = client.get("/api/v1/repositories/nonexistent-id")
        assert response.status_code == 404
    
    def test_delete_repository_not_found(self, client):
        """Test deleting non-existent repository."""
        response = client.delete("/api/v1/repositories/nonexistent-id")
        assert response.status_code == 404
    
    def test_trigger_analysis_not_found(self, client):
        """Test triggering analysis for non-existent repository."""
        response = client.post("/api/v1/repositories/nonexistent-id/analyze")
        assert response.status_code == 404


class TestUserEndpoints:
    """Test user management endpoints."""
    
    def test_get_user_profile_unauthorized(self, client):
        """Test getting user profile without authentication."""
        response = client.get("/api/v1/users/me")
        assert response.status_code == 401
    
    def test_update_preferences_unauthorized(self, client):
        """Test updating preferences without authentication."""
        preferences = {
            "theme": "dark",
            "notifications_enabled": True
        }
        
        response = client.put("/api/v1/users/me/preferences", json=preferences)
        assert response.status_code == 401
    
    def test_get_user_stats_unauthorized(self, client):
        """Test getting user stats without authentication."""
        response = client.get("/api/v1/users/me/stats")
        assert response.status_code == 401


class TestAPIInfo:
    """Test API information endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Code Intelligence API"
        assert data["version"] == "1.0.0"
        assert "docs_url" in data
    
    def test_api_info_endpoint(self, client):
        """Test API info endpoint."""
        response = client.get("/api/v1/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Code Intelligence API"
        assert data["version"] == "1.0.0"
        assert "features" in data
        assert "endpoints" in data


class TestErrorHandling:
    """Test error handling."""
    
    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed error."""
        response = client.patch("/api/v1/health/")
        assert response.status_code == 405
    
    def test_validation_error(self, client):
        """Test validation error handling."""
        # Submit invalid JSON
        response = client.post(
            "/api/v1/queries/",
            json={"invalid": "data"}
        )
        assert response.status_code == 422


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/v1/health/")
        assert response.status_code == 200
        
        # Check for CORS headers
        headers = response.headers
        assert "access-control-allow-origin" in headers
        assert "access-control-allow-methods" in headers


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test async endpoint behavior."""
    
    async def test_concurrent_requests(self, client):
        """Test handling concurrent requests."""
        import asyncio
        import httpx
        
        async with httpx.AsyncClient(app=app, base_url="http://test") as async_client:
            # Make multiple concurrent health check requests
            tasks = [
                async_client.get("/api/v1/health/")
                for _ in range(10)
            ]
            
            responses = await asyncio.gather(*tasks)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200