"""Tests for web interface components (integration tests)."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from src.code_intelligence.api.main import app


@pytest.fixture
def client():
    """Create test client for web interface testing."""
    return TestClient(app)


class TestWebInterfaceIntegration:
    """Test web interface integration with API."""
    
    def test_serve_static_files(self, client):
        """Test serving static web interface files."""
        # In a real implementation, FastAPI would serve static files
        # For now, we test that the API endpoints work for the web interface
        
        # Test API info endpoint that web interface uses
        response = client.get("/api/v1/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "endpoints" in data
        assert data["endpoints"]["queries"] == "/api/v1/queries"
        assert data["endpoints"]["repositories"] == "/api/v1/repositories"
        assert data["endpoints"]["websocket"] == "/ws"
    
    def test_cors_headers_for_web_interface(self, client):
        """Test CORS headers are properly set for web interface."""
        response = client.options("/api/v1/queries/")
        assert response.status_code == 200
        
        headers = response.headers
        assert "access-control-allow-origin" in headers
        assert "access-control-allow-methods" in headers
        assert "access-control-allow-headers" in headers
    
    def test_web_interface_query_workflow(self, client):
        """Test complete query workflow that web interface would use."""
        # 1. Submit query (as web interface would)
        query_data = {
            "repository_url": "https://github.com/test/repo.git",
            "query": "What changed in the authentication system?",
            "options": {"max_commits": 100}
        }
        
        response = client.post("/api/v1/queries/", json=query_data)
        assert response.status_code == 200
        
        query_result = response.json()
        query_id = query_result["query_id"]
        
        # 2. Check query status (as web interface would poll)
        response = client.get(f"/api/v1/queries/{query_id}")
        assert response.status_code == 200
        
        status_data = response.json()
        assert status_data["query_id"] == query_id
        assert "status" in status_data
    
    def test_web_interface_repository_workflow(self, client):
        """Test repository management workflow for web interface."""
        # 1. List repositories (initially empty)
        response = client.get("/api/v1/repositories/")
        assert response.status_code == 200
        assert response.json() == []
        
        # 2. Add repository
        repo_data = {
            "url": "https://github.com/test/repo.git",
            "name": "test-repo",
            "auto_sync": True
        }
        
        response = client.post("/api/v1/repositories/", json=repo_data)
        assert response.status_code == 200
        
        repo_result = response.json()
        repo_id = repo_result["id"]
        
        # 3. Get repository details
        response = client.get(f"/api/v1/repositories/{repo_id}")
        assert response.status_code == 200
        
        repo_details = response.json()
        assert repo_details["name"] == "test-repo"
        assert repo_details["url"] == repo_data["url"]
    
    def test_websocket_connection_for_web_interface(self, client):
        """Test WebSocket connection that web interface would use."""
        with client.websocket_connect("/ws") as websocket:
            # Should receive welcome message
            data = websocket.receive_json()
            assert data["type"] == "connection_established"
            assert "connection_id" in data["data"]
            
            # Test subscription (as web interface would do)
            websocket.send_json({
                "type": "subscribe_query",
                "data": {"query_id": "test-query-id"}
            })
            
            response = websocket.receive_json()
            assert response["type"] == "subscription_response"
            assert response["data"]["subscribed"] is True
    
    def test_error_handling_for_web_interface(self, client):
        """Test error handling that web interface needs to handle."""
        # Test 404 error
        response = client.get("/api/v1/queries/nonexistent-id")
        assert response.status_code == 404
        
        error_data = response.json()
        assert "error" in error_data
        assert "message" in error_data
        
        # Test validation error
        response = client.post("/api/v1/queries/", json={"invalid": "data"})
        assert response.status_code == 422
        
        error_data = response.json()
        assert "detail" in error_data  # FastAPI validation error format


class TestWebInterfaceDataFormats:
    """Test data formats expected by web interface."""
    
    def test_query_response_format(self, client):
        """Test query response format matches web interface expectations."""
        query_data = {
            "repository_url": "https://github.com/test/repo.git",
            "query": "Test query"
        }
        
        response = client.post("/api/v1/queries/", json=query_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields for web interface
        required_fields = ["query_id", "status", "message"]
        for field in required_fields:
            assert field in data
        
        # Check optional fields
        if "estimated_duration_seconds" in data:
            assert isinstance(data["estimated_duration_seconds"], (int, type(None)))
    
    def test_repository_response_format(self, client):
        """Test repository response format matches web interface expectations."""
        repo_data = {
            "url": "https://github.com/test/repo.git",
            "name": "test-repo"
        }
        
        response = client.post("/api/v1/repositories/", json=repo_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields for web interface
        required_fields = [
            "id", "name", "url", "status", "commit_count", 
            "supported_languages", "file_count", "lines_of_code",
            "created_at", "updated_at"
        ]
        for field in required_fields:
            assert field in data
        
        # Check data types
        assert isinstance(data["commit_count"], int)
        assert isinstance(data["supported_languages"], list)
        assert isinstance(data["file_count"], int)
        assert isinstance(data["lines_of_code"], int)
    
    def test_websocket_message_format(self, client):
        """Test WebSocket message format matches web interface expectations."""
        with client.websocket_connect("/ws") as websocket:
            # Receive welcome message
            data = websocket.receive_json()
            
            # Check message structure
            required_fields = ["type", "data", "timestamp"]
            for field in required_fields:
                assert field in data
            
            # Check welcome message specific fields
            assert data["type"] == "connection_established"
            assert "connection_id" in data["data"]
            assert "available_message_types" in data["data"]
    
    def test_health_check_format(self, client):
        """Test health check format for web interface status display."""
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
        
        data = response.json()
        
        required_fields = ["status", "version", "services"]
        for field in required_fields:
            assert field in data
        
        assert isinstance(data["services"], dict)


class TestWebInterfacePagination:
    """Test pagination support for web interface."""
    
    def test_query_history_pagination(self, client):
        """Test query history pagination parameters."""
        # Test with pagination parameters
        response = client.get("/api/v1/queries/?page=1&page_size=10")
        assert response.status_code == 200
        
        data = response.json()
        
        # Check pagination structure
        required_fields = ["queries", "total_count", "page", "page_size"]
        for field in required_fields:
            assert field in data
        
        assert data["page"] == 1
        assert data["page_size"] == 10
        assert isinstance(data["queries"], list)
        assert isinstance(data["total_count"], int)
    
    def test_repository_list_pagination(self, client):
        """Test repository list pagination parameters."""
        response = client.get("/api/v1/repositories/?page=1&page_size=5")
        assert response.status_code == 200
        
        # Should return list (current implementation)
        data = response.json()
        assert isinstance(data, list)


class TestWebInterfaceRealTimeUpdates:
    """Test real-time update functionality for web interface."""
    
    def test_websocket_query_progress_updates(self, client):
        """Test WebSocket query progress updates for web interface."""
        with client.websocket_connect("/ws") as websocket:
            # Skip welcome message
            websocket.receive_json()
            
            # Subscribe to query updates
            websocket.send_json({
                "type": "subscribe_query",
                "data": {"query_id": "test-query-id"}
            })
            
            # Receive subscription confirmation
            response = websocket.receive_json()
            assert response["type"] == "subscription_response"
            
            # Test that web interface can handle different message types
            expected_message_types = [
                "query_progress",
                "partial_results", 
                "query_completed",
                "query_failed"
            ]
            
            # These would be sent by the system during actual query processing
            # For now, just verify the subscription works
            assert response["data"]["subscribed"] is True
    
    def test_websocket_connection_recovery(self, client):
        """Test WebSocket connection recovery for web interface."""
        # Test multiple connections (simulating reconnection)
        for i in range(3):
            with client.websocket_connect("/ws") as websocket:
                data = websocket.receive_json()
                assert data["type"] == "connection_established"
                
                # Each connection should get a unique ID
                connection_id = data["data"]["connection_id"]
                assert connection_id is not None


class TestWebInterfaceAccessibility:
    """Test accessibility features for web interface."""
    
    def test_api_response_structure_for_screen_readers(self, client):
        """Test API responses have proper structure for accessibility."""
        # Test that error messages are descriptive
        response = client.get("/api/v1/queries/nonexistent-id")
        assert response.status_code == 404
        
        error_data = response.json()
        assert "message" in error_data
        assert len(error_data["message"]) > 0  # Non-empty error message
    
    def test_consistent_field_naming(self, client):
        """Test consistent field naming across API for web interface."""
        # All timestamp fields should follow same format
        query_data = {
            "repository_url": "https://github.com/test/repo.git",
            "query": "Test query"
        }
        
        response = client.post("/api/v1/queries/", json=query_data)
        data = response.json()
        query_id = data["query_id"]
        
        # Check query status response
        response = client.get(f"/api/v1/queries/{query_id}")
        status_data = response.json()
        
        # Timestamp fields should be consistently named
        timestamp_fields = ["created_at", "updated_at"]
        for field in timestamp_fields:
            if field in status_data:
                # Should be ISO format string
                assert isinstance(status_data[field], str)


class TestWebInterfacePerformance:
    """Test performance considerations for web interface."""
    
    def test_api_response_size_limits(self, client):
        """Test API responses are reasonably sized for web interface."""
        # Test that list endpoints don't return excessive data
        response = client.get("/api/v1/repositories/")
        assert response.status_code == 200
        
        # Response should be reasonable size (less than 1MB for empty list)
        content_length = len(response.content)
        assert content_length < 1024 * 1024  # 1MB limit
    
    def test_websocket_message_size(self, client):
        """Test WebSocket messages are reasonably sized."""
        with client.websocket_connect("/ws") as websocket:
            data = websocket.receive_json()
            
            # Welcome message should be reasonable size
            message_size = len(json.dumps(data))
            assert message_size < 10 * 1024  # 10KB limit for welcome message
    
    def test_concurrent_api_requests(self, client):
        """Test API can handle concurrent requests from web interface."""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get("/api/v1/health/")
            results.append(response.status_code)
        
        # Create multiple threads to simulate concurrent web interface requests
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5