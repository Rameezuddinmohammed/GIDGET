"""Tests for WebSocket functionality."""

import pytest
import json
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

from src.code_intelligence.api.main import app
from src.code_intelligence.api.websocket import ConnectionManager, manager


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def connection_manager():
    """Create fresh connection manager for testing."""
    return ConnectionManager()


class TestConnectionManager:
    """Test WebSocket connection manager."""
    
    def test_connection_manager_initialization(self, connection_manager):
        """Test connection manager initialization."""
        assert len(connection_manager.active_connections) == 0
        assert len(connection_manager.user_connections) == 0
        assert len(connection_manager.query_subscriptions) == 0
        assert len(connection_manager.connection_metadata) == 0
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self, connection_manager):
        """Test WebSocket connection."""
        # Mock WebSocket
        mock_websocket = AsyncMock()
        
        connection_id = await connection_manager.connect(mock_websocket, "user123")
        
        assert connection_id in connection_manager.active_connections
        assert connection_manager.active_connections[connection_id] == mock_websocket
        assert "user123" in connection_manager.user_connections
        assert connection_id in connection_manager.user_connections["user123"]
        assert connection_id in connection_manager.connection_metadata
        
        mock_websocket.accept.assert_called_once()
    
    def test_disconnect_websocket(self, connection_manager):
        """Test WebSocket disconnection."""
        # Setup connection
        mock_websocket = Mock()
        connection_id = "test-connection-id"
        user_id = "user123"
        
        connection_manager.active_connections[connection_id] = mock_websocket
        connection_manager.user_connections[user_id] = {connection_id}
        connection_manager.connection_metadata[connection_id] = {"user_id": user_id}
        
        # Disconnect
        connection_manager.disconnect(connection_id)
        
        assert connection_id not in connection_manager.active_connections
        assert user_id not in connection_manager.user_connections
        assert connection_id not in connection_manager.connection_metadata
    
    @pytest.mark.asyncio
    async def test_send_personal_message(self, connection_manager):
        """Test sending personal message."""
        # Setup connection
        mock_websocket = AsyncMock()
        connection_id = "test-connection-id"
        connection_manager.active_connections[connection_id] = mock_websocket
        
        message = {"type": "test", "data": {"message": "hello"}}
        
        await connection_manager.send_personal_message(message, connection_id)
        
        mock_websocket.send_text.assert_called_once_with(json.dumps(message))
    
    @pytest.mark.asyncio
    async def test_send_to_user(self, connection_manager):
        """Test sending message to all user connections."""
        # Setup multiple connections for user
        user_id = "user123"
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        connection_id1 = "conn1"
        connection_id2 = "conn2"
        
        connection_manager.active_connections[connection_id1] = mock_websocket1
        connection_manager.active_connections[connection_id2] = mock_websocket2
        connection_manager.user_connections[user_id] = {connection_id1, connection_id2}
        
        message = {"type": "test", "data": {"message": "hello"}}
        
        await connection_manager.send_to_user(message, user_id)
        
        mock_websocket1.send_text.assert_called_once_with(json.dumps(message))
        mock_websocket2.send_text.assert_called_once_with(json.dumps(message))
    
    def test_subscribe_to_query(self, connection_manager):
        """Test subscribing to query updates."""
        connection_id = "test-connection-id"
        query_id = "test-query-id"
        
        # Setup connection
        connection_manager.active_connections[connection_id] = Mock()
        connection_manager.connection_metadata[connection_id] = {"subscriptions": set()}
        
        success = connection_manager.subscribe_to_query(connection_id, query_id)
        
        assert success is True
        assert query_id in connection_manager.query_subscriptions
        assert connection_id in connection_manager.query_subscriptions[query_id]
        assert query_id in connection_manager.connection_metadata[connection_id]["subscriptions"]
    
    def test_unsubscribe_from_query(self, connection_manager):
        """Test unsubscribing from query updates."""
        connection_id = "test-connection-id"
        query_id = "test-query-id"
        
        # Setup subscription
        connection_manager.query_subscriptions[query_id] = {connection_id}
        connection_manager.connection_metadata[connection_id] = {"subscriptions": {query_id}}
        
        connection_manager.unsubscribe_from_query(connection_id, query_id)
        
        assert query_id not in connection_manager.query_subscriptions
        assert query_id not in connection_manager.connection_metadata[connection_id]["subscriptions"]
    
    @pytest.mark.asyncio
    async def test_broadcast_to_query_subscribers(self, connection_manager):
        """Test broadcasting to query subscribers."""
        query_id = "test-query-id"
        connection_id1 = "conn1"
        connection_id2 = "conn2"
        mock_websocket1 = AsyncMock()
        mock_websocket2 = AsyncMock()
        
        # Setup subscriptions
        connection_manager.active_connections[connection_id1] = mock_websocket1
        connection_manager.active_connections[connection_id2] = mock_websocket2
        connection_manager.query_subscriptions[query_id] = {connection_id1, connection_id2}
        
        message = {"type": "query_progress", "data": {"progress": 50}}
        
        await connection_manager.broadcast_to_query_subscribers(message, query_id)
        
        mock_websocket1.send_text.assert_called_once_with(json.dumps(message))
        mock_websocket2.send_text.assert_called_once_with(json.dumps(message))
    
    def test_get_connection_stats(self, connection_manager):
        """Test getting connection statistics."""
        # Setup some connections
        connection_manager.active_connections["conn1"] = Mock()
        connection_manager.active_connections["conn2"] = Mock()
        connection_manager.user_connections["user1"] = {"conn1"}
        connection_manager.user_connections["user2"] = {"conn2"}
        connection_manager.query_subscriptions["query1"] = {"conn1", "conn2"}
        
        stats = connection_manager.get_connection_stats()
        
        assert stats["total_connections"] == 2
        assert stats["unique_users"] == 2
        assert stats["active_subscriptions"] == 1
        assert stats["total_subscriptions"] == 2


class TestWebSocketEndpoint:
    """Test WebSocket endpoint functionality."""
    
    def test_websocket_connection(self, client):
        """Test WebSocket connection establishment."""
        with client.websocket_connect("/ws") as websocket:
            # Should receive welcome message
            data = websocket.receive_json()
            assert data["type"] == "connection_established"
            assert "connection_id" in data["data"]
            assert "server_time" in data["data"]
            assert "available_message_types" in data["data"]
    
    def test_websocket_ping_pong(self, client):
        """Test WebSocket ping/pong functionality."""
        with client.websocket_connect("/ws") as websocket:
            # Skip welcome message
            websocket.receive_json()
            
            # Send ping
            websocket.send_json({"type": "ping"})
            
            # Should receive pong
            data = websocket.receive_json()
            assert data["type"] == "pong"
            assert "server_time" in data["data"]
    
    def test_websocket_subscribe_query(self, client):
        """Test subscribing to query updates."""
        with client.websocket_connect("/ws") as websocket:
            # Skip welcome message
            websocket.receive_json()
            
            # Subscribe to query
            websocket.send_json({
                "type": "subscribe_query",
                "data": {"query_id": "test-query-id"}
            })
            
            # Should receive subscription confirmation
            data = websocket.receive_json()
            assert data["type"] == "subscription_response"
            assert data["query_id"] == "test-query-id"
            assert data["data"]["subscribed"] is True
    
    def test_websocket_unsubscribe_query(self, client):
        """Test unsubscribing from query updates."""
        with client.websocket_connect("/ws") as websocket:
            # Skip welcome message
            websocket.receive_json()
            
            # Subscribe first
            websocket.send_json({
                "type": "subscribe_query",
                "data": {"query_id": "test-query-id"}
            })
            websocket.receive_json()  # Skip subscription response
            
            # Unsubscribe
            websocket.send_json({
                "type": "unsubscribe_query",
                "data": {"query_id": "test-query-id"}
            })
            
            # Should receive unsubscription confirmation
            data = websocket.receive_json()
            assert data["type"] == "unsubscription_response"
            assert data["query_id"] == "test-query-id"
            assert data["data"]["unsubscribed"] is True
    
    def test_websocket_get_stats(self, client):
        """Test getting WebSocket statistics."""
        with client.websocket_connect("/ws") as websocket:
            # Skip welcome message
            websocket.receive_json()
            
            # Request stats
            websocket.send_json({"type": "get_stats"})
            
            # Should receive stats
            data = websocket.receive_json()
            assert data["type"] == "stats"
            assert "total_connections" in data["data"]
            assert "unique_users" in data["data"]
    
    def test_websocket_invalid_message(self, client):
        """Test handling invalid WebSocket messages."""
        with client.websocket_connect("/ws") as websocket:
            # Skip welcome message
            websocket.receive_json()
            
            # Send invalid JSON
            websocket.send_text("invalid json")
            
            # Should receive error message
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "Invalid JSON format" in data["data"]["message"]
    
    def test_websocket_unknown_message_type(self, client):
        """Test handling unknown message types."""
        with client.websocket_connect("/ws") as websocket:
            # Skip welcome message
            websocket.receive_json()
            
            # Send unknown message type
            websocket.send_json({"type": "unknown_type"})
            
            # Should receive error message
            data = websocket.receive_json()
            assert data["type"] == "error"
            assert "Unknown message type" in data["data"]["message"]


class TestWebSocketBroadcasting:
    """Test WebSocket broadcasting functions."""
    
    @pytest.mark.asyncio
    async def test_broadcast_query_progress(self):
        """Test broadcasting query progress."""
        from src.code_intelligence.api.websocket import broadcast_query_progress
        from src.code_intelligence.api.models import QueryProgress
        
        # Mock the manager
        with patch('src.code_intelligence.api.websocket.manager') as mock_manager:
            progress = QueryProgress(
                current_agent="analyst",
                completed_steps=["orchestrator"],
                total_steps=5,
                progress_percentage=40.0,
                current_step="analyzing code structure"
            )
            
            await broadcast_query_progress("test-query-id", progress)
            
            mock_manager.broadcast_to_query_subscribers.assert_called_once()
            call_args = mock_manager.broadcast_to_query_subscribers.call_args
            message, query_id = call_args[0]
            
            assert query_id == "test-query-id"
            assert message["type"] == "query_progress"
            assert message["query_id"] == "test-query-id"
    
    @pytest.mark.asyncio
    async def test_broadcast_partial_results(self):
        """Test broadcasting partial results."""
        from src.code_intelligence.api.websocket import broadcast_partial_results
        
        with patch('src.code_intelligence.api.websocket.manager') as mock_manager:
            findings = [{"content": "test finding", "confidence": 0.9}]
            
            await broadcast_partial_results("test-query-id", "analyst", findings)
            
            mock_manager.broadcast_to_query_subscribers.assert_called_once()
            call_args = mock_manager.broadcast_to_query_subscribers.call_args
            message, query_id = call_args[0]
            
            assert query_id == "test-query-id"
            assert message["type"] == "partial_results"
            assert message["data"]["agent"] == "analyst"
            assert message["data"]["findings"] == findings
    
    @pytest.mark.asyncio
    async def test_broadcast_query_completed(self):
        """Test broadcasting query completion."""
        from src.code_intelligence.api.websocket import broadcast_query_completed
        
        with patch('src.code_intelligence.api.websocket.manager') as mock_manager:
            results = {"summary": "Analysis complete", "confidence": 0.95}
            
            await broadcast_query_completed("test-query-id", results)
            
            mock_manager.broadcast_to_query_subscribers.assert_called_once()
            call_args = mock_manager.broadcast_to_query_subscribers.call_args
            message, query_id = call_args[0]
            
            assert query_id == "test-query-id"
            assert message["type"] == "query_completed"
            assert message["data"] == results
    
    @pytest.mark.asyncio
    async def test_broadcast_query_failed(self):
        """Test broadcasting query failure."""
        from src.code_intelligence.api.websocket import broadcast_query_failed
        
        with patch('src.code_intelligence.api.websocket.manager') as mock_manager:
            error = "Analysis failed due to repository access error"
            
            await broadcast_query_failed("test-query-id", error)
            
            mock_manager.broadcast_to_query_subscribers.assert_called_once()
            call_args = mock_manager.broadcast_to_query_subscribers.call_args
            message, query_id = call_args[0]
            
            assert query_id == "test-query-id"
            assert message["type"] == "query_failed"
            assert message["data"]["error"] == error


class TestWebSocketStats:
    """Test WebSocket statistics endpoint."""
    
    def test_websocket_stats_endpoint(self, client):
        """Test WebSocket stats endpoint."""
        response = client.get("/ws/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_connections" in data
        assert "unique_users" in data
        assert "active_subscriptions" in data
        assert "total_subscriptions" in data


@pytest.mark.asyncio
class TestWebSocketConcurrency:
    """Test WebSocket concurrency handling."""
    
    async def test_multiple_connections(self, client):
        """Test handling multiple WebSocket connections."""
        connections = []
        
        try:
            # Create multiple connections
            for i in range(5):
                websocket = client.websocket_connect("/ws")
                websocket.__enter__()
                connections.append(websocket)
                
                # Receive welcome message
                data = websocket.receive_json()
                assert data["type"] == "connection_established"
            
            # All connections should be active
            assert len(connections) == 5
            
        finally:
            # Clean up connections
            for websocket in connections:
                try:
                    websocket.__exit__(None, None, None)
                except:
                    pass
    
    async def test_concurrent_subscriptions(self, client):
        """Test concurrent query subscriptions."""
        with client.websocket_connect("/ws") as websocket:
            # Skip welcome message
            websocket.receive_json()
            
            # Subscribe to multiple queries concurrently
            query_ids = ["query1", "query2", "query3"]
            
            for query_id in query_ids:
                websocket.send_json({
                    "type": "subscribe_query",
                    "data": {"query_id": query_id}
                })
            
            # Should receive all subscription confirmations
            for _ in query_ids:
                data = websocket.receive_json()
                assert data["type"] == "subscription_response"
                assert data["data"]["subscribed"] is True