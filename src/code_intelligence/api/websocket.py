"""WebSocket implementation for real-time updates."""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.routing import APIRouter
import asyncio

from .models import WebSocketMessage, QueryProgress
from .dependencies import get_current_user
from ..logging import get_logger


logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""
    
    def __init__(self):
        # Active connections: {connection_id: WebSocket}
        self.active_connections: Dict[str, WebSocket] = {}
        
        # User connections: {user_id: Set[connection_id]}
        self.user_connections: Dict[str, Set[str]] = {}
        
        # Query subscriptions: {query_id: Set[connection_id]}
        self.query_subscriptions: Dict[str, Set[str]] = {}
        
        # Connection metadata: {connection_id: metadata}
        self.connection_metadata: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, user_id: Optional[str] = None) -> str:
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        connection_id = str(uuid.uuid4())
        self.active_connections[connection_id] = websocket
        
        # Track user connections
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = set()
            self.user_connections[user_id].add(connection_id)
        
        # Store connection metadata
        self.connection_metadata[connection_id] = {
            "user_id": user_id,
            "connected_at": datetime.utcnow(),
            "subscriptions": set()
        }
        
        logger.info(f"WebSocket connected: {connection_id}", extra={
            "connection_id": connection_id,
            "user_id": user_id,
            "total_connections": len(self.active_connections)
        })
        
        return connection_id
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection."""
        if connection_id not in self.active_connections:
            return
        
        metadata = self.connection_metadata.get(connection_id, {})
        user_id = metadata.get("user_id")
        
        # Remove from active connections
        del self.active_connections[connection_id]
        
        # Remove from user connections
        if user_id and user_id in self.user_connections:
            self.user_connections[user_id].discard(connection_id)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        # Remove from query subscriptions
        for query_id, subscribers in self.query_subscriptions.items():
            subscribers.discard(connection_id)
        
        # Clean up empty subscriptions
        self.query_subscriptions = {
            query_id: subscribers 
            for query_id, subscribers in self.query_subscriptions.items()
            if subscribers
        }
        
        # Remove metadata
        if connection_id in self.connection_metadata:
            del self.connection_metadata[connection_id]
        
        logger.info(f"WebSocket disconnected: {connection_id}", extra={
            "connection_id": connection_id,
            "user_id": user_id,
            "total_connections": len(self.active_connections)
        })
    
    async def send_personal_message(self, message: dict, connection_id: str):
        """Send a message to a specific connection."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to {connection_id}: {str(e)}")
                self.disconnect(connection_id)
    
    async def send_to_user(self, message: dict, user_id: str):
        """Send a message to all connections for a specific user."""
        if user_id in self.user_connections:
            connection_ids = list(self.user_connections[user_id])
            for connection_id in connection_ids:
                await self.send_personal_message(message, connection_id)
    
    async def broadcast_to_query_subscribers(self, message: dict, query_id: str):
        """Broadcast a message to all subscribers of a query."""
        if query_id in self.query_subscriptions:
            connection_ids = list(self.query_subscriptions[query_id])
            for connection_id in connection_ids:
                await self.send_personal_message(message, connection_id)
    
    async def broadcast_to_all(self, message: dict):
        """Broadcast a message to all active connections."""
        connection_ids = list(self.active_connections.keys())
        for connection_id in connection_ids:
            await self.send_personal_message(message, connection_id)
    
    def subscribe_to_query(self, connection_id: str, query_id: str):
        """Subscribe a connection to query updates."""
        if connection_id not in self.active_connections:
            return False
        
        if query_id not in self.query_subscriptions:
            self.query_subscriptions[query_id] = set()
        
        self.query_subscriptions[query_id].add(connection_id)
        
        # Update connection metadata
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscriptions"].add(query_id)
        
        logger.info(f"Connection subscribed to query: {connection_id} -> {query_id}")
        return True
    
    def unsubscribe_from_query(self, connection_id: str, query_id: str):
        """Unsubscribe a connection from query updates."""
        if query_id in self.query_subscriptions:
            self.query_subscriptions[query_id].discard(connection_id)
            
            if not self.query_subscriptions[query_id]:
                del self.query_subscriptions[query_id]
        
        # Update connection metadata
        if connection_id in self.connection_metadata:
            self.connection_metadata[connection_id]["subscriptions"].discard(query_id)
        
        logger.info(f"Connection unsubscribed from query: {connection_id} -> {query_id}")
    
    def get_connection_stats(self) -> dict:
        """Get connection statistics."""
        return {
            "total_connections": len(self.active_connections),
            "unique_users": len(self.user_connections),
            "active_subscriptions": len(self.query_subscriptions),
            "total_subscriptions": sum(len(subs) for subs in self.query_subscriptions.values())
        }


# Global connection manager
manager = ConnectionManager()


# WebSocket router
router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time updates."""
    connection_id = await manager.connect(websocket)
    
    try:
        # Send welcome message
        welcome_message = WebSocketMessage(
            type="connection_established",
            data={
                "connection_id": connection_id,
                "server_time": datetime.utcnow().isoformat(),
                "available_message_types": [
                    "subscribe_query",
                    "unsubscribe_query", 
                    "query_progress",
                    "partial_results",
                    "query_completed",
                    "query_failed"
                ]
            }
        )
        await manager.send_personal_message(welcome_message.model_dump(), connection_id)
        
        # Message handling loop
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_websocket_message(connection_id, message)
                
            except json.JSONDecodeError:
                error_message = WebSocketMessage(
                    type="error",
                    data={"message": "Invalid JSON format"}
                )
                await manager.send_personal_message(error_message.model_dump(), connection_id)
                
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {str(e)}")
                error_message = WebSocketMessage(
                    type="error",
                    data={"message": f"Message handling error: {str(e)}"}
                )
                await manager.send_personal_message(error_message.model_dump(), connection_id)
    
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(connection_id)


async def handle_websocket_message(connection_id: str, message: dict):
    """Handle incoming WebSocket messages."""
    message_type = message.get("type")
    data = message.get("data", {})
    
    if message_type == "subscribe_query":
        query_id = data.get("query_id")
        if query_id:
            success = manager.subscribe_to_query(connection_id, query_id)
            response = WebSocketMessage(
                type="subscription_response",
                query_id=query_id,
                data={
                    "subscribed": success,
                    "query_id": query_id
                }
            )
            await manager.send_personal_message(response.model_dump(), connection_id)
    
    elif message_type == "unsubscribe_query":
        query_id = data.get("query_id")
        if query_id:
            manager.unsubscribe_from_query(connection_id, query_id)
            response = WebSocketMessage(
                type="unsubscription_response",
                query_id=query_id,
                data={
                    "unsubscribed": True,
                    "query_id": query_id
                }
            )
            await manager.send_personal_message(response.model_dump(), connection_id)
    
    elif message_type == "ping":
        pong_message = WebSocketMessage(
            type="pong",
            data={"server_time": datetime.utcnow().isoformat()}
        )
        await manager.send_personal_message(pong_message.model_dump(), connection_id)
    
    elif message_type == "get_stats":
        stats = manager.get_connection_stats()
        stats_message = WebSocketMessage(
            type="stats",
            data=stats
        )
        await manager.send_personal_message(stats_message.model_dump(), connection_id)
    
    else:
        error_message = WebSocketMessage(
            type="error",
            data={"message": f"Unknown message type: {message_type}"}
        )
        await manager.send_personal_message(error_message.model_dump(), connection_id)


# Utility functions for broadcasting updates
async def broadcast_query_progress(query_id: str, progress: QueryProgress):
    """Broadcast query progress to subscribers."""
    message = WebSocketMessage(
        type="query_progress",
        query_id=query_id,
        data=progress.model_dump()
    )
    await manager.broadcast_to_query_subscribers(message.model_dump(), query_id)


async def broadcast_partial_results(query_id: str, agent_name: str, findings: List[dict]):
    """Broadcast partial results to subscribers."""
    message = WebSocketMessage(
        type="partial_results",
        query_id=query_id,
        data={
            "agent": agent_name,
            "findings": findings,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    await manager.broadcast_to_query_subscribers(message.model_dump(), query_id)


async def broadcast_query_completed(query_id: str, results: dict):
    """Broadcast query completion to subscribers."""
    message = WebSocketMessage(
        type="query_completed",
        query_id=query_id,
        data=results
    )
    await manager.broadcast_to_query_subscribers(message.model_dump(), query_id)


async def broadcast_query_failed(query_id: str, error: str):
    """Broadcast query failure to subscribers."""
    message = WebSocketMessage(
        type="query_failed",
        query_id=query_id,
        data={
            "error": error,
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    await manager.broadcast_to_query_subscribers(message.model_dump(), query_id)


# Health check for WebSocket connections
@router.get("/ws/stats")
async def websocket_stats():
    """Get WebSocket connection statistics."""
    return manager.get_connection_stats()