"""Connection pool management for database and LLM clients."""

import asyncio
from typing import Any, Dict, Type, TypeVar, Generic
from contextlib import asynccontextmanager

import logging
from .singleton import SingletonMeta

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConnectionPool(Generic[T]):
    """Generic connection pool for any client type."""
    
    def __init__(self, client_class: Type[T], max_connections: int = 10, **client_kwargs):
        """Initialize connection pool."""
        self.client_class = client_class
        self.client_kwargs = client_kwargs
        self.max_connections = max_connections
        self._pool: asyncio.Queue[T] = asyncio.Queue(maxsize=max_connections)
        self._created_connections = 0
        self._lock = asyncio.Lock()
        
    async def _create_connection(self) -> T:
        """Create a new connection instance."""
        try:
            client = self.client_class(**self.client_kwargs)
            # If client has async connect method, call it
            if hasattr(client, 'connect') and asyncio.iscoroutinefunction(client.connect):
                await client.connect()
            logger.info(f"Created new {self.client_class.__name__} connection")
            return client
        except Exception as e:
            logger.error(f"Failed to create {self.client_class.__name__} connection: {str(e)}")
            # Decrement counter on failure to prevent resource leak
            async with self._lock:
                self._created_connections -= 1
            raise
            
    async def _is_connection_healthy(self, connection: T) -> bool:
        """Check if a connection is still healthy."""
        try:
            # Check if connection has a health check method
            if hasattr(connection, 'health_check'):
                if asyncio.iscoroutinefunction(connection.health_check):
                    return await connection.health_check()
                else:
                    return connection.health_check()
            # If no health check method, assume healthy
            return True
        except Exception:
            return False

    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool."""
        connection = None
        try:
            # Try to get existing connection
            try:
                connection = self._pool.get_nowait()
                # Validate connection health
                if not await self._is_connection_healthy(connection):
                    logger.warning(f"Unhealthy {self.client_class.__name__} connection detected, creating new one")
                    # Close unhealthy connection
                    if hasattr(connection, 'close'):
                        if asyncio.iscoroutinefunction(connection.close):
                            await connection.close()
                        else:
                            connection.close()
                    async with self._lock:
                        self._created_connections -= 1
                    connection = None  # Force creation of new connection
            except asyncio.QueueEmpty:
                connection = None
                
            if connection is None:
                # Create new connection if under limit
                should_create = False
                async with self._lock:
                    if self._created_connections < self.max_connections:
                        self._created_connections += 1  # Reserve slot before creating
                        should_create = True
                
                if should_create:
                    try:
                        connection = await self._create_connection()
                    except Exception:
                        # Connection creation failed, slot already decremented in _create_connection
                        raise
                else:
                    # Wait for available connection
                    connection = await self._pool.get()
                        
            yield connection
            
        except Exception as e:
            logger.error(f"Connection pool error: {str(e)}")
            raise
        finally:
            # Return connection to pool
            if connection:
                try:
                    self._pool.put_nowait(connection)
                except asyncio.QueueFull:
                    # Pool is full, close connection
                    if hasattr(connection, 'close'):
                        if asyncio.iscoroutinefunction(connection.close):
                            await connection.close()
                        else:
                            connection.close()
                    async with self._lock:
                        self._created_connections -= 1
                        
    async def close_all(self) -> None:
        """Close all connections in the pool."""
        connections_closed = 0
        while self._pool.qsize() > 0:
            try:
                connection = self._pool.get_nowait()
                if hasattr(connection, 'close'):
                    if asyncio.iscoroutinefunction(connection.close):
                        await connection.close()
                    else:
                        connection.close()
                connections_closed += 1
            except asyncio.QueueEmpty:
                break
        self._created_connections = 0
        logger.info(f"Closed {connections_closed} {self.client_class.__name__} connections")


class ConnectionPoolManager(metaclass=SingletonMeta):
    """Singleton manager for all connection pools."""
    
    def __init__(self):
        """Initialize connection pool manager."""
        self._pools: Dict[str, ConnectionPool] = {}
        
    def get_pool(self, name: str, client_class: Type[T], max_connections: int = 10, **client_kwargs) -> ConnectionPool[T]:
        """Get or create a connection pool."""
        if name not in self._pools:
            self._pools[name] = ConnectionPool(
                client_class=client_class,
                max_connections=max_connections,
                **client_kwargs
            )
            logger.info(f"Created connection pool '{name}' for {client_class.__name__}")
        return self._pools[name]
        
    async def close_all_pools(self) -> None:
        """Close all connection pools."""
        for name, pool in self._pools.items():
            await pool.close_all()
            logger.info(f"Closed connection pool '{name}'")
        self._pools.clear()
        
    def get_pool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all pools."""
        stats = {}
        for name, pool in self._pools.items():
            stats[name] = {
                "max_connections": pool.max_connections,
                "created_connections": pool._created_connections,
                "available_connections": pool._pool.qsize(),
                "client_class": pool.client_class.__name__
            }
        return stats