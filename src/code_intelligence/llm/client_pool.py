"""LLM client connection pooling and management."""

import asyncio
from typing import Dict, Optional, Type
from threading import Lock

from ..logging import get_logger
from .client import LLMClient
from .azure_client import AzureOpenAIClient

logger = get_logger(__name__)


class LLMClientPool:
    """Thread-safe singleton pool for LLM clients."""
    
    _instance: Optional['LLMClientPool'] = None
    _lock = Lock()
    
    def __new__(cls) -> 'LLMClientPool':
        """Ensure singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the client pool."""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._clients: Dict[str, LLMClient] = {}
        self._client_locks: Dict[str, asyncio.Lock] = {}
        self._initialized = True
        logger.info("LLM client pool initialized")
    
    async def get_client(self, client_type: str = "azure_openai") -> LLMClient:
        """Get or create an LLM client instance."""
        if client_type not in self._client_locks:
            self._client_locks[client_type] = asyncio.Lock()
            
        async with self._client_locks[client_type]:
            if client_type not in self._clients:
                self._clients[client_type] = self._create_client(client_type)
                logger.info(f"Created new LLM client: {client_type}")
            
            return self._clients[client_type]
    
    def _create_client(self, client_type: str) -> LLMClient:
        """Create a new client instance."""
        client_factories = {
            "azure_openai": AzureOpenAIClient,
        }
        
        if client_type not in client_factories:
            raise ValueError(f"Unknown client type: {client_type}")
            
        return client_factories[client_type]()
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all clients."""
        results = {}
        
        for client_type, client in self._clients.items():
            try:
                results[client_type] = await client.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {client_type}: {str(e)}")
                results[client_type] = False
                
        return results
    
    async def cleanup(self) -> None:
        """Cleanup all client connections."""
        for client_type, client in self._clients.items():
            try:
                if hasattr(client, 'close'):
                    await client.close()
                logger.info(f"Cleaned up client: {client_type}")
            except Exception as e:
                logger.error(f"Failed to cleanup client {client_type}: {str(e)}")
        
        self._clients.clear()
        self._client_locks.clear()


# Global pool instance
llm_client_pool = LLMClientPool()