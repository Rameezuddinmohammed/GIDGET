"""Base LLM client interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any
    ) -> str:
        """Generate a chat completion."""
        pass
        
    @abstractmethod
    async def embedding(
        self,
        text: str,
        **kwargs: Any
    ) -> List[float]:
        """Generate text embedding."""
        pass
        
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM service is available."""
        pass