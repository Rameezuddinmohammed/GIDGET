"""Azure OpenAI client implementation."""

from typing import Any, Dict, List

from openai import AsyncAzureOpenAI
from openai.types.chat import ChatCompletion

from ..config import config
from ..logging import get_logger
from ..core.connection_pool import ConnectionPoolManager
from .client import LLMClient

logger = get_logger(__name__)


class AzureOpenAIClient(LLMClient):
    """Azure OpenAI client implementation with connection pooling."""
    
    def __init__(self):
        """Initialize Azure OpenAI client."""
        self._pool_manager = ConnectionPoolManager()
        self.deployment_name = config.llm.azure_openai_deployment_name
        
    def _get_client_pool(self):
        """Get or create the Azure OpenAI client pool."""
        return self._pool_manager.get_pool(
            name="azure_openai",
            client_class=AsyncAzureOpenAI,
            max_connections=5,
            api_key=config.llm.azure_openai_api_key,
            api_version=config.llm.azure_openai_api_version,
            azure_endpoint=config.llm.azure_openai_endpoint
        )
        
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2000,
        **kwargs: Any
    ) -> str:
        """Generate a chat completion using Azure OpenAI with connection pooling."""
        pool = self._get_client_pool()
        
        async with pool.get_connection() as client:
            try:
                logger.info(
                    "Requesting chat completion",
                    deployment=self.deployment_name,
                    message_count=len(messages),
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                response: ChatCompletion = await client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                content = response.choices[0].message.content
                
                logger.info(
                    "Chat completion successful",
                    deployment=self.deployment_name,
                    response_length=len(content) if content else 0,
                    usage=response.usage.model_dump() if response.usage else None
                )
                
                return content or ""
                
            except Exception as e:
                logger.error(
                    "Chat completion failed",
                    deployment=self.deployment_name,
                    error=str(e)
                )
                raise
            
    async def embedding(
        self,
        text: str,
        model: str = "text-embedding-3-small",
        **kwargs: Any
    ) -> List[float]:
        """Generate text embedding using Azure OpenAI with connection pooling."""
        pool = self._get_client_pool()
        
        async with pool.get_connection() as client:
            try:
                logger.info(
                    "Requesting text embedding",
                    model=model,
                    text_length=len(text)
                )
                
                response = await client.embeddings.create(
                    model=model,
                    input=text,
                    **kwargs
                )
                
                embedding = response.data[0].embedding
                
                logger.info(
                    "Embedding generation successful",
                    model=model,
                    embedding_dimension=len(embedding)
                )
                
                return embedding
                
            except Exception as e:
                logger.error(
                    "Embedding generation failed",
                    model=model,
                    error=str(e)
                )
                raise
            
    async def health_check(self) -> bool:
        """Check if Azure OpenAI service is available."""
        try:
            # Simple test with minimal tokens
            test_messages = [
                {"role": "user", "content": "Hello"}
            ]
            
            await self.chat_completion(
                messages=test_messages,
                max_tokens=5,
                temperature=0
            )
            
            logger.info("Azure OpenAI health check passed")
            return True
            
        except Exception as e:
            logger.error("Azure OpenAI health check failed", error=str(e))
            return False