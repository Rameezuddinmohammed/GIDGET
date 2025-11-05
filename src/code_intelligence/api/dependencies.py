"""API dependencies and dependency injection."""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..database.supabase_client import SupabaseClient
from ..database.neo4j_client import Neo4jClient
from ..agents.orchestrator_agent import OrchestratorAgent
from ..config import config


# Security
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """Get current authenticated user."""
    if not credentials:
        return None
    
    # In a real implementation, validate the JWT token
    # For now, return a mock user ID
    return "user_123"


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> str:
    """Require authentication and return user ID."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # In a real implementation, validate the JWT token
    # For now, return a mock user ID
    return "user_123"


async def get_supabase_client() -> SupabaseClient:
    """Get Supabase client instance."""
    try:
        client = SupabaseClient()
        await client.initialize()
        return client
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database service unavailable: {str(e)}"
        )


async def get_neo4j_client() -> Neo4jClient:
    """Get Neo4j client instance."""
    try:
        client = Neo4jClient()
        await client.initialize()
        return client
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Graph database service unavailable: {str(e)}"
        )


async def get_orchestrator() -> OrchestratorAgent:
    """Get orchestrator agent instance."""
    try:
        return OrchestratorAgent()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Agent service unavailable: {str(e)}"
        )