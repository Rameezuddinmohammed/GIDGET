"""Health check API routes."""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status

from ..models import HealthCheck
from ..dependencies import get_supabase_client, get_neo4j_client
from ...database.supabase_client import SupabaseClient
from ...database.neo4j_client import Neo4jClient
from ...logging import get_logger


router = APIRouter(prefix="/health", tags=["health"])
logger = get_logger(__name__)


@router.get("/", response_model=HealthCheck)
async def health_check() -> HealthCheck:
    """Basic health check endpoint."""
    return HealthCheck(
        status="healthy",
        version="1.0.0",
        services={
            "api": "healthy",
            "database": "unknown",
            "graph_db": "unknown",
            "agents": "healthy"
        }
    )


@router.get("/detailed")
async def detailed_health_check() -> dict:
    """Detailed health check with service dependencies."""
    services = {}
    overall_status = "healthy"
    
    # Check Supabase connection
    try:
        supabase = SupabaseClient()
        await supabase.initialize()
        services["supabase"] = "healthy"
    except Exception as e:
        services["supabase"] = f"unhealthy: {str(e)}"
        overall_status = "degraded"
    
    # Check Neo4j connection
    try:
        neo4j = Neo4jClient()
        await neo4j.initialize()
        services["neo4j"] = "healthy"
    except Exception as e:
        services["neo4j"] = f"unhealthy: {str(e)}"
        overall_status = "degraded"
    
    # Check agent system
    try:
        from ...agents.orchestrator_agent import OrchestratorAgent
        orchestrator = OrchestratorAgent()
        services["agents"] = "healthy"
    except Exception as e:
        services["agents"] = f"unhealthy: {str(e)}"
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "version": "1.0.0",
        "timestamp": datetime.utcnow(),
        "services": services,
        "uptime_seconds": 3600,  # Mock uptime
        "memory_usage_mb": 512,  # Mock memory usage
        "active_queries": 0,     # Mock active queries
        "total_repositories": len(getattr(health_check, '_repo_storage', {}))
    }


@router.get("/ready")
async def readiness_check() -> dict:
    """Readiness check for Kubernetes/container orchestration."""
    try:
        # Check critical dependencies
        supabase = SupabaseClient()
        await supabase.initialize()
        
        neo4j = Neo4jClient()
        await neo4j.initialize()
        
        return {"status": "ready"}
        
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


@router.get("/live")
async def liveness_check() -> dict:
    """Liveness check for Kubernetes/container orchestration."""
    return {"status": "alive", "timestamp": datetime.utcnow()}