"""Health check API routes."""

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status

from ..models import HealthCheck
from ..dependencies import get_supabase_client, get_neo4j_client
from ...database.supabase_client import SupabaseClient
from ...database.neo4j_client import Neo4jClient
from ...monitoring.agent_monitor import agent_monitor
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


@router.get("/metrics")
async def get_metrics() -> dict:
    """Get comprehensive performance metrics from the monitoring system."""
    try:
        # Get agent metrics
        agent_health = agent_monitor.get_health_summary()
        
        # Get cache metrics
        from ...caching.cache_manager import cache_manager
        cache_stats = await cache_manager.get_statistics()
        
        # Get overall health
        from ...monitoring.health_checker import health_checker
        overall_health = health_checker.get_overall_health()
        
        # Calculate derived metrics
        total_executions = sum(stats.get("total_executions", 0) for stats in agent_health.values())
        total_errors = sum(stats.get("error_count", 0) for stats in agent_health.values())
        success_rate = (total_executions - total_errors) / total_executions if total_executions > 0 else 1.0
        
        # Calculate average response time
        response_times = [comp.get("response_time", 0) for comp in overall_health["components"].values() 
                         if comp.get("response_time")]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "health": {
                "status": overall_health["status"],
                "success_rate": success_rate,
                "avg_response_time_ms": avg_response_time * 1000,
                "active_executions": 0  # Mock data
            },
            "system": {
                "total_executions": total_executions,
                "success_rate": success_rate,
                "avg_duration_ms": avg_response_time * 1000,
                "agents": agent_health
            },
            "cache": {
                "cache_hits": cache_stats.get("hits", 0),
                "cache_misses": cache_stats.get("misses", 0),
                "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
                "cache_size_bytes": cache_stats.get("size_bytes", 0)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        # Return basic metrics if detailed collection fails
        return {
            "health": {
                "status": "unknown",
                "success_rate": 0.0,
                "avg_response_time_ms": 0.0,
                "active_executions": 0
            },
            "system": {
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "agents": {}
            },
            "cache": {
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_hit_rate": 0.0,
                "cache_size_bytes": 0
            },
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get("/prometheus")
async def prometheus_metrics():
    """Prometheus-formatted metrics endpoint."""
    from fastapi import Response
    from ...monitoring.metrics import metrics
    
    try:
        metrics_data = metrics.get_metrics()
        return Response(content=metrics_data, media_type="text/plain")
    except Exception as e:
        logger.error(f"Failed to get Prometheus metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve Prometheus metrics: {str(e)}"
        )


@router.get("/components/{component_name}")
async def component_health(component_name: str):
    """Get health status for a specific component."""
    from ...monitoring.health_checker import health_checker
    
    component_health = health_checker.get_component_health(component_name)
    
    if not component_health:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": f"Component '{component_name}' not found",
                "available_components": list(health_checker.checks.keys())
            }
        )
    
    return component_health.to_dict()


@router.get("/components/{component_name}/history")
async def component_health_history(component_name: str, limit: int = 10):
    """Get health history for a specific component."""
    from ...monitoring.health_checker import health_checker
    
    history = health_checker.get_component_history(component_name, limit)
    
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": f"No history found for component '{component_name}'",
                "available_components": list(health_checker.check_history.keys())
            }
        )
    
    return {
        "component": component_name,
        "history": [check.to_dict() for check in history],
        "count": len(history)
    }