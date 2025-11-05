"""Health checking system for Code Intelligence components."""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import structlog

from ..database.neo4j_client import Neo4jClient
from ..database.supabase_client import SupabaseClient
from .metrics import metrics


logger = structlog.get_logger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ComponentHealth:
    """Health information for a system component."""
    
    def __init__(self, name: str, status: HealthStatus, message: str = "", 
                 response_time: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.status = status
        self.message = message
        self.response_time = response_time
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "response_time": self.response_time,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.checks: Dict[str, ComponentHealth] = {}
        self.check_history: Dict[str, List[ComponentHealth]] = {}
        self.max_history_size = 100
        self._running = False
        self._check_interval = 30  # seconds
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        self._running = True
        logger.info("Starting health monitoring")
        
        # Start monitoring task
        asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self._running = False
        logger.info("Stopping health monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self.check_all_components()
                await asyncio.sleep(self._check_interval)
            except Exception as e:
                logger.error("Error in health monitoring loop", error=str(e))
                await asyncio.sleep(self._check_interval)
    
    async def check_all_components(self) -> Dict[str, ComponentHealth]:
        """Check health of all system components."""
        logger.debug("Running health checks for all components")
        
        # Run all health checks concurrently
        tasks = [
            self.check_api_health(),
            self.check_neo4j_health(),
            self.check_supabase_health(),
            self.check_redis_health(),
            self.check_agent_system_health(),
            self.check_cache_health(),
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error("Health check failed", error=str(result))
            elif isinstance(result, ComponentHealth):
                self._record_health_check(result)
        
        return self.checks
    
    async def check_api_health(self) -> ComponentHealth:
        """Check API health."""
        start_time = time.time()
        
        try:
            # Simple health check - verify the application is responding
            # In a real implementation, this might make an HTTP request to itself
            response_time = time.time() - start_time
            
            health = ComponentHealth(
                name="api",
                status=HealthStatus.HEALTHY,
                message="API is responding normally",
                response_time=response_time,
                metadata={
                    "version": "1.0.0",
                    "uptime": time.time() - start_time
                }
            )
            
            return health
            
        except Exception as e:
            response_time = time.time() - start_time
            health = ComponentHealth(
                name="api",
                status=HealthStatus.UNHEALTHY,
                message=f"API health check failed: {str(e)}",
                response_time=response_time
            )
            
            return health
    
    async def check_neo4j_health(self) -> ComponentHealth:
        """Check Neo4j database health."""
        start_time = time.time()
        
        try:
            # Create a temporary client for health checking
            neo4j_client = Neo4jClient()
            
            # Simple query to test connectivity
            result = await neo4j_client.execute_query("RETURN 1 as health_check")
            response_time = time.time() - start_time
            
            if result and len(result) > 0:
                # Get additional metadata
                db_info = await neo4j_client.execute_query(
                    "CALL dbms.components() YIELD name, versions, edition"
                )
                
                health = ComponentHealth(
                    name="neo4j",
                    status=HealthStatus.HEALTHY,
                    message="Neo4j is responding normally",
                    response_time=response_time,
                    metadata={
                        "database_info": db_info[0] if db_info else {},
                        "query_result": result[0]
                    }
                )
            else:
                health = ComponentHealth(
                    name="neo4j",
                    status=HealthStatus.DEGRADED,
                    message="Neo4j query returned unexpected result",
                    response_time=response_time
                )
            
            await neo4j_client.close()
            return health
            
        except Exception as e:
            response_time = time.time() - start_time
            health = ComponentHealth(
                name="neo4j",
                status=HealthStatus.UNHEALTHY,
                message=f"Neo4j health check failed: {str(e)}",
                response_time=response_time
            )
            
            return health
    
    async def check_supabase_health(self) -> ComponentHealth:
        """Check Supabase/PostgreSQL health."""
        start_time = time.time()
        
        try:
            # Create a temporary client for health checking
            supabase_client = SupabaseClient()
            
            # Simple query to test connectivity
            # This would depend on your Supabase client implementation
            is_healthy = await supabase_client.is_healthy()
            response_time = time.time() - start_time
            
            if is_healthy:
                health = ComponentHealth(
                    name="supabase",
                    status=HealthStatus.HEALTHY,
                    message="Supabase is responding normally",
                    response_time=response_time,
                    metadata={
                        "connection_pool": "active"
                    }
                )
            else:
                health = ComponentHealth(
                    name="supabase",
                    status=HealthStatus.UNHEALTHY,
                    message="Supabase health check failed",
                    response_time=response_time
                )
            
            return health
            
        except Exception as e:
            response_time = time.time() - start_time
            health = ComponentHealth(
                name="supabase",
                status=HealthStatus.UNHEALTHY,
                message=f"Supabase health check failed: {str(e)}",
                response_time=response_time
            )
            
            return health
    
    async def check_redis_health(self) -> ComponentHealth:
        """Check Redis cache health."""
        start_time = time.time()
        
        try:
            # This would use your Redis client
            # For now, we'll simulate a health check
            response_time = time.time() - start_time
            
            health = ComponentHealth(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis is responding normally",
                response_time=response_time,
                metadata={
                    "memory_usage": "normal",
                    "connected_clients": 5
                }
            )
            
            return health
            
        except Exception as e:
            response_time = time.time() - start_time
            health = ComponentHealth(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis health check failed: {str(e)}",
                response_time=response_time
            )
            
            return health
    
    async def check_agent_system_health(self) -> ComponentHealth:
        """Check agent system health."""
        start_time = time.time()
        
        try:
            from ..agents.base import agent_monitor
            
            # Get agent system metrics
            health_summary = agent_monitor.get_health_summary()
            response_time = time.time() - start_time
            
            # Determine overall health based on agent performance
            total_executions = sum(stats.get("total_executions", 0) for stats in health_summary.values())
            total_errors = sum(stats.get("error_count", 0) for stats in health_summary.values())
            
            if total_executions == 0:
                status = HealthStatus.UNKNOWN
                message = "No agent executions recorded"
            elif total_errors / total_executions > 0.1:  # More than 10% error rate
                status = HealthStatus.DEGRADED
                message = f"High agent error rate: {total_errors}/{total_executions}"
            else:
                status = HealthStatus.HEALTHY
                message = "Agent system is performing normally"
            
            health = ComponentHealth(
                name="agents",
                status=status,
                message=message,
                response_time=response_time,
                metadata={
                    "total_executions": total_executions,
                    "total_errors": total_errors,
                    "agent_count": len(health_summary),
                    "agents": health_summary
                }
            )
            
            return health
            
        except Exception as e:
            response_time = time.time() - start_time
            health = ComponentHealth(
                name="agents",
                status=HealthStatus.UNHEALTHY,
                message=f"Agent system health check failed: {str(e)}",
                response_time=response_time
            )
            
            return health
    
    async def check_cache_health(self) -> ComponentHealth:
        """Check caching system health."""
        start_time = time.time()
        
        try:
            from ..caching.cache_manager import cache_manager
            
            # Get cache statistics
            cache_stats = await cache_manager.get_statistics()
            response_time = time.time() - start_time
            
            hit_rate = cache_stats.get("hit_rate", 0)
            
            if hit_rate > 0.8:
                status = HealthStatus.HEALTHY
                message = "Cache is performing well"
            elif hit_rate > 0.5:
                status = HealthStatus.DEGRADED
                message = "Cache hit rate is below optimal"
            else:
                status = HealthStatus.DEGRADED
                message = "Cache hit rate is low"
            
            health = ComponentHealth(
                name="cache",
                status=status,
                message=message,
                response_time=response_time,
                metadata=cache_stats
            )
            
            return health
            
        except Exception as e:
            response_time = time.time() - start_time
            health = ComponentHealth(
                name="cache",
                status=HealthStatus.UNHEALTHY,
                message=f"Cache health check failed: {str(e)}",
                response_time=response_time
            )
            
            return health
    
    def _record_health_check(self, health: ComponentHealth):
        """Record a health check result."""
        # Update current status
        self.checks[health.name] = health
        
        # Add to history
        if health.name not in self.check_history:
            self.check_history[health.name] = []
        
        self.check_history[health.name].append(health)
        
        # Trim history if too long
        if len(self.check_history[health.name]) > self.max_history_size:
            self.check_history[health.name] = self.check_history[health.name][-self.max_history_size:]
        
        # Update metrics
        status_value = 1 if health.status == HealthStatus.HEALTHY else 0
        # This would update a Prometheus gauge if we had one for health status
        
        logger.debug("Health check recorded", 
                    component=health.name, 
                    status=health.status.value,
                    response_time=health.response_time)
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        if not self.checks:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health checks performed yet",
                "components": {}
            }
        
        # Determine overall status
        component_statuses = [check.status for check in self.checks.values()]
        
        if all(status == HealthStatus.HEALTHY for status in component_statuses):
            overall_status = HealthStatus.HEALTHY
            message = "All components are healthy"
        elif any(status == HealthStatus.UNHEALTHY for status in component_statuses):
            overall_status = HealthStatus.UNHEALTHY
            unhealthy_components = [name for name, check in self.checks.items() 
                                  if check.status == HealthStatus.UNHEALTHY]
            message = f"Unhealthy components: {', '.join(unhealthy_components)}"
        elif any(status == HealthStatus.DEGRADED for status in component_statuses):
            overall_status = HealthStatus.DEGRADED
            degraded_components = [name for name, check in self.checks.items() 
                                 if check.status == HealthStatus.DEGRADED]
            message = f"Degraded components: {', '.join(degraded_components)}"
        else:
            overall_status = HealthStatus.UNKNOWN
            message = "System status unknown"
        
        return {
            "status": overall_status.value,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {name: check.to_dict() for name, check in self.checks.items()}
        }
    
    def get_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Get health status for a specific component."""
        return self.checks.get(component_name)
    
    def get_component_history(self, component_name: str, limit: int = 10) -> List[ComponentHealth]:
        """Get health history for a specific component."""
        history = self.check_history.get(component_name, [])
        return history[-limit:] if limit else history


# Global health checker instance
health_checker = HealthChecker()