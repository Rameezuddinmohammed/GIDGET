"""Agent performance monitoring and metrics collection."""

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import asyncio

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentExecution:
    """Record of a single agent execution."""
    agent_name: str
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None
    success: bool = False
    error_message: Optional[str] = None
    findings_count: int = 0
    confidence_score: float = 0.0
    memory_usage_mb: Optional[float] = None


@dataclass
class AgentMetrics:
    """Aggregated metrics for an agent."""
    agent_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_duration_ms: float = 0.0
    min_duration_ms: int = 0
    max_duration_ms: int = 0
    avg_confidence: float = 0.0
    avg_findings_count: float = 0.0
    last_execution: Optional[datetime] = None
    error_rate: float = 0.0


class AgentMonitor:
    """Monitor and collect performance metrics for agents."""
    
    def __init__(self, max_history_size: int = 10000):
        """Initialize the agent monitor."""
        self.max_history_size = max_history_size
        self.executions: List[AgentExecution] = []
        self.active_executions: Dict[str, AgentExecution] = {}
        self._lock = asyncio.Lock()
        
    async def start_execution(
        self, 
        agent_name: str, 
        session_id: str
    ) -> str:
        """Start tracking an agent execution."""
        execution_id = f"{agent_name}_{session_id}_{int(time.time() * 1000)}"
        
        execution = AgentExecution(
            agent_name=agent_name,
            session_id=session_id,
            start_time=datetime.now()
        )
        
        async with self._lock:
            self.active_executions[execution_id] = execution
            
        logger.debug(f"Started tracking execution: {execution_id}")
        return execution_id
        
    async def record_execution(
        self,
        execution_id: str,
        success: bool,
        findings_count: int = 0,
        confidence_score: float = 0.0,
        error_message: Optional[str] = None,
        memory_usage_mb: Optional[float] = None
    ) -> None:
        """Record the completion of an agent execution."""
        async with self._lock:
            if execution_id not in self.active_executions:
                logger.warning(f"Unknown execution ID: {execution_id}")
                return
                
            execution = self.active_executions.pop(execution_id)
            execution.end_time = datetime.now()
            execution.duration_ms = int(
                (execution.end_time - execution.start_time).total_seconds() * 1000
            )
            execution.success = success
            execution.findings_count = findings_count
            execution.confidence_score = confidence_score
            execution.error_message = error_message
            execution.memory_usage_mb = memory_usage_mb
            
            # Add to history
            self.executions.append(execution)
            
            # Trim history if needed
            if len(self.executions) > self.max_history_size:
                self.executions = self.executions[-self.max_history_size:]
                
        logger.debug(f"Recorded execution: {execution_id}, success: {success}")
        
    async def get_agent_metrics(self, agent_name: str) -> AgentMetrics:
        """Get aggregated metrics for a specific agent."""
        async with self._lock:
            agent_executions = [
                e for e in self.executions 
                if e.agent_name == agent_name and e.end_time is not None
            ]
            
        if not agent_executions:
            return AgentMetrics(agent_name=agent_name)
            
        successful = [e for e in agent_executions if e.success]
        failed = [e for e in agent_executions if not e.success]
        
        durations = [e.duration_ms for e in agent_executions if e.duration_ms is not None]
        confidences = [e.confidence_score for e in agent_executions]
        findings_counts = [e.findings_count for e in agent_executions]
        
        return AgentMetrics(
            agent_name=agent_name,
            total_executions=len(agent_executions),
            successful_executions=len(successful),
            failed_executions=len(failed),
            avg_duration_ms=sum(durations) / len(durations) if durations else 0.0,
            min_duration_ms=min(durations) if durations else 0,
            max_duration_ms=max(durations) if durations else 0,
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            avg_findings_count=sum(findings_counts) / len(findings_counts) if findings_counts else 0.0,
            last_execution=max(e.end_time for e in agent_executions),
            error_rate=len(failed) / len(agent_executions) if agent_executions else 0.0
        )
        
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide performance metrics."""
        async with self._lock:
            completed_executions = [
                e for e in self.executions 
                if e.end_time is not None
            ]
            
        if not completed_executions:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "avg_duration_ms": 0.0,
                "active_executions": len(self.active_executions),
                "agents": {}
            }
            
        successful = [e for e in completed_executions if e.success]
        durations = [e.duration_ms for e in completed_executions if e.duration_ms is not None]
        
        # Get per-agent metrics
        agent_names = set(e.agent_name for e in completed_executions)
        agent_metrics = {}
        
        for agent_name in agent_names:
            metrics = await self.get_agent_metrics(agent_name)
            agent_metrics[agent_name] = {
                "total_executions": metrics.total_executions,
                "success_rate": (metrics.successful_executions / metrics.total_executions) if metrics.total_executions > 0 else 0.0,
                "avg_duration_ms": metrics.avg_duration_ms,
                "avg_confidence": metrics.avg_confidence,
                "error_rate": metrics.error_rate
            }
            
        return {
            "total_executions": len(completed_executions),
            "success_rate": len(successful) / len(completed_executions),
            "avg_duration_ms": sum(durations) / len(durations) if durations else 0.0,
            "active_executions": len(self.active_executions),
            "agents": agent_metrics,
            "last_updated": datetime.now().isoformat()
        }
        
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get a health summary for monitoring dashboards."""
        system_metrics = await self.get_system_metrics()
        
        # Determine health status
        success_rate = system_metrics["success_rate"]
        avg_duration = system_metrics["avg_duration_ms"]
        active_count = system_metrics["active_executions"]
        
        health_status = "healthy"
        if success_rate < 0.8:
            health_status = "degraded"
        elif success_rate < 0.5:
            health_status = "unhealthy"
            
        if avg_duration > 30000:  # 30 seconds
            health_status = "slow" if health_status == "healthy" else "unhealthy"
            
        return {
            "status": health_status,
            "success_rate": success_rate,
            "avg_response_time_ms": avg_duration,
            "active_executions": active_count,
            "total_executions_24h": len([
                e for e in self.executions 
                if e.start_time > datetime.now() - timedelta(hours=24)
            ]),
            "timestamp": datetime.now().isoformat()
        }
        
    async def get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache-related performance metrics."""
        try:
            # Import here to avoid circular imports
            from ..caching.cache_manager import cache_manager
            
            cache_stats = await cache_manager.get_cache_stats()
            
            return {
                "cache_hits": cache_stats.get("hits", 0),
                "cache_misses": cache_stats.get("misses", 0),
                "cache_hit_rate": cache_stats.get("hit_rate", 0.0),
                "avg_cache_lookup_ms": 0.0,  # This would need timing data from cache_manager
                "cache_size": 0,  # This would need size data from Supabase
                "cache_evictions": cache_stats.get("invalidations", 0)
            }
        except Exception as e:
            logger.warning(f"Failed to get cache metrics: {str(e)}")
            return {
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_hit_rate": 0.0,
                "avg_cache_lookup_ms": 0.0,
                "cache_size": 0,
                "cache_evictions": 0
            }


# Global monitor instance
agent_monitor = AgentMonitor()