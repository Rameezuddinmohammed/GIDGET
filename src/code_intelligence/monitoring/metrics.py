"""Prometheus metrics for Code Intelligence System."""

import time
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, generate_latest
from prometheus_client.core import REGISTRY
import structlog

from ..config import config


logger = structlog.get_logger(__name__)


class MetricsCollector:
    """Centralized metrics collection for the Code Intelligence System."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector."""
        self.registry = registry or REGISTRY
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Set up all Prometheus metrics."""
        # API Metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration_seconds = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Agent Metrics
        self.agent_executions_total = Counter(
            'agent_executions_total',
            'Total agent executions',
            ['agent_name', 'status'],
            registry=self.registry
        )
        
        self.agent_executions_failed_total = Counter(
            'agent_executions_failed_total',
            'Total failed agent executions',
            ['agent_name', 'error_type'],
            registry=self.registry
        )
        
        self.agent_execution_duration_seconds = Histogram(
            'agent_execution_duration_seconds',
            'Agent execution duration in seconds',
            ['agent_name'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
            registry=self.registry
        )
        
        self.agent_queue_size = Gauge(
            'agent_queue_size',
            'Number of tasks in agent queue',
            ['queue_type'],
            registry=self.registry
        )
        
        self.agent_memory_usage_bytes = Gauge(
            'agent_memory_usage_bytes',
            'Agent memory usage in bytes',
            ['agent_name'],
            registry=self.registry
        )
        
        self.agent_memory_limit_bytes = Gauge(
            'agent_memory_limit_bytes',
            'Agent memory limit in bytes',
            ['agent_name'],
            registry=self.registry
        )
        
        # Query Metrics
        self.queries_submitted_total = Counter(
            'queries_submitted_total',
            'Total queries submitted',
            ['query_type'],
            registry=self.registry
        )
        
        self.queries_completed_total = Counter(
            'queries_completed_total',
            'Total queries completed',
            ['query_type', 'status'],
            registry=self.registry
        )
        
        self.query_duration_seconds = Histogram(
            'query_duration_seconds',
            'Query processing duration in seconds',
            ['query_type'],
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0],
            registry=self.registry
        )
        
        # Database Metrics
        self.database_connections_active = Gauge(
            'database_connections_active',
            'Active database connections',
            ['database_type'],
            registry=self.registry
        )
        
        self.database_query_duration_seconds = Histogram(
            'database_query_duration_seconds',
            'Database query duration in seconds',
            ['database_type', 'operation'],
            registry=self.registry
        )
        
        self.database_errors_total = Counter(
            'database_errors_total',
            'Total database errors',
            ['database_type', 'error_type'],
            registry=self.registry
        )
        
        # Cache Metrics
        self.cache_operations_total = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation', 'result'],
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'cache_hit_rate',
            'Cache hit rate (0-1)',
            registry=self.registry
        )
        
        self.cache_size_bytes = Gauge(
            'cache_size_bytes',
            'Cache size in bytes',
            registry=self.registry
        )
        
        # Repository Metrics
        self.repository_ingestion_total = Counter(
            'repository_ingestion_total',
            'Total repository ingestions',
            ['status'],
            registry=self.registry
        )
        
        self.repository_ingestion_failures_total = Counter(
            'repository_ingestion_failures_total',
            'Total repository ingestion failures',
            ['error_type'],
            registry=self.registry
        )
        
        self.repository_size_bytes = Gauge(
            'repository_size_bytes',
            'Repository size in bytes',
            ['repository_id'],
            registry=self.registry
        )
        
        # System Info
        self.system_info = Info(
            'code_intelligence_system_info',
            'System information',
            registry=self.registry
        )
        
        # Set system info
        self.system_info.info({
            'version': '1.0.0',
            'environment': config.app.environment,
            'python_version': '3.11'
        })
    
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        self.http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_agent_execution(self, agent_name: str, duration: float, success: bool, error_type: Optional[str] = None):
        """Record agent execution metrics."""
        status = 'success' if success else 'failure'
        
        self.agent_executions_total.labels(
            agent_name=agent_name,
            status=status
        ).inc()
        
        if not success and error_type:
            self.agent_executions_failed_total.labels(
                agent_name=agent_name,
                error_type=error_type
            ).inc()
        
        self.agent_execution_duration_seconds.labels(
            agent_name=agent_name
        ).observe(duration)
    
    def update_agent_queue_size(self, queue_type: str, size: int):
        """Update agent queue size."""
        self.agent_queue_size.labels(queue_type=queue_type).set(size)
    
    def update_agent_memory_usage(self, agent_name: str, usage_bytes: int, limit_bytes: int):
        """Update agent memory usage."""
        self.agent_memory_usage_bytes.labels(agent_name=agent_name).set(usage_bytes)
        self.agent_memory_limit_bytes.labels(agent_name=agent_name).set(limit_bytes)
    
    def record_query_submission(self, query_type: str):
        """Record query submission."""
        self.queries_submitted_total.labels(query_type=query_type).inc()
    
    def record_query_completion(self, query_type: str, duration: float, success: bool):
        """Record query completion."""
        status = 'success' if success else 'failure'
        
        self.queries_completed_total.labels(
            query_type=query_type,
            status=status
        ).inc()
        
        self.query_duration_seconds.labels(query_type=query_type).observe(duration)
    
    def update_database_connections(self, database_type: str, active_connections: int):
        """Update database connection count."""
        self.database_connections_active.labels(database_type=database_type).set(active_connections)
    
    def record_database_query(self, database_type: str, operation: str, duration: float):
        """Record database query metrics."""
        self.database_query_duration_seconds.labels(
            database_type=database_type,
            operation=operation
        ).observe(duration)
    
    def record_database_error(self, database_type: str, error_type: str):
        """Record database error."""
        self.database_errors_total.labels(
            database_type=database_type,
            error_type=error_type
        ).inc()
    
    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache operation."""
        result = 'hit' if hit else 'miss'
        self.cache_operations_total.labels(
            operation=operation,
            result=result
        ).inc()
    
    def update_cache_hit_rate(self, hit_rate: float):
        """Update cache hit rate."""
        self.cache_hit_rate.set(hit_rate)
    
    def update_cache_size(self, size_bytes: int):
        """Update cache size."""
        self.cache_size_bytes.set(size_bytes)
    
    def record_repository_ingestion(self, success: bool, error_type: Optional[str] = None):
        """Record repository ingestion."""
        status = 'success' if success else 'failure'
        self.repository_ingestion_total.labels(status=status).inc()
        
        if not success and error_type:
            self.repository_ingestion_failures_total.labels(error_type=error_type).inc()
    
    def update_repository_size(self, repository_id: str, size_bytes: int):
        """Update repository size."""
        self.repository_size_bytes.labels(repository_id=repository_id).set(size_bytes)
    
    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format."""
        return generate_latest(self.registry).decode('utf-8')


# Global metrics collector instance
metrics = MetricsCollector()


class MetricsMiddleware:
    """Middleware to collect HTTP request metrics."""
    
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        start_time = time.time()
        method = scope["method"]
        path = scope["path"]
        
        # Normalize endpoint for metrics
        endpoint = self._normalize_endpoint(path)
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status = message["status"]
                duration = time.time() - start_time
                
                # Record metrics
                metrics.record_http_request(method, endpoint, status, duration)
            
            await send(message)
        
        await self.app(scope, receive, send_wrapper)
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalize endpoint path for metrics."""
        # Remove query parameters
        if '?' in path:
            path = path.split('?')[0]
        
        # Normalize common patterns
        import re
        
        # Replace UUIDs with placeholder
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{id}', path)
        
        # Replace other IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        return path


def track_execution_time(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to track execution time of functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record successful execution
                if hasattr(metrics, metric_name):
                    metric = getattr(metrics, metric_name)
                    if labels:
                        metric.labels(**labels).observe(duration)
                    else:
                        metric.observe(duration)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failed execution
                logger.error(f"Function {func.__name__} failed", error=str(e), duration=duration)
                raise
        
        return wrapper
    return decorator


async def collect_system_metrics():
    """Collect system-wide metrics periodically."""
    import psutil
    import asyncio
    
    while True:
        try:
            # Collect system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Update system metrics (if we had system-specific metrics)
            logger.debug("System metrics collected", 
                        cpu_percent=cpu_percent,
                        memory_percent=memory.percent,
                        disk_percent=(disk.used / disk.total) * 100)
            
            # Sleep for 30 seconds
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
            await asyncio.sleep(60)  # Wait longer on error