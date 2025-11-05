"""Load testing for API endpoints and concurrent operations."""

import asyncio
import time
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest
import httpx
from unittest.mock import Mock, patch, AsyncMock

from src.code_intelligence.api.main import app
from src.code_intelligence.monitoring.agent_monitor import agent_monitor


class LoadTestResults:
    """Container for load test results."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.success_count: int = 0
        self.error_count: int = 0
        self.errors: List[str] = []
        self.start_time: datetime = datetime.now()
        self.end_time: datetime = None
        
    def add_result(self, response_time: float, success: bool, error: str = None):
        """Add a test result."""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
            if error:
                self.errors.append(error)
                
    def finalize(self):
        """Finalize the test results."""
        self.end_time = datetime.now()
        
    @property
    def total_requests(self) -> int:
        """Total number of requests."""
        return self.success_count + self.error_count
        
    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.success_count / self.total_requests) * 100
        
    @property
    def avg_response_time(self) -> float:
        """Average response time in milliseconds."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
        
    @property
    def median_response_time(self) -> float:
        """Median response time in milliseconds."""
        if not self.response_times:
            return 0.0
        return statistics.median(self.response_times)
        
    @property
    def p95_response_time(self) -> float:
        """95th percentile response time in milliseconds."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]
        
    @property
    def requests_per_second(self) -> float:
        """Requests per second."""
        if not self.end_time:
            return 0.0
        duration = (self.end_time - self.start_time).total_seconds()
        if duration == 0:
            return 0.0
        return self.total_requests / duration
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "total_requests": self.total_requests,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
            "avg_response_time_ms": self.avg_response_time,
            "median_response_time_ms": self.median_response_time,
            "p95_response_time_ms": self.p95_response_time,
            "requests_per_second": self.requests_per_second,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else 0,
            "errors": self.errors[:10]  # First 10 errors
        }


class APILoadTester:
    """Load tester for API endpoints."""
    
    def __init__(self, base_url: str = "http://testserver"):
        """Initialize the load tester."""
        self.base_url = base_url
        
    async def test_endpoint_load(
        self,
        endpoint: str,
        method: str = "GET",
        payload: Dict[str, Any] = None,
        concurrent_requests: int = 50,
        total_requests: int = 200,
        timeout: float = 30.0
    ) -> LoadTestResults:
        """Test load on a specific endpoint."""
        results = LoadTestResults()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def make_request(session: httpx.AsyncClient) -> Tuple[float, bool, str]:
            """Make a single request."""
            async with semaphore:
                start_time = time.time()
                try:
                    if method.upper() == "POST":
                        response = await session.post(endpoint, json=payload, timeout=timeout)
                    elif method.upper() == "PUT":
                        response = await session.put(endpoint, json=payload, timeout=timeout)
                    elif method.upper() == "DELETE":
                        response = await session.delete(endpoint, timeout=timeout)
                    else:
                        response = await session.get(endpoint, timeout=timeout)
                    
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Consider 2xx and 3xx as success
                    success = 200 <= response.status_code < 400
                    error_msg = f"HTTP {response.status_code}" if not success else None
                    
                    return response_time, success, error_msg
                    
                except httpx.TimeoutException:
                    response_time = timeout * 1000
                    return response_time, False, "Timeout"
                except Exception as e:
                    response_time = (time.time() - start_time) * 1000
                    return response_time, False, str(e)
        
        # Execute load test
        async with httpx.AsyncClient(app=app, base_url=self.base_url) as client:
            tasks = [make_request(client) for _ in range(total_requests)]
            
            for task in asyncio.as_completed(tasks):
                response_time, success, error = await task
                results.add_result(response_time, success, error)
        
        results.finalize()
        return results
        
    async def test_query_submission_load(self, concurrent_requests: int = 50) -> LoadTestResults:
        """Test load on query submission endpoint."""
        query_payload = {
            "repository_url": "https://github.com/test/repo.git",
            "query": "What functions were changed in the last week?",
            "options": {
                "max_commits": 50,
                "include_tests": False
            }
        }
        
        with patch('src.code_intelligence.api.dependencies.get_orchestrator') as mock_orchestrator:
            # Mock the orchestrator to return quickly
            mock_agent = Mock()
            mock_agent.execute_query = AsyncMock(return_value=Mock(
                session_id="test-session",
                get_all_findings=Mock(return_value=[]),
                verification={"overall_confidence": 0.8}
            ))
            mock_orchestrator.return_value = mock_agent
            
            return await self.test_endpoint_load(
                endpoint="/api/v1/queries/",
                method="POST",
                payload=query_payload,
                concurrent_requests=concurrent_requests,
                total_requests=concurrent_requests * 2  # 2 requests per concurrent user
            )
            
    async def test_health_endpoint_load(self, concurrent_requests: int = 100) -> LoadTestResults:
        """Test load on health endpoint."""
        return await self.test_endpoint_load(
            endpoint="/api/v1/health/",
            method="GET",
            concurrent_requests=concurrent_requests,
            total_requests=concurrent_requests * 3  # 3 requests per concurrent user
        )
        
    async def test_metrics_endpoint_load(self, concurrent_requests: int = 30) -> LoadTestResults:
        """Test load on metrics endpoint."""
        return await self.test_endpoint_load(
            endpoint="/api/v1/health/metrics",
            method="GET",
            concurrent_requests=concurrent_requests,
            total_requests=concurrent_requests * 2
        )


@pytest.mark.load
class TestAPILoadTesting:
    """Load testing for API endpoints."""
    
    @pytest.fixture
    def load_tester(self):
        """Create load tester instance."""
        return APILoadTester()
    
    @pytest.mark.asyncio
    async def test_health_endpoint_load(self, load_tester):
        """Test health endpoint under load."""
        results = await load_tester.test_health_endpoint_load(concurrent_requests=50)
        
        # Performance assertions
        assert results.success_rate >= 95.0, f"Success rate too low: {results.success_rate}%"
        assert results.avg_response_time <= 1000, f"Average response time too high: {results.avg_response_time}ms"
        assert results.p95_response_time <= 2000, f"95th percentile too high: {results.p95_response_time}ms"
        assert results.requests_per_second >= 10, f"Throughput too low: {results.requests_per_second} req/s"
        
        print(f"Health endpoint load test results: {results.to_dict()}")
        
    @pytest.mark.asyncio
    async def test_metrics_endpoint_load(self, load_tester):
        """Test metrics endpoint under load."""
        results = await load_tester.test_metrics_endpoint_load(concurrent_requests=20)
        
        # Metrics endpoint might be slower due to data aggregation
        assert results.success_rate >= 90.0, f"Success rate too low: {results.success_rate}%"
        assert results.avg_response_time <= 3000, f"Average response time too high: {results.avg_response_time}ms"
        assert results.p95_response_time <= 5000, f"95th percentile too high: {results.p95_response_time}ms"
        
        print(f"Metrics endpoint load test results: {results.to_dict()}")
        
    @pytest.mark.asyncio
    async def test_query_submission_load(self, load_tester):
        """Test query submission under load."""
        results = await load_tester.test_query_submission_load(concurrent_requests=10)
        
        # Query submission is more complex, so we allow for higher response times
        assert results.success_rate >= 80.0, f"Success rate too low: {results.success_rate}%"
        assert results.avg_response_time <= 10000, f"Average response time too high: {results.avg_response_time}ms"
        assert results.error_count <= results.total_requests * 0.2, "Too many errors"
        
        print(f"Query submission load test results: {results.to_dict()}")
        
    @pytest.mark.asyncio
    async def test_mixed_endpoint_load(self, load_tester):
        """Test mixed load across multiple endpoints."""
        # Run multiple endpoint tests concurrently
        tasks = [
            load_tester.test_health_endpoint_load(concurrent_requests=20),
            load_tester.test_metrics_endpoint_load(concurrent_requests=10),
            load_tester.test_query_submission_load(concurrent_requests=5)
        ]
        
        results = await asyncio.gather(*tasks)
        health_results, metrics_results, query_results = results
        
        # All endpoints should maintain reasonable performance under mixed load
        assert health_results.success_rate >= 90.0
        assert metrics_results.success_rate >= 85.0
        assert query_results.success_rate >= 75.0
        
        # Combined throughput should be reasonable
        total_rps = sum(r.requests_per_second for r in results)
        assert total_rps >= 15, f"Combined throughput too low: {total_rps} req/s"
        
        print(f"Mixed load test - Health: {health_results.success_rate}%, "
              f"Metrics: {metrics_results.success_rate}%, "
              f"Query: {query_results.success_rate}%")


@pytest.mark.load
class TestConnectionPoolLoad:
    """Test connection pool performance under load."""
    
    @pytest.mark.asyncio
    async def test_connection_pool_concurrent_access(self):
        """Test connection pool under concurrent access."""
        from src.code_intelligence.core.connection_pool import ConnectionPoolManager
        
        # Mock client class
        class MockClient:
            def __init__(self, **kwargs):
                self.connected = False
                
            async def connect(self):
                await asyncio.sleep(0.01)  # Simulate connection time
                self.connected = True
                
            async def health_check(self):
                return self.connected
                
            async def close(self):
                self.connected = False
        
        pool_manager = ConnectionPoolManager()
        pool = pool_manager.get_pool("test_pool", MockClient, max_connections=10)
        
        results = LoadTestResults()
        
        async def use_connection():
            """Use a connection from the pool."""
            start_time = time.time()
            try:
                async with pool.get_connection() as conn:
                    # Simulate some work
                    await asyncio.sleep(0.05)
                    assert conn.connected
                
                response_time = (time.time() - start_time) * 1000
                return response_time, True, None
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                return response_time, False, str(e)
        
        # Test with high concurrency
        tasks = [use_connection() for _ in range(50)]
        
        for task in asyncio.as_completed(tasks):
            response_time, success, error = await task
            results.add_result(response_time, success, error)
        
        results.finalize()
        
        # Connection pool should handle concurrent access efficiently
        assert results.success_rate >= 95.0, f"Connection pool success rate too low: {results.success_rate}%"
        assert results.avg_response_time <= 200, f"Connection pool response time too high: {results.avg_response_time}ms"
        
        # Clean up
        await pool_manager.close_all_pools()
        
        print(f"Connection pool load test results: {results.to_dict()}")


@pytest.mark.load
class TestAgentMonitoringLoad:
    """Test agent monitoring system under load."""
    
    @pytest.mark.asyncio
    async def test_agent_monitor_concurrent_executions(self):
        """Test agent monitor with many concurrent executions."""
        results = LoadTestResults()
        
        async def simulate_agent_execution(agent_name: str, session_id: str):
            """Simulate an agent execution."""
            start_time = time.time()
            try:
                # Start execution tracking
                execution_id = await agent_monitor.start_execution(agent_name, session_id)
                
                # Simulate some work
                await asyncio.sleep(0.02)
                
                # Record completion
                await agent_monitor.record_execution(
                    execution_id=execution_id,
                    success=True,
                    findings_count=5,
                    confidence_score=0.8
                )
                
                response_time = (time.time() - start_time) * 1000
                return response_time, True, None
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                return response_time, False, str(e)
        
        # Simulate many concurrent agent executions
        tasks = []
        for i in range(100):
            agent_name = f"agent_{i % 5}"  # 5 different agents
            session_id = f"session_{i}"
            tasks.append(simulate_agent_execution(agent_name, session_id))
        
        for task in asyncio.as_completed(tasks):
            response_time, success, error = await task
            results.add_result(response_time, success, error)
        
        results.finalize()
        
        # Agent monitoring should handle high concurrency
        assert results.success_rate >= 98.0, f"Agent monitor success rate too low: {results.success_rate}%"
        assert results.avg_response_time <= 100, f"Agent monitor response time too high: {results.avg_response_time}ms"
        
        # Verify metrics are properly aggregated
        system_metrics = await agent_monitor.get_system_metrics()
        assert system_metrics["total_executions"] >= 98  # At least 98% should have succeeded
        
        print(f"Agent monitor load test results: {results.to_dict()}")
        print(f"System metrics after load test: {system_metrics}")


@pytest.mark.load
class TestCacheSystemLoad:
    """Test caching system under load."""
    
    @pytest.mark.asyncio
    async def test_cache_concurrent_operations(self):
        """Test cache system with concurrent reads and writes."""
        from src.code_intelligence.caching.cache_manager import cache_manager
        
        results = LoadTestResults()
        
        async def cache_operation(operation_type: str, key: str):
            """Perform a cache operation."""
            start_time = time.time()
            try:
                if operation_type == "read":
                    result = await cache_manager.get_cached_result(
                        query=f"test query {key}",
                        repository_id="test_repo"
                    )
                    # Result might be None, which is fine
                    
                elif operation_type == "write":
                    await cache_manager.store_result(
                        query=f"test query {key}",
                        repository_id="test_repo",
                        result_data={"findings": [{"content": f"test finding {key}"}]},
                        confidence_score=0.8
                    )
                
                response_time = (time.time() - start_time) * 1000
                return response_time, True, None
                
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                return response_time, False, str(e)
        
        # Mix of read and write operations
        tasks = []
        for i in range(50):
            # 70% reads, 30% writes
            operation = "read" if i % 10 < 7 else "write"
            tasks.append(cache_operation(operation, str(i)))
        
        for task in asyncio.as_completed(tasks):
            response_time, success, error = await task
            results.add_result(response_time, success, error)
        
        results.finalize()
        
        # Cache system should handle concurrent operations efficiently
        assert results.success_rate >= 90.0, f"Cache system success rate too low: {results.success_rate}%"
        assert results.avg_response_time <= 500, f"Cache system response time too high: {results.avg_response_time}ms"
        
        print(f"Cache system load test results: {results.to_dict()}")


def run_load_tests():
    """Run all load tests and generate a report."""
    import subprocess
    import json
    from pathlib import Path
    
    # Run load tests
    result = subprocess.run([
        "python", "-m", "pytest", 
        "tests/test_load_testing.py", 
        "-m", "load",
        "-v",
        "--tb=short"
    ], capture_output=True, text=True)
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_results": {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        },
        "summary": {
            "passed": result.stdout.count("PASSED"),
            "failed": result.stdout.count("FAILED"),
            "errors": result.stdout.count("ERROR")
        }
    }
    
    # Save report
    report_file = Path(f"load_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    report_file.write_text(json.dumps(report, indent=2))
    
    print(f"Load test report saved to: {report_file}")
    return report


if __name__ == "__main__":
    # Run load tests when executed directly
    asyncio.run(run_load_tests())