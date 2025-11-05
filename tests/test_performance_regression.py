"""Performance regression testing to detect performance degradation."""

import asyncio
import json
import time
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.code_intelligence.api.main import app
from src.code_intelligence.agents.orchestrator import AgentOrchestrator
from src.code_intelligence.agents.state import AgentState
from src.code_intelligence.database.query_optimizer import db_optimizer
from src.code_intelligence.monitoring.agent_monitor import agent_monitor


class PerformanceBaseline:
    """Manages performance baselines for regression testing."""
    
    def __init__(self, baseline_file: Path = None):
        self.baseline_file = baseline_file or Path("performance_baselines.json")
        self.baselines = self._load_baselines()
        
    def _load_baselines(self) -> Dict[str, Any]:
        """Load performance baselines from file."""
        if self.baseline_file.exists():
            try:
                return json.loads(self.baseline_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}
        
    def save_baselines(self):
        """Save baselines to file."""
        try:
            self.baseline_file.write_text(json.dumps(self.baselines, indent=2))
        except OSError as e:
            print(f"Failed to save baselines: {e}")
            
    def set_baseline(self, test_name: str, metric: str, value: float, tolerance: float = 0.2):
        """Set a performance baseline."""
        if test_name not in self.baselines:
            self.baselines[test_name] = {}
            
        self.baselines[test_name][metric] = {
            "value": value,
            "tolerance": tolerance,
            "updated": datetime.now().isoformat()
        }
        
    def check_regression(self, test_name: str, metric: str, current_value: float) -> Tuple[bool, str]:
        """Check if current value represents a regression."""
        if test_name not in self.baselines or metric not in self.baselines[test_name]:
            return False, f"No baseline found for {test_name}.{metric}"
            
        baseline_data = self.baselines[test_name][metric]
        baseline_value = baseline_data["value"]
        tolerance = baseline_data.get("tolerance", 0.2)
        
        # Calculate acceptable threshold (baseline + tolerance)
        threshold = baseline_value * (1 + tolerance)
        
        if current_value > threshold:
            regression_pct = ((current_value - baseline_value) / baseline_value) * 100
            return True, f"Regression detected: {current_value:.2f} > {threshold:.2f} ({regression_pct:.1f}% increase)"
        
        return False, f"Performance within acceptable range: {current_value:.2f} <= {threshold:.2f}"
        
    def get_baseline(self, test_name: str, metric: str) -> Optional[float]:
        """Get baseline value for a metric."""
        if test_name in self.baselines and metric in self.baselines[test_name]:
            return self.baselines[test_name][metric]["value"]
        return None


class PerformanceRegressionTester:
    """Test suite for performance regression detection."""
    
    def __init__(self, baseline_file: Path = None):
        self.baseline = PerformanceBaseline(baseline_file)
        
    async def test_api_endpoint_performance(self) -> Dict[str, Any]:
        """Test API endpoint performance and check for regressions."""
        import httpx
        
        results = {
            "test_name": "api_endpoint_performance",
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "regressions": []
        }
        
        endpoints = [
            ("/api/v1/health/", "GET", None),
            ("/api/v1/health/metrics", "GET", None),
            ("/api/v1/health/detailed", "GET", None)
        ]
        
        async with httpx.AsyncClient(app=app, base_url="http://testserver") as client:
            for endpoint, method, payload in endpoints:
                # Warm up
                for _ in range(3):
                    if method == "POST":
                        await client.post(endpoint, json=payload)
                    else:
                        await client.get(endpoint)
                
                # Measure performance
                times = []
                for _ in range(10):
                    start_time = time.time()
                    
                    if method == "POST":
                        response = await client.post(endpoint, json=payload)
                    else:
                        response = await client.get(endpoint)
                    
                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    times.append(response_time)
                    
                    # Ensure request succeeded
                    assert 200 <= response.status_code < 400
                
                # Calculate statistics
                avg_time = statistics.mean(times)
                median_time = statistics.median(times)
                p95_time = sorted(times)[int(0.95 * len(times))]
                
                endpoint_key = endpoint.replace("/", "_").replace("api_v1_", "")
                
                results["metrics"][f"{endpoint_key}_avg_ms"] = avg_time
                results["metrics"][f"{endpoint_key}_median_ms"] = median_time
                results["metrics"][f"{endpoint_key}_p95_ms"] = p95_time
                
                # Check for regressions
                for metric_suffix, value in [("avg_ms", avg_time), ("p95_ms", p95_time)]:
                    metric_name = f"{endpoint_key}_{metric_suffix}"
                    is_regression, message = self.baseline.check_regression(
                        "api_endpoint_performance", metric_name, value
                    )
                    
                    if is_regression:
                        results["regressions"].append({
                            "metric": metric_name,
                            "message": message,
                            "current_value": value,
                            "baseline_value": self.baseline.get_baseline("api_endpoint_performance", metric_name)
                        })
        
        return results
        
    async def test_agent_orchestration_performance(self) -> Dict[str, Any]:
        """Test agent orchestration performance."""
        results = {
            "test_name": "agent_orchestration_performance",
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "regressions": []
        }
        
        # Mock database clients
        with patch('src.code_intelligence.agents.orchestrator.neo4j_client') as mock_neo4j, \
             patch('src.code_intelligence.agents.orchestrator.supabase_client') as mock_supabase:
            
            mock_neo4j.execute_query = AsyncMock(return_value=[])
            mock_supabase.get_cached_result = AsyncMock(return_value=None)
            
            orchestrator = AgentOrchestrator()
            
            # Test different query complexities
            test_queries = [
                ("simple", "What is the main function in this repository?"),
                ("medium", "Find all functions that were modified in the last week and analyze their complexity."),
                ("complex", "Analyze the evolution of the authentication system, identify security vulnerabilities, and trace dependencies across all modules.")
            ]
            
            for complexity, query in test_queries:
                times = []
                
                # Run multiple iterations
                for i in range(5):
                    start_time = time.time()
                    
                    try:
                        result = await asyncio.wait_for(
                            orchestrator.execute_query(query, "/mock/repo/path"),
                            timeout=30.0
                        )
                        
                        execution_time = (time.time() - start_time) * 1000  # Convert to ms
                        times.append(execution_time)
                        
                    except asyncio.TimeoutError:
                        times.append(30000)  # 30 second timeout
                    except Exception as e:
                        print(f"Query execution failed: {e}")
                        times.append(30000)  # Treat as timeout
                
                # Calculate metrics
                avg_time = statistics.mean(times)
                median_time = statistics.median(times)
                
                results["metrics"][f"{complexity}_query_avg_ms"] = avg_time
                results["metrics"][f"{complexity}_query_median_ms"] = median_time
                
                # Check for regressions
                for metric_suffix, value in [("avg_ms", avg_time), ("median_ms", median_time)]:
                    metric_name = f"{complexity}_query_{metric_suffix}"
                    is_regression, message = self.baseline.check_regression(
                        "agent_orchestration_performance", metric_name, value
                    )
                    
                    if is_regression:
                        results["regressions"].append({
                            "metric": metric_name,
                            "message": message,
                            "current_value": value,
                            "baseline_value": self.baseline.get_baseline("agent_orchestration_performance", metric_name)
                        })
        
        return results
        
    async def test_database_query_performance(self) -> Dict[str, Any]:
        """Test database query performance."""
        results = {
            "test_name": "database_query_performance",
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "regressions": []
        }
        
        try:
            # Run database optimization validation
            validation_report = await db_optimizer.run_comprehensive_validation()
            
            # Extract key performance metrics
            if "neo4j_analysis" in validation_report:
                neo4j_data = validation_report["neo4j_analysis"]
                if "benchmark" in neo4j_data:
                    benchmark = neo4j_data["benchmark"]
                    summary = benchmark.get("summary", {})
                    
                    results["metrics"]["neo4j_avg_query_time_ms"] = summary.get("avg_execution_time_ms", 0)
                    results["metrics"]["neo4j_success_rate"] = summary.get("success_rate", 0)
                    results["metrics"]["neo4j_fastest_query_ms"] = summary.get("fastest_query_ms", 0)
                    results["metrics"]["neo4j_slowest_query_ms"] = summary.get("slowest_query_ms", 0)
            
            if "supabase_analysis" in validation_report:
                supabase_data = validation_report["supabase_analysis"]
                if "vector_search" in supabase_data:
                    vs = supabase_data["vector_search"]
                    results["metrics"]["vector_search_time_ms"] = vs.get("execution_time_ms", 0)
                    results["metrics"]["vector_search_confidence"] = vs.get("confidence_score", 0)
                
                if "cache_performance" in supabase_data:
                    cache = supabase_data["cache_performance"]
                    summary = cache.get("summary", {})
                    results["metrics"]["cache_avg_time_ms"] = summary.get("avg_execution_time_ms", 0)
            
            # Check for regressions
            for metric_name, value in results["metrics"].items():
                if isinstance(value, (int, float)) and value > 0:
                    is_regression, message = self.baseline.check_regression(
                        "database_query_performance", metric_name, value
                    )
                    
                    if is_regression:
                        results["regressions"].append({
                            "metric": metric_name,
                            "message": message,
                            "current_value": value,
                            "baseline_value": self.baseline.get_baseline("database_query_performance", metric_name)
                        })
        
        except Exception as e:
            results["error"] = str(e)
        
        return results
        
    async def test_memory_usage_performance(self) -> Dict[str, Any]:
        """Test memory usage patterns."""
        results = {
            "test_name": "memory_usage_performance",
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "regressions": []
        }
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate workload
        with patch('src.code_intelligence.agents.orchestrator.neo4j_client') as mock_neo4j:
            mock_neo4j.execute_query = AsyncMock(return_value=[])
            
            orchestrator = AgentOrchestrator()
            
            # Run multiple queries to stress memory
            for i in range(10):
                await orchestrator.execute_query(
                    f"Test query {i} for memory usage analysis",
                    "/mock/repo/path"
                )
                
                # Measure memory after each query
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                if i == 4:  # After 5 queries
                    results["metrics"]["memory_after_5_queries_mb"] = memory_increase
                elif i == 9:  # After 10 queries
                    results["metrics"]["memory_after_10_queries_mb"] = memory_increase
        
        # Final memory measurement
        final_memory = process.memory_info().rss / 1024 / 1024
        results["metrics"]["peak_memory_increase_mb"] = final_memory - initial_memory
        
        # Check for memory regressions
        for metric_name, value in results["metrics"].items():
            is_regression, message = self.baseline.check_regression(
                "memory_usage_performance", metric_name, value
            )
            
            if is_regression:
                results["regressions"].append({
                    "metric": metric_name,
                    "message": message,
                    "current_value": value,
                    "baseline_value": self.baseline.get_baseline("memory_usage_performance", metric_name)
                })
        
        return results
        
    def update_baselines_from_results(self, results: Dict[str, Any]):
        """Update baselines from test results (use when establishing new baselines)."""
        test_name = results["test_name"]
        
        for metric_name, value in results["metrics"].items():
            if isinstance(value, (int, float)):
                # Set baseline with 20% tolerance by default
                tolerance = 0.3 if "memory" in metric_name else 0.2  # More tolerance for memory metrics
                self.baseline.set_baseline(test_name, metric_name, value, tolerance)
        
        self.baseline.save_baselines()
        
    async def run_all_regression_tests(self) -> Dict[str, Any]:
        """Run all performance regression tests."""
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {
                "total_regressions": 0,
                "tests_with_regressions": 0,
                "total_tests": 0
            }
        }
        
        # List of test methods
        test_methods = [
            self.test_api_endpoint_performance,
            self.test_agent_orchestration_performance,
            self.test_database_query_performance,
            self.test_memory_usage_performance
        ]
        
        for test_method in test_methods:
            try:
                result = await test_method()
                test_name = result["test_name"]
                all_results["tests"][test_name] = result
                
                # Update summary
                all_results["summary"]["total_tests"] += 1
                regression_count = len(result.get("regressions", []))
                all_results["summary"]["total_regressions"] += regression_count
                
                if regression_count > 0:
                    all_results["summary"]["tests_with_regressions"] += 1
                    
            except Exception as e:
                print(f"Test {test_method.__name__} failed: {e}")
        
        return all_results


@pytest.mark.regression
class TestPerformanceRegression:
    """Performance regression test suite."""
    
    @pytest.fixture
    def regression_tester(self):
        """Create regression tester instance."""
        return PerformanceRegressionTester()
    
    @pytest.mark.asyncio
    async def test_api_endpoint_regression(self, regression_tester):
        """Test API endpoint performance regression."""
        results = await regression_tester.test_api_endpoint_performance()
        
        # Should have metrics for all tested endpoints
        assert len(results["metrics"]) > 0, "No performance metrics collected"
        
        # Check for regressions
        regressions = results.get("regressions", [])
        if regressions:
            regression_details = "\n".join([r["message"] for r in regressions])
            pytest.fail(f"Performance regressions detected:\n{regression_details}")
        
        print(f"API endpoint performance: {len(results['metrics'])} metrics, {len(regressions)} regressions")
        
    @pytest.mark.asyncio
    async def test_agent_orchestration_regression(self, regression_tester):
        """Test agent orchestration performance regression."""
        results = await regression_tester.test_agent_orchestration_performance()
        
        # Should have metrics for different query complexities
        assert len(results["metrics"]) >= 6, "Missing orchestration performance metrics"
        
        # Check for regressions
        regressions = results.get("regressions", [])
        if regressions:
            regression_details = "\n".join([r["message"] for r in regressions])
            pytest.fail(f"Agent orchestration regressions detected:\n{regression_details}")
        
        print(f"Agent orchestration performance: {len(results['metrics'])} metrics, {len(regressions)} regressions")
        
    @pytest.mark.asyncio
    async def test_database_query_regression(self, regression_tester):
        """Test database query performance regression."""
        results = await regression_tester.test_database_query_performance()
        
        # Should have some database metrics (unless database is unavailable)
        if "error" not in results:
            assert len(results["metrics"]) > 0, "No database performance metrics collected"
            
            # Check for regressions
            regressions = results.get("regressions", [])
            if regressions:
                regression_details = "\n".join([r["message"] for r in regressions])
                pytest.fail(f"Database performance regressions detected:\n{regression_details}")
        else:
            pytest.skip(f"Database performance test skipped: {results['error']}")
        
        print(f"Database performance: {len(results.get('metrics', {}))} metrics, {len(results.get('regressions', []))} regressions")
        
    @pytest.mark.asyncio
    async def test_memory_usage_regression(self, regression_tester):
        """Test memory usage regression."""
        results = await regression_tester.test_memory_usage_performance()
        
        # Should have memory metrics
        assert len(results["metrics"]) >= 2, "Missing memory usage metrics"
        
        # Memory usage should be reasonable
        peak_memory = results["metrics"].get("peak_memory_increase_mb", 0)
        assert peak_memory < 500, f"Peak memory increase too high: {peak_memory}MB"
        
        # Check for regressions
        regressions = results.get("regressions", [])
        if regressions:
            regression_details = "\n".join([r["message"] for r in regressions])
            pytest.fail(f"Memory usage regressions detected:\n{regression_details}")
        
        print(f"Memory usage performance: {len(results['metrics'])} metrics, {len(regressions)} regressions")
        
    @pytest.mark.asyncio
    async def test_comprehensive_regression_suite(self, regression_tester):
        """Run comprehensive regression test suite."""
        all_results = await regression_tester.run_all_regression_tests()
        
        # Should have run multiple tests
        assert all_results["summary"]["total_tests"] >= 3, "Not enough regression tests executed"
        
        # Check overall regression status
        total_regressions = all_results["summary"]["total_regressions"]
        tests_with_regressions = all_results["summary"]["tests_with_regressions"]
        
        if total_regressions > 0:
            # Collect all regression messages
            all_regressions = []
            for test_name, test_results in all_results["tests"].items():
                for regression in test_results.get("regressions", []):
                    all_regressions.append(f"{test_name}: {regression['message']}")
            
            regression_summary = f"{total_regressions} regressions in {tests_with_regressions} tests:\n" + "\n".join(all_regressions)
            pytest.fail(f"Performance regressions detected:\n{regression_summary}")
        
        print(f"Comprehensive regression test: {all_results['summary']['total_tests']} tests, {total_regressions} regressions")


@pytest.mark.regression
class TestBaselineManagement:
    """Test baseline management functionality."""
    
    def test_baseline_creation_and_retrieval(self, tmp_path):
        """Test creating and retrieving performance baselines."""
        baseline_file = tmp_path / "test_baselines.json"
        baseline = PerformanceBaseline(baseline_file)
        
        # Set some baselines
        baseline.set_baseline("test_api", "response_time_ms", 100.0, 0.2)
        baseline.set_baseline("test_api", "throughput_rps", 50.0, 0.15)
        baseline.save_baselines()
        
        # Create new instance and verify baselines are loaded
        baseline2 = PerformanceBaseline(baseline_file)
        
        assert baseline2.get_baseline("test_api", "response_time_ms") == 100.0
        assert baseline2.get_baseline("test_api", "throughput_rps") == 50.0
        
    def test_regression_detection(self, tmp_path):
        """Test regression detection logic."""
        baseline_file = tmp_path / "test_baselines.json"
        baseline = PerformanceBaseline(baseline_file)
        
        # Set baseline: 100ms with 20% tolerance
        baseline.set_baseline("test", "response_time", 100.0, 0.2)
        
        # Test cases
        test_cases = [
            (110.0, False),  # 10% increase - within tolerance
            (120.0, False),  # 20% increase - at tolerance limit
            (125.0, True),   # 25% increase - regression
            (150.0, True),   # 50% increase - clear regression
            (90.0, False),   # Improvement - not a regression
        ]
        
        for value, should_be_regression in test_cases:
            is_regression, message = baseline.check_regression("test", "response_time", value)
            assert is_regression == should_be_regression, f"Failed for value {value}: {message}"


def establish_performance_baselines():
    """Utility function to establish performance baselines."""
    async def _establish():
        tester = PerformanceRegressionTester()
        
        print("Establishing performance baselines...")
        
        # Run all tests to collect baseline data
        all_results = await tester.run_all_regression_tests()
        
        # Update baselines from results
        for test_name, results in all_results["tests"].items():
            tester.update_baselines_from_results(results)
            print(f"Updated baselines for {test_name}: {len(results.get('metrics', {}))} metrics")
        
        print("Performance baselines established successfully!")
        
    return asyncio.run(_establish())


def run_regression_tests():
    """Run performance regression tests and generate report."""
    import subprocess
    import json
    from pathlib import Path
    
    # Run regression tests
    result = subprocess.run([
        "python", "-m", "pytest", 
        "tests/test_performance_regression.py", 
        "-m", "regression",
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
            "errors": result.stdout.count("ERROR"),
            "regressions_detected": result.returncode != 0
        }
    }
    
    # Save report
    report_file = Path(f"regression_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    report_file.write_text(json.dumps(report, indent=2))
    
    print(f"Regression test report saved to: {report_file}")
    return report


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "establish-baselines":
        establish_performance_baselines()
    else:
        run_regression_tests()