"""Scalability tests for large repository processing and complex queries."""

import asyncio
import time
import tempfile
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.code_intelligence.ingestion.pipeline import IngestionPipeline
from src.code_intelligence.git.repository import GitRepository
from src.code_intelligence.agents.orchestrator import AgentOrchestrator
from src.code_intelligence.agents.state import AgentState
from src.code_intelligence.database.neo4j_client import neo4j_client
from src.code_intelligence.monitoring.agent_monitor import agent_monitor


class ScalabilityTestResults:
    """Container for scalability test results."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.metrics: Dict[str, Any] = {}
        self.performance_data: List[Dict[str, Any]] = []
        self.memory_usage: List[float] = []
        self.errors: List[str] = []
        
    def add_metric(self, key: str, value: Any):
        """Add a metric."""
        self.metrics[key] = value
        
    def add_performance_point(self, **kwargs):
        """Add a performance data point."""
        self.performance_data.append({
            "timestamp": datetime.now().isoformat(),
            **kwargs
        })
        
    def add_memory_usage(self, usage_mb: float):
        """Add memory usage data point."""
        self.memory_usage.append(usage_mb)
        
    def add_error(self, error: str):
        """Add an error."""
        self.errors.append(error)
        
    def finalize(self):
        """Finalize the test results."""
        self.end_time = datetime.now()
        self.metrics["duration_seconds"] = (self.end_time - self.start_time).total_seconds()
        self.metrics["peak_memory_mb"] = max(self.memory_usage) if self.memory_usage else 0
        self.metrics["avg_memory_mb"] = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
        self.metrics["error_count"] = len(self.errors)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "test_name": self.test_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metrics": self.metrics,
            "performance_data": self.performance_data,
            "memory_usage": self.memory_usage,
            "errors": self.errors[:10]  # First 10 errors
        }


class LargeRepositoryGenerator:
    """Generate large test repositories for scalability testing."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        
    def create_large_python_repository(
        self, 
        num_files: int = 1000,
        num_commits: int = 500,
        functions_per_file: int = 10
    ) -> Path:
        """Create a large Python repository for testing."""
        repo_path = self.base_path / f"large_repo_{num_files}_{num_commits}"
        repo_path.mkdir(exist_ok=True)
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        
        # Create initial file structure
        for i in range(num_files):
            module_path = repo_path / f"module_{i:04d}.py"
            self._create_python_file(module_path, functions_per_file, i)
            
        # Create initial commit
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True, capture_output=True)
        
        # Create additional commits with modifications
        for commit_num in range(1, num_commits):
            # Modify some files
            files_to_modify = min(10, num_files // 10)  # Modify 10% of files or 10 files, whichever is smaller
            
            for j in range(files_to_modify):
                file_index = (commit_num * files_to_modify + j) % num_files
                module_path = repo_path / f"module_{file_index:04d}.py"
                self._modify_python_file(module_path, commit_num)
                
            subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", f"Commit {commit_num}: Modified {files_to_modify} files"],
                cwd=repo_path, check=True, capture_output=True
            )
            
        return repo_path
        
    def _create_python_file(self, file_path: Path, num_functions: int, module_index: int):
        """Create a Python file with specified number of functions."""
        content = f'''"""Module {module_index} - Generated for scalability testing."""

import os
import sys
import json
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

'''
        
        for func_index in range(num_functions):
            content += f'''
def function_{module_index:04d}_{func_index:02d}(param1: str, param2: int = 0) -> Dict[str, Any]:
    """Function {func_index} in module {module_index}.
    
    Args:
        param1: String parameter
        param2: Integer parameter with default value
        
    Returns:
        Dictionary with processed results
    """
    result = {{
        "module": {module_index},
        "function": {func_index},
        "param1": param1,
        "param2": param2,
        "timestamp": datetime.now().isoformat()
    }}
    
    # Simulate some processing
    for i in range(param2):
        result[f"processed_{{i}}"] = f"{{param1}}_{{i}}"
        
    return result


def helper_function_{module_index:04d}_{func_index:02d}(data: List[Any]) -> int:
    """Helper function for function_{module_index:04d}_{func_index:02d}."""
    return len([item for item in data if item is not None])

'''
        
        # Add a class
        content += f'''
class DataProcessor_{module_index:04d}:
    """Data processor class for module {module_index}."""
    
    def __init__(self, name: str):
        self.name = name
        self.processed_count = 0
        
    def process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a list of data items."""
        processed = []
        for item in data:
            processed_item = {{
                **item,
                "processed_by": self.name,
                "processed_at": datetime.now().isoformat(),
                "processor_module": {module_index}
            }}
            processed.append(processed_item)
            
        self.processed_count += len(processed)
        return processed
        
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {{
            "name": self.name,
            "processed_count": self.processed_count,
            "module": {module_index}
        }}
'''
        
        file_path.write_text(content)
        
    def _modify_python_file(self, file_path: Path, commit_num: int):
        """Modify an existing Python file."""
        if not file_path.exists():
            return
            
        content = file_path.read_text()
        
        # Add a new function
        new_function = f'''

def new_function_commit_{commit_num}(value: Any) -> str:
    """Function added in commit {commit_num}."""
    return f"Processed in commit {commit_num}: {{value}}"
'''
        
        content += new_function
        file_path.write_text(content)


class RepositoryScalabilityTester:
    """Test repository processing scalability."""
    
    def __init__(self):
        self.temp_dir = None
        self.repo_generator = None
        
    def setup(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.repo_generator = LargeRepositoryGenerator(self.temp_dir)
        
    def teardown(self):
        """Clean up test environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            
    async def test_large_repository_ingestion(
        self,
        num_files: int = 500,
        num_commits: int = 100
    ) -> ScalabilityTestResults:
        """Test ingestion of a large repository."""
        results = ScalabilityTestResults(f"large_repo_ingestion_{num_files}_{num_commits}")
        
        try:
            # Generate large repository
            print(f"Generating repository with {num_files} files and {num_commits} commits...")
            repo_path = self.repo_generator.create_large_python_repository(
                num_files=num_files,
                num_commits=num_commits,
                functions_per_file=5
            )
            results.add_metric("repository_path", str(repo_path))
            results.add_metric("num_files", num_files)
            results.add_metric("num_commits", num_commits)
            
            # Mock database clients to avoid actual database operations
            with patch('src.code_intelligence.ingestion.pipeline.neo4j_client') as mock_neo4j:
                mock_neo4j.execute_query = AsyncMock(return_value=[])
                
                # Create ingestion pipeline
                pipeline = IngestionPipeline(mock_neo4j, Mock())
                
                # Create git repository wrapper
                git_repo = GitRepository(str(repo_path))
                
                # Monitor memory usage
                import psutil
                import os
                process = psutil.Process(os.getpid())
                
                # Start ingestion
                ingestion_start = time.time()
                
                try:
                    # Get commit history
                    commits = git_repo.get_commit_history(max_commits=num_commits)
                    results.add_metric("actual_commits_found", len(commits))
                    
                    processed_commits = 0
                    for i, commit in enumerate(commits[:50]):  # Process first 50 commits for testing
                        commit_start = time.time()
                        
                        # Checkout commit
                        git_repo.checkout(commit.sha)
                        
                        # Parse files
                        parsed_files = pipeline.parser.parse_directory(str(repo_path))
                        
                        # Record performance data
                        commit_time = time.time() - commit_start
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        
                        results.add_performance_point(
                            commit_index=i,
                            commit_sha=commit.sha,
                            files_parsed=len(parsed_files),
                            processing_time_seconds=commit_time,
                            memory_mb=memory_mb
                        )
                        results.add_memory_usage(memory_mb)
                        
                        processed_commits += 1
                        
                        # Break if taking too long (for testing purposes)
                        if time.time() - ingestion_start > 300:  # 5 minutes max
                            break
                            
                    results.add_metric("processed_commits", processed_commits)
                    
                except Exception as e:
                    results.add_error(f"Ingestion error: {str(e)}")
                    
                ingestion_time = time.time() - ingestion_start
                results.add_metric("total_ingestion_time_seconds", ingestion_time)
                
        except Exception as e:
            results.add_error(f"Setup error: {str(e)}")
            
        results.finalize()
        return results
        
    async def test_complex_query_performance(
        self,
        repository_size: str = "medium"
    ) -> ScalabilityTestResults:
        """Test complex query performance on large datasets."""
        results = ScalabilityTestResults(f"complex_query_{repository_size}")
        
        # Define repository sizes
        size_configs = {
            "small": {"files": 100, "commits": 50},
            "medium": {"files": 500, "commits": 100},
            "large": {"files": 1000, "commits": 200}
        }
        
        config = size_configs.get(repository_size, size_configs["medium"])
        
        try:
            # Generate repository
            repo_path = self.repo_generator.create_large_python_repository(
                num_files=config["files"],
                num_commits=config["commits"]
            )
            
            # Mock orchestrator and agents
            with patch('src.code_intelligence.agents.orchestrator.neo4j_client') as mock_neo4j, \
                 patch('src.code_intelligence.agents.orchestrator.supabase_client') as mock_supabase:
                
                # Mock database responses
                mock_neo4j.execute_query = AsyncMock(return_value=[
                    {"name": f"function_{i}", "file_path": f"module_{i}.py", "start_line": i * 10}
                    for i in range(100)  # Mock 100 functions
                ])
                
                orchestrator = AgentOrchestrator()
                
                # Test complex queries
                complex_queries = [
                    "Find all functions that call database operations and were modified in the last 30 days",
                    "Identify circular dependencies between modules and their impact on performance",
                    "Analyze code complexity trends over the last 50 commits",
                    "Find functions with high cyclomatic complexity that haven't been tested",
                    "Trace the evolution of the authentication system across all commits"
                ]
                
                for i, query in enumerate(complex_queries):
                    query_start = time.time()
                    
                    try:
                        # Create initial state
                        initial_state = AgentState(
                            session_id=f"scalability_test_{i}",
                            query={"original": query},
                            repository={"path": str(repo_path)}
                        )
                        
                        # Execute query (with timeout)
                        result = await asyncio.wait_for(
                            orchestrator.execute_query(query, str(repo_path)),
                            timeout=60.0  # 1 minute timeout per query
                        )
                        
                        query_time = time.time() - query_start
                        
                        # Record results
                        results.add_performance_point(
                            query_index=i,
                            query=query[:50] + "...",
                            processing_time_seconds=query_time,
                            findings_count=len(result.get_all_findings()),
                            success=not result.has_errors()
                        )
                        
                        results.add_metric(f"query_{i}_time", query_time)
                        results.add_metric(f"query_{i}_findings", len(result.get_all_findings()))
                        
                    except asyncio.TimeoutError:
                        results.add_error(f"Query {i} timed out after 60 seconds")
                        results.add_performance_point(
                            query_index=i,
                            query=query[:50] + "...",
                            processing_time_seconds=60.0,
                            findings_count=0,
                            success=False,
                            error="timeout"
                        )
                    except Exception as e:
                        results.add_error(f"Query {i} failed: {str(e)}")
                        
        except Exception as e:
            results.add_error(f"Test setup error: {str(e)}")
            
        results.finalize()
        return results


@pytest.mark.scalability
class TestRepositoryScalability:
    """Scalability tests for repository processing."""
    
    @pytest.fixture
    def scalability_tester(self):
        """Create scalability tester instance."""
        tester = RepositoryScalabilityTester()
        tester.setup()
        yield tester
        tester.teardown()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_medium_repository_ingestion(self, scalability_tester):
        """Test ingestion of medium-sized repository."""
        results = await scalability_tester.test_large_repository_ingestion(
            num_files=200,
            num_commits=50
        )
        
        # Performance assertions
        assert results.metrics["error_count"] == 0, f"Ingestion had {results.metrics['error_count']} errors"
        assert results.metrics["total_ingestion_time_seconds"] < 300, "Ingestion took too long (>5 minutes)"
        assert results.metrics["peak_memory_mb"] < 1000, f"Peak memory usage too high: {results.metrics['peak_memory_mb']}MB"
        
        # Should process reasonable number of commits
        assert results.metrics["processed_commits"] >= 10, "Too few commits processed"
        
        print(f"Medium repository ingestion results: {results.to_dict()}")
        
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_large_repository_ingestion(self, scalability_tester):
        """Test ingestion of large repository."""
        results = await scalability_tester.test_large_repository_ingestion(
            num_files=500,
            num_commits=100
        )
        
        # More lenient assertions for large repositories
        assert results.metrics["error_count"] <= 2, f"Too many ingestion errors: {results.metrics['error_count']}"
        assert results.metrics["total_ingestion_time_seconds"] < 600, "Ingestion took too long (>10 minutes)"
        assert results.metrics["peak_memory_mb"] < 2000, f"Peak memory usage too high: {results.metrics['peak_memory_mb']}MB"
        
        print(f"Large repository ingestion results: {results.to_dict()}")
        
    @pytest.mark.asyncio
    async def test_complex_query_performance_medium(self, scalability_tester):
        """Test complex query performance on medium repository."""
        results = await scalability_tester.test_complex_query_performance("medium")
        
        # Query performance assertions
        successful_queries = len([p for p in results.performance_data if p.get("success", False)])
        assert successful_queries >= 3, f"Too few successful queries: {successful_queries}/5"
        
        # Average query time should be reasonable
        query_times = [p["processing_time_seconds"] for p in results.performance_data if "processing_time_seconds" in p]
        if query_times:
            avg_query_time = sum(query_times) / len(query_times)
            assert avg_query_time < 30, f"Average query time too high: {avg_query_time}s"
        
        print(f"Complex query performance results: {results.to_dict()}")
        
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_usage_scaling(self, scalability_tester):
        """Test memory usage scaling with repository size."""
        sizes = ["small", "medium"]
        memory_results = {}
        
        for size in sizes:
            results = await scalability_tester.test_complex_query_performance(size)
            memory_results[size] = {
                "peak_memory_mb": results.metrics.get("peak_memory_mb", 0),
                "avg_memory_mb": results.metrics.get("avg_memory_mb", 0)
            }
        
        # Memory usage should scale reasonably
        small_peak = memory_results["small"]["peak_memory_mb"]
        medium_peak = memory_results["medium"]["peak_memory_mb"]
        
        # Medium should use more memory than small, but not excessively
        if small_peak > 0:
            memory_ratio = medium_peak / small_peak
            assert memory_ratio < 5, f"Memory usage scaling too high: {memory_ratio}x"
        
        print(f"Memory scaling results: {memory_results}")


@pytest.mark.scalability
class TestSystemScalability:
    """System-wide scalability tests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_repository_processing(self):
        """Test processing multiple repositories concurrently."""
        results = ScalabilityTestResults("concurrent_repository_processing")
        
        async def process_repository(repo_id: str):
            """Process a single repository."""
            start_time = time.time()
            try:
                # Simulate repository processing
                await asyncio.sleep(0.5)  # Simulate processing time
                
                processing_time = time.time() - start_time
                return processing_time, True, None
                
            except Exception as e:
                processing_time = time.time() - start_time
                return processing_time, False, str(e)
        
        # Process multiple repositories concurrently
        num_repos = 10
        tasks = [process_repository(f"repo_{i}") for i in range(num_repos)]
        
        completed_tasks = 0
        total_time = 0
        
        for task in asyncio.as_completed(tasks):
            processing_time, success, error = await task
            total_time += processing_time
            
            if success:
                completed_tasks += 1
            else:
                results.add_error(error or "Unknown error")
                
            results.add_performance_point(
                processing_time_seconds=processing_time,
                success=success
            )
        
        results.add_metric("completed_repositories", completed_tasks)
        results.add_metric("total_repositories", num_repos)
        results.add_metric("success_rate", completed_tasks / num_repos)
        results.add_metric("avg_processing_time", total_time / num_repos)
        
        results.finalize()
        
        # Concurrent processing should be efficient
        assert results.metrics["success_rate"] >= 0.9, "Too many repository processing failures"
        assert results.metrics["duration_seconds"] < 2.0, "Concurrent processing took too long"
        
        print(f"Concurrent repository processing results: {results.to_dict()}")


def run_scalability_tests():
    """Run all scalability tests and generate a report."""
    import subprocess
    import json
    from pathlib import Path
    
    # Run scalability tests
    result = subprocess.run([
        "python", "-m", "pytest", 
        "tests/test_scalability.py", 
        "-m", "scalability",
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
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
            "skipped": result.stdout.count("SKIPPED")
        }
    }
    
    # Save report
    report_file = Path(f"scalability_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    report_file.write_text(json.dumps(report, indent=2))
    
    print(f"Scalability test report saved to: {report_file}")
    return report


if __name__ == "__main__":
    # Run scalability tests when executed directly
    run_scalability_tests()