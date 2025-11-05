"""Performance tests for ingestion pipeline and API load testing."""

import pytest
import time
import asyncio
import concurrent.futures
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
import httpx
import json
from typing import List, Dict, Any

from src.code_intelligence.ingestion.pipeline import IngestionPipeline
from src.code_intelligence.git.models import CommitInfo


class TestPerformance:
    """Performance tests for the ingestion system."""
    
    @pytest.mark.performance
    def test_parsing_performance(self, temp_dir):
        """Test parsing performance with multiple files."""
        from src.code_intelligence.parsing.parser import MultiLanguageParser
        
        # Create multiple test files
        python_code = '''
def function_{i}():
    """Function {i}."""
    return {i} * 2

class Class_{i}:
    """Class {i}."""
    def method_{i}(self):
        return function_{i}()
'''
        
        # Create 50 Python files
        for i in range(50):
            file_path = temp_dir / f"module_{i}.py"
            file_path.write_text(python_code.format(i=i))
        
        parser = MultiLanguageParser()
        
        start_time = time.time()
        parsed_files = parser.parse_directory(str(temp_dir))
        end_time = time.time()
        
        parsing_time = end_time - start_time
        
        # Verify results
        assert len(parsed_files) == 50
        total_elements = sum(len(f.elements) for f in parsed_files)
        assert total_elements > 0
        
        # Performance assertion (should parse 50 files in under 10 seconds)
        assert parsing_time < 10.0, f"Parsing took {parsing_time:.2f}s, expected < 10s"
        
        # Log performance metrics
        print(f"Parsed {len(parsed_files)} files with {total_elements} elements in {parsing_time:.2f}s")
        print(f"Average: {parsing_time/len(parsed_files):.3f}s per file")
    
    @pytest.mark.performance
    def test_graph_population_performance(self, mock_neo4j_client):
        """Test graph population performance with large batches."""
        from src.code_intelligence.ingestion.graph_populator import GraphPopulator
        from src.code_intelligence.parsing.models import ParsedFile, FunctionElement
        
        populator = GraphPopulator(mock_neo4j_client)
        
        # Create mock parsed files with many elements
        parsed_files = []
        for i in range(10):  # 10 files
            elements = []
            for j in range(100):  # 100 functions per file
                elements.append(FunctionElement(
                    name=f"function_{j}",
                    file_path=f"file_{i}.py",
                    start_line=j * 5,
                    end_line=j * 5 + 3,
                    language="python",
                    parameters=[f"param_{k}" for k in range(3)]
                ))
            
            parsed_files.append(ParsedFile(
                file_path=f"file_{i}.py",
                language="python",
                elements=elements
            ))
        
        # Mock successful batch operations
        mock_neo4j_client.execute_query.return_value = [{'created': 100}]
        
        start_time = time.time()
        stats = populator.ingest_parsed_files(parsed_files, 'commit123', 'test_repo')
        end_time = time.time()
        
        ingestion_time = end_time - start_time
        
        # Verify results
        assert stats['nodes'] > 0
        assert stats['relationships'] > 0
        
        # Performance assertion (should handle 1000 elements quickly)
        total_elements = sum(len(f.elements) for f in parsed_files)
        assert ingestion_time < 5.0, f"Graph population took {ingestion_time:.2f}s, expected < 5s"
        
        print(f"Ingested {total_elements} elements in {ingestion_time:.2f}s")
        print(f"Rate: {total_elements/ingestion_time:.0f} elements/second")
    
    @pytest.mark.performance
    def test_commit_history_processing(self, mock_neo4j_client, repository_manager):
        """Test performance with large commit histories."""
        pipeline = IngestionPipeline(mock_neo4j_client, repository_manager)
        
        # Mock a repository with many commits
        mock_repo = Mock()
        mock_repo.repo_path = "/mock/path"
        mock_repo.remote_url = "https://github.com/test/repo.git"
        mock_repo.get_supported_languages.return_value = {'python'}
        
        # Create mock commit history (500 commits)
        mock_commits = []
        for i in range(500):
            commit = CommitInfo(
                sha=f'commit_{i:04d}',
                message=f'Commit {i}',
                author_name='Test Author',
                author_email='test@example.com',
                committer_name='Test Author',
                committer_email='test@example.com',
                authored_date=datetime.now(),
                committed_date=datetime.now(),
                parents=[f'parent_{i-1:04d}'] if i > 0 else [],
                stats={'insertions': 10, 'deletions': 5, 'files': 2}
            )
            mock_commits.append(commit)
        
        mock_repo.get_commit_history.return_value = mock_commits
        mock_repo.checkout.return_value = None
        
        # Mock parsing to return minimal results
        with patch.object(pipeline.parser, 'parse_directory') as mock_parse:
            mock_parse.return_value = []  # No files to parse for speed
            
            repository_manager._repositories['test_repo'] = mock_repo
            
            start_time = time.time()
            
            # Create a job and execute repository ingestion
            from src.code_intelligence.ingestion.models import IngestionJob
            job = IngestionJob(
                id='perf_test',
                repository_id='test_repo',
                repository_path='/mock/path',
                max_commits=500
            )
            
            pipeline._execute_repository_ingestion(job, mock_repo)
            
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            # Verify job completion
            assert job.status.value == 'completed'
            assert job.processed_commits == 500
            
            # Performance assertion (should process 500 commits in reasonable time)
            assert processing_time < 30.0, f"Commit processing took {processing_time:.2f}s, expected < 30s"
            
            print(f"Processed {job.processed_commits} commits in {processing_time:.2f}s")
            print(f"Rate: {job.processed_commits/processing_time:.1f} commits/second")
    
    @pytest.mark.performance
    def test_memory_usage(self, temp_dir):
        """Test memory usage during parsing."""
        import psutil
        import os
        
        from src.code_intelligence.parsing.parser import MultiLanguageParser
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large files to test memory efficiency
        large_python_code = '''
import os
import sys
from typing import List, Dict, Optional, Any

''' + '\n'.join([f'''
def function_{i}(param1: str, param2: int, param3: Optional[List[str]] = None) -> Dict[str, Any]:
    """Function {i} with comprehensive docstring.
    
    Args:
        param1: String parameter
        param2: Integer parameter  
        param3: Optional list parameter
        
    Returns:
        Dictionary with results
    """
    result = {{}}
    for j in range(param2):
        result[f"key_{{j}}"] = f"{{param1}}_{{j}}"
    
    if param3:
        result["param3_length"] = len(param3)
    
    return result

class Class_{i}:
    """Class {i} for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.data = []
    
    def method_{i}(self, value: Any) -> None:
        """Method {i}."""
        self.data.append(value)
        
    def get_data(self) -> List[Any]:
        """Get stored data."""
        return self.data.copy()
''' for i in range(20)])  # 20 functions and classes per file
        
        # Create 10 large files
        for i in range(10):
            file_path = temp_dir / f"large_module_{i}.py"
            file_path.write_text(large_python_code)
        
        parser = MultiLanguageParser()
        
        # Parse files and monitor memory
        parsed_files = parser.parse_directory(str(temp_dir))
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Verify parsing worked
        assert len(parsed_files) == 10
        total_elements = sum(len(f.elements) for f in parsed_files)
        assert total_elements > 0
        
        # Memory usage should be reasonable (less than 500MB increase)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase:.1f}MB, expected < 500MB"
        
        print(f"Parsed {total_elements} elements from {len(parsed_files)} files")
        print(f"Memory usage: {initial_memory:.1f}MB -> {peak_memory:.1f}MB (+{memory_increase:.1f}MB)")
    
    @pytest.mark.performance  
    def test_concurrent_parsing(self, temp_dir):
        """Test concurrent parsing performance."""
        import concurrent.futures
        from src.code_intelligence.parsing.parser import CodeParser
        
        # Create test files
        test_code = '''
def test_function():
    """Test function."""
    return "test"

class TestClass:
    """Test class."""
    def method(self):
        return test_function()
'''
        
        files = []
        for i in range(20):
            file_path = temp_dir / f"concurrent_{i}.py"
            file_path.write_text(test_code)
            files.append(str(file_path))
        
        parser = CodeParser()
        
        # Sequential parsing
        start_time = time.time()
        sequential_results = []
        for file_path in files:
            result = parser.parse_file(file_path)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Concurrent parsing (simulate with threading)
        start_time = time.time()
        concurrent_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(parser.parse_file, file_path): file_path 
                            for file_path in files}
            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()
                concurrent_results.append(result)
        concurrent_time = time.time() - start_time
        
        # Verify results are equivalent
        assert len(sequential_results) == len(concurrent_results) == 20
        
        # Concurrent should be faster (or at least not much slower due to GIL)
        speedup = sequential_time / concurrent_time
        
        print(f"Sequential: {sequential_time:.2f}s, Concurrent: {concurrent_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")
        
        # Even with Python's GIL, we shouldn't be much slower
        assert concurrent_time < sequential_time * 1.5, "Concurrent parsing is significantly slower"


@pytest.mark.performance
class TestResourceConsumption:
    """Test resource consumption and memory usage patterns."""
    
    def test_memory_usage_during_large_file_parsing(self, temp_dir):
        """Test memory usage when parsing large files."""
        import psutil
        import os
        from src.code_intelligence.parsing.parser import MultiLanguageParser
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create a very large Python file
        large_code = '''
import os
import sys
import json
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta

''' + '\n'.join([f'''
def large_function_{i}(param1: str, param2: int, param3: Optional[List[str]] = None) -> Dict[str, Any]:
    """Large function {i} for memory testing."""
    result = {{}}
    for j in range(param2):
        result[f"key_{{j}}"] = f"{{param1}}_{{j}}"
        
    if param3:
        result["param3_data"] = [item.upper() for item in param3]
        
    # Simulate complex processing
    nested_data = {{}}
    for k in range(10):
        nested_data[f"level_{{k}}"] = {{
            f"item_{{m}}": f"value_{{i}}_{{k}}_{{m}}" for m in range(20)
        }}
    result["nested"] = nested_data
    
    return result

class LargeClass_{i}:
    """Large class {i} for memory testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.data = []
        self.metadata = {{
            "created": datetime.now().isoformat(),
            "class_id": {i},
            "large_data": [f"item_{{j}}" for j in range(100)]
        }}
        
    def process_large_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process large dataset."""
        processed = []
        for item in dataset:
            processed_item = {{
                **item,
                "processed_by": self.name,
                "class_id": {i},
                "processing_metadata": {{
                    "timestamp": datetime.now().isoformat(),
                    "processor_version": "1.0.0",
                    "additional_data": [f"meta_{{k}}" for k in range(50)]
                }}
            }}
            processed.append(processed_item)
            
        return processed
''' for i in range(200)])  # 200 functions and classes
        
        large_file = temp_dir / "large_file.py"
        large_file.write_text(large_code)
        
        parser = MultiLanguageParser()
        
        # Monitor memory during parsing
        memory_samples = []
        
        # Parse the large file
        start_time = time.time()
        parsed_files = parser.parse_directory(str(temp_dir))
        parsing_time = time.time() - start_time
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Verify parsing worked
        assert len(parsed_files) == 1
        parsed_file = parsed_files[0]
        assert len(parsed_file.elements) >= 400  # Should have 200 functions + 200 classes
        
        # Memory usage assertions
        assert memory_increase < 200, f"Memory usage increased by {memory_increase:.1f}MB, expected < 200MB"
        assert parsing_time < 30, f"Parsing took {parsing_time:.1f}s, expected < 30s"
        
        print(f"Large file parsing: {len(parsed_file.elements)} elements")
        print(f"Memory usage: {initial_memory:.1f}MB -> {peak_memory:.1f}MB (+{memory_increase:.1f}MB)")
        print(f"Parsing time: {parsing_time:.2f}s")
        
    @pytest.mark.asyncio
    async def test_memory_usage_during_agent_execution(self):
        """Test memory usage during agent execution."""
        import psutil
        import os
        from src.code_intelligence.agents.orchestrator import AgentOrchestrator
        from unittest.mock import patch, AsyncMock
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_samples = []
        
        # Mock database clients to avoid external dependencies
        with patch('src.code_intelligence.agents.orchestrator.neo4j_client') as mock_neo4j, \
             patch('src.code_intelligence.agents.orchestrator.supabase_client') as mock_supabase:
            
            mock_neo4j.execute_query = AsyncMock(return_value=[])
            mock_supabase.get_cached_result = AsyncMock(return_value=None)
            
            orchestrator = AgentOrchestrator()
            
            # Execute multiple queries and monitor memory
            queries = [
                "What functions were modified in the last week?",
                "Find all classes that implement authentication logic.",
                "Analyze the complexity of database access patterns.",
                "Identify potential security vulnerabilities in user input handling.",
                "Trace the evolution of the API endpoints over time."
            ]
            
            for i, query in enumerate(queries):
                # Sample memory before query
                pre_query_memory = process.memory_info().rss / 1024 / 1024
                memory_samples.append(("before_query", i, pre_query_memory))
                
                # Execute query
                try:
                    result = await asyncio.wait_for(
                        orchestrator.execute_query(query, "/mock/repo/path"),
                        timeout=10.0
                    )
                    
                    # Sample memory after query
                    post_query_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(("after_query", i, post_query_memory))
                    
                except asyncio.TimeoutError:
                    print(f"Query {i} timed out")
                except Exception as e:
                    print(f"Query {i} failed: {e}")
        
        # Analyze memory usage
        final_memory = process.memory_info().rss / 1024 / 1024
        total_memory_increase = final_memory - initial_memory
        
        # Calculate memory increase per query
        if len(memory_samples) >= 2:
            max_memory = max(sample[2] for sample in memory_samples)
            peak_memory_increase = max_memory - initial_memory
        else:
            peak_memory_increase = total_memory_increase
        
        # Memory usage should be reasonable
        assert total_memory_increase < 100, f"Total memory increase too high: {total_memory_increase:.1f}MB"
        assert peak_memory_increase < 150, f"Peak memory increase too high: {peak_memory_increase:.1f}MB"
        
        print(f"Agent execution memory usage:")
        print(f"Initial: {initial_memory:.1f}MB, Final: {final_memory:.1f}MB")
        print(f"Total increase: {total_memory_increase:.1f}MB, Peak increase: {peak_memory_increase:.1f}MB")
        
    def test_connection_pool_resource_usage(self):
        """Test connection pool resource usage."""
        import psutil
        import os
        from src.code_intelligence.core.connection_pool import ConnectionPoolManager
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Mock client class
        class MockClient:
            def __init__(self, **kwargs):
                self.data = [f"data_{i}" for i in range(1000)]  # Some memory usage
                self.connected = False
                
            async def connect(self):
                self.connected = True
                
            async def health_check(self):
                return self.connected
                
            async def close(self):
                self.connected = False
                self.data = []
        
        pool_manager = ConnectionPoolManager()
        
        # Create multiple pools
        pools = []
        for i in range(5):
            pool = pool_manager.get_pool(f"test_pool_{i}", MockClient, max_connections=10)
            pools.append(pool)
        
        # Use connections from all pools
        async def use_pools():
            tasks = []
            for pool in pools:
                for _ in range(5):  # 5 connections per pool
                    async def use_connection(p=pool):
                        async with p.get_connection() as conn:
                            await asyncio.sleep(0.01)  # Simulate work
                    tasks.append(use_connection())
            
            await asyncio.gather(*tasks)
        
        # Run the connection usage
        asyncio.run(use_pools())
        
        # Check memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory
        
        # Clean up
        asyncio.run(pool_manager.close_all_pools())
        
        # Memory usage should be reasonable for connection pools
        assert memory_increase < 50, f"Connection pool memory usage too high: {memory_increase:.1f}MB"
        
        print(f"Connection pool resource usage: {memory_increase:.1f}MB increase")
        
    @pytest.mark.asyncio
    async def test_cache_system_memory_usage(self):
        """Test cache system memory usage."""
        import psutil
        import os
        from src.code_intelligence.caching.cache_manager import cache_manager
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        # Store many items in cache
        large_result_data = {
            "findings": [
                {
                    "content": f"Large finding {i} with lots of content " * 100,
                    "metadata": {f"key_{j}": f"value_{i}_{j}" for j in range(50)},
                    "citations": [f"file_{k}.py" for k in range(10)]
                }
                for i in range(20)  # 20 findings per result
            ],
            "analysis": {f"analysis_key_{i}": f"analysis_value_{i}" * 50 for i in range(100)}
        }
        
        # Store multiple large results
        for i in range(50):
            await cache_manager.store_result(
                query=f"Large query {i} with complex analysis requirements",
                repository_id=f"repo_{i}",
                result_data=large_result_data,
                confidence_score=0.8 + (i % 20) * 0.01
            )
        
        # Check memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = peak_memory - initial_memory
        
        # Retrieve some cached results
        retrieved_count = 0
        for i in range(0, 50, 5):  # Every 5th item
            result = await cache_manager.get_cached_result(
                query=f"Large query {i} with complex analysis requirements",
                repository_id=f"repo_{i}"
            )
            if result:
                retrieved_count += 1
        
        # Memory usage should be reasonable even with large cache
        assert memory_increase < 200, f"Cache system memory usage too high: {memory_increase:.1f}MB"
        
        print(f"Cache system memory usage: {memory_increase:.1f}MB increase")
        print(f"Retrieved {retrieved_count} cached results")
        
    def test_file_handle_usage(self, temp_dir):
        """Test file handle usage during parsing."""
        import psutil
        import os
        from src.code_intelligence.parsing.parser import MultiLanguageParser
        
        process = psutil.Process(os.getpid())
        initial_handles = process.num_fds() if hasattr(process, 'num_fds') else 0
        
        # Create many files
        for i in range(100):
            file_path = temp_dir / f"file_{i:03d}.py"
            file_path.write_text(f'''
def function_{i}():
    """Function {i}."""
    return {i}

class Class_{i}:
    """Class {i}."""
    def method(self):
        return function_{i}()
''')
        
        parser = MultiLanguageParser()
        
        # Parse all files
        parsed_files = parser.parse_directory(str(temp_dir))
        
        # Check file handle usage
        final_handles = process.num_fds() if hasattr(process, 'num_fds') else 0
        handle_increase = final_handles - initial_handles if initial_handles > 0 else 0
        
        # Verify parsing worked
        assert len(parsed_files) == 100
        
        # File handles should not leak
        if initial_handles > 0:
            assert handle_increase <= 10, f"Too many file handles opened: {handle_increase}"
        
        print(f"File handle usage: {initial_handles} -> {final_handles} (+{handle_increase})")
        print(f"Parsed {len(parsed_files)} files successfully")

@pyt
est.mark.load_test
class TestAPILoadTesting:
    """Load testing for API endpoints with concurrent requests."""
    
    @pytest.fixture
    def api_base_url(self):
        """Get API base URL for testing."""
        return "http://localhost:8000/api/v1"
    
    @pytest.fixture
    def mock_api_server(self):
        """Mock API server for load testing."""
        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            
            # Mock successful query submission
            mock_instance.post.return_value.status_code = 200
            mock_instance.post.return_value.json.return_value = {
                "query_id": "test-query-123",
                "status": "pending",
                "message": "Query submitted successfully"
            }
            
            # Mock query status check
            mock_instance.get.return_value.status_code = 200
            mock_instance.get.return_value.json.return_value = {
                "query_id": "test-query-123",
                "status": "completed",
                "results": {"summary": "Test results"}
            }
            
            yield mock_instance
    
    @pytest.mark.asyncio
    async def test_concurrent_query_submissions(self, api_base_url, mock_api_server):
        """Test concurrent query submissions to /api/v1/queries/ endpoint."""
        concurrent_requests = 50
        
        async def submit_query(session_id: int) -> Dict[str, Any]:
            """Submit a single query request."""
            async with httpx.AsyncClient(base_url=api_base_url, timeout=30.0) as client:
                query_data = {
                    "repository_url": f"https://github.com/test/repo-{session_id}.git",
                    "query": f"What changed in session {session_id}?",
                    "options": {
                        "max_commits": 50,
                        "include_tests": False
                    }
                }
                
                start_time = time.time()
                try:
                    response = await client.post("/queries/", json=query_data)
                    end_time = time.time()
                    
                    return {
                        "session_id": session_id,
   