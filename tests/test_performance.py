"""Performance tests for ingestion pipeline."""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime

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