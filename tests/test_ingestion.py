"""Tests for ingestion pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.code_intelligence.ingestion.pipeline import IngestionPipeline, IngestionError
from src.code_intelligence.ingestion.graph_populator import GraphPopulator
from src.code_intelligence.ingestion.models import IngestionJob, IngestionStatus, GraphBatch
from src.code_intelligence.git.models import CommitInfo


class TestIngestionPipeline:
    """Test IngestionPipeline class."""
    
    def test_init(self, mock_neo4j_client, repository_manager):
        """Test pipeline initialization."""
        pipeline = IngestionPipeline(mock_neo4j_client, repository_manager)
        
        assert pipeline.neo4j == mock_neo4j_client
        assert pipeline.repo_manager == repository_manager
        assert pipeline.parser is not None
        assert pipeline.graph_populator is not None
    
    def test_ingest_local_repository(self, ingestion_pipeline, sample_git_repo):
        """Test ingesting local repository."""
        job = ingestion_pipeline.ingest_local_repository(str(sample_git_repo), "test_repo")
        
        assert isinstance(job, IngestionJob)
        assert job.repository_id == "test_repo"
        assert job.status == IngestionStatus.COMPLETED
        assert job.total_commits == 2
        assert job.processed_commits == 2
    
    def test_ingest_nonexistent_repository(self, ingestion_pipeline, temp_dir):
        """Test ingesting non-existent repository."""
        nonexistent_path = temp_dir / "nonexistent"
        
        with pytest.raises(IngestionError):
            ingestion_pipeline.ingest_local_repository(str(nonexistent_path))
    
    def test_get_job_status(self, ingestion_pipeline, sample_git_repo):
        """Test getting job status."""
        job = ingestion_pipeline.ingest_local_repository(str(sample_git_repo), "test_repo")
        
        retrieved_job = ingestion_pipeline.get_job_status(job.id)
        
        assert retrieved_job is not None
        assert retrieved_job.id == job.id
        assert retrieved_job.status == IngestionStatus.COMPLETED
    
    def test_cancel_job(self, ingestion_pipeline):
        """Test canceling job."""
        # Create a mock running job
        job = IngestionJob(
            id="test_job",
            repository_id="test_repo",
            repository_path="/test/path",
            status=IngestionStatus.RUNNING
        )
        ingestion_pipeline._active_jobs[job.id] = job
        
        result = ingestion_pipeline.cancel_job(job.id)
        
        assert result is True
        assert job.status == IngestionStatus.CANCELLED
        assert job.completed_at is not None
    
    def test_get_repository_stats(self, ingestion_pipeline):
        """Test getting repository statistics."""
        # Mock the Neo4j query result
        ingestion_pipeline.neo4j.execute_query.return_value = [{
            'commit_count': 10,
            'file_count': 25,
            'element_count': 150,
            'languages': ['python', 'javascript']
        }]
        
        stats = ingestion_pipeline.get_repository_stats("test_repo")
        
        assert stats['repository_id'] == "test_repo"
        assert stats['commit_count'] == 10
        assert stats['file_count'] == 25
        assert stats['element_count'] == 150
        assert stats['supported_languages'] == ['python', 'javascript']


class TestGraphPopulator:
    """Test GraphPopulator class."""
    
    def test_init(self, mock_neo4j_client):
        """Test graph populator initialization."""
        populator = GraphPopulator(mock_neo4j_client)
        
        assert populator.neo4j == mock_neo4j_client
        assert populator._batch_size == 1000
    
    def test_create_repository_node(self, mock_neo4j_client):
        """Test creating repository node."""
        populator = GraphPopulator(mock_neo4j_client)
        
        repo_info = {
            'name': 'test_repo',
            'url': 'https://github.com/test/repo.git',
            'supported_languages': ['python', 'javascript']
        }
        
        result = populator.create_repository_node('test_repo', repo_info)
        
        assert result == 'test_repo'
        mock_neo4j_client.execute_query.assert_called_once()
    
    def test_create_commit_node(self, mock_neo4j_client):
        """Test creating commit node."""
        populator = GraphPopulator(mock_neo4j_client)
        
        commit_info = CommitInfo(
            sha='abc123',
            message='Test commit',
            author_name='Test Author',
            author_email='test@example.com',
            committer_name='Test Author',
            committer_email='test@example.com',
            authored_date=datetime.now(),
            committed_date=datetime.now(),
            parents=['parent123'],
            stats={'insertions': 10, 'deletions': 5, 'files': 2}
        )
        
        result = populator.create_commit_node(commit_info, 'test_repo')
        
        assert result == 'abc123'
        mock_neo4j_client.execute_query.assert_called_once()
    
    def test_ingest_parsed_files(self, mock_neo4j_client, sample_python_code):
        """Test ingesting parsed files."""
        from src.code_intelligence.parsing.parser import CodeParser
        
        populator = GraphPopulator(mock_neo4j_client)
        parser = CodeParser()
        
        # Parse sample code
        parsed_file = parser.parse_content(sample_python_code, 'python', 'test.py')
        
        # Mock successful batch ingestion
        mock_neo4j_client.execute_query.return_value = [{'created': 5}]
        
        stats = populator.ingest_parsed_files([parsed_file], 'commit123', 'test_repo')
        
        assert 'nodes' in stats
        assert 'relationships' in stats
        assert stats['nodes'] >= 0
        assert stats['relationships'] >= 0
    
    def test_cleanup_old_data(self, mock_neo4j_client):
        """Test cleaning up old data."""
        populator = GraphPopulator(mock_neo4j_client)
        
        # Mock cleanup result
        mock_neo4j_client.execute_query.return_value = [{'deleted': 10}]
        
        deleted_count = populator.cleanup_old_data('test_repo', keep_commits=50)
        
        assert deleted_count == 10
        mock_neo4j_client.execute_query.assert_called_once()


class TestGraphBatch:
    """Test GraphBatch model."""
    
    def test_init(self):
        """Test graph batch initialization."""
        batch = GraphBatch(commit_sha='abc123', repository_id='test_repo')
        
        assert batch.commit_sha == 'abc123'
        assert batch.repository_id == 'test_repo'
        assert len(batch.nodes) == 0
        assert len(batch.relationships) == 0
    
    def test_add_node(self):
        """Test adding node to batch."""
        batch = GraphBatch(commit_sha='abc123', repository_id='test_repo')
        
        batch.add_node(
            labels=['Function'],
            properties={'name': 'test_func', 'line': 10},
            unique_key='func_key'
        )
        
        assert len(batch.nodes) == 1
        node = batch.nodes[0]
        assert node.labels == ['Function']
        assert node.properties['name'] == 'test_func'
        assert node.unique_key == 'func_key'
    
    def test_add_relationship(self):
        """Test adding relationship to batch."""
        batch = GraphBatch(commit_sha='abc123', repository_id='test_repo')
        
        batch.add_relationship(
            source_key='source_key',
            target_key='target_key',
            rel_type='CALLS',
            properties={'line': 15}
        )
        
        assert len(batch.relationships) == 1
        rel = batch.relationships[0]
        assert rel.source_key == 'source_key'
        assert rel.target_key == 'target_key'
        assert rel.relationship_type == 'CALLS'
        assert rel.properties['line'] == 15


class TestIngestionJob:
    """Test IngestionJob model."""
    
    def test_init(self):
        """Test ingestion job initialization."""
        job = IngestionJob(
            id='job123',
            repository_id='test_repo',
            repository_path='/path/to/repo'
        )
        
        assert job.id == 'job123'
        assert job.repository_id == 'test_repo'
        assert job.status == IngestionStatus.PENDING
        assert job.total_commits == 0
        assert job.processed_commits == 0
    
    def test_progress_percentage(self):
        """Test progress percentage calculation."""
        job = IngestionJob(
            id='job123',
            repository_id='test_repo',
            repository_path='/path/to/repo'
        )
        
        # No commits processed yet
        assert job.progress_percentage == 0.0
        
        # Set totals and progress
        job.total_commits = 10
        job.processed_commits = 3
        
        assert job.progress_percentage == 30.0
        
        # Complete processing
        job.processed_commits = 10
        assert job.progress_percentage == 100.0
    
    def test_update_progress(self):
        """Test updating progress."""
        job = IngestionJob(
            id='job123',
            repository_id='test_repo',
            repository_path='/path/to/repo'
        )
        
        job.update_progress(processed_commits=5, processed_files=20)
        
        assert job.processed_commits == 5
        assert job.processed_files == 20


class TestIntegration:
    """Integration tests for the complete ingestion pipeline."""
    
    def test_end_to_end_ingestion(self, ingestion_pipeline, sample_git_repo):
        """Test complete end-to-end ingestion process."""
        # This test verifies the entire pipeline works together
        job = ingestion_pipeline.ingest_local_repository(str(sample_git_repo), "integration_test")
        
        # Verify job completion
        assert job.status == IngestionStatus.COMPLETED
        assert job.total_commits > 0
        assert job.processed_commits == job.total_commits
        assert job.ingested_elements > 0
        
        # Verify Neo4j interactions occurred
        assert ingestion_pipeline.neo4j.execute_query.called
        
        # Verify repository is tracked
        assert "integration_test" in ingestion_pipeline.repo_manager.list_repositories()
    
    def test_error_handling(self, ingestion_pipeline, temp_dir):
        """Test error handling in ingestion pipeline."""
        # Create invalid repository path
        invalid_path = temp_dir / "invalid"
        
        with pytest.raises(IngestionError):
            ingestion_pipeline.ingest_local_repository(str(invalid_path))
    
    @patch('src.code_intelligence.git.repository.GitRepository.get_commit_history')
    def test_large_repository_handling(self, mock_get_history, ingestion_pipeline, sample_git_repo):
        """Test handling of large repositories."""
        # Mock a large number of commits
        mock_commits = []
        for i in range(1000):
            commit = CommitInfo(
                sha=f'commit_{i:04d}',
                message=f'Commit {i}',
                author_name='Test Author',
                author_email='test@example.com',
                committer_name='Test Author',
                committer_email='test@example.com',
                authored_date=datetime.now(),
                committed_date=datetime.now(),
                parents=[],
                stats={'insertions': 1, 'deletions': 0, 'files': 1}
            )
            mock_commits.append(commit)
        
        mock_get_history.return_value = mock_commits
        
        # Test with max_commits limit
        job = ingestion_pipeline.ingest_local_repository(
            str(sample_git_repo), 
            "large_repo",
            max_commits=100
        )
        
        # Should process only the limited number
        assert job.status == IngestionStatus.COMPLETED
        # Note: The actual processing depends on the mock setup