"""Main ingestion pipeline orchestrating git analysis and graph population."""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
import uuid

from ..database.neo4j_client import Neo4jClient
from ..git.repository import GitRepository, RepositoryManager
from ..parsing.parser import MultiLanguageParser
from .graph_populator import GraphPopulator
from .models import IngestionJob, IngestionStatus
from ..exceptions import CodeIntelligenceError

logger = logging.getLogger(__name__)


class IngestionError(CodeIntelligenceError):
    """Ingestion pipeline related errors."""
    pass


class IngestionPipeline:
    """Orchestrates the complete ingestion pipeline from git to graph database."""
    
    def __init__(
        self, 
        neo4j_client: Neo4jClient,
        repository_manager: RepositoryManager = None,
        progress_callback: Optional[Callable[[IngestionJob], None]] = None
    ):
        """Initialize ingestion pipeline."""
        self.neo4j = neo4j_client
        self.repo_manager = repository_manager or RepositoryManager()
        self.parser = MultiLanguageParser()
        self.graph_populator = GraphPopulator(neo4j_client)
        self.progress_callback = progress_callback
        
        # Active jobs tracking
        self._active_jobs: Dict[str, IngestionJob] = {}
    
    def ingest_repository(
        self,
        repository_url: str,
        repository_name: Optional[str] = None,
        max_commits: Optional[int] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> IngestionJob:
        """Start ingestion of a repository."""
        
        # Create ingestion job
        job = IngestionJob(
            id=str(uuid.uuid4()),
            repository_id=repository_name or self._extract_repo_name(repository_url),
            repository_path="",  # Will be set after cloning
            max_commits=max_commits,
            include_patterns=include_patterns or ['*.py', '*.js', '*.jsx', '*.ts', '*.tsx'],
            exclude_patterns=exclude_patterns or ['node_modules/**', '.git/**', '__pycache__/**']
        )
        
        self._active_jobs[job.id] = job
        
        try:
            # Execute ingestion
            self._execute_ingestion(job, repository_url)
            
        except Exception as e:
            job.status = IngestionStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            logger.error(f"Ingestion failed for {repository_url}: {e}")
            raise IngestionError(f"Ingestion failed: {e}")
        
        return job
    
    def ingest_local_repository(
        self,
        repository_path: str,
        repository_name: Optional[str] = None,
        max_commits: Optional[int] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> IngestionJob:
        """Ingest a local repository."""
        
        repo_path = Path(repository_path)
        if not repo_path.exists():
            raise IngestionError(f"Repository path does not exist: {repository_path}")
        
        # Create ingestion job
        job = IngestionJob(
            id=str(uuid.uuid4()),
            repository_id=repository_name or repo_path.name,
            repository_path=str(repo_path),
            max_commits=max_commits,
            include_patterns=include_patterns or ['*.py', '*.js', '*.jsx', '*.ts', '*.tsx'],
            exclude_patterns=exclude_patterns or ['node_modules/**', '.git/**', '__pycache__/**']
        )
        
        self._active_jobs[job.id] = job
        
        try:
            # Load existing repository
            git_repo = self.repo_manager.load_repository(str(repo_path), job.repository_id)
            job.repository_path = str(git_repo.repo_path)
            
            # Execute ingestion
            self._execute_repository_ingestion(job, git_repo)
            
        except Exception as e:
            job.status = IngestionStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            logger.error(f"Ingestion failed for {repository_path}: {e}")
            raise IngestionError(f"Ingestion failed: {e}")
        
        return job
    
    def get_job_status(self, job_id: str) -> Optional[IngestionJob]:
        """Get status of an ingestion job."""
        return self._active_jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel an active ingestion job."""
        job = self._active_jobs.get(job_id)
        if job and job.status == IngestionStatus.RUNNING:
            job.status = IngestionStatus.CANCELLED
            job.completed_at = datetime.now()
            return True
        return False
    
    def _execute_ingestion(self, job: IngestionJob, repository_url: str):
        """Execute the complete ingestion process."""
        job.status = IngestionStatus.RUNNING
        job.started_at = datetime.now()
        self._notify_progress(job)
        
        # Clone repository
        logger.info(f"Cloning repository {repository_url}")
        git_repo = self.repo_manager.clone_repository(repository_url, job.repository_id)
        job.repository_path = str(git_repo.repo_path)
        
        # Execute repository ingestion
        self._execute_repository_ingestion(job, git_repo)
    
    def _execute_repository_ingestion(self, job: IngestionJob, git_repo: GitRepository):
        """Execute ingestion for a loaded git repository."""
        
        # Create repository node in graph
        repo_info = {
            'name': job.repository_id,
            'url': git_repo.remote_url or '',
            'local_path': str(git_repo.repo_path),
            'supported_languages': list(git_repo.get_supported_languages())
        }
        self.graph_populator.create_repository_node(job.repository_id, repo_info)
        
        # Get commit history
        logger.info(f"Analyzing commit history for {job.repository_id}")
        commits = list(git_repo.get_commit_history(max_count=job.max_commits))
        job.total_commits = len(commits)
        self._notify_progress(job)
        
        # Process commits in reverse chronological order (oldest first)
        commits.reverse()
        
        total_elements = 0
        total_relationships = 0
        
        for i, commit_info in enumerate(commits):
            if job.status == IngestionStatus.CANCELLED:
                break
            
            try:
                # Create commit node
                self.graph_populator.create_commit_node(commit_info, job.repository_id)
                
                # Checkout commit to analyze its state
                git_repo.checkout(commit_info.sha)
                
                # Parse files at this commit
                parsed_files = self.parser.parse_directory(
                    str(git_repo.repo_path),
                    include_patterns=job.include_patterns,
                    exclude_patterns=job.exclude_patterns
                )
                
                # Filter out files with errors for now
                valid_files = [f for f in parsed_files if not f.parse_errors]
                job.total_files += len(parsed_files)
                job.processed_files += len(valid_files)
                
                # Ingest parsed files
                if valid_files:
                    stats = self.graph_populator.ingest_parsed_files(
                        valid_files, commit_info.sha, job.repository_id
                    )
                    total_elements += stats['nodes']
                    total_relationships += stats['relationships']
                
                # Update progress
                job.processed_commits = i + 1
                self._notify_progress(job)
                
                logger.debug(
                    f"Processed commit {commit_info.sha[:8]} "
                    f"({i+1}/{len(commits)}): {len(valid_files)} files"
                )
                
            except Exception as e:
                logger.error(f"Failed to process commit {commit_info.sha}: {e}")
                # Continue with next commit
                continue
        
        # Return to latest commit
        git_repo.checkout('HEAD')
        
        # Update job completion
        job.ingested_elements = total_elements
        job.ingested_relationships = total_relationships
        job.status = IngestionStatus.COMPLETED
        job.completed_at = datetime.now()
        
        logger.info(
            f"Ingestion completed for {job.repository_id}: "
            f"{total_elements} elements, {total_relationships} relationships"
        )
        
        self._notify_progress(job)
    
    def update_repository(self, repository_id: str) -> IngestionJob:
        """Update an existing repository with new commits."""
        
        # Get existing repository
        git_repo = self.repo_manager.get_repository(repository_id)
        if not git_repo:
            raise IngestionError(f"Repository not found: {repository_id}")
        
        # Pull latest changes
        git_repo.pull()
        
        # Create incremental ingestion job
        job = IngestionJob(
            id=str(uuid.uuid4()),
            repository_id=repository_id,
            repository_path=str(git_repo.repo_path)
        )
        
        self._active_jobs[job.id] = job
        
        try:
            # Get new commits since last ingestion
            # This would require tracking last ingested commit
            # For now, we'll do a full re-ingestion
            self._execute_repository_ingestion(job, git_repo)
            
        except Exception as e:
            job.status = IngestionStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            raise IngestionError(f"Update failed: {e}")
        
        return job
    
    def _notify_progress(self, job: IngestionJob):
        """Notify progress callback if available."""
        if self.progress_callback:
            try:
                self.progress_callback(job)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def _extract_repo_name(self, url: str) -> str:
        """Extract repository name from URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        
        if path.endswith('.git'):
            path = path[:-4]
        
        return path.split('/')[-1]
    
    def cleanup_repository(self, repository_id: str, keep_commits: int = 100):
        """Clean up old data for a repository."""
        return self.graph_populator.cleanup_old_data(repository_id, keep_commits)
    
    def get_repository_stats(self, repository_id: str) -> Dict[str, Any]:
        """Get statistics for a repository."""
        query = """
        MATCH (r:Repository {id: $repo_id})
        OPTIONAL MATCH (r)-[:HAS_COMMIT]->(c:Commit)
        OPTIONAL MATCH (r)-[:CONTAINS]->(f:File)
        OPTIONAL MATCH (f)-[:CONTAINS]->(e:CodeElement)
        
        RETURN 
            count(DISTINCT c) as commit_count,
            count(DISTINCT f) as file_count,
            count(DISTINCT e) as element_count,
            r.supported_languages as languages
        """
        
        result = self.neo4j.execute_query_sync(query, {'repo_id': repository_id})
        
        if result:
            return {
                'repository_id': repository_id,
                'commit_count': result[0]['commit_count'],
                'file_count': result[0]['file_count'],
                'element_count': result[0]['element_count'],
                'supported_languages': result[0]['languages'] or []
            }
        
        return {
            'repository_id': repository_id,
            'commit_count': 0,
            'file_count': 0,
            'element_count': 0,
            'supported_languages': []
        }