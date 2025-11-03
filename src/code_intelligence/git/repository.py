"""Git repository management and analysis."""

import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Iterator, Dict, Set
from urllib.parse import urlparse
import logging

import git
from git import Repo, InvalidGitRepositoryError, GitCommandError

from ..exceptions import CodeIntelligenceError
from .models import CommitInfo, FileChange, ChangeType, RepositoryStatus, RepositoryInfo


logger = logging.getLogger(__name__)


class GitRepositoryError(CodeIntelligenceError):
    """Git repository related errors."""
    pass


class GitRepository:
    """Manages a single git repository for analysis."""
    
    def __init__(self, repo_path: str):
        """Initialize with path to existing git repository."""
        self.repo_path = Path(repo_path)
        try:
            self.repo = Repo(repo_path)
        except InvalidGitRepositoryError:
            raise GitRepositoryError(f"Invalid git repository: {repo_path}")
        
        self._commit_cache: Dict[str, CommitInfo] = {}
    
    @property
    def name(self) -> str:
        """Get repository name from path."""
        return self.repo_path.name
    
    @property
    def current_commit(self) -> str:
        """Get current HEAD commit SHA."""
        return self.repo.head.commit.hexsha
    
    @property
    def remote_url(self) -> Optional[str]:
        """Get remote origin URL if available."""
        try:
            return self.repo.remotes.origin.url
        except AttributeError:
            return None
    
    def get_commit_info(self, commit_sha: str) -> CommitInfo:
        """Get detailed information about a specific commit."""
        if commit_sha in self._commit_cache:
            return self._commit_cache[commit_sha]
        
        try:
            commit = self.repo.commit(commit_sha)
        except git.BadName:
            raise GitRepositoryError(f"Commit not found: {commit_sha}")
        
        # Get file changes
        file_changes = []
        if commit.parents:
            # Compare with first parent
            parent = commit.parents[0]
            diffs = parent.diff(commit, create_patch=True)
            
            for diff in diffs:
                change_type = self._map_change_type(diff.change_type)
                file_change = FileChange(
                    file_path=diff.b_path or diff.a_path or "",
                    change_type=change_type,
                    old_file_path=diff.a_path if diff.renamed_file else None,
                    binary=diff.b_blob is None or diff.a_blob is None,
                )
                
                # Calculate insertions/deletions for text files
                if not file_change.binary and diff.diff:
                    lines = diff.diff.decode('utf-8', errors='ignore').split('\n')
                    insertions = sum(1 for line in lines if line.startswith('+') and not line.startswith('+++'))
                    deletions = sum(1 for line in lines if line.startswith('-') and not line.startswith('---'))
                    file_change.insertions = insertions
                    file_change.deletions = deletions
                
                file_changes.append(file_change)
        
        commit_info = CommitInfo(
            sha=commit.hexsha,
            message=commit.message.strip(),
            author_name=commit.author.name,
            author_email=commit.author.email,
            committer_name=commit.committer.name,
            committer_email=commit.committer.email,
            authored_date=datetime.fromtimestamp(commit.authored_date),
            committed_date=datetime.fromtimestamp(commit.committed_date),
            parents=[p.hexsha for p in commit.parents],
            file_changes=file_changes,
            stats={
                'insertions': sum(fc.insertions for fc in file_changes),
                'deletions': sum(fc.deletions for fc in file_changes),
                'files': len(file_changes)
            }
        )
        
        self._commit_cache[commit_sha] = commit_info
        return commit_info
    
    def get_commit_history(
        self, 
        max_count: Optional[int] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        paths: Optional[List[str]] = None
    ) -> Iterator[CommitInfo]:
        """Get commit history with optional filtering."""
        kwargs = {}
        if max_count:
            kwargs['max_count'] = max_count
        if since:
            kwargs['since'] = since
        if until:
            kwargs['until'] = until
        if paths:
            kwargs['paths'] = paths
        
        try:
            for commit in self.repo.iter_commits(**kwargs):
                yield self.get_commit_info(commit.hexsha)
        except GitCommandError as e:
            raise GitRepositoryError(f"Failed to get commit history: {e}")
    
    def get_file_history(self, file_path: str, max_count: Optional[int] = None) -> Iterator[CommitInfo]:
        """Get commit history for a specific file."""
        return self.get_commit_history(max_count=max_count, paths=[file_path])
    
    def get_branches(self) -> List[str]:
        """Get list of all branches."""
        return [branch.name for branch in self.repo.branches]
    
    def get_tags(self) -> List[str]:
        """Get list of all tags."""
        return [tag.name for tag in self.repo.tags]
    
    def checkout(self, ref: str) -> None:
        """Checkout a specific branch, tag, or commit."""
        try:
            self.repo.git.checkout(ref)
        except GitCommandError as e:
            raise GitRepositoryError(f"Failed to checkout {ref}: {e}")
    
    def pull(self) -> None:
        """Pull latest changes from remote."""
        try:
            origin = self.repo.remotes.origin
            origin.pull()
        except GitCommandError as e:
            raise GitRepositoryError(f"Failed to pull changes: {e}")
    
    def get_file_content(self, file_path: str, commit_sha: Optional[str] = None) -> str:
        """Get file content at specific commit (or current if None)."""
        try:
            if commit_sha:
                commit = self.repo.commit(commit_sha)
                blob = commit.tree / file_path
                return blob.data_stream.read().decode('utf-8', errors='ignore')
            else:
                full_path = self.repo_path / file_path
                return full_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            raise GitRepositoryError(f"Failed to read file {file_path}: {e}")
    
    def get_supported_languages(self) -> Set[str]:
        """Detect supported programming languages in repository."""
        languages = set()
        
        # Walk through repository files
        for root, dirs, files in os.walk(self.repo_path):
            # Skip .git directory
            if '.git' in dirs:
                dirs.remove('.git')
            
            for file in files:
                ext = Path(file).suffix.lower()
                if ext == '.py':
                    languages.add('python')
                elif ext in ['.js', '.jsx']:
                    languages.add('javascript')
                elif ext in ['.ts', '.tsx']:
                    languages.add('typescript')
        
        return languages
    
    def _map_change_type(self, git_change_type: str) -> ChangeType:
        """Map GitPython change type to our enum."""
        mapping = {
            'A': ChangeType.ADDED,
            'M': ChangeType.MODIFIED,
            'D': ChangeType.DELETED,
            'R': ChangeType.RENAMED,
            'C': ChangeType.COPIED,
            'U': ChangeType.UNMERGED,
            'X': ChangeType.UNKNOWN,
            'B': ChangeType.BROKEN,
        }
        return mapping.get(git_change_type, ChangeType.UNKNOWN)


class RepositoryManager:
    """Manages multiple git repositories and their analysis status."""
    
    def __init__(self, base_path: str = None):
        """Initialize repository manager."""
        self.base_path = Path(base_path or tempfile.gettempdir()) / "code_intelligence_repos"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._repositories: Dict[str, GitRepository] = {}
    
    def clone_repository(self, url: str, name: Optional[str] = None) -> GitRepository:
        """Clone a repository from URL."""
        if not name:
            name = self._extract_repo_name(url)
        
        repo_path = self.base_path / name
        
        # Remove existing directory if it exists
        if repo_path.exists():
            shutil.rmtree(repo_path)
        
        try:
            logger.info(f"Cloning repository {url} to {repo_path}")
            git.Repo.clone_from(url, repo_path)
            git_repo = GitRepository(str(repo_path))
            self._repositories[name] = git_repo
            return git_repo
        except GitCommandError as e:
            raise GitRepositoryError(f"Failed to clone repository {url}: {e}")
    
    def load_repository(self, path: str, name: Optional[str] = None) -> GitRepository:
        """Load existing repository from local path."""
        if not name:
            name = Path(path).name
        
        git_repo = GitRepository(path)
        self._repositories[name] = git_repo
        return git_repo
    
    def get_repository(self, name: str) -> Optional[GitRepository]:
        """Get repository by name."""
        return self._repositories.get(name)
    
    def list_repositories(self) -> List[str]:
        """List all managed repository names."""
        return list(self._repositories.keys())
    
    def update_repository(self, name: str) -> None:
        """Update repository by pulling latest changes."""
        repo = self.get_repository(name)
        if not repo:
            raise GitRepositoryError(f"Repository not found: {name}")
        
        repo.pull()
    
    def get_repository_info(self, name: str) -> RepositoryInfo:
        """Get comprehensive repository information."""
        repo = self.get_repository(name)
        if not repo:
            raise GitRepositoryError(f"Repository not found: {name}")
        
        # Count total commits
        commit_count = sum(1 for _ in repo.get_commit_history())
        
        return RepositoryInfo(
            id=name,
            name=name,
            url=repo.remote_url or "",
            local_path=str(repo.repo_path),
            status=RepositoryStatus.NOT_ANALYZED,  # Default status
            last_commit_sha=repo.current_commit,
            commit_count=commit_count,
            supported_languages=list(repo.get_supported_languages())
        )
    
    def remove_repository(self, name: str) -> None:
        """Remove repository from management and delete local copy."""
        if name in self._repositories:
            repo = self._repositories[name]
            # Remove from memory
            del self._repositories[name]
            
            # Remove local directory if it's under our base path
            if repo.repo_path.is_relative_to(self.base_path):
                shutil.rmtree(repo.repo_path)
    
    def _extract_repo_name(self, url: str) -> str:
        """Extract repository name from URL."""
        parsed = urlparse(url)
        path = parsed.path.strip('/')
        
        # Remove .git suffix if present
        if path.endswith('.git'):
            path = path[:-4]
        
        # Get last part of path
        return path.split('/')[-1]