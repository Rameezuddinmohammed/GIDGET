"""Tests for git repository management."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.code_intelligence.git.repository import GitRepository, RepositoryManager, GitRepositoryError
from src.code_intelligence.git.models import CommitInfo, FileChange, ChangeType


class TestGitRepository:
    """Test GitRepository class."""
    
    def test_init_with_valid_repo(self, sample_git_repo):
        """Test initialization with valid git repository."""
        git_repo = GitRepository(str(sample_git_repo))
        
        assert git_repo.name == "sample_repo"
        assert git_repo.current_commit is not None
        assert len(git_repo.current_commit) == 40  # SHA length
    
    def test_init_with_invalid_repo(self, temp_dir):
        """Test initialization with invalid repository."""
        invalid_path = temp_dir / "not_a_repo"
        invalid_path.mkdir()
        
        with pytest.raises(GitRepositoryError):
            GitRepository(str(invalid_path))
    
    def test_get_commit_info(self, sample_git_repo):
        """Test getting commit information."""
        git_repo = GitRepository(str(sample_git_repo))
        
        # Get latest commit
        commit_sha = git_repo.current_commit
        commit_info = git_repo.get_commit_info(commit_sha)
        
        assert isinstance(commit_info, CommitInfo)
        assert commit_info.sha == commit_sha
        assert commit_info.message is not None
        assert commit_info.author_name is not None
        assert isinstance(commit_info.authored_date, datetime)
    
    def test_get_commit_history(self, sample_git_repo):
        """Test getting commit history."""
        git_repo = GitRepository(str(sample_git_repo))
        
        commits = list(git_repo.get_commit_history(max_count=10))
        
        assert len(commits) == 2  # We created 2 commits
        assert all(isinstance(commit, CommitInfo) for commit in commits)
        
        # Check commits are in reverse chronological order
        assert commits[0].committed_date >= commits[1].committed_date
    
    def test_get_file_history(self, sample_git_repo):
        """Test getting file history."""
        git_repo = GitRepository(str(sample_git_repo))
        
        file_commits = list(git_repo.get_file_history("calculator.py"))
        
        assert len(file_commits) == 2  # File was modified in 2 commits
        assert all(isinstance(commit, CommitInfo) for commit in file_commits)
    
    def test_get_supported_languages(self, sample_git_repo):
        """Test language detection."""
        git_repo = GitRepository(str(sample_git_repo))
        
        languages = git_repo.get_supported_languages()
        
        assert "python" in languages
        assert "javascript" in languages
    
    def test_get_file_content(self, sample_git_repo):
        """Test getting file content."""
        git_repo = GitRepository(str(sample_git_repo))
        
        content = git_repo.get_file_content("calculator.py")
        
        assert "class Calculator" in content
        assert "def add" in content


class TestRepositoryManager:
    """Test RepositoryManager class."""
    
    def test_init(self, temp_dir):
        """Test repository manager initialization."""
        manager = RepositoryManager(str(temp_dir))
        
        assert manager.base_path.exists()
        assert manager.base_path.is_dir()
    
    def test_load_repository(self, repository_manager, sample_git_repo):
        """Test loading existing repository."""
        git_repo = repository_manager.load_repository(str(sample_git_repo), "test_repo")
        
        assert isinstance(git_repo, GitRepository)
        assert git_repo.name == "sample_repo"
        assert "test_repo" in repository_manager.list_repositories()
    
    def test_get_repository_info(self, repository_manager, sample_git_repo):
        """Test getting repository information."""
        repository_manager.load_repository(str(sample_git_repo), "test_repo")
        
        repo_info = repository_manager.get_repository_info("test_repo")
        
        assert repo_info.id == "test_repo"
        assert repo_info.name == "test_repo"
        assert repo_info.commit_count == 2
        assert "python" in repo_info.supported_languages
        assert "javascript" in repo_info.supported_languages
    
    def test_remove_repository(self, repository_manager, sample_git_repo):
        """Test removing repository."""
        repository_manager.load_repository(str(sample_git_repo), "test_repo")
        
        assert "test_repo" in repository_manager.list_repositories()
        
        repository_manager.remove_repository("test_repo")
        
        assert "test_repo" not in repository_manager.list_repositories()
    
    @patch('src.code_intelligence.git.repository.GitRepository')
    @patch('git.Repo.clone_from')
    def test_clone_repository(self, mock_clone, mock_git_repo_class, repository_manager, sample_git_repo):
        """Test cloning repository."""
        # Mock the clone operation
        mock_clone.return_value = None
        
        # Mock GitRepository constructor
        mock_git_repo_instance = Mock(spec=GitRepository)
        mock_git_repo_class.return_value = mock_git_repo_instance
        
        git_repo = repository_manager.clone_repository("https://github.com/test/test_repo.git", "test_repo")
        
        assert git_repo == mock_git_repo_instance
        assert "test_repo" in repository_manager.list_repositories()
        mock_clone.assert_called_once()