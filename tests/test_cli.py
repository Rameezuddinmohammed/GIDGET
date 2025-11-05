"""Tests for CLI functionality."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typer.testing import CliRunner
import httpx

from src.code_intelligence.cli import app, CLIConfig


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_config_file():
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_data = {
            "api_url": "http://localhost:8000/api/v1",
            "output_format": "table",
            "timeout": 300
        }
        json.dump(config_data, f)
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for API requests."""
    with patch('src.code_intelligence.cli.get_api_client') as mock:
        mock_client = MagicMock()
        mock.__enter__ = Mock(return_value=mock_client)
        mock.__exit__ = Mock(return_value=None)
        mock.return_value = mock_client
        yield mock_client


class TestCLIConfig:
    """Test CLI configuration management."""
    
    def test_config_initialization(self, temp_config_file):
        """Test config initialization with existing file."""
        with patch.object(CLIConfig, 'config_file', temp_config_file):
            config = CLIConfig()
            assert config.get("api_url") == "http://localhost:8000/api/v1"
            assert config.get("output_format") == "table"
            assert config.get("timeout") == 300
    
    def test_config_default_values(self):
        """Test config initialization with default values."""
        with patch.object(Path, 'exists', return_value=False):
            config = CLIConfig()
            assert config.get("output_format") == "table"
            assert config.get("timeout") == 300
    
    def test_config_set_and_save(self, temp_config_file):
        """Test setting and saving configuration."""
        with patch.object(CLIConfig, 'config_file', temp_config_file):
            config = CLIConfig()
            config.set("new_key", "new_value")
            
            # Reload config to verify persistence
            config2 = CLIConfig()
            assert config2.get("new_key") == "new_value"


class TestVersionCommand:
    """Test version command."""
    
    def test_version_command(self, runner):
        """Test version command output."""
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "Multi-Agent Code Intelligence System v1.0.0" in result.stdout


class TestConfigCommand:
    """Test config command."""
    
    def test_config_list_all(self, runner):
        """Test listing all configuration."""
        with patch('src.code_intelligence.cli.cli_config') as mock_config:
            mock_config._config = {"key1": "value1", "key2": "value2"}
            
            result = runner.invoke(app, ["config-cmd", "--list"])
            assert result.exit_code == 0
            assert "CLI Configuration" in result.stdout
    
    def test_config_get_existing_key(self, runner):
        """Test getting existing configuration key."""
        with patch('src.code_intelligence.cli.cli_config') as mock_config:
            mock_config.get.return_value = "test_value"
            
            result = runner.invoke(app, ["config-cmd", "test_key"])
            assert result.exit_code == 0
            assert "test_key: test_value" in result.stdout
    
    def test_config_get_nonexistent_key(self, runner):
        """Test getting non-existent configuration key."""
        with patch('src.code_intelligence.cli.cli_config') as mock_config:
            mock_config.get.return_value = None
            
            result = runner.invoke(app, ["config-cmd", "nonexistent_key"])
            assert result.exit_code == 1
            assert "not found" in result.stdout
    
    def test_config_set_key(self, runner):
        """Test setting configuration key."""
        with patch('src.code_intelligence.cli.cli_config') as mock_config:
            result = runner.invoke(app, ["config-cmd", "test_key", "test_value"])
            assert result.exit_code == 0
            mock_config.set.assert_called_once_with("test_key", "test_value")
            assert "Set test_key = test_value" in result.stdout


class TestQueryCommand:
    """Test query command."""
    
    def test_query_submit_success(self, runner, mock_http_client):
        """Test successful query submission."""
        # Mock API responses
        mock_response = Mock()
        mock_response.json.return_value = {
            "query_id": "test-query-id",
            "status": "pending",
            "message": "Query submitted successfully"
        }
        mock_response.raise_for_status.return_value = None
        mock_http_client.post.return_value = mock_response
        
        result = runner.invoke(app, [
            "query",
            "What changed in the code?",
            "--repo", "https://github.com/test/repo.git",
            "--no-wait"
        ])
        
        assert result.exit_code == 0
        assert "Query submitted successfully!" in result.stdout
        assert "test-query-id" in result.stdout
    
    def test_query_submit_with_wait(self, runner, mock_http_client):
        """Test query submission with wait for completion."""
        # Mock query submission
        submit_response = Mock()
        submit_response.json.return_value = {
            "query_id": "test-query-id",
            "status": "pending"
        }
        submit_response.raise_for_status.return_value = None
        
        # Mock status check (completed)
        status_response = Mock()
        status_response.json.return_value = {
            "status": "completed",
            "results": {
                "summary": "Test analysis complete",
                "confidence_score": 0.95,
                "processing_time_seconds": 30.5,
                "findings": []
            }
        }
        status_response.raise_for_status.return_value = None
        
        mock_http_client.post.return_value = submit_response
        mock_http_client.get.return_value = status_response
        
        result = runner.invoke(app, [
            "query",
            "What changed in the code?",
            "--repo", "https://github.com/test/repo.git",
            "--wait"
        ])
        
        assert result.exit_code == 0
        assert "Query submitted successfully!" in result.stdout
        assert "Analysis Summary" in result.stdout
    
    def test_query_submit_api_error(self, runner, mock_http_client):
        """Test query submission with API error."""
        mock_http_client.post.side_effect = httpx.HTTPError("API Error")
        
        result = runner.invoke(app, [
            "query",
            "What changed in the code?",
            "--repo", "https://github.com/test/repo.git"
        ])
        
        assert result.exit_code == 1
        assert "API request failed" in result.stdout
    
    def test_query_failed_status(self, runner, mock_http_client):
        """Test query with failed status."""
        # Mock query submission
        submit_response = Mock()
        submit_response.json.return_value = {
            "query_id": "test-query-id",
            "status": "pending"
        }
        submit_response.raise_for_status.return_value = None
        
        # Mock status check (failed)
        status_response = Mock()
        status_response.json.return_value = {
            "status": "failed",
            "error": "Repository not accessible"
        }
        status_response.raise_for_status.return_value = None
        
        mock_http_client.post.return_value = submit_response
        mock_http_client.get.return_value = status_response
        
        result = runner.invoke(app, [
            "query",
            "What changed in the code?",
            "--repo", "https://github.com/test/repo.git",
            "--wait"
        ])
        
        assert result.exit_code == 1
        assert "Query failed" in result.stdout
        assert "Repository not accessible" in result.stdout


class TestStatusCommand:
    """Test status command."""
    
    def test_status_success(self, runner, mock_http_client):
        """Test successful status check."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "processing",
            "created_at": "2024-01-01T10:00:00Z",
            "updated_at": "2024-01-01T10:05:00Z",
            "progress": {
                "current_agent": "analyst",
                "progress_percentage": 60.0,
                "current_step": "analyzing code structure"
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_http_client.get.return_value = mock_response
        
        result = runner.invoke(app, ["status", "test-query-id"])
        
        assert result.exit_code == 0
        assert "Query Status" in result.stdout
        assert "processing" in result.stdout
        assert "analyst" in result.stdout
    
    def test_status_json_output(self, runner, mock_http_client):
        """Test status command with JSON output."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "completed"}
        mock_response.raise_for_status.return_value = None
        mock_http_client.get.return_value = mock_response
        
        result = runner.invoke(app, ["status", "test-query-id", "--output", "json"])
        
        assert result.exit_code == 0
        assert '"status": "completed"' in result.stdout
    
    def test_status_not_found(self, runner, mock_http_client):
        """Test status command with non-existent query."""
        mock_http_client.get.side_effect = httpx.HTTPError("Not found")
        
        result = runner.invoke(app, ["status", "nonexistent-id"])
        
        assert result.exit_code == 1
        assert "API request failed" in result.stdout


class TestRepositoriesCommand:
    """Test repositories command."""
    
    def test_repositories_list_empty(self, runner, mock_http_client):
        """Test listing repositories when empty."""
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status.return_value = None
        mock_http_client.get.return_value = mock_response
        
        result = runner.invoke(app, ["repositories", "list"])
        
        assert result.exit_code == 0
        assert "No repositories found" in result.stdout
    
    def test_repositories_list_with_data(self, runner, mock_http_client):
        """Test listing repositories with data."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "id": "repo-id-123",
                "name": "test-repo",
                "status": "analyzed",
                "commit_count": 100,
                "supported_languages": ["Python", "JavaScript"]
            }
        ]
        mock_response.raise_for_status.return_value = None
        mock_http_client.get.return_value = mock_response
        
        result = runner.invoke(app, ["repositories", "list"])
        
        assert result.exit_code == 0
        assert "Repositories" in result.stdout
        assert "test-repo" in result.stdout
        assert "analyzed" in result.stdout
    
    def test_repositories_add_success(self, runner, mock_http_client):
        """Test adding repository successfully."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "new-repo-id",
            "name": "new-repo"
        }
        mock_response.raise_for_status.return_value = None
        mock_http_client.post.return_value = mock_response
        
        result = runner.invoke(app, [
            "repositories", "add",
            "--url", "https://github.com/test/repo.git",
            "--name", "test-repo"
        ])
        
        assert result.exit_code == 0
        assert "Repository added successfully!" in result.stdout
        assert "new-repo-id" in result.stdout
    
    def test_repositories_add_missing_url(self, runner):
        """Test adding repository without URL."""
        result = runner.invoke(app, ["repositories", "add"])
        
        assert result.exit_code == 1
        assert "Repository URL is required" in result.stdout
    
    def test_repositories_delete_success(self, runner, mock_http_client):
        """Test deleting repository successfully."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_http_client.delete.return_value = mock_response
        
        result = runner.invoke(app, [
            "repositories", "delete",
            "--id", "repo-id-123"
        ])
        
        assert result.exit_code == 0
        assert "Repository deleted successfully!" in result.stdout
    
    def test_repositories_analyze_success(self, runner, mock_http_client):
        """Test triggering repository analysis."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_http_client.post.return_value = mock_response
        
        result = runner.invoke(app, [
            "repositories", "analyze",
            "--id", "repo-id-123"
        ])
        
        assert result.exit_code == 0
        assert "Repository analysis started!" in result.stdout


class TestHistoryCommand:
    """Test history command."""
    
    def test_history_empty(self, runner, mock_http_client):
        """Test history command with no queries."""
        mock_response = Mock()
        mock_response.json.return_value = {"queries": []}
        mock_response.raise_for_status.return_value = None
        mock_http_client.get.return_value = mock_response
        
        result = runner.invoke(app, ["history"])
        
        assert result.exit_code == 0
        assert "No queries found" in result.stdout
    
    def test_history_with_data(self, runner, mock_http_client):
        """Test history command with query data."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "queries": [
                {
                    "query_id": "query-123",
                    "query": "What changed in the authentication system?",
                    "repository_name": "test-repo",
                    "status": "completed",
                    "created_at": "2024-01-01T10:00:00Z"
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_http_client.get.return_value = mock_response
        
        result = runner.invoke(app, ["history"])
        
        assert result.exit_code == 0
        assert "Query History" in result.stdout
        assert "authentication system" in result.stdout
        assert "completed" in result.stdout
    
    def test_history_json_output(self, runner, mock_http_client):
        """Test history command with JSON output."""
        mock_response = Mock()
        mock_response.json.return_value = {"queries": []}
        mock_response.raise_for_status.return_value = None
        mock_http_client.get.return_value = mock_response
        
        result = runner.invoke(app, ["history", "--output", "json"])
        
        assert result.exit_code == 0
        assert '"queries": []' in result.stdout


class TestExportCommand:
    """Test export command."""
    
    def test_export_success(self, runner, mock_http_client):
        """Test successful export."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "export_id": "export-123",
            "download_url": "/api/v1/exports/export-123/download",
            "expires_at": "2024-01-02T10:00:00Z"
        }
        mock_response.raise_for_status.return_value = None
        mock_http_client.post.return_value = mock_response
        
        result = runner.invoke(app, [
            "export", "query-123",
            "--format", "json"
        ])
        
        assert result.exit_code == 0
        assert "Export created successfully!" in result.stdout
        assert "export-123" in result.stdout
    
    def test_export_api_error(self, runner, mock_http_client):
        """Test export with API error."""
        mock_http_client.post.side_effect = httpx.HTTPError("Export failed")
        
        result = runner.invoke(app, ["export", "query-123"])
        
        assert result.exit_code == 1
        assert "API request failed" in result.stdout


class TestHealthCommand:
    """Test health command."""
    
    def test_health_success(self, runner, mock_http_client):
        """Test successful health check."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "healthy",
            "version": "1.0.0",
            "services": {
                "supabase": "healthy",
                "neo4j": "healthy",
                "agents": "healthy"
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_http_client.get.return_value = mock_response
        
        result = runner.invoke(app, ["health"])
        
        assert result.exit_code == 0
        assert "API Status: healthy" in result.stdout
        assert "Service Health" in result.stdout
    
    def test_health_api_error(self, runner, mock_http_client):
        """Test health check with API error."""
        mock_http_client.get.side_effect = httpx.HTTPError("Health check failed")
        
        result = runner.invoke(app, ["health"])
        
        assert result.exit_code == 1
        assert "API health check failed" in result.stdout


class TestSetupCommand:
    """Test setup command."""
    
    def test_setup_success(self, runner):
        """Test successful setup."""
        with patch('src.code_intelligence.cli.db_initializer') as mock_initializer:
            mock_initializer.initialize_sync.return_value = None
            
            result = runner.invoke(app, ["setup"])
            
            assert result.exit_code == 0
            assert "Database setup completed!" in result.stdout
    
    def test_setup_neo4j_only(self, runner):
        """Test setup with Neo4j only."""
        with patch('src.code_intelligence.cli.db_initializer') as mock_initializer:
            with patch('asyncio.run') as mock_run:
                result = runner.invoke(app, ["setup", "--neo4j-only"])
                
                assert result.exit_code == 0
                assert "Neo4j setup completed!" in result.stdout
    
    def test_setup_with_reset_confirmation(self, runner):
        """Test setup with reset and user confirmation."""
        with patch('src.code_intelligence.cli.db_initializer') as mock_initializer:
            mock_initializer.initialize_sync.return_value = None
            
            result = runner.invoke(app, ["setup", "--reset"], input="y\n")
            
            assert result.exit_code == 0
            assert "Database setup completed!" in result.stdout
    
    def test_setup_with_reset_cancelled(self, runner):
        """Test setup with reset cancelled by user."""
        result = runner.invoke(app, ["setup", "--reset"], input="n\n")
        
        assert result.exit_code == 0
        assert "Setup cancelled." in result.stdout