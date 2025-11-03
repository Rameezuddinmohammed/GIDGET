"""Pytest configuration and fixtures."""

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock

from src.code_intelligence.database.neo4j_client import Neo4jClient
from src.code_intelligence.git.repository import RepositoryManager
from src.code_intelligence.parsing.parser import MultiLanguageParser
from src.code_intelligence.ingestion.pipeline import IngestionPipeline


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    # Windows-safe cleanup with multiple strategies
    _safe_rmtree(temp_path)

def _safe_rmtree(path):
    """Safely remove directory tree on Windows."""
    import stat
    import time
    import gc
    
    def handle_remove_readonly(func, path, exc):
        """Handle read-only files on Windows."""
        if os.path.exists(path):
            # Make the file writable and try again
            os.chmod(path, stat.S_IWRITE)
            func(path)
    
    # Strategy 1: Normal removal
    try:
        shutil.rmtree(path)
        return
    except PermissionError:
        pass
    
    # Strategy 2: Force garbage collection and retry
    try:
        gc.collect()
        time.sleep(0.1)
        shutil.rmtree(path)
        return
    except PermissionError:
        pass
    
    # Strategy 3: Make files writable and retry
    try:
        shutil.rmtree(path, onerror=handle_remove_readonly)
        return
    except PermissionError:
        pass
    
    # Strategy 4: Walk through and force remove each file
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                try:
                    os.chmod(file_path, stat.S_IWRITE)
                    os.remove(file_path)
                except:
                    pass
            for name in dirs:
                dir_path = os.path.join(root, name)
                try:
                    os.chmod(dir_path, stat.S_IWRITE)
                    os.rmdir(dir_path)
                except:
                    pass
        os.rmdir(path)
    except:
        # If all else fails, just ignore the error
        # The temp directory will be cleaned up by the OS eventually
        pass


@pytest.fixture
def mock_neo4j_client():
    """Mock Neo4j client for testing."""
    client = Mock(spec=Neo4jClient)
    # Mock both sync and async query methods
    client.execute_query_sync.return_value = [{'id': 'test_repo', 'created': 1}]
    client.execute_query.return_value = [{'id': 'test_repo', 'created': 1}]
    client.close.return_value = None
    return client


@pytest.fixture
def repository_manager(temp_dir):
    """Repository manager with temporary directory."""
    return RepositoryManager(str(temp_dir))


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return '''
"""Sample module for testing."""

import os
from typing import List, Optional

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        result = a + b
        self.history.append(f"add({a}, {b}) = {result}")
        return result
    
    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"multiply({a}, {b}) = {result}")
        return result

def main():
    """Main function."""
    calc = Calculator()
    print(calc.add(2, 3))
    print(calc.multiply(4, 5))

if __name__ == "__main__":
    main()
'''


@pytest.fixture
def sample_javascript_code():
    """Sample JavaScript code for testing."""
    return '''
/**
 * Sample JavaScript module for testing
 */

import { EventEmitter } from 'events';
import fs from 'fs';

class TaskManager extends EventEmitter {
    constructor() {
        super();
        this.tasks = [];
    }
    
    async addTask(name, description) {
        const task = {
            id: Date.now(),
            name,
            description,
            completed: false
        };
        
        this.tasks.push(task);
        this.emit('taskAdded', task);
        return task;
    }
    
    completeTask(taskId) {
        const task = this.tasks.find(t => t.id === taskId);
        if (task) {
            task.completed = true;
            this.emit('taskCompleted', task);
        }
        return task;
    }
}

export default TaskManager;
'''


@pytest.fixture
def sample_git_repo(temp_dir, sample_python_code, sample_javascript_code):
    """Create a sample git repository for testing."""
    import git
    
    repo_path = temp_dir / "sample_repo"
    repo_path.mkdir()
    
    # Initialize git repo
    repo = git.Repo.init(repo_path)
    
    # Create sample files
    (repo_path / "calculator.py").write_text(sample_python_code)
    (repo_path / "task_manager.js").write_text(sample_javascript_code)
    (repo_path / "README.md").write_text("# Sample Repository\n\nThis is a test repository.")
    
    # Add and commit files
    repo.index.add(["calculator.py", "task_manager.js", "README.md"])
    repo.index.commit("Initial commit")
    
    # Create second commit with changes
    updated_python = sample_python_code + "\n\n# Added comment\n"
    (repo_path / "calculator.py").write_text(updated_python)
    repo.index.add(["calculator.py"])
    repo.index.commit("Update calculator with comment")
    
    return repo_path


@pytest.fixture
def ingestion_pipeline(mock_neo4j_client, repository_manager):
    """Ingestion pipeline with mocked dependencies."""
    return IngestionPipeline(
        neo4j_client=mock_neo4j_client,
        repository_manager=repository_manager
    )