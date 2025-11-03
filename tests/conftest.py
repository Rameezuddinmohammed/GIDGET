"""Pytest configuration and fixtures."""

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
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_neo4j_client():
    """Mock Neo4j client for testing."""
    client = Mock(spec=Neo4jClient)
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