"""Shared tools and utilities for agents."""

import os
import subprocess
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from ..database.neo4j_client import Neo4jClient
from ..database.supabase_client import SupabaseClient
from ..git.repository import GitRepository
from ..logging import get_logger
from .base import AgentTool


logger = get_logger(__name__)


class GitTool(AgentTool):
    """Tool for git operations."""
    
    def __init__(self):
        super().__init__(
            name="git_tool",
            description="Tool for git repository operations and history analysis"
        )
        
    async def execute(
        self, 
        repository_path: str, 
        operation: str, 
        **kwargs: Any
    ) -> Any:
        """Execute git operations."""
        try:
            git_repo = GitRepository(repository_path)
            
            if operation == "get_commits":
                return await self._get_commits(git_repo, **kwargs)
            elif operation == "get_file_history":
                return await self._get_file_history(git_repo, **kwargs)
            elif operation == "get_diff":
                return await self._get_diff(git_repo, **kwargs)
            elif operation == "get_blame":
                return await self._get_blame(git_repo, **kwargs)
            else:
                raise ValueError(f"Unknown git operation: {operation}")
                
        except Exception as e:
            logger.error(f"Git tool error: {str(e)}")
            raise
            
    async def _get_commits(
        self, 
        git_repo: GitRepository, 
        max_count: int = 100,
        since: Optional[str] = None,
        until: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get commit history."""
        commits = git_repo.get_commits(
            max_count=max_count,
            since=since,
            until=until,
            file_path=file_path
        )
        
        return [
            {
                "sha": commit.sha,
                "message": commit.message,
                "author": commit.author,
                "date": commit.date,
                "files_changed": commit.files_changed
            }
            for commit in commits
        ]
        
    async def _get_file_history(
        self, 
        git_repo: GitRepository, 
        file_path: str
    ) -> List[Dict[str, Any]]:
        """Get history for a specific file."""
        history = git_repo.get_file_history(file_path)
        
        return [
            {
                "sha": entry.sha,
                "message": entry.message,
                "author": entry.author,
                "date": entry.date,
                "changes": entry.changes
            }
            for entry in history
        ]
        
    async def _get_diff(
        self, 
        git_repo: GitRepository, 
        commit1: str, 
        commit2: str,
        file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get diff between commits."""
        diff = git_repo.get_diff(commit1, commit2, file_path)
        
        return {
            "commit1": commit1,
            "commit2": commit2,
            "file_path": file_path,
            "diff": diff
        }
        
    async def _get_blame(
        self, 
        git_repo: GitRepository, 
        file_path: str,
        commit: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get blame information for a file."""
        blame = git_repo.get_blame(file_path, commit)
        
        return [
            {
                "line_number": entry.line_number,
                "content": entry.content,
                "commit_sha": entry.commit_sha,
                "author": entry.author,
                "date": entry.date
            }
            for entry in blame
        ]


class Neo4jTool(AgentTool):
    """Tool for Neo4j graph database operations."""
    
    def __init__(self, neo4j_client: Neo4jClient):
        super().__init__(
            name="neo4j_tool",
            description="Tool for graph database queries and analysis"
        )
        self.client = neo4j_client
        
    async def execute(
        self, 
        operation: str, 
        **kwargs: Any
    ) -> Any:
        """Execute Neo4j operations."""
        try:
            if operation == "query":
                return await self._execute_query(**kwargs)
            elif operation == "find_functions":
                return await self._find_functions(**kwargs)
            elif operation == "find_dependencies":
                return await self._find_dependencies(**kwargs)
            elif operation == "trace_calls":
                return await self._trace_calls(**kwargs)
            else:
                raise ValueError(f"Unknown Neo4j operation: {operation}")
                
        except Exception as e:
            logger.error(f"Neo4j tool error: {str(e)}")
            raise
            
    async def _execute_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query."""
        return await self.client.execute_query(query, parameters or {})
        
    async def _find_functions(
        self, 
        name_pattern: Optional[str] = None,
        file_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find functions matching criteria."""
        query = """
        MATCH (f:Function)
        WHERE ($name_pattern IS NULL OR f.name CONTAINS $name_pattern)
        AND ($file_path IS NULL OR f.file_path = $file_path)
        RETURN f.name as name, f.file_path as file_path, 
               f.start_line as start_line, f.end_line as end_line
        """
        
        return await self._execute_query(
            query, 
            {"name_pattern": name_pattern, "file_path": file_path}
        )
        
    async def _find_dependencies(
        self, 
        element_name: str, 
        element_type: str = "Function"
    ) -> List[Dict[str, Any]]:
        """Find dependencies for a code element."""
        query = f"""
        MATCH (e:{element_type} {{name: $element_name}})
        MATCH (e)-[:DEPENDS_ON|CALLS|IMPORTS*1..3]->(dep)
        RETURN DISTINCT dep.name as name, dep.type as type, 
               dep.file_path as file_path
        """
        
        return await self._execute_query(query, {"element_name": element_name})
        
    async def _trace_calls(
        self, 
        function_name: str, 
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """Trace function calls."""
        query = """
        MATCH (f:Function {name: $function_name})
        MATCH path = (f)-[:CALLS*1..%d]->(called)
        RETURN [node in nodes(path) | {
            name: node.name, 
            file_path: node.file_path,
            start_line: node.start_line
        }] as call_path
        """ % max_depth
        
        return await self._execute_query(query, {"function_name": function_name})


class VectorSearchTool(AgentTool):
    """Tool for semantic vector search operations."""
    
    def __init__(self, supabase_client: SupabaseClient):
        super().__init__(
            name="vector_search_tool",
            description="Tool for semantic code search using vector embeddings"
        )
        self.client = supabase_client
        
    async def execute(
        self, 
        operation: str, 
        **kwargs: Any
    ) -> Any:
        """Execute vector search operations."""
        try:
            if operation == "semantic_search":
                return await self._semantic_search(**kwargs)
            elif operation == "find_similar":
                return await self._find_similar(**kwargs)
            elif operation == "hybrid_search":
                return await self._hybrid_search(**kwargs)
            else:
                raise ValueError(f"Unknown vector search operation: {operation}")
                
        except Exception as e:
            logger.error(f"Vector search tool error: {str(e)}")
            raise
            
    async def _semantic_search(
        self, 
        query: str, 
        limit: int = 10,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform semantic search."""
        # This would integrate with your embedding model
        # For now, return placeholder results
        return [
            {
                "content": f"Semantic match for: {query}",
                "similarity": 0.85,
                "file_path": "example.py",
                "line_number": 42
            }
        ]
        
    async def _find_similar(
        self, 
        code_snippet: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find similar code snippets."""
        # Placeholder implementation
        return [
            {
                "content": f"Similar to: {code_snippet[:50]}...",
                "similarity": 0.78,
                "file_path": "similar.py",
                "line_number": 15
            }
        ]
        
    async def _hybrid_search(
        self, 
        query: str, 
        structural_filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and structural queries."""
        # Placeholder implementation
        return [
            {
                "content": f"Hybrid match for: {query}",
                "similarity": 0.82,
                "file_path": "hybrid.py",
                "line_number": 28,
                "structural_match": True
            }
        ]


class FileSystemTool(AgentTool):
    """Tool for file system operations."""
    
    def __init__(self):
        super().__init__(
            name="filesystem_tool",
            description="Tool for file system operations and code reading"
        )
        
    async def execute(
        self, 
        operation: str, 
        **kwargs: Any
    ) -> Any:
        """Execute file system operations."""
        try:
            if operation == "read_file":
                return await self._read_file(**kwargs)
            elif operation == "list_files":
                return await self._list_files(**kwargs)
            elif operation == "find_files":
                return await self._find_files(**kwargs)
            elif operation == "get_file_info":
                return await self._get_file_info(**kwargs)
            else:
                raise ValueError(f"Unknown filesystem operation: {operation}")
                
        except Exception as e:
            logger.error(f"Filesystem tool error: {str(e)}")
            raise
            
    async def _read_file(
        self, 
        file_path: str, 
        start_line: Optional[int] = None,
        end_line: Optional[int] = None
    ) -> Dict[str, Any]:
        """Read file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            if start_line is not None or end_line is not None:
                start = (start_line - 1) if start_line else 0
                end = end_line if end_line else len(lines)
                lines = lines[start:end]
                
            return {
                "file_path": file_path,
                "content": ''.join(lines),
                "line_count": len(lines),
                "start_line": start_line,
                "end_line": end_line
            }
            
        except Exception as e:
            raise Exception(f"Failed to read file {file_path}: {str(e)}")
            
    async def _list_files(
        self, 
        directory: str, 
        pattern: Optional[str] = None,
        recursive: bool = False
    ) -> List[str]:
        """List files in directory."""
        path = Path(directory)
        
        if recursive:
            if pattern:
                files = list(path.rglob(pattern))
            else:
                files = [f for f in path.rglob("*") if f.is_file()]
        else:
            if pattern:
                files = list(path.glob(pattern))
            else:
                files = [f for f in path.iterdir() if f.is_file()]
                
        return [str(f) for f in files]
        
    async def _find_files(
        self, 
        directory: str, 
        name_pattern: str,
        content_pattern: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Find files matching patterns."""
        matching_files = []
        path = Path(directory)
        
        for file_path in path.rglob(name_pattern):
            if file_path.is_file():
                match_info = {
                    "file_path": str(file_path),
                    "name_match": True,
                    "content_match": False
                }
                
                if content_pattern:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if content_pattern in content:
                                match_info["content_match"] = True
                    except:
                        pass  # Skip files that can't be read
                        
                matching_files.append(match_info)
                
        return matching_files
        
    async def _get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        stat = path.stat()
        
        return {
            "file_path": str(path),
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "is_file": path.is_file(),
            "is_directory": path.is_dir(),
            "extension": path.suffix,
            "name": path.name
        }