"""Data models for ingestion pipeline."""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class IngestionStatus(str, Enum):
    """Status of ingestion jobs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IngestionJob(BaseModel):
    """Represents an ingestion job for a repository."""
    id: str
    repository_id: str
    repository_path: str
    status: IngestionStatus = IngestionStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Progress tracking
    total_commits: int = 0
    processed_commits: int = 0
    total_files: int = 0
    processed_files: int = 0
    
    # Configuration
    max_commits: Optional[int] = None
    include_patterns: List[str] = Field(default_factory=lambda: ['*.py', '*.js', '*.jsx', '*.ts', '*.tsx'])
    exclude_patterns: List[str] = Field(default_factory=lambda: ['node_modules/**', '.git/**', '__pycache__/**'])
    
    # Results
    ingested_elements: int = 0
    ingested_relationships: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def progress_percentage(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_commits == 0:
            return 0.0
        return (self.processed_commits / self.total_commits) * 100
    
    def update_progress(self, processed_commits: int = None, processed_files: int = None):
        """Update progress counters."""
        if processed_commits is not None:
            self.processed_commits = processed_commits
        if processed_files is not None:
            self.processed_files = processed_files


class GraphNode(BaseModel):
    """Represents a node to be created in the graph database."""
    labels: List[str]  # Neo4j node labels
    properties: Dict[str, Any]
    unique_key: str  # Unique identifier for the node


class GraphRelationship(BaseModel):
    """Represents a relationship to be created in the graph database."""
    source_key: str  # Unique key of source node
    target_key: str  # Unique key of target node
    relationship_type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class GraphBatch(BaseModel):
    """Represents a batch of nodes and relationships for graph ingestion."""
    nodes: List[GraphNode] = Field(default_factory=list)
    relationships: List[GraphRelationship] = Field(default_factory=list)
    commit_sha: str
    repository_id: str
    
    def add_node(self, labels: List[str], properties: Dict[str, Any], unique_key: str):
        """Add a node to the batch."""
        self.nodes.append(GraphNode(
            labels=labels,
            properties=properties,
            unique_key=unique_key
        ))
    
    def add_relationship(self, source_key: str, target_key: str, rel_type: str, properties: Dict[str, Any] = None):
        """Add a relationship to the batch."""
        self.relationships.append(GraphRelationship(
            source_key=source_key,
            target_key=target_key,
            relationship_type=rel_type,
            properties=properties or {}
        ))