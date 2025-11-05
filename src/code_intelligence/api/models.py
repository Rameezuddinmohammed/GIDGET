"""API data models and schemas."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict

from ..agents.state import QueryScope, AgentFinding, Citation


class QueryStatus(str, Enum):
    """Status of a query execution."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RepositoryStatus(str, Enum):
    """Status of repository analysis."""
    NOT_ANALYZED = "not_analyzed"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    ANALYSIS_FAILED = "analysis_failed"
    UPDATING = "updating"


class QueryRequest(BaseModel):
    """Request model for submitting a query."""
    repository_url: str = Field(description="Git repository URL")
    query: str = Field(description="Natural language query")
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Query options and preferences"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "repository_url": "https://github.com/user/repo.git",
                "query": "What changed in the authentication system since last week?",
                "options": {
                    "max_commits": 100,
                    "languages": ["python", "javascript"],
                    "include_tests": False
                }
            }
        }
    )


class QueryResponse(BaseModel):
    """Response model for query submission."""
    query_id: str = Field(description="Unique query identifier")
    status: QueryStatus = Field(description="Current query status")
    message: str = Field(description="Status message")
    estimated_duration_seconds: Optional[int] = Field(
        default=None,
        description="Estimated completion time in seconds"
    )


class QueryProgress(BaseModel):
    """Progress information for a query."""
    current_agent: str = Field(description="Currently executing agent")
    completed_steps: List[str] = Field(description="Completed processing steps")
    total_steps: int = Field(description="Total number of steps")
    progress_percentage: float = Field(description="Progress as percentage (0-100)")
    estimated_remaining_seconds: Optional[int] = Field(
        default=None,
        description="Estimated remaining time in seconds"
    )
    current_step: str = Field(description="Current processing step")


class AnalysisResult(BaseModel):
    """Complete analysis result."""
    summary: str = Field(description="Executive summary of findings")
    findings: List[AgentFinding] = Field(description="Detailed findings from agents")
    confidence_score: float = Field(description="Overall confidence score (0-1)")
    citations: List[Citation] = Field(description="Source citations")
    metadata: Dict[str, Any] = Field(description="Additional metadata")
    processing_time_seconds: float = Field(description="Total processing time")


class QueryResult(BaseModel):
    """Complete query result with status and findings."""
    query_id: str = Field(description="Query identifier")
    status: QueryStatus = Field(description="Query status")
    progress: Optional[QueryProgress] = Field(default=None, description="Progress information")
    results: Optional[AnalysisResult] = Field(default=None, description="Analysis results")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    created_at: datetime = Field(description="Query creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")


class RepositoryRequest(BaseModel):
    """Request model for repository registration."""
    url: str = Field(description="Git repository URL")
    name: Optional[str] = Field(default=None, description="Repository display name")
    auto_sync: bool = Field(default=True, description="Enable automatic synchronization")
    analysis_options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Repository analysis options"
    )


class RepositoryInfo(BaseModel):
    """Repository information model."""
    id: str = Field(description="Repository identifier")
    name: str = Field(description="Repository name")
    url: str = Field(description="Repository URL")
    status: RepositoryStatus = Field(description="Analysis status")
    last_analyzed: Optional[datetime] = Field(default=None, description="Last analysis timestamp")
    commit_count: int = Field(description="Number of commits analyzed")
    supported_languages: List[str] = Field(description="Detected programming languages")
    file_count: int = Field(description="Number of files analyzed")
    lines_of_code: int = Field(description="Total lines of code")
    created_at: datetime = Field(description="Repository registration timestamp")
    updated_at: datetime = Field(description="Last update timestamp")


class UserProfile(BaseModel):
    """User profile model."""
    user_id: str = Field(description="User identifier")
    email: str = Field(description="User email")
    preferences: Dict[str, Any] = Field(description="User preferences")
    created_at: datetime = Field(description="Account creation timestamp")
    last_active: datetime = Field(description="Last activity timestamp")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class HealthCheck(BaseModel):
    """Health check response model."""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    services: Dict[str, str] = Field(description="Service component statuses")


class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str = Field(description="Message type")
    query_id: Optional[str] = Field(default=None, description="Associated query ID")
    data: Dict[str, Any] = Field(description="Message data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")


class QueryHistoryItem(BaseModel):
    """Query history item model."""
    query_id: str = Field(description="Query identifier")
    query: str = Field(description="Original query text")
    repository_name: str = Field(description="Repository name")
    status: QueryStatus = Field(description="Query status")
    confidence_score: Optional[float] = Field(default=None, description="Result confidence")
    created_at: datetime = Field(description="Query creation timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")


class QueryHistory(BaseModel):
    """Query history response model."""
    queries: List[QueryHistoryItem] = Field(description="Query history items")
    total_count: int = Field(description="Total number of queries")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Items per page")


class ExportRequest(BaseModel):
    """Request model for result export."""
    query_id: str = Field(description="Query identifier to export")
    format: str = Field(description="Export format (json, markdown, pdf)")
    include_citations: bool = Field(default=True, description="Include citations in export")
    include_metadata: bool = Field(default=False, description="Include metadata in export")


class ExportResponse(BaseModel):
    """Response model for result export."""
    export_id: str = Field(description="Export identifier")
    download_url: str = Field(description="Download URL for exported file")
    expires_at: datetime = Field(description="URL expiration timestamp")
    format: str = Field(description="Export format")
    file_size_bytes: int = Field(description="File size in bytes")