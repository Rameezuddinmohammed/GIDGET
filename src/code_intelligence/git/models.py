"""Data models for git repository analysis."""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChangeType(str, Enum):
    """Types of file changes in git commits."""
    ADDED = "A"
    MODIFIED = "M"
    DELETED = "D"
    RENAMED = "R"
    COPIED = "C"
    UNMERGED = "U"
    UNKNOWN = "X"
    BROKEN = "B"


class FileChange(BaseModel):
    """Represents a file change in a git commit."""
    file_path: str
    change_type: ChangeType
    old_file_path: Optional[str] = None  # For renames/copies
    insertions: int = 0
    deletions: int = 0
    binary: bool = False


class CommitInfo(BaseModel):
    """Represents git commit information."""
    sha: str
    message: str
    author_name: str
    author_email: str
    committer_name: str
    committer_email: str
    authored_date: datetime
    committed_date: datetime
    parents: List[str] = Field(default_factory=list)
    file_changes: List[FileChange] = Field(default_factory=list)
    stats: Dict[str, int] = Field(default_factory=dict)  # insertions, deletions, files


class RepositoryStatus(str, Enum):
    """Repository analysis status."""
    NOT_ANALYZED = "not_analyzed"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    ERROR = "error"
    OUTDATED = "outdated"


class RepositoryInfo(BaseModel):
    """Repository metadata and status."""
    id: str
    name: str
    url: str
    local_path: str
    status: RepositoryStatus
    last_analyzed: Optional[datetime] = None
    last_commit_sha: Optional[str] = None
    commit_count: int = 0
    supported_languages: List[str] = Field(default_factory=list)
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)