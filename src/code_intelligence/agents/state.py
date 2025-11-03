"""Agent state management and data models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class QueryScope(str, Enum):
    """Scope of the query analysis."""
    FUNCTION = "function"
    CLASS = "class"
    FILE = "file"
    MODULE = "module"
    REPOSITORY = "repository"


class ParsedQuery(BaseModel):
    """Parsed natural language query structure."""
    intent: str = Field(description="Primary intent of the query")
    entities: List[str] = Field(default_factory=list, description="Code entities mentioned")
    time_range: Optional[str] = Field(None, description="Time range for analysis")
    scope: QueryScope = Field(default=QueryScope.REPOSITORY, description="Analysis scope")
    keywords: List[str] = Field(default_factory=list, description="Key terms extracted")


class CodeElement(BaseModel):
    """Represents a code element in the analysis."""
    name: str
    type: str  # function, class, file, etc.
    file_path: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    signature_hash: Optional[str] = None


class TimeRange(BaseModel):
    """Time range for temporal analysis."""
    start_commit: Optional[str] = None
    end_commit: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


class Citation(BaseModel):
    """Citation reference for findings."""
    file_path: str
    line_number: Optional[int] = None
    commit_sha: Optional[str] = None
    url: Optional[str] = None
    description: str


class AgentFinding(BaseModel):
    """Finding from an individual agent."""
    agent_name: str
    finding_type: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: List[Citation] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentProgress(BaseModel):
    """Progress tracking for agent execution."""
    current_agent: str
    completed_steps: List[str] = Field(default_factory=list)
    total_steps: int = 0
    estimated_remaining_seconds: Optional[int] = None
    status: str = "initializing"


class AgentState(BaseModel):
    """Central state object shared between all agents."""
    
    # Query information
    query: Dict[str, Any] = Field(default_factory=dict)
    
    # Repository context
    repository: Dict[str, Any] = Field(default_factory=dict)
    
    # Analysis data
    analysis: Dict[str, Any] = Field(default_factory=dict)
    
    # Verification results
    verification: Dict[str, Any] = Field(default_factory=dict)
    
    # Progress tracking
    progress: Dict[str, Any] = Field(default_factory=dict)
    
    # Agent communication
    agent_results: Dict[str, List[AgentFinding]] = Field(default_factory=dict)
    
    # Error handling
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Metadata
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    model_config = {"arbitrary_types_allowed": True}
        
    def update_progress(self, agent_name: str, step: str, status: str = "processing") -> None:
        """Update progress tracking."""
        self.progress.update({
            "current_agent": agent_name,
            "current_step": step,
            "status": status,
            "updated_at": datetime.utcnow()
        })
        self.updated_at = datetime.utcnow()
        
    def add_finding(self, agent_name: str, finding: AgentFinding) -> None:
        """Add a finding from an agent."""
        if agent_name not in self.agent_results:
            self.agent_results[agent_name] = []
        self.agent_results[agent_name].append(finding)
        self.updated_at = datetime.utcnow()
        
    def add_error(self, error: str, agent_name: Optional[str] = None) -> None:
        """Add an error to the state."""
        error_msg = f"[{agent_name}] {error}" if agent_name else error
        self.errors.append(error_msg)
        self.updated_at = datetime.utcnow()
        
    def add_warning(self, warning: str, agent_name: Optional[str] = None) -> None:
        """Add a warning to the state."""
        warning_msg = f"[{agent_name}] {warning}" if agent_name else warning
        self.warnings.append(warning_msg)
        self.updated_at = datetime.utcnow()
        
    def get_findings_by_agent(self, agent_name: str) -> List[AgentFinding]:
        """Get all findings from a specific agent."""
        return self.agent_results.get(agent_name, [])
        
    def get_all_findings(self) -> List[AgentFinding]:
        """Get all findings from all agents."""
        all_findings = []
        for findings in self.agent_results.values():
            all_findings.extend(findings)
        return all_findings
        
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
        
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0