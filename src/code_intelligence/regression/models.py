"""Data models for regression debugging and analysis."""

from enum import Enum
from typing import List, Optional, Dict, Any, Set
from datetime import datetime
from pydantic import BaseModel, Field

from ..parsing.models import CodeElement


class ChangeType(str, Enum):
    """Types of changes between versions."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    RENAMED = "renamed"
    MOVED = "moved"


class ChangeSignificance(str, Enum):
    """Significance levels for code changes."""
    CRITICAL = "critical"      # Breaking changes, API modifications
    MAJOR = "major"           # Significant functionality changes
    MINOR = "minor"           # Small improvements, bug fixes
    TRIVIAL = "trivial"       # Formatting, comments, documentation


class RegressionType(str, Enum):
    """Types of regressions that can be detected."""
    FUNCTIONAL = "functional"     # Function behavior changed
    PERFORMANCE = "performance"   # Performance degradation
    API_BREAKING = "api_breaking" # API contract broken
    DEPENDENCY = "dependency"     # Dependency issues
    LOGIC_ERROR = "logic_error"   # Logic bugs introduced
    INTEGRATION = "integration"   # Integration failures


class StructuralDiff(BaseModel):
    """Represents a structural difference between code versions."""
    element_name: str
    element_type: str
    change_type: ChangeType
    old_version: Optional[CodeElement] = None
    new_version: Optional[CodeElement] = None
    file_path: str
    significance: ChangeSignificance = ChangeSignificance.MINOR
    impact_score: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BehavioralChange(BaseModel):
    """Represents a behavioral change in code functionality."""
    element_name: str
    change_description: str
    change_type: ChangeType
    file_path: str
    line_numbers: List[int] = Field(default_factory=list)
    impact_analysis: str = ""
    confidence: float = 0.0  # 0.0 to 1.0
    evidence: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DependencyImpact(BaseModel):
    """Represents the impact of changes on dependencies."""
    source_element: str
    affected_elements: List[str] = Field(default_factory=list)
    impact_type: str  # "direct", "transitive", "circular"
    severity: ChangeSignificance = ChangeSignificance.MINOR
    ripple_effects: List[str] = Field(default_factory=list)
    file_paths: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VersionComparisonResult(BaseModel):
    """Complete result of version comparison analysis."""
    old_commit: str
    new_commit: str
    structural_diffs: List[StructuralDiff] = Field(default_factory=list)
    behavioral_changes: List[BehavioralChange] = Field(default_factory=list)
    dependency_impacts: List[DependencyImpact] = Field(default_factory=list)
    overall_significance: ChangeSignificance = ChangeSignificance.MINOR
    change_summary: str = ""
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RegressionPattern(BaseModel):
    """Represents a pattern that indicates potential regression."""
    pattern_id: str
    pattern_name: str
    description: str
    regression_type: RegressionType
    detection_rules: List[str] = Field(default_factory=list)
    confidence_threshold: float = 0.7
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RegressionFinding(BaseModel):
    """Represents a detected regression."""
    finding_id: str
    regression_type: RegressionType
    pattern_matched: str
    confidence: float
    description: str
    affected_elements: List[str] = Field(default_factory=list)
    file_paths: List[str] = Field(default_factory=list)
    commit_range: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    severity: ChangeSignificance = ChangeSignificance.MINOR
    root_cause_analysis: Optional[str] = None
    suggested_fixes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RootCauseAnalysis(BaseModel):
    """Detailed root cause analysis for a regression."""
    regression_id: str
    primary_cause: str
    contributing_factors: List[str] = Field(default_factory=list)
    failure_chain: List[str] = Field(default_factory=list)
    commit_sequence: List[str] = Field(default_factory=list)
    code_changes: List[StructuralDiff] = Field(default_factory=list)
    confidence: float = 0.0
    analysis_method: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CommitIntent(BaseModel):
    """Represents the extracted intent from a commit."""
    commit_sha: str
    message: str
    extracted_intent: str
    intent_categories: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    keywords: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChangeCorrelation(BaseModel):
    """Represents correlation between code changes and stated intent."""
    commit_sha: str
    stated_intent: str
    actual_changes: List[str] = Field(default_factory=list)
    correlation_score: float = 0.0  # -1.0 to 1.0
    discrepancies: List[str] = Field(default_factory=list)
    alignment_analysis: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChangeTimeline(BaseModel):
    """Represents a timeline of changes for regression analysis."""
    element_name: str
    file_path: str
    timeline_events: List[Dict[str, Any]] = Field(default_factory=list)
    regression_points: List[str] = Field(default_factory=list)  # Commit SHAs
    impact_progression: List[float] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)