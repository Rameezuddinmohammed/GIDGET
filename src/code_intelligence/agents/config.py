"""Configuration management for agents."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import os


@dataclass
class AgentLimits:
    """Configuration for agent processing limits."""
    max_commits: int = 50
    max_findings_per_agent: int = 100
    max_citations_per_finding: int = 10
    max_elements_per_analysis: int = 20
    max_llm_retries: int = 3
    max_database_retries: int = 2


@dataclass
class AgentTimeouts:
    """Configuration for agent timeouts."""
    llm_timeout_seconds: int = 30
    database_timeout_seconds: int = 15
    file_operation_timeout_seconds: int = 5
    agent_execution_timeout_seconds: int = 300


@dataclass
class AgentThresholds:
    """Configuration for agent decision thresholds."""
    confidence_threshold: float = 0.7
    conflict_detection_threshold: float = 0.3
    coupling_normalization_factor: float = 20.0
    content_overlap_threshold: float = 0.3
    validation_score_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "strong": 0.8,
        "moderate": 0.6,
        "weak": 0.4
    })


@dataclass
class AgentConfiguration:
    """Central configuration for all agents."""
    limits: AgentLimits = field(default_factory=AgentLimits)
    timeouts: AgentTimeouts = field(default_factory=AgentTimeouts)
    thresholds: AgentThresholds = field(default_factory=AgentThresholds)
    
    # Environment-specific settings
    environment: str = "development"
    debug_mode: bool = False
    enable_caching: bool = True
    enable_metrics: bool = True
    
    # Database settings
    neo4j_enabled: bool = True
    supabase_enabled: bool = True
    
    @classmethod
    def from_environment(cls) -> 'AgentConfiguration':
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        config.environment = os.getenv("AGENT_ENVIRONMENT", "development")
        config.debug_mode = os.getenv("AGENT_DEBUG", "false").lower() == "true"
        config.enable_caching = os.getenv("AGENT_ENABLE_CACHING", "true").lower() == "true"
        config.enable_metrics = os.getenv("AGENT_ENABLE_METRICS", "true").lower() == "true"
        
        # Limits
        if os.getenv("AGENT_MAX_COMMITS"):
            config.limits.max_commits = int(os.getenv("AGENT_MAX_COMMITS"))
        if os.getenv("AGENT_MAX_FINDINGS"):
            config.limits.max_findings_per_agent = int(os.getenv("AGENT_MAX_FINDINGS"))
            
        # Timeouts
        if os.getenv("AGENT_LLM_TIMEOUT"):
            config.timeouts.llm_timeout_seconds = int(os.getenv("AGENT_LLM_TIMEOUT"))
        if os.getenv("AGENT_DB_TIMEOUT"):
            config.timeouts.database_timeout_seconds = int(os.getenv("AGENT_DB_TIMEOUT"))
            
        # Thresholds
        if os.getenv("AGENT_CONFIDENCE_THRESHOLD"):
            config.thresholds.confidence_threshold = float(os.getenv("AGENT_CONFIDENCE_THRESHOLD"))
            
        return config


# Global configuration instance
_global_config: Optional[AgentConfiguration] = None


def get_agent_config() -> AgentConfiguration:
    """Get the global agent configuration."""
    global _global_config
    if _global_config is None:
        _global_config = AgentConfiguration.from_environment()
    return _global_config


def set_agent_config(config: AgentConfiguration) -> None:
    """Set the global agent configuration."""
    global _global_config
    _global_config = config