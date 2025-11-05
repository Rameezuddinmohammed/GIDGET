"""Configuration for regression debugging system."""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class RegressionConfig:
    """Configuration for regression debugging system."""
    
    # Version comparison settings
    significance_weights: Dict[str, float] = field(default_factory=lambda: {
        'function_signature_change': 0.8,
        'function_removal': 0.9,
        'class_hierarchy_change': 0.7,
        'api_change': 0.9,
        'dependency_change': 0.6,
        'logic_change': 0.5
    })
    
    # Regression pattern confidence thresholds
    pattern_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'func_signature_change': 0.9,
        'logic_error_pattern': 0.7,
        'dependency_break': 0.8,
        'performance_degradation': 0.6,
        'integration_failure': 0.8
    })
    
    # Intent analysis settings
    intent_weights: Dict[str, float] = field(default_factory=lambda: {
        'fix': 0.9,
        'feature': 0.8,
        'refactor': 0.7,
        'performance': 0.8,
        'documentation': 0.5,
        'test': 0.6,
        'security': 0.9,
        'dependency': 0.6
    })
    
    # Correlation scoring settings
    correlation_weights: Dict[str, float] = field(default_factory=lambda: {
        'keyword_weight': 0.4,
        'category_weight': 0.6,
        'category_match_score': 0.3
    })
    
    # Performance settings
    max_dependency_depth: int = 3
    max_ripple_effects: int = 10
    max_timeline_events: int = 100
    
    # File processing settings
    supported_extensions: set = field(default_factory=lambda: {
        '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h'
    })
    
    # Language support
    supported_languages: set = field(default_factory=lambda: {
        'python', 'javascript', 'typescript', 'java', 'cpp', 'c'
    })


# Default configuration instance
DEFAULT_CONFIG = RegressionConfig()


def get_config() -> RegressionConfig:
    """Get the current regression configuration."""
    return DEFAULT_CONFIG


def update_config(**kwargs) -> None:
    """Update configuration values."""
    for key, value in kwargs.items():
        if hasattr(DEFAULT_CONFIG, key):
            setattr(DEFAULT_CONFIG, key, value)
        else:
            raise ValueError(f"Unknown configuration key: {key}")


def reset_config() -> None:
    """Reset configuration to defaults."""
    global DEFAULT_CONFIG
    DEFAULT_CONFIG = RegressionConfig()