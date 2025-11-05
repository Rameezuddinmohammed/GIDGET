"""Regression debugging and analysis capabilities."""

from .version_comparison import VersionComparator
from .regression_detector import RegressionDetector, RegressionPattern, RegressionFinding
from .intent_analyzer import IntentAnalyzer, CommitIntent, ChangeCorrelation
from .models import (
    VersionComparisonResult, DependencyImpact, ChangeSignificance,
    RegressionType, RootCauseAnalysis, ChangeTimeline,
    StructuralDiff, BehavioralChange, ChangeType
)
from .config import RegressionConfig, get_config, update_config, reset_config

__all__ = [
    'VersionComparator', 'StructuralDiff', 'BehavioralChange', 'ChangeType',
    'RegressionDetector', 'RegressionPattern', 'RegressionFinding',
    'IntentAnalyzer', 'CommitIntent', 'ChangeCorrelation',
    'VersionComparisonResult', 'DependencyImpact', 'ChangeSignificance',
    'RegressionType', 'RootCauseAnalysis', 'ChangeTimeline',
    'RegressionConfig', 'get_config', 'update_config', 'reset_config'
]