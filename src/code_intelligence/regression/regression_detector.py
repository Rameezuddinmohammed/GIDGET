"""Automated regression detection and root cause analysis system."""

import logging
import uuid
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from ..git.repository import GitRepository
from .version_comparison import VersionComparator
from .models import (
    RegressionPattern, RegressionFinding, RootCauseAnalysis, ChangeTimeline,
    RegressionType, ChangeSignificance, StructuralDiff, BehavioralChange
)
from .config import get_config

logger = logging.getLogger(__name__)


class RegressionDetector:
    """Detects regressions using pattern matching and analysis."""
    
    def __init__(self, git_repo: GitRepository, version_comparator: VersionComparator):
        """Initialize with git repository and version comparator."""
        self.git_repo = git_repo
        self.version_comparator = version_comparator
        self.config = get_config()
        self.patterns = self._initialize_patterns()
    
    def detect_regressions(
        self, 
        commit_range: List[str],
        focus_files: Optional[List[str]] = None
    ) -> List[RegressionFinding]:
        """Detect regressions across a range of commits."""
        logger.info(f"Detecting regressions across {len(commit_range)} commits")
        
        findings = []
        
        # Analyze each consecutive pair of commits
        for i in range(len(commit_range) - 1):
            old_commit = commit_range[i]
            new_commit = commit_range[i + 1]
            
            # Compare versions
            comparison = self.version_comparator.compare_versions(
                old_commit, new_commit, focus_files
            )
            
            # Apply regression patterns
            commit_findings = self._apply_patterns(comparison, [old_commit, new_commit])
            findings.extend(commit_findings)
        
        # Perform root cause analysis for high-confidence findings
        for finding in findings:
            if finding.confidence >= 0.8:
                finding.root_cause_analysis = self._perform_root_cause_analysis(
                    finding, commit_range
                )
        
        return findings
    
    def _initialize_patterns(self) -> List[RegressionPattern]:
        """Initialize regression detection patterns."""
        return [
            # Functional regression patterns
            RegressionPattern(
                pattern_id="func_signature_change",
                pattern_name="Function Signature Change",
                description="Function signature changed, potentially breaking API",
                regression_type=RegressionType.API_BREAKING,
                detection_rules=[
                    "function_parameters_changed",
                    "function_return_type_changed",
                    "function_removed"
                ],
                confidence_threshold=self.config.pattern_thresholds['func_signature_change']
            ),
            
            RegressionPattern(
                pattern_id="logic_error_pattern",
                pattern_name="Logic Error Pattern",
                description="Changes that commonly introduce logic errors",
                regression_type=RegressionType.LOGIC_ERROR,
                detection_rules=[
                    "conditional_logic_changed",
                    "loop_condition_changed",
                    "exception_handling_changed"
                ],
                confidence_threshold=self.config.pattern_thresholds['logic_error_pattern']
            ),
            
            RegressionPattern(
                pattern_id="dependency_break",
                pattern_name="Dependency Breaking Change",
                description="Changes that break dependency relationships",
                regression_type=RegressionType.DEPENDENCY,
                detection_rules=[
                    "import_removed",
                    "called_function_removed",
                    "class_inheritance_changed"
                ],
                confidence_threshold=self.config.pattern_thresholds['dependency_break']
            ),
            
            RegressionPattern(
                pattern_id="performance_degradation",
                pattern_name="Performance Degradation Pattern",
                description="Changes that commonly cause performance issues",
                regression_type=RegressionType.PERFORMANCE,
                detection_rules=[
                    "nested_loop_added",
                    "synchronous_to_async_change",
                    "database_query_in_loop"
                ],
                confidence_threshold=self.config.pattern_thresholds['performance_degradation']
            ),
            
            RegressionPattern(
                pattern_id="integration_failure",
                pattern_name="Integration Failure Pattern",
                description="Changes that break integration points",
                regression_type=RegressionType.INTEGRATION,
                detection_rules=[
                    "api_endpoint_changed",
                    "data_format_changed",
                    "protocol_changed"
                ],
                confidence_threshold=self.config.pattern_thresholds['integration_failure']
            )
        ]
    
    def _apply_patterns(
        self, 
        comparison, 
        commit_range: List[str]
    ) -> List[RegressionFinding]:
        """Apply regression patterns to version comparison results."""
        findings = []
        
        for pattern in self.patterns:
            pattern_findings = self._apply_single_pattern(
                pattern, comparison, commit_range
            )
            findings.extend(pattern_findings)
        
        return findings 
   
    def _apply_single_pattern(
        self, 
        pattern: RegressionPattern, 
        comparison, 
        commit_range: List[str]
    ) -> List[RegressionFinding]:
        """Apply a single regression pattern to comparison results."""
        findings = []
        
        # Check structural diffs against pattern rules
        for diff in comparison.structural_diffs:
            confidence = self._calculate_pattern_confidence(pattern, diff, comparison)
            
            if confidence >= pattern.confidence_threshold:
                finding = RegressionFinding(
                    finding_id=str(uuid.uuid4()),
                    regression_type=pattern.regression_type,
                    pattern_matched=pattern.pattern_id,
                    confidence=confidence,
                    description=self._generate_finding_description(pattern, diff),
                    affected_elements=[diff.element_name],
                    file_paths=[diff.file_path],
                    commit_range=commit_range,
                    evidence=self._collect_evidence(diff, comparison),
                    severity=diff.significance,
                    suggested_fixes=self._generate_suggested_fixes(pattern, diff)
                )
                findings.append(finding)
        
        # Check behavioral changes against pattern rules
        for behavior_change in comparison.behavioral_changes:
            confidence = self._calculate_behavior_pattern_confidence(
                pattern, behavior_change, comparison
            )
            
            if confidence >= pattern.confidence_threshold:
                finding = RegressionFinding(
                    finding_id=str(uuid.uuid4()),
                    regression_type=pattern.regression_type,
                    pattern_matched=pattern.pattern_id,
                    confidence=confidence,
                    description=self._generate_behavior_finding_description(
                        pattern, behavior_change
                    ),
                    affected_elements=[behavior_change.element_name],
                    file_paths=[behavior_change.file_path],
                    commit_range=commit_range,
                    evidence=behavior_change.evidence,
                    severity=ChangeSignificance.MAJOR,
                    suggested_fixes=self._generate_behavior_suggested_fixes(
                        pattern, behavior_change
                    )
                )
                findings.append(finding)
        
        return findings
    
    def _calculate_pattern_confidence(
        self, 
        pattern: RegressionPattern, 
        diff: StructuralDiff, 
        comparison
    ) -> float:
        """Calculate confidence that a structural diff matches a pattern."""
        confidence = 0.0
        
        if pattern.pattern_id == "func_signature_change":
            if diff.element_type == "function":
                if diff.change_type.value in ["removed", "modified"]:
                    confidence = 0.9
                    # Higher confidence for public functions
                    if not diff.element_name.startswith('_'):
                        confidence = 0.95
        
        elif pattern.pattern_id == "dependency_break":
            if diff.change_type.value == "removed":
                confidence = 0.8
                # Check if other elements depend on this
                for impact in comparison.dependency_impacts:
                    if impact.source_element == diff.element_name:
                        confidence = 0.9
                        break
        
        elif pattern.pattern_id == "logic_error_pattern":
            if diff.change_type.value == "modified":
                # Look for specific indicators in metadata
                if "conditional" in str(diff.metadata).lower():
                    confidence = 0.7
                elif "loop" in str(diff.metadata).lower():
                    confidence = 0.75
        
        elif pattern.pattern_id == "performance_degradation":
            if diff.change_type.value == "modified":
                # Check for performance-related changes
                if diff.impact_score > 0.7:
                    confidence = 0.6
        
        elif pattern.pattern_id == "integration_failure":
            if diff.element_type in ["function", "class"]:
                if diff.significance == ChangeSignificance.CRITICAL:
                    confidence = 0.8
        
        return confidence
    
    def _calculate_behavior_pattern_confidence(
        self, 
        pattern: RegressionPattern, 
        behavior_change: BehavioralChange, 
        comparison
    ) -> float:
        """Calculate confidence for behavioral change patterns."""
        confidence = 0.0
        
        if pattern.pattern_id == "func_signature_change":
            if "signature changed" in behavior_change.change_description.lower():
                confidence = 0.9
        
        elif pattern.pattern_id == "logic_error_pattern":
            if any(keyword in behavior_change.change_description.lower() 
                   for keyword in ["calls", "behavior", "logic"]):
                confidence = 0.7
        
        elif pattern.pattern_id == "performance_degradation":
            if "async" in behavior_change.change_description.lower():
                confidence = 0.6
        
        return confidence
    
    def _generate_finding_description(
        self, 
        pattern: RegressionPattern, 
        diff: StructuralDiff
    ) -> str:
        """Generate description for a regression finding."""
        base_desc = f"{pattern.pattern_name} detected in {diff.element_name}"
        
        if pattern.pattern_id == "func_signature_change":
            return f"{base_desc}: Function signature was {diff.change_type.value}"
        elif pattern.pattern_id == "dependency_break":
            return f"{base_desc}: Dependency relationship broken"
        elif pattern.pattern_id == "logic_error_pattern":
            return f"{base_desc}: Logic change detected that may introduce errors"
        elif pattern.pattern_id == "performance_degradation":
            return f"{base_desc}: Change may cause performance degradation"
        elif pattern.pattern_id == "integration_failure":
            return f"{base_desc}: Integration point may be broken"
        
        return base_desc
    
    def _generate_behavior_finding_description(
        self, 
        pattern: RegressionPattern, 
        behavior_change: BehavioralChange
    ) -> str:
        """Generate description for behavioral change finding."""
        return f"{pattern.pattern_name} in {behavior_change.element_name}: {behavior_change.change_description}"    

    def _collect_evidence(self, diff: StructuralDiff, comparison) -> List[str]:
        """Collect evidence for a regression finding."""
        evidence = []
        
        # Basic change evidence
        evidence.append(f"Element {diff.element_name} was {diff.change_type.value}")
        evidence.append(f"Change significance: {diff.significance.value}")
        
        if diff.impact_score > 0:
            evidence.append(f"Impact score: {diff.impact_score:.2f}")
        
        # Dependency evidence
        for impact in comparison.dependency_impacts:
            if impact.source_element == diff.element_name:
                evidence.append(f"Affects {len(impact.affected_elements)} dependent elements")
                break
        
        # File location evidence
        evidence.append(f"Location: {diff.file_path}")
        
        return evidence
    
    def _generate_suggested_fixes(
        self, 
        pattern: RegressionPattern, 
        diff: StructuralDiff
    ) -> List[str]:
        """Generate suggested fixes for a regression."""
        fixes = []
        
        if pattern.pattern_id == "func_signature_change":
            if diff.change_type.value == "removed":
                fixes.append("Consider adding a deprecated wrapper function")
                fixes.append("Update all callers to use alternative function")
            elif diff.change_type.value == "modified":
                fixes.append("Maintain backward compatibility with default parameters")
                fixes.append("Version the API to support both old and new signatures")
        
        elif pattern.pattern_id == "dependency_break":
            fixes.append("Restore the removed dependency")
            fixes.append("Update dependent code to use alternative")
            fixes.append("Add migration guide for affected components")
        
        elif pattern.pattern_id == "logic_error_pattern":
            fixes.append("Add comprehensive unit tests for the changed logic")
            fixes.append("Review the change with domain experts")
            fixes.append("Consider reverting and implementing incrementally")
        
        elif pattern.pattern_id == "performance_degradation":
            fixes.append("Profile the performance impact")
            fixes.append("Add performance benchmarks")
            fixes.append("Consider optimizing the implementation")
        
        elif pattern.pattern_id == "integration_failure":
            fixes.append("Test integration points thoroughly")
            fixes.append("Update integration documentation")
            fixes.append("Coordinate with dependent systems")
        
        return fixes
    
    def _generate_behavior_suggested_fixes(
        self, 
        pattern: RegressionPattern, 
        behavior_change: BehavioralChange
    ) -> List[str]:
        """Generate suggested fixes for behavioral changes."""
        fixes = []
        
        if "signature changed" in behavior_change.change_description.lower():
            fixes.append("Maintain backward compatibility")
            fixes.append("Update all callers")
        
        if "calls" in behavior_change.change_description.lower():
            fixes.append("Verify new function calls are correct")
            fixes.append("Test integration with called functions")
        
        if "async" in behavior_change.change_description.lower():
            fixes.append("Update all callers to handle async behavior")
            fixes.append("Add proper error handling for async operations")
        
        return fixes or ["Review the behavioral change carefully"]
    
    def _perform_root_cause_analysis(
        self, 
        finding: RegressionFinding, 
        commit_range: List[str]
    ) -> str:
        """Perform root cause analysis for a regression finding."""
        logger.info(f"Performing root cause analysis for {finding.finding_id}")
        
        # Analyze commit sequence
        commit_analysis = self._analyze_commit_sequence(
            finding.affected_elements, commit_range
        )
        
        # Identify primary cause
        primary_cause = self._identify_primary_cause(finding, commit_analysis)
        
        # Build failure chain
        failure_chain = self._build_failure_chain(finding, commit_analysis)
        
        # Generate analysis summary
        analysis = f"Root Cause Analysis:\n"
        analysis += f"Primary Cause: {primary_cause}\n"
        analysis += f"Failure Chain: {' -> '.join(failure_chain)}\n"
        analysis += f"Analysis Method: Pattern matching with commit history analysis"
        
        return analysis
    
    def _analyze_commit_sequence(
        self, 
        affected_elements: List[str], 
        commit_range: List[str]
    ) -> Dict[str, List[str]]:
        """Analyze commit sequence for affected elements."""
        analysis = defaultdict(list)
        
        for commit_sha in commit_range:
            commit_info = self.git_repo.get_commit_info(commit_sha)
            
            # Check if commit affects any of the elements
            for file_change in commit_info.file_changes:
                for element in affected_elements:
                    # Simple heuristic: check if element name appears in commit
                    if element in commit_info.message or element in file_change.file_path:
                        analysis[element].append(commit_sha)
        
        return dict(analysis)
    
    def _identify_primary_cause(
        self, 
        finding: RegressionFinding, 
        commit_analysis: Dict[str, List[str]]
    ) -> str:
        """Identify the primary cause of the regression."""
        if finding.regression_type == RegressionType.API_BREAKING:
            return "API contract was modified without maintaining backward compatibility"
        elif finding.regression_type == RegressionType.DEPENDENCY:
            return "Dependency relationship was broken by removing or modifying required component"
        elif finding.regression_type == RegressionType.LOGIC_ERROR:
            return "Logic change introduced error in business logic or control flow"
        elif finding.regression_type == RegressionType.PERFORMANCE:
            return "Implementation change introduced performance bottleneck"
        elif finding.regression_type == RegressionType.INTEGRATION:
            return "Integration contract was modified without coordinating with dependent systems"
        else:
            return "Functional behavior was modified in an incompatible way"
    
    def _build_failure_chain(
        self, 
        finding: RegressionFinding, 
        commit_analysis: Dict[str, List[str]]
    ) -> List[str]:
        """Build the failure chain for the regression."""
        chain = []
        
        # Start with the change
        chain.append(f"Code change in {', '.join(finding.affected_elements)}")
        
        # Add pattern-specific chain elements
        if finding.regression_type == RegressionType.API_BREAKING:
            chain.extend([
                "API signature modified",
                "Existing callers become incompatible",
                "Runtime errors or compilation failures"
            ])
        elif finding.regression_type == RegressionType.DEPENDENCY:
            chain.extend([
                "Dependency removed or modified",
                "Dependent components lose required functionality",
                "Integration failures occur"
            ])
        elif finding.regression_type == RegressionType.LOGIC_ERROR:
            chain.extend([
                "Business logic modified",
                "Incorrect behavior in edge cases",
                "Functional regression manifests"
            ])
        
        return chain
    
    def build_regression_timeline(
        self, 
        element_name: str, 
        commit_range: List[str]
    ) -> ChangeTimeline:
        """Build a timeline of changes for regression analysis."""
        timeline_events = []
        regression_points = []
        impact_progression = []
        
        for i, commit_sha in enumerate(commit_range):
            commit_info = self.git_repo.get_commit_info(commit_sha)
            
            # Check if this commit affects the element
            element_affected = False
            for file_change in commit_info.file_changes:
                if element_name in file_change.file_path or element_name in commit_info.message:
                    element_affected = True
                    break
            
            if element_affected:
                event = {
                    'commit': commit_sha,
                    'timestamp': commit_info.committed_date.isoformat(),
                    'message': commit_info.message,
                    'author': commit_info.author_name,
                    'changes': [fc.file_path for fc in commit_info.file_changes]
                }
                timeline_events.append(event)
                
                # Simple heuristic for regression points
                if any(keyword in commit_info.message.lower() 
                       for keyword in ['fix', 'bug', 'error', 'issue']):
                    regression_points.append(commit_sha)
                
                # Calculate impact progression (simplified)
                impact_score = len(commit_info.file_changes) * 0.1
                impact_progression.append(min(impact_score, 1.0))
        
        return ChangeTimeline(
            element_name=element_name,
            file_path="",  # Will be filled by caller
            timeline_events=timeline_events,
            regression_points=regression_points,
            impact_progression=impact_progression
        )