"""Comprehensive version comparison and diff analysis system."""

import logging
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import difflib

from ..git.repository import GitRepository
from ..parsing.models import CodeElement, FunctionElement, ClassElement, ParsedFile
from ..parsing.parser import CodeParser
from .models import (
    VersionComparisonResult, StructuralDiff, BehavioralChange, DependencyImpact,
    ChangeType, ChangeSignificance
)
from .config import get_config

logger = logging.getLogger(__name__)


class RegressionAnalysisError(Exception):
    """Regression analysis related errors."""
    pass


class VersionComparator:
    """Compares code versions and analyzes structural and behavioral changes."""
    
    def __init__(self, git_repo: GitRepository, parser: CodeParser):
        """Initialize with git repository and code parser."""
        self.git_repo = git_repo
        self.parser = parser
        self.config = get_config()
        self._significance_weights = self.config.significance_weights
        self._parsed_files_cache: Dict[str, Dict[str, ParsedFile]] = {}  # Cache for parsed files
    
    def compare_versions(
        self, 
        old_commit: str, 
        new_commit: str,
        file_patterns: Optional[List[str]] = None
    ) -> VersionComparisonResult:
        """Compare two versions and analyze all changes."""
        logger.info(f"Comparing versions {old_commit[:8]} -> {new_commit[:8]}")
        
        # Parse both versions
        old_files = self._parse_version(old_commit, file_patterns)
        new_files = self._parse_version(new_commit, file_patterns)
        
        # Analyze structural differences
        structural_diffs = self._analyze_structural_diffs(old_files, new_files)
        
        # Analyze behavioral changes
        behavioral_changes = self._analyze_behavioral_changes(
            old_files, new_files, structural_diffs
        )
        
        # Analyze dependency impacts
        dependency_impacts = self._analyze_dependency_impacts(
            old_files, new_files, structural_diffs
        )
        
        # Calculate overall significance
        overall_significance = self._calculate_overall_significance(
            structural_diffs, behavioral_changes, dependency_impacts
        )
        
        # Generate change summary
        change_summary = self._generate_change_summary(
            structural_diffs, behavioral_changes, dependency_impacts
        )
        
        return VersionComparisonResult(
            old_commit=old_commit,
            new_commit=new_commit,
            structural_diffs=structural_diffs,
            behavioral_changes=behavioral_changes,
            dependency_impacts=dependency_impacts,
            overall_significance=overall_significance,
            change_summary=change_summary
        )
    
    def _parse_version(
        self, 
        commit_sha: str, 
        file_patterns: Optional[List[str]] = None
    ) -> Dict[str, ParsedFile]:
        """Parse all files in a specific version."""
        # Check cache first
        cache_key = f"{commit_sha}:{str(file_patterns)}"
        if cache_key in self._parsed_files_cache:
            logger.debug(f"Using cached parsed files for {commit_sha[:8]}")
            return self._parsed_files_cache[cache_key]
        
        # Checkout the specific commit
        original_commit = self.git_repo.current_commit
        try:
            self.git_repo.checkout(commit_sha)
            
            # Get all supported files
            files_to_parse = self._get_files_to_parse(file_patterns)
            
            parsed_files = {}
            for file_path in files_to_parse:
                try:
                    parsed_file = self.parser.parse_file(file_path)
                    parsed_files[file_path] = parsed_file
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path}: {e}")
            
            # Cache the results
            self._parsed_files_cache[cache_key] = parsed_files
            
            return parsed_files
            
        finally:
            # Restore original commit
            self.git_repo.checkout(original_commit)
    
    def _get_files_to_parse(self, file_patterns: Optional[List[str]] = None) -> List[str]:
        """Get list of files to parse based on patterns."""
        supported_extensions = self.config.supported_extensions
        files = []
        
        import os
        for root, dirs, filenames in os.walk(self.git_repo.repo_path):
            # Skip .git directory
            if '.git' in dirs:
                dirs.remove('.git')
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, self.git_repo.repo_path)
                
                # Check extension
                if any(rel_path.endswith(ext) for ext in supported_extensions):
                    # Apply file patterns if specified
                    if not file_patterns or any(
                        pattern in rel_path for pattern in file_patterns
                    ):
                        files.append(rel_path)
        
        return files   
 
    def _analyze_structural_diffs(
        self, 
        old_files: Dict[str, ParsedFile], 
        new_files: Dict[str, ParsedFile]
    ) -> List[StructuralDiff]:
        """Analyze structural differences between versions."""
        diffs = []
        
        # Get all file paths from both versions
        all_files = set(old_files.keys()) | set(new_files.keys())
        
        for file_path in all_files:
            old_file = old_files.get(file_path)
            new_file = new_files.get(file_path)
            
            if not old_file and new_file:
                # File added
                for element in new_file.elements:
                    diffs.append(StructuralDiff(
                        element_name=element.name,
                        element_type=element.element_type.value,
                        change_type=ChangeType.ADDED,
                        new_version=element,
                        file_path=file_path,
                        significance=self._calculate_element_significance(element, ChangeType.ADDED)
                    ))
            
            elif old_file and not new_file:
                # File removed
                for element in old_file.elements:
                    diffs.append(StructuralDiff(
                        element_name=element.name,
                        element_type=element.element_type.value,
                        change_type=ChangeType.REMOVED,
                        old_version=element,
                        file_path=file_path,
                        significance=self._calculate_element_significance(element, ChangeType.REMOVED)
                    ))
            
            elif old_file and new_file:
                # File exists in both versions - compare elements
                diffs.extend(self._compare_file_elements(old_file, new_file, file_path))
        
        return diffs
    
    def _compare_file_elements(
        self, 
        old_file: ParsedFile, 
        new_file: ParsedFile, 
        file_path: str
    ) -> List[StructuralDiff]:
        """Compare elements within a single file."""
        diffs = []
        
        # Create element maps by name and type
        old_elements = {(elem.name, elem.element_type): elem for elem in old_file.elements}
        new_elements = {(elem.name, elem.element_type): elem for elem in new_file.elements}
        
        all_element_keys = set(old_elements.keys()) | set(new_elements.keys())
        
        for element_key in all_element_keys:
            old_elem = old_elements.get(element_key)
            new_elem = new_elements.get(element_key)
            
            if not old_elem and new_elem:
                # Element added
                diffs.append(StructuralDiff(
                    element_name=new_elem.name,
                    element_type=new_elem.element_type.value,
                    change_type=ChangeType.ADDED,
                    new_version=new_elem,
                    file_path=file_path,
                    significance=self._calculate_element_significance(new_elem, ChangeType.ADDED)
                ))
            
            elif old_elem and not new_elem:
                # Element removed
                diffs.append(StructuralDiff(
                    element_name=old_elem.name,
                    element_type=old_elem.element_type.value,
                    change_type=ChangeType.REMOVED,
                    old_version=old_elem,
                    file_path=file_path,
                    significance=self._calculate_element_significance(old_elem, ChangeType.REMOVED)
                ))
            
            elif old_elem and new_elem:
                # Element exists in both - check for modifications
                if self._elements_differ(old_elem, new_elem):
                    change_type = ChangeType.MOVED if (
                        old_elem.start_line != new_elem.start_line or 
                        old_elem.end_line != new_elem.end_line
                    ) else ChangeType.MODIFIED
                    
                    significance = self._calculate_modification_significance(old_elem, new_elem)
                    
                    diffs.append(StructuralDiff(
                        element_name=old_elem.name,
                        element_type=old_elem.element_type.value,
                        change_type=change_type,
                        old_version=old_elem,
                        new_version=new_elem,
                        file_path=file_path,
                        significance=significance,
                        impact_score=self._calculate_impact_score(old_elem, new_elem)
                    ))
        
        return diffs
    
    def _elements_differ(self, old_elem: CodeElement, new_elem: CodeElement) -> bool:
        """Check if two elements are structurally different."""
        # Compare signature hashes first
        if old_elem.signature_hash != new_elem.signature_hash:
            return True
        
        # Compare positions
        if (old_elem.start_line != new_elem.start_line or 
            old_elem.end_line != new_elem.end_line):
            return True
        
        # Type-specific comparisons
        if isinstance(old_elem, FunctionElement) and isinstance(new_elem, FunctionElement):
            return (
                old_elem.parameters != new_elem.parameters or
                old_elem.return_type != new_elem.return_type or
                old_elem.is_async != new_elem.is_async or
                old_elem.decorators != new_elem.decorators
            )
        
        elif isinstance(old_elem, ClassElement) and isinstance(new_elem, ClassElement):
            return (
                old_elem.base_classes != new_elem.base_classes or
                old_elem.methods != new_elem.methods or
                old_elem.attributes != new_elem.attributes
            )
        
        return False    

    def _analyze_behavioral_changes(
        self,
        old_files: Dict[str, ParsedFile],
        new_files: Dict[str, ParsedFile],
        structural_diffs: List[StructuralDiff]
    ) -> List[BehavioralChange]:
        """Analyze behavioral changes based on structural diffs."""
        behavioral_changes = []
        
        for diff in structural_diffs:
            if diff.change_type == ChangeType.MODIFIED and diff.old_version and diff.new_version:
                # Analyze function behavior changes
                if isinstance(diff.old_version, FunctionElement) and isinstance(diff.new_version, FunctionElement):
                    behavior_change = self._analyze_function_behavior_change(
                        diff.old_version, diff.new_version, diff.file_path
                    )
                    if behavior_change:
                        behavioral_changes.append(behavior_change)
                
                # Analyze class behavior changes
                elif isinstance(diff.old_version, ClassElement) and isinstance(diff.new_version, ClassElement):
                    behavior_change = self._analyze_class_behavior_change(
                        diff.old_version, diff.new_version, diff.file_path
                    )
                    if behavior_change:
                        behavioral_changes.append(behavior_change)
        
        return behavioral_changes
    
    def _analyze_function_behavior_change(
        self, 
        old_func: FunctionElement, 
        new_func: FunctionElement, 
        file_path: str
    ) -> Optional[BehavioralChange]:
        """Analyze behavioral changes in a function."""
        changes = []
        evidence = []
        
        # Parameter changes
        if old_func.parameters != new_func.parameters:
            changes.append("Function signature changed")
            evidence.append(f"Parameters: {old_func.parameters} -> {new_func.parameters}")
        
        # Return type changes
        if old_func.return_type != new_func.return_type:
            changes.append("Return type changed")
            evidence.append(f"Return type: {old_func.return_type} -> {new_func.return_type}")
        
        # Async/sync changes
        if old_func.is_async != new_func.is_async:
            changes.append("Async/sync behavior changed")
            evidence.append(f"Async: {old_func.is_async} -> {new_func.is_async}")
        
        # Function calls changes
        if old_func.calls != new_func.calls:
            added_calls = new_func.calls - old_func.calls
            removed_calls = old_func.calls - new_func.calls
            if added_calls:
                changes.append("New function calls added")
                evidence.append(f"Added calls: {list(added_calls)}")
            if removed_calls:
                changes.append("Function calls removed")
                evidence.append(f"Removed calls: {list(removed_calls)}")
        
        if not changes:
            return None
        
        return BehavioralChange(
            element_name=old_func.name,
            change_description="; ".join(changes),
            change_type=ChangeType.MODIFIED,
            file_path=file_path,
            line_numbers=[old_func.start_line, new_func.start_line],
            impact_analysis=self._generate_impact_analysis(changes),
            confidence=0.8,
            evidence=evidence
        )
    
    def _analyze_class_behavior_change(
        self, 
        old_class: ClassElement, 
        new_class: ClassElement, 
        file_path: str
    ) -> Optional[BehavioralChange]:
        """Analyze behavioral changes in a class."""
        changes = []
        evidence = []
        
        # Inheritance changes
        if old_class.base_classes != new_class.base_classes:
            changes.append("Class inheritance changed")
            evidence.append(f"Base classes: {old_class.base_classes} -> {new_class.base_classes}")
        
        # Method changes
        old_methods = set(old_class.methods)
        new_methods = set(new_class.methods)
        
        added_methods = new_methods - old_methods
        removed_methods = old_methods - new_methods
        
        if added_methods:
            changes.append("Methods added")
            evidence.append(f"Added methods: {list(added_methods)}")
        
        if removed_methods:
            changes.append("Methods removed")
            evidence.append(f"Removed methods: {list(removed_methods)}")
        
        if not changes:
            return None
        
        return BehavioralChange(
            element_name=old_class.name,
            change_description="; ".join(changes),
            change_type=ChangeType.MODIFIED,
            file_path=file_path,
            line_numbers=[old_class.start_line, new_class.start_line],
            impact_analysis=self._generate_impact_analysis(changes),
            confidence=0.7,
            evidence=evidence
        )    

    def _analyze_dependency_impacts(
        self,
        old_files: Dict[str, ParsedFile],
        new_files: Dict[str, ParsedFile],
        structural_diffs: List[StructuralDiff]
    ) -> List[DependencyImpact]:
        """Analyze dependency impacts of changes."""
        impacts = []
        
        # Build dependency graphs for both versions
        old_deps = self._build_dependency_graph(old_files)
        new_deps = self._build_dependency_graph(new_files)
        
        # Analyze impact of each structural change
        for diff in structural_diffs:
            if diff.significance in [ChangeSignificance.MAJOR, ChangeSignificance.CRITICAL]:
                impact = self._calculate_dependency_impact(
                    diff, old_deps, new_deps
                )
                if impact:
                    impacts.append(impact)
        
        return impacts
    
    def _build_dependency_graph(self, files: Dict[str, ParsedFile]) -> Dict[str, Set[str]]:
        """Build a dependency graph from parsed files."""
        deps = defaultdict(set)
        
        for file_path, parsed_file in files.items():
            for dependency in parsed_file.dependencies:
                deps[dependency.source_element].add(dependency.target_element)
        
        return dict(deps)
    
    def _calculate_dependency_impact(
        self,
        diff: StructuralDiff,
        old_deps: Dict[str, Set[str]],
        new_deps: Dict[str, Set[str]]
    ) -> Optional[DependencyImpact]:
        """Calculate the dependency impact of a structural change."""
        element_name = diff.element_name
        
        # Find elements that depend on this changed element
        affected_elements = []
        
        # Check direct dependencies
        for source, targets in old_deps.items():
            if element_name in targets:
                affected_elements.append(source)
        
        # Check transitive dependencies (up to 2 levels)
        transitive_affected = set()
        for affected in affected_elements:
            for source, targets in old_deps.items():
                if affected in targets:
                    transitive_affected.add(source)
        
        affected_elements.extend(list(transitive_affected))
        
        if not affected_elements:
            return None
        
        # Determine impact type
        impact_type = "direct"
        if len(transitive_affected) > 0:
            impact_type = "transitive"
        
        # Check for circular dependencies
        if self._has_circular_dependency(element_name, old_deps):
            impact_type = "circular"
        
        return DependencyImpact(
            source_element=element_name,
            affected_elements=affected_elements,
            impact_type=impact_type,
            severity=diff.significance,
            ripple_effects=self._calculate_ripple_effects(element_name, old_deps),
            file_paths=[diff.file_path]
        )
    
    def _has_circular_dependency(self, element: str, deps: Dict[str, Set[str]]) -> bool:
        """Check if an element has circular dependencies."""
        visited = set()
        path = set()
        
        def dfs(node: str) -> bool:
            if node in path:
                return True
            if node in visited:
                return False
            
            visited.add(node)
            path.add(node)
            
            for neighbor in deps.get(node, set()):
                if dfs(neighbor):
                    return True
            
            path.remove(node)
            return False
        
        return dfs(element)
    
    def _calculate_ripple_effects(self, element: str, deps: Dict[str, Set[str]]) -> List[str]:
        """Calculate ripple effects of changing an element."""
        effects = []
        visited = set()
        
        def find_effects(node: str, depth: int = 0):
            if depth > self.config.max_dependency_depth or node in visited:  # Configurable depth limit
                return
            
            visited.add(node)
            
            for source, targets in deps.items():
                if node in targets and source not in visited:
                    effects.append(f"Change in {node} affects {source}")
                    find_effects(source, depth + 1)
        
        find_effects(element)
        return effects[:self.config.max_ripple_effects]  # Configurable limit   
 
    def _calculate_element_significance(
        self, 
        element: CodeElement, 
        change_type: ChangeType
    ) -> ChangeSignificance:
        """Calculate significance of an element change."""
        # Base significance on element type and change type
        if isinstance(element, FunctionElement):
            if change_type == ChangeType.REMOVED:
                return ChangeSignificance.MAJOR
            elif element.name.startswith('_'):  # Private function
                return ChangeSignificance.MINOR
            else:
                return ChangeSignificance.MAJOR
        
        elif isinstance(element, ClassElement):
            if change_type == ChangeType.REMOVED:
                return ChangeSignificance.CRITICAL
            else:
                return ChangeSignificance.MAJOR
        
        else:
            return ChangeSignificance.MINOR
    
    def _calculate_modification_significance(
        self, 
        old_elem: CodeElement, 
        new_elem: CodeElement
    ) -> ChangeSignificance:
        """Calculate significance of element modification."""
        if isinstance(old_elem, FunctionElement) and isinstance(new_elem, FunctionElement):
            # Signature changes are critical
            if old_elem.parameters != new_elem.parameters:
                return ChangeSignificance.CRITICAL
            elif old_elem.return_type != new_elem.return_type:
                return ChangeSignificance.MAJOR
            elif old_elem.is_async != new_elem.is_async:
                return ChangeSignificance.MAJOR
            else:
                return ChangeSignificance.MINOR
        
        elif isinstance(old_elem, ClassElement) and isinstance(new_elem, ClassElement):
            # Inheritance changes are major
            if old_elem.base_classes != new_elem.base_classes:
                return ChangeSignificance.MAJOR
            else:
                return ChangeSignificance.MINOR
        
        return ChangeSignificance.MINOR
    
    def _calculate_impact_score(self, old_elem: CodeElement, new_elem: CodeElement) -> float:
        """Calculate impact score for element modification."""
        score = 0.0
        
        if isinstance(old_elem, FunctionElement) and isinstance(new_elem, FunctionElement):
            # Parameter changes
            if old_elem.parameters != new_elem.parameters:
                score += self._significance_weights['function_signature_change']
            
            # Return type changes
            if old_elem.return_type != new_elem.return_type:
                score += self._significance_weights['api_change']
            
            # Call changes
            if old_elem.calls != new_elem.calls:
                score += self._significance_weights['dependency_change']
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_overall_significance(
        self,
        structural_diffs: List[StructuralDiff],
        behavioral_changes: List[BehavioralChange],
        dependency_impacts: List[DependencyImpact]
    ) -> ChangeSignificance:
        """Calculate overall significance of all changes."""
        # Count changes by significance
        significance_counts = {
            ChangeSignificance.CRITICAL: 0,
            ChangeSignificance.MAJOR: 0,
            ChangeSignificance.MINOR: 0,
            ChangeSignificance.TRIVIAL: 0
        }
        
        for diff in structural_diffs:
            significance_counts[diff.significance] += 1
        
        for impact in dependency_impacts:
            significance_counts[impact.severity] += 1
        
        # Determine overall significance
        if significance_counts[ChangeSignificance.CRITICAL] > 0:
            return ChangeSignificance.CRITICAL
        elif significance_counts[ChangeSignificance.MAJOR] > 2:
            return ChangeSignificance.CRITICAL
        elif significance_counts[ChangeSignificance.MAJOR] > 0:
            return ChangeSignificance.MAJOR
        elif significance_counts[ChangeSignificance.MINOR] > 5:
            return ChangeSignificance.MAJOR
        elif significance_counts[ChangeSignificance.MINOR] > 0:
            return ChangeSignificance.MINOR
        else:
            return ChangeSignificance.TRIVIAL
    
    def _generate_change_summary(
        self,
        structural_diffs: List[StructuralDiff],
        behavioral_changes: List[BehavioralChange],
        dependency_impacts: List[DependencyImpact]
    ) -> str:
        """Generate a human-readable summary of changes."""
        summary_parts = []
        
        # Structural changes summary
        if structural_diffs:
            added = sum(1 for d in structural_diffs if d.change_type == ChangeType.ADDED)
            removed = sum(1 for d in structural_diffs if d.change_type == ChangeType.REMOVED)
            modified = sum(1 for d in structural_diffs if d.change_type == ChangeType.MODIFIED)
            
            if added:
                summary_parts.append(f"{added} elements added")
            if removed:
                summary_parts.append(f"{removed} elements removed")
            if modified:
                summary_parts.append(f"{modified} elements modified")
        
        # Behavioral changes summary
        if behavioral_changes:
            summary_parts.append(f"{len(behavioral_changes)} behavioral changes detected")
        
        # Dependency impacts summary
        if dependency_impacts:
            total_affected = sum(len(impact.affected_elements) for impact in dependency_impacts)
            summary_parts.append(f"{total_affected} elements affected by dependency changes")
        
        return "; ".join(summary_parts) if summary_parts else "No significant changes detected"
    
    def _generate_impact_analysis(self, changes: List[str]) -> str:
        """Generate impact analysis for behavioral changes."""
        if not changes:
            return "No impact analysis available"
        
        impact_levels = []
        
        for change in changes:
            if "signature changed" in change.lower():
                impact_levels.append("High impact: API compatibility may be broken")
            elif "calls" in change.lower():
                impact_levels.append("Medium impact: Dependencies changed")
            elif "async" in change.lower():
                impact_levels.append("High impact: Execution model changed")
            else:
                impact_levels.append("Low impact: Internal implementation changed")
        
        return "; ".join(impact_levels)