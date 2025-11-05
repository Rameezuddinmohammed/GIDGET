#!/usr/bin/env python3
"""Comprehensive functionality test for Task 8 regression debugging implementation."""

import sys
import traceback
from unittest.mock import Mock, MagicMock
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_imports():
    """Test all imports work correctly."""
    print("Testing imports...")
    try:
        from code_intelligence.regression import (
            VersionComparator, RegressionDetector, IntentAnalyzer,
            StructuralDiff, BehavioralChange, VersionComparisonResult,
            RegressionFinding, CommitIntent, ChangeCorrelation
        )
        from code_intelligence.regression.models import (
            ChangeType, ChangeSignificance, RegressionType
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model instantiation and validation."""
    print("\nTesting model creation...")
    try:
        from code_intelligence.regression.models import (
            StructuralDiff, ChangeType, ChangeSignificance,
            RegressionFinding, RegressionType, CommitIntent
        )
        
        # Test StructuralDiff
        diff = StructuralDiff(
            element_name="test_function",
            element_type="function",
            change_type=ChangeType.MODIFIED,
            file_path="test.py",
            significance=ChangeSignificance.MAJOR
        )
        assert diff.element_name == "test_function"
        assert diff.change_type == ChangeType.MODIFIED
        
        # Test RegressionFinding
        finding = RegressionFinding(
            finding_id="test_id",
            regression_type=RegressionType.API_BREAKING,
            pattern_matched="test_pattern",
            confidence=0.9,
            description="Test finding"
        )
        assert finding.confidence == 0.9
        assert finding.regression_type == RegressionType.API_BREAKING
        
        # Test CommitIntent
        intent = CommitIntent(
            commit_sha="abc123",
            message="Fix bug",
            extracted_intent="Fix a bug",
            confidence=0.8
        )
        assert intent.commit_sha == "abc123"
        assert intent.confidence == 0.8
        
        print("✅ Model creation successful")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        traceback.print_exc()
        return False

def test_version_comparator():
    """Test VersionComparator functionality."""
    print("\nTesting VersionComparator...")
    try:
        from code_intelligence.regression.version_comparison import VersionComparator
        from code_intelligence.git.repository import GitRepository
        from code_intelligence.parsing.parser import CodeParser
        
        # Mock dependencies
        mock_git_repo = Mock(spec=GitRepository)
        mock_git_repo.current_commit = "abc123"
        mock_git_repo.repo_path = Path("/test/repo")
        mock_git_repo.checkout.return_value = None
        
        mock_parser = Mock(spec=CodeParser)
        
        # Create comparator
        comparator = VersionComparator(mock_git_repo, mock_parser)
        
        # Test significance weights
        assert 'function_signature_change' in comparator._significance_weights
        assert comparator._significance_weights['function_signature_change'] == 0.8
        
        # Test element significance calculation
        from code_intelligence.parsing.models import FunctionElement, CodeElementType
        from code_intelligence.regression.models import ChangeType, ChangeSignificance
        
        func = FunctionElement(
            name="test_func",
            element_type=CodeElementType.FUNCTION,
            file_path="test.py",
            start_line=1,
            end_line=5,
            language="python"
        )
        
        significance = comparator._calculate_element_significance(func, ChangeType.REMOVED)
        assert significance == ChangeSignificance.MAJOR
        
        print("✅ VersionComparator functionality working")
        return True
    except Exception as e:
        print(f"❌ VersionComparator test failed: {e}")
        traceback.print_exc()
        return False

def test_regression_detector():
    """Test RegressionDetector functionality."""
    print("\nTesting RegressionDetector...")
    try:
        from code_intelligence.regression.regression_detector import RegressionDetector
        from code_intelligence.regression.version_comparison import VersionComparator
        from code_intelligence.git.repository import GitRepository
        
        # Mock dependencies
        mock_git_repo = Mock(spec=GitRepository)
        mock_version_comparator = Mock(spec=VersionComparator)
        
        # Create detector
        detector = RegressionDetector(mock_git_repo, mock_version_comparator)
        
        # Test pattern initialization
        assert len(detector.patterns) == 5
        pattern_ids = [p.pattern_id for p in detector.patterns]
        expected_patterns = [
            "func_signature_change", "logic_error_pattern", "dependency_break",
            "performance_degradation", "integration_failure"
        ]
        for expected in expected_patterns:
            assert expected in pattern_ids
        
        # Test pattern confidence thresholds
        func_pattern = next(p for p in detector.patterns if p.pattern_id == "func_signature_change")
        assert func_pattern.confidence_threshold == 0.9
        
        print("✅ RegressionDetector functionality working")
        return True
    except Exception as e:
        print(f"❌ RegressionDetector test failed: {e}")
        traceback.print_exc()
        return False

def test_intent_analyzer():
    """Test IntentAnalyzer functionality."""
    print("\nTesting IntentAnalyzer...")
    try:
        from code_intelligence.regression.intent_analyzer import IntentAnalyzer
        from code_intelligence.git.repository import GitRepository
        from code_intelligence.git.models import CommitInfo
        
        # Mock git repository
        mock_git_repo = Mock(spec=GitRepository)
        
        # Mock commit info
        commit_info = CommitInfo(
            sha="abc123",
            message="Fix bug in authentication system",
            author_name="Developer",
            author_email="dev@example.com",
            committer_name="Developer", 
            committer_email="dev@example.com",
            authored_date=datetime.now(),
            committed_date=datetime.now(),
            parents=[],
            file_changes=[],
            stats={}
        )
        mock_git_repo.get_commit_info.return_value = commit_info
        
        # Create analyzer
        analyzer = IntentAnalyzer(mock_git_repo)
        
        # Test intent keyword initialization
        assert 'fix' in analyzer.intent_keywords
        assert 'feature' in analyzer.intent_keywords
        assert len(analyzer.intent_keywords['fix']['keywords']) > 0
        
        # Test intent analysis
        intent = analyzer.analyze_commit_intent("abc123")
        assert intent.commit_sha == "abc123"
        assert "fix" in intent.extracted_intent.lower()
        assert "fix" in intent.intent_categories
        assert intent.confidence > 0.5
        
        # Test communication pattern extraction
        patterns = analyzer._extract_communication_patterns("Fix urgent issue #123")
        assert "issue_references" in patterns
        assert "123" in patterns["issue_references"]
        assert "urgency" in patterns
        
        print("✅ IntentAnalyzer functionality working")
        return True
    except Exception as e:
        print(f"❌ IntentAnalyzer test failed: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between components."""
    print("\nTesting component integration...")
    try:
        from code_intelligence.regression import (
            VersionComparator, RegressionDetector, IntentAnalyzer
        )
        from code_intelligence.regression.models import (
            VersionComparisonResult, StructuralDiff, ChangeType, ChangeSignificance
        )
        from code_intelligence.git.repository import GitRepository
        from code_intelligence.parsing.parser import CodeParser
        
        # Mock dependencies
        mock_git_repo = Mock(spec=GitRepository)
        mock_parser = Mock(spec=CodeParser)
        
        # Create components
        version_comparator = VersionComparator(mock_git_repo, mock_parser)
        regression_detector = RegressionDetector(mock_git_repo, version_comparator)
        intent_analyzer = IntentAnalyzer(mock_git_repo)
        
        # Test that components can work together
        assert regression_detector.version_comparator == version_comparator
        assert regression_detector.git_repo == mock_git_repo
        assert intent_analyzer.git_repo == mock_git_repo
        
        # Test pattern application with mock data
        diff = StructuralDiff(
            element_name="test_function",
            element_type="function", 
            change_type=ChangeType.REMOVED,
            file_path="test.py",
            significance=ChangeSignificance.MAJOR
        )
        
        comparison = VersionComparisonResult(
            old_commit="old",
            new_commit="new",
            structural_diffs=[diff]
        )
        
        # Test pattern matching
        pattern = regression_detector.patterns[0]  # func_signature_change
        confidence = regression_detector._calculate_pattern_confidence(pattern, diff, comparison)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        
        print("✅ Component integration working")
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling and edge cases."""
    print("\nTesting error handling...")
    try:
        from code_intelligence.regression.intent_analyzer import IntentAnalyzer
        from code_intelligence.git.repository import GitRepository
        
        # Mock git repository
        mock_git_repo = Mock(spec=GitRepository)
        analyzer = IntentAnalyzer(mock_git_repo)
        
        # Test with empty commit message
        result = analyzer._extract_intent_keywords("")
        assert result['primary_intent'] == "General code change"
        assert result['confidence'] == 0.3
        
        # Test with None values
        discrepancies = analyzer._identify_discrepancies(
            Mock(extracted_intent="", keywords=[]), []
        )
        assert isinstance(discrepancies, list)
        
        # Test correlation with empty changes
        correlation_score = analyzer._calculate_correlation_score(
            Mock(extracted_intent="test", keywords=[], intent_categories=[], confidence=1.0),
            []
        )
        assert correlation_score == 0.0
        
        print("✅ Error handling working")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all functionality tests."""
    print("🔍 COMPREHENSIVE TASK 8 FUNCTIONALITY ANALYSIS")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_creation,
        test_version_comparator,
        test_regression_detector,
        test_intent_analyzer,
        test_integration,
        test_error_handling
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 ALL FUNCTIONALITY TESTS PASSED!")
        return True
    else:
        print(f"⚠️  {failed} tests failed - see details above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)