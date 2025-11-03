"""Test runner script for the ingestion pipeline tests."""

import sys
import pytest
from pathlib import Path

def run_tests():
    """Run all tests with appropriate configuration."""
    
    # Add src to Python path
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
    
    # Test configuration
    args = [
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "-x",  # Stop on first failure
        str(Path(__file__).parent),  # Test directory
    ]
    
    # Add performance tests if requested
    if "--performance" in sys.argv:
        args.extend(["-m", "performance"])
    else:
        # Skip performance tests by default
        args.extend(["-m", "not performance"])
    
    # Run tests
    exit_code = pytest.main(args)
    return exit_code

def run_unit_tests():
    """Run only unit tests (fast)."""
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
    
    args = [
        "-v",
        "--tb=short", 
        "-m", "not performance",
        str(Path(__file__).parent / "test_git_repository.py"),
        str(Path(__file__).parent / "test_parsing.py"),
    ]
    
    return pytest.main(args)

def run_integration_tests():
    """Run integration tests."""
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
    
    args = [
        "-v",
        "--tb=short",
        "-m", "not performance", 
        str(Path(__file__).parent / "test_ingestion.py"),
    ]
    
    return pytest.main(args)

def run_performance_tests():
    """Run performance tests."""
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
    
    args = [
        "-v",
        "--tb=short",
        "-m", "performance",
        str(Path(__file__).parent / "test_performance.py"),
    ]
    
    return pytest.main(args)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "unit":
            exit_code = run_unit_tests()
        elif sys.argv[1] == "integration":
            exit_code = run_integration_tests()
        elif sys.argv[1] == "performance":
            exit_code = run_performance_tests()
        else:
            exit_code = run_tests()
    else:
        exit_code = run_tests()
    
    sys.exit(exit_code)