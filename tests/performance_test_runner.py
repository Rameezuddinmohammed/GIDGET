"""Performance test runner for CI/CD integration and comprehensive testing."""

import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


class PerformanceTestSuite:
    """Comprehensive performance test suite runner."""
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("performance_test_results")
        self.output_dir.mkdir(exist_ok=True)
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_suites": {},
            "summary": {
                "total_suites": 0,
                "passed_suites": 0,
                "failed_suites": 0,
                "total_duration_seconds": 0,
                "performance_regressions": 0
            }
        }
        
    def run_test_suite(self, suite_name: str, test_command: List[str], timeout: int = 1800) -> Dict[str, Any]:
        """Run a single test suite."""
        print(f"\n{'='*60}")
        print(f"Running {suite_name} test suite...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                test_command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            duration = time.time() - start_time
            
            suite_result = {
                "suite_name": suite_name,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "duration_seconds": duration,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
                "timeout": False
            }
            
            # Parse test results from output
            suite_result.update(self._parse_test_output(result.stdout))
            
            print(f"✅ {suite_name} completed in {duration:.1f}s" if suite_result["success"] else f"❌ {suite_name} failed after {duration:.1f}s")
            
            return suite_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ {suite_name} timed out after {timeout}s")
            
            return {
                "suite_name": suite_name,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "duration_seconds": duration,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Test suite timed out after {timeout} seconds",
                "success": False,
                "timeout": True
            }
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 {suite_name} crashed: {str(e)}")
            
            return {
                "suite_name": suite_name,
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "duration_seconds": duration,
                "exit_code": -2,
                "stdout": "",
                "stderr": f"Test suite crashed: {str(e)}",
                "success": False,
                "timeout": False
            }
    
    def _parse_test_output(self, stdout: str) -> Dict[str, Any]:
        """Parse test output to extract metrics."""
        parsed = {
            "tests_passed": stdout.count("PASSED"),
            "tests_failed": stdout.count("FAILED"),
            "tests_skipped": stdout.count("SKIPPED"),
            "tests_errors": stdout.count("ERROR"),
            "warnings": stdout.count("WARNING")
        }
        
        # Look for performance regression indicators
        if "regression" in stdout.lower():
            parsed["performance_regressions"] = stdout.lower().count("regression")
        else:
            parsed["performance_regressions"] = 0
            
        return parsed
    
    def run_load_tests(self) -> Dict[str, Any]:
        """Run load testing suite."""
        return self.run_test_suite(
            "Load Testing",
            [
                sys.executable, "-m", "pytest",
                "tests/test_load_testing.py",
                "-m", "load",
                "-v",
                "--tb=short",
                "--maxfail=3"
            ],
            timeout=900  # 15 minutes
        )
    
    def run_scalability_tests(self) -> Dict[str, Any]:
        """Run scalability testing suite."""
        return self.run_test_suite(
            "Scalability Testing",
            [
                sys.executable, "-m", "pytest",
                "tests/test_scalability.py",
                "-m", "scalability",
                "-v",
                "--tb=short",
                "--maxfail=2"
            ],
            timeout=1800  # 30 minutes
        )
    
    def run_regression_tests(self) -> Dict[str, Any]:
        """Run performance regression testing suite."""
        return self.run_test_suite(
            "Performance Regression",
            [
                sys.executable, "-m", "pytest",
                "tests/test_performance_regression.py",
                "-m", "regression",
                "-v",
                "--tb=short"
            ],
            timeout=600  # 10 minutes
        )
    
    def run_resource_tests(self) -> Dict[str, Any]:
        """Run resource consumption tests."""
        return self.run_test_suite(
            "Resource Consumption",
            [
                sys.executable, "-m", "pytest",
                "tests/test_performance.py::TestResourceConsumption",
                "-v",
                "--tb=short"
            ],
            timeout=600  # 10 minutes
        )
    
    def run_database_performance_tests(self) -> Dict[str, Any]:
        """Run database performance tests."""
        return self.run_test_suite(
            "Database Performance",
            [
                sys.executable, "-m", "pytest",
                "tests/test_database_optimization.py",
                "-v",
                "--tb=short"
            ],
            timeout=300  # 5 minutes
        )
    
    def run_all_performance_tests(self, include_slow: bool = False) -> Dict[str, Any]:
        """Run all performance test suites."""
        print("🚀 Starting comprehensive performance test suite...")
        print(f"Output directory: {self.output_dir}")
        
        overall_start_time = time.time()
        
        # Define test suites
        test_suites = [
            ("Database Performance", self.run_database_performance_tests),
            ("Load Testing", self.run_load_tests),
            ("Resource Consumption", self.run_resource_tests),
            ("Performance Regression", self.run_regression_tests),
        ]
        
        # Add slow tests if requested
        if include_slow:
            test_suites.append(("Scalability Testing", self.run_scalability_tests))
        
        # Run each test suite
        for suite_name, suite_runner in test_suites:
            suite_result = suite_runner()
            self.results["test_suites"][suite_name] = suite_result
            
            # Update summary
            self.results["summary"]["total_suites"] += 1
            if suite_result["success"]:
                self.results["summary"]["passed_suites"] += 1
            else:
                self.results["summary"]["failed_suites"] += 1
                
            # Count performance regressions
            self.results["summary"]["performance_regressions"] += suite_result.get("performance_regressions", 0)
        
        # Calculate total duration
        self.results["summary"]["total_duration_seconds"] = time.time() - overall_start_time
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _save_results(self):
        """Save test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.output_dir / f"performance_test_results_{timestamp}.json"
        json_file.write_text(json.dumps(self.results, indent=2))
        
        # Save HTML report
        html_file = self.output_dir / f"performance_test_report_{timestamp}.html"
        html_content = self._generate_html_report()
        html_file.write_text(html_content, encoding='utf-8')
        
        # Save latest results (for CI)
        latest_json = self.output_dir / "latest_performance_results.json"
        latest_json.write_text(json.dumps(self.results, indent=2))
        
        print(f"\n📊 Results saved:")
        print(f"  JSON: {json_file}")
        print(f"  HTML: {html_file}")
        print(f"  Latest: {latest_json}")
    
    def _generate_html_report(self) -> str:
        """Generate HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Performance Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .suite {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .suite-header {{ background-color: #f9f9f9; padding: 10px; font-weight: bold; }}
        .suite-content {{ padding: 10px; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .warning {{ color: orange; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
        .metric {{ background-color: #f5f5f5; padding: 10px; border-radius: 3px; }}
        pre {{ background-color: #f0f0f0; padding: 10px; border-radius: 3px; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Performance Test Report</h1>
        <p>Generated: {self.results['timestamp']}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <div class="metrics">
            <div class="metric">
                <strong>Total Suites:</strong> {self.results['summary']['total_suites']}
            </div>
            <div class="metric">
                <strong>Passed:</strong> <span class="success">{self.results['summary']['passed_suites']}</span>
            </div>
            <div class="metric">
                <strong>Failed:</strong> <span class="failure">{self.results['summary']['failed_suites']}</span>
            </div>
            <div class="metric">
                <strong>Duration:</strong> {self.results['summary']['total_duration_seconds']:.1f}s
            </div>
            <div class="metric">
                <strong>Regressions:</strong> <span class="{'failure' if self.results['summary']['performance_regressions'] > 0 else 'success'}">{self.results['summary']['performance_regressions']}</span>
            </div>
        </div>
    </div>
"""
        
        # Add test suite details
        for suite_name, suite_result in self.results["test_suites"].items():
            status_class = "success" if suite_result["success"] else "failure"
            status_text = "✅ PASSED" if suite_result["success"] else "❌ FAILED"
            
            html += f"""
    <div class="suite">
        <div class="suite-header">
            <span class="{status_class}">{status_text}</span> {suite_name}
            <span style="float: right;">{suite_result['duration_seconds']:.1f}s</span>
        </div>
        <div class="suite-content">
            <div class="metrics">
                <div class="metric">
                    <strong>Tests Passed:</strong> {suite_result.get('tests_passed', 0)}
                </div>
                <div class="metric">
                    <strong>Tests Failed:</strong> {suite_result.get('tests_failed', 0)}
                </div>
                <div class="metric">
                    <strong>Tests Skipped:</strong> {suite_result.get('tests_skipped', 0)}
                </div>
                <div class="metric">
                    <strong>Regressions:</strong> {suite_result.get('performance_regressions', 0)}
                </div>
            </div>
"""
            
            if suite_result.get("stderr"):
                html += f"""
            <h4>Errors:</h4>
            <pre>{suite_result['stderr'][:1000]}{'...' if len(suite_result['stderr']) > 1000 else ''}</pre>
"""
            
            html += """
        </div>
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html
    
    def _print_summary(self):
        """Print test summary."""
        print(f"\n{'='*60}")
        print("PERFORMANCE TEST SUMMARY")
        print(f"{'='*60}")
        
        summary = self.results["summary"]
        
        print(f"Total Test Suites: {summary['total_suites']}")
        print(f"Passed: {summary['passed_suites']} ✅")
        print(f"Failed: {summary['failed_suites']} ❌")
        print(f"Total Duration: {summary['total_duration_seconds']:.1f}s")
        print(f"Performance Regressions: {summary['performance_regressions']} {'⚠️' if summary['performance_regressions'] > 0 else '✅'}")
        
        print(f"\nTest Suite Details:")
        for suite_name, suite_result in self.results["test_suites"].items():
            status = "✅ PASS" if suite_result["success"] else "❌ FAIL"
            duration = suite_result["duration_seconds"]
            tests_passed = suite_result.get("tests_passed", 0)
            tests_failed = suite_result.get("tests_failed", 0)
            
            print(f"  {status} {suite_name:<25} {duration:>6.1f}s  ({tests_passed} passed, {tests_failed} failed)")
        
        # Overall result
        overall_success = summary["failed_suites"] == 0 and summary["performance_regressions"] == 0
        print(f"\n{'='*60}")
        print(f"OVERALL RESULT: {'✅ SUCCESS' if overall_success else '❌ FAILURE'}")
        print(f"{'='*60}")


def main():
    """Main entry point for performance test runner."""
    parser = argparse.ArgumentParser(description="Performance Test Runner")
    parser.add_argument("--suite", choices=["load", "scalability", "regression", "resource", "database", "all"], 
                       default="all", help="Test suite to run")
    parser.add_argument("--include-slow", action="store_true", help="Include slow tests (scalability)")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")
    parser.add_argument("--establish-baselines", action="store_true", help="Establish performance baselines")
    parser.add_argument("--ci-mode", action="store_true", help="Run in CI mode (fail on regressions)")
    
    args = parser.parse_args()
    
    # Establish baselines if requested
    if args.establish_baselines:
        print("🎯 Establishing performance baselines...")
        from test_performance_regression import establish_performance_baselines
        establish_performance_baselines()
        return 0
    
    # Create test runner
    runner = PerformanceTestSuite(args.output_dir)
    
    # Run specific test suite
    if args.suite == "load":
        result = runner.run_load_tests()
    elif args.suite == "scalability":
        result = runner.run_scalability_tests()
    elif args.suite == "regression":
        result = runner.run_regression_tests()
    elif args.suite == "resource":
        result = runner.run_resource_tests()
    elif args.suite == "database":
        result = runner.run_database_performance_tests()
    else:  # all
        result = runner.run_all_performance_tests(include_slow=args.include_slow)
    
    # Save individual suite result if not running all
    if args.suite != "all":
        runner.results["test_suites"][args.suite] = result
        runner.results["summary"]["total_suites"] = 1
        runner.results["summary"]["passed_suites"] = 1 if result["success"] else 0
        runner.results["summary"]["failed_suites"] = 0 if result["success"] else 1
        runner.results["summary"]["performance_regressions"] = result.get("performance_regressions", 0)
        runner._save_results()
        runner._print_summary()
    
    # Exit with appropriate code for CI
    if args.ci_mode:
        # Fail if there are test failures or performance regressions
        if runner.results["summary"]["failed_suites"] > 0:
            print("❌ CI FAILURE: Test suite failures detected")
            return 1
        if runner.results["summary"]["performance_regressions"] > 0:
            print("❌ CI FAILURE: Performance regressions detected")
            return 1
        print("✅ CI SUCCESS: All performance tests passed")
        return 0
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)