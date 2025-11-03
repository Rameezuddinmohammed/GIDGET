#!/usr/bin/env python3
"""
Demo of Optimized Multi-Agent Code Intelligence System

This demo showcases the enhanced agents with real code extraction,
integration analysis, executable solutions, and high-confidence validation.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any

from src.code_intelligence.agents.state import AgentState
from src.code_intelligence.agents.historian_agent import HistorianAgent
from src.code_intelligence.agents.analyst_agent import AnalystAgent
from src.code_intelligence.agents.synthesizer_agent import SynthesizerAgent
from src.code_intelligence.agents.verification_agent import VerificationAgent


class OptimizedAgentDemo:
    """Demo showcasing the optimized multi-agent system."""
    
    def __init__(self):
        """Initialize the demo with optimized agents."""
        self.historian = HistorianAgent()
        self.analyst = AnalystAgent()
        self.synthesizer = SynthesizerAgent()
        self.verifier = VerificationAgent()
        
    async def run_developer_scenario(self, query: str, repository_path: str) -> Dict[str, Any]:
        """Run a complete developer problem-solving scenario."""
        
        print(f"🎯 **DEVELOPER QUERY**: {query}")
        print(f"📁 **REPOSITORY**: {repository_path}")
        print("=" * 80)
        
        # Initialize state
        state = AgentState(
            session_id="demo_optimized_agents",
            query={"original": query},
            repository={"path": repository_path}
        )
        
        results = {}
        
        # Step 1: Historian - Extract working code
        print("🔍 **STEP 1: HISTORIAN AGENT - EXTRACTING WORKING CODE**")
        try:
            state = await self.historian.execute(state)
            historian_findings = state.get_findings_by_agent("historian")
            
            working_code_found = any(f.finding_type == "working_code_extraction" for f in historian_findings)
            
            if working_code_found:
                working_code_finding = next(f for f in historian_findings if f.finding_type == "working_code_extraction")
                print(f"✅ Working code extracted with {working_code_finding.confidence:.1%} confidence")
                print(f"📄 File: {working_code_finding.metadata.get('file_path', 'N/A')}")
                print(f"🔗 Commit: {working_code_finding.metadata.get('commit_sha', 'N/A')[:8]}")
                results["working_code"] = {
                    "found": True,
                    "confidence": working_code_finding.confidence,
                    "file_path": working_code_finding.metadata.get('file_path'),
                    "commit_sha": working_code_finding.metadata.get('commit_sha')
                }
            else:
                print("❌ No working code found in git history")
                results["working_code"] = {"found": False}
                
        except Exception as e:
            print(f"❌ Historian failed: {str(e)}")
            results["working_code"] = {"found": False, "error": str(e)}
            
        print()
        
        # Step 2: Analyst - Analyze integration requirements
        print("🔬 **STEP 2: ANALYST AGENT - ANALYZING INTEGRATION REQUIREMENTS**")
        try:
            state = await self.analyst.execute(state)
            analyst_findings = state.get_findings_by_agent("analyst")
            
            integration_found = any(f.finding_type == "integration_analysis" for f in analyst_findings)
            
            if integration_found:
                integration_finding = next(f for f in analyst_findings if f.finding_type == "integration_analysis")
                metadata = integration_finding.metadata
                
                dependencies = metadata.get("dependencies", [])
                compatibility = metadata.get("compatibility", [])
                integration_steps = metadata.get("integration_steps", [])
                
                print(f"✅ Integration analysis completed with {integration_finding.confidence:.1%} confidence")
                print(f"📦 Dependencies: {len(dependencies)} identified")
                print(f"🔗 Compatibility: {len([c for c in compatibility if c.get('compatible', False)])}/{len(compatibility)} available")
                print(f"📋 Integration steps: {len(integration_steps)} steps required")
                
                results["integration"] = {
                    "available": True,
                    "confidence": integration_finding.confidence,
                    "dependencies_count": len(dependencies),
                    "compatibility_ratio": len([c for c in compatibility if c.get('compatible', False)]) / len(compatibility) if compatibility else 1.0,
                    "steps_count": len(integration_steps)
                }
            else:
                print("ℹ️ No integration analysis available (fallback to graph-based analysis)")
                results["integration"] = {"available": False}
                
        except Exception as e:
            print(f"❌ Analyst failed: {str(e)}")
            results["integration"] = {"available": False, "error": str(e)}
            
        print()
        
        # Step 3: Synthesizer - Generate executable solution
        print("🔧 **STEP 3: SYNTHESIZER AGENT - GENERATING EXECUTABLE SOLUTION**")
        try:
            state = await self.synthesizer.execute(state)
            synthesizer_findings = state.get_findings_by_agent("synthesizer")
            
            if synthesizer_findings:
                synthesis_finding = synthesizer_findings[0]  # Main synthesis
                content = synthesis_finding.content
                
                # Check if it's a solution-oriented synthesis
                if "executable solution steps" in content.lower():
                    print(f"✅ Executable solution generated with {synthesis_finding.confidence:.1%} confidence")
                    print("📋 Solution includes:")
                    
                    if "step 1:" in content.lower():
                        steps = content.lower().split("step ")[1:]  # Skip first empty split
                        print(f"   • {len(steps)} executable steps")
                        
                    if "installation commands" in content.lower():
                        print("   • Dependency installation commands")
                        
                    if "validation points" in content.lower():
                        print("   • Testing and validation checklist")
                        
                    results["solution"] = {
                        "type": "executable",
                        "confidence": synthesis_finding.confidence,
                        "steps_count": len(steps) if 'steps' in locals() else 0
                    }
                else:
                    print(f"ℹ️ Analysis report generated with {synthesis_finding.confidence:.1%} confidence")
                    results["solution"] = {
                        "type": "analysis",
                        "confidence": synthesis_finding.confidence
                    }
            else:
                print("❌ No synthesis generated")
                results["solution"] = {"type": "none"}
                
        except Exception as e:
            print(f"❌ Synthesizer failed: {str(e)}")
            results["solution"] = {"type": "error", "error": str(e)}
            
        print()
        
        # Step 4: Verifier - Validate solution
        print("✅ **STEP 4: VERIFICATION AGENT - VALIDATING SOLUTION**")
        try:
            state = await self.verifier.execute(state)
            verifier_findings = state.get_findings_by_agent("verifier")
            
            if verifier_findings:
                verification_finding = next((f for f in verifier_findings if f.finding_type == "solution_validation"), None)
                
                if verification_finding:
                    confidence = verification_finding.confidence
                    validation_status = verification_finding.metadata.get("validation_status", "UNKNOWN")
                    
                    if validation_status == "SOLUTION_APPROVED":
                        print(f"🟢 **SOLUTION APPROVED** with {confidence:.1%} confidence")
                        print("   Ready for implementation!")
                    elif validation_status == "SOLUTION_NEEDS_REVIEW":
                        print(f"🟡 **SOLUTION NEEDS REVIEW** - {confidence:.1%} confidence (below 80% threshold)")
                        print("   Consider additional validation before implementation")
                    else:
                        print(f"⚪ **VALIDATION COMPLETED** - {confidence:.1%} confidence")
                        
                    results["validation"] = {
                        "status": validation_status,
                        "confidence": confidence,
                        "approved": validation_status == "SOLUTION_APPROVED"
                    }
                else:
                    print("ℹ️ Individual findings validated (no complete solution found)")
                    results["validation"] = {"status": "INDIVIDUAL_VALIDATION"}
            else:
                print("❌ No validation performed")
                results["validation"] = {"status": "NONE"}
                
        except Exception as e:
            print(f"❌ Verifier failed: {str(e)}")
            results["validation"] = {"status": "ERROR", "error": str(e)}
            
        print()
        print("=" * 80)
        
        # Summary
        self._print_summary(results)
        
        return results
        
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of the demo results."""
        
        print("📊 **DEMO SUMMARY**")
        print()
        
        # Working Code
        working_code = results.get("working_code", {})
        if working_code.get("found"):
            print(f"🔍 **Working Code**: ✅ Found ({working_code.get('confidence', 0):.1%} confidence)")
        else:
            print("🔍 **Working Code**: ❌ Not found")
            
        # Integration Analysis
        integration = results.get("integration", {})
        if integration.get("available"):
            compatibility = integration.get("compatibility_ratio", 0)
            print(f"🔬 **Integration**: ✅ Analyzed ({integration.get('confidence', 0):.1%} confidence, {compatibility:.1%} compatible)")
        else:
            print("🔬 **Integration**: ℹ️ Fallback analysis")
            
        # Solution Type
        solution = results.get("solution", {})
        solution_type = solution.get("type", "none")
        if solution_type == "executable":
            print(f"🔧 **Solution**: ✅ Executable ({solution.get('confidence', 0):.1%} confidence)")
        elif solution_type == "analysis":
            print(f"🔧 **Solution**: ℹ️ Analysis report ({solution.get('confidence', 0):.1%} confidence)")
        else:
            print("🔧 **Solution**: ❌ None generated")
            
        # Validation
        validation = results.get("validation", {})
        status = validation.get("status", "NONE")
        if status == "SOLUTION_APPROVED":
            print(f"✅ **Validation**: 🟢 APPROVED ({validation.get('confidence', 0):.1%} confidence)")
        elif status == "SOLUTION_NEEDS_REVIEW":
            print(f"✅ **Validation**: 🟡 NEEDS REVIEW ({validation.get('confidence', 0):.1%} confidence)")
        else:
            print(f"✅ **Validation**: ⚪ {status}")
            
        print()
        
        # Overall Assessment
        if (working_code.get("found") and 
            integration.get("available") and 
            solution_type == "executable" and 
            validation.get("approved")):
            print("🎯 **OVERALL**: 🟢 **COMPLETE SOLUTION READY FOR IMPLEMENTATION**")
        elif working_code.get("found") or solution_type == "executable":
            print("🎯 **OVERALL**: 🟡 **PARTIAL SOLUTION AVAILABLE**")
        else:
            print("🎯 **OVERALL**: 🔴 **ANALYSIS ONLY - NO EXECUTABLE SOLUTION**")


async def main():
    """Run the optimized agents demo."""
    
    print("🚀 **OPTIMIZED MULTI-AGENT CODE INTELLIGENCE DEMO**")
    print("=" * 80)
    print()
    
    demo = OptimizedAgentDemo()
    
    # Demo scenarios
    scenarios = [
        {
            "query": "How do I fix the deadlock in the user authentication system?",
            "repository": "."  # Current repository
        },
        {
            "query": "Find the working implementation of the payment processing feature",
            "repository": "."
        },
        {
            "query": "How to optimize the database connection pooling performance issue?",
            "repository": "."
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"🎬 **SCENARIO {i}**")
        print()
        
        try:
            results = await demo.run_developer_scenario(
                scenario["query"], 
                scenario["repository"]
            )
            
            # Save results for analysis
            results_file = f"demo_results_scenario_{i}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"📄 Results saved to {results_file}")
            
        except Exception as e:
            print(f"❌ Scenario {i} failed: {str(e)}")
            
        print()
        print("=" * 80)
        print()
        
    print("✅ **DEMO COMPLETED**")
    print()
    print("The optimized agents demonstrate:")
    print("• Real code extraction from git history")
    print("• Actual dependency analysis from source code")
    print("• Executable solutions with step-by-step guides")
    print("• High-confidence validation (80-90% thresholds)")
    print("• Developer-focused problem solving")


if __name__ == "__main__":
    asyncio.run(main())