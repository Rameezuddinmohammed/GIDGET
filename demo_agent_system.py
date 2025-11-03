#!/usr/bin/env python3
"""
Demonstration of the Multi-Agent Code Intelligence System.

This script shows how the core agent system works with LangGraph orchestration.
"""

import asyncio
from src.code_intelligence.agents.orchestrator import AgentOrchestrator, OrchestrationConfig
from src.code_intelligence.agents.base import BaseAgent, AgentConfig
from src.code_intelligence.agents.state import AgentState, AgentFinding


class DemoAgent(BaseAgent):
    """Demo agent for demonstration purposes."""
    
    def __init__(self, name: str, demo_findings: list = None):
        config = AgentConfig(name=name, description=f"Demo {name} agent")
        super().__init__(config)
        self.demo_findings = demo_findings or []
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute demo agent logic."""
        print(f"🤖 {self.config.name} agent executing...")
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Add demo findings
        for finding_data in self.demo_findings:
            finding = self._create_finding(**finding_data)
            state.add_finding(self.config.name, finding)
            print(f"   ✅ Added finding: {finding.content}")
            
        return state


async def main():
    """Demonstrate the multi-agent system."""
    print("🚀 Multi-Agent Code Intelligence System Demo")
    print("=" * 50)
    
    # Create orchestrator
    config = OrchestrationConfig(
        max_execution_time_seconds=30,
        graceful_degradation=True
    )
    orchestrator = AgentOrchestrator(config)
    
    # Create demo agents
    orchestrator_agent = DemoAgent("orchestrator", [
        {
            "finding_type": "query_analysis",
            "content": "Parsed query: 'How did the Calculator class evolve?'",
            "confidence": 0.95
        }
    ])
    
    analyst_agent = DemoAgent("analyst", [
        {
            "finding_type": "code_analysis", 
            "content": "Found 3 method changes in Calculator class across 2 commits",
            "confidence": 0.88
        }
    ])
    
    synthesizer_agent = DemoAgent("synthesizer", [
        {
            "finding_type": "synthesis",
            "content": "Calculator class evolved from basic arithmetic to advanced operations",
            "confidence": 0.92
        }
    ])
    
    verifier_agent = DemoAgent("verifier", [
        {
            "finding_type": "verification",
            "content": "All findings verified against git history and source code",
            "confidence": 0.94
        }
    ])
    
    # Register agents
    orchestrator.register_agent("orchestrator", orchestrator_agent)
    orchestrator.register_agent("analyst", analyst_agent)
    orchestrator.register_agent("synthesizer", synthesizer_agent)
    orchestrator.register_agent("verifier", verifier_agent)
    
    print("\n📋 Registered Agents:")
    for name in orchestrator.agents.keys():
        print(f"   • {name}")
    
    # Execute query
    print("\n🔍 Executing Query: 'How did the Calculator class evolve over time?'")
    print("-" * 50)
    
    result = await orchestrator.execute_query(
        "How did the Calculator class evolve over time?",
        "/demo/repo"
    )
    
    # Convert result back to AgentState if needed
    if isinstance(result, dict):
        result = AgentState(**result)
    
    # Display results
    print("\n📊 Results Summary:")
    print(f"   Session ID: {result.session_id}")
    print(f"   Status: {result.progress.get('status', 'unknown')}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Warnings: {len(result.warnings)}")
    
    all_findings = result.get_all_findings()
    print(f"\n🔍 Findings ({len(all_findings)} total):")
    
    for finding in all_findings:
        print(f"   [{finding.agent_name}] {finding.finding_type}")
        print(f"      Content: {finding.content}")
        print(f"      Confidence: {finding.confidence:.2f}")
        print()
    
    # Calculate overall confidence
    if all_findings:
        avg_confidence = sum(f.confidence for f in all_findings) / len(all_findings)
        print(f"📈 Overall Confidence: {avg_confidence:.2f}")
    
    print("\n✅ Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())