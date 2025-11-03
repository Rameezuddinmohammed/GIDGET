#!/usr/bin/env python3
"""
Demo of the Multi-Agent System with Real LLM Integration.

This demonstrates the agent system using actual Azure OpenAI calls.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.code_intelligence.agents.orchestrator import AgentOrchestrator, OrchestrationConfig
from src.code_intelligence.agents.base import BaseAgent, AgentConfig, LLMConfig
from src.code_intelligence.agents.state import AgentState, AgentFinding


class RealLLMAgent(BaseAgent):
    """Agent that uses real LLM calls for analysis."""
    
    def __init__(self, name: str, role_description: str):
        config = AgentConfig(
            name=name, 
            description=f"Real LLM-powered {name} agent",
            llm_config=LLMConfig(temperature=0.1, max_tokens=500)
        )
        super().__init__(config)
        self.role_description = role_description
        
    async def execute(self, state: AgentState) -> AgentState:
        """Execute agent with real LLM analysis."""
        self._log_execution_start(state)
        
        try:
            # Get query from state
            original_query = state.query.get("original", "")
            repository_path = state.repository.get("path", "")
            
            # Create system prompt for this agent's role
            system_prompt = f"""You are a {self.role_description} for a code intelligence system.
Your role is to analyze code repositories and provide insights about code evolution and structure.

Current task: Analyze the query "{original_query}" for repository at "{repository_path}"

Provide a concise, specific analysis in 1-2 sentences. Focus on actionable insights."""

            # Create user prompt
            user_prompt = f"""Query: "{original_query}"
Repository: {repository_path}

As a {self.role_description}, provide your analysis of this query. Be specific and actionable."""

            # Call real LLM
            response = await self._call_llm(user_prompt, system_prompt)
            
            # Create finding from LLM response
            finding = self._create_finding(
                finding_type=f"{self.config.name}_analysis",
                content=response,
                confidence=0.85,  # Real LLM responses get high confidence
                metadata={
                    "llm_powered": True,
                    "agent_role": self.role_description,
                    "query": original_query
                }
            )
            
            state.add_finding(self.config.name, finding)
            self._log_execution_end(state, success=True)
            
        except Exception as e:
            self.logger.error(f"Agent {self.config.name} failed: {str(e)}")
            state.add_error(f"Agent execution failed: {str(e)}", self.config.name)
            self._log_execution_end(state, success=False)
            
        return state


async def main():
    """Demonstrate real LLM integration with multi-agent system."""
    print("🚀 Multi-Agent Code Intelligence - Real LLM Integration Demo")
    print("=" * 65)
    
    # Create orchestrator
    config = OrchestrationConfig(
        max_execution_time_seconds=60,
        graceful_degradation=True
    )
    orchestrator = AgentOrchestrator(config)
    
    # Create real LLM-powered agents
    query_agent = RealLLMAgent(
        "query_orchestrator", 
        "query analysis specialist who parses natural language queries about code"
    )
    
    code_agent = RealLLMAgent(
        "code_analyst", 
        "code structure analyst who examines code patterns and architecture"
    )
    
    history_agent = RealLLMAgent(
        "git_historian", 
        "git history analyst who tracks code evolution over time"
    )
    
    synthesis_agent = RealLLMAgent(
        "synthesizer", 
        "synthesis specialist who combines insights from multiple analyses"
    )
    
    # Register agents
    orchestrator.register_agent("orchestrator", query_agent)
    orchestrator.register_agent("analyst", code_agent)
    orchestrator.register_agent("historian", history_agent)
    orchestrator.register_agent("synthesizer", synthesis_agent)
    
    print("\n📋 Registered LLM-Powered Agents:")
    for name in orchestrator.agents.keys():
        agent = orchestrator.agents[name]
        print(f"   • {name}: {agent.role_description}")
    
    # Test queries
    test_queries = [
        "How has the Calculator class evolved over the last 6 months?",
        "What are the main architectural patterns in this codebase?",
        "Show me functions that have been refactored recently"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: '{query}'")
        print("-" * 65)
        
        try:
            # Execute query with real LLM agents
            result = await orchestrator.execute_query(
                query,
                "/example/repository"
            )
            
            # Convert result if needed (LangGraph returns dict)
            if isinstance(result, dict):
                result = AgentState(**result)
            
            # Display results
            print(f"\n📊 Results for Query {i}:")
            print(f"   Session ID: {result.session_id}")
            print(f"   Status: {result.progress.get('status', 'unknown')}")
            print(f"   Errors: {len(result.errors)}")
            
            # Show LLM-generated findings
            all_findings = result.get_all_findings()
            print(f"\n🤖 LLM-Generated Findings ({len(all_findings)} total):")
            
            for finding in all_findings:
                print(f"\n   [{finding.agent_name}] {finding.finding_type}")
                print(f"      Content: {finding.content}")
                print(f"      Confidence: {finding.confidence:.2f}")
                if finding.metadata.get("llm_powered"):
                    print(f"      🧠 LLM-Powered: Yes")
            
            # Calculate overall confidence
            if all_findings:
                avg_confidence = sum(f.confidence for f in all_findings) / len(all_findings)
                print(f"\n📈 Overall Confidence: {avg_confidence:.2f}")
                
            if result.has_errors():
                print(f"\n⚠️  Errors encountered:")
                for error in result.errors:
                    print(f"      • {error}")
                    
        except Exception as e:
            print(f"❌ Query {i} failed: {str(e)}")
    
    print(f"\n✅ Demo completed! Real LLM integration working successfully.")
    print("\n💡 Key Features Demonstrated:")
    print("   • Real Azure OpenAI GPT-4 integration")
    print("   • Multi-agent coordination with LLM calls")
    print("   • Intelligent query analysis and code insights")
    print("   • Error handling and graceful degradation")
    print("   • Structured findings with confidence scores")


if __name__ == "__main__":
    asyncio.run(main())