#!/usr/bin/env python3
"""Demo script showing specialized agents working together."""

import asyncio
from src.code_intelligence.agents.orchestrator_agent import OrchestratorAgent
from src.code_intelligence.agents.historian_agent import HistorianAgent
from src.code_intelligence.agents.analyst_agent import AnalystAgent
from src.code_intelligence.agents.synthesizer_agent import SynthesizerAgent
from src.code_intelligence.agents.verification_agent import VerificationAgent
from src.code_intelligence.agents.state import AgentState


async def demo_specialized_agents():
    """Demonstrate specialized agents working together."""
    print("🤖 Multi-Agent Code Intelligence System Demo")
    print("=" * 50)
    
    # Create initial state
    state = AgentState(
        session_id="demo-session-001",
        query={
            "original": "How has the user authentication system evolved over the last month?",
            "options": {"detail_level": "high"}
        },
        repository={
            "path": "/demo/repo",
            "name": "demo-project"
        }
    )
    
    print(f"📝 Original Query: {state.query['original']}")
    print(f"🔍 Session ID: {state.session_id}")
    print()
    
    # Initialize agents
    agents = {
        "orchestrator": OrchestratorAgent(),
        "historian": HistorianAgent(),
        "analyst": AnalystAgent(),
        "synthesizer": SynthesizerAgent(),
        "verifier": VerificationAgent()
    }
    
    print("🚀 Initializing specialized agents...")
    for name, agent in agents.items():
        print(f"  ✓ {name.title()} Agent: {agent.config.description}")
    print()
    
    # Step 1: Orchestrator parses query and plans workflow
    print("🎯 Step 1: Query Orchestration")
    print("-" * 30)
    
    # Mock the LLM calls for demo
    async def mock_llm_orchestrator(prompt, system_prompt=None, **kwargs):
        if "parse" in prompt.lower():
            return '''
            {
                "intent": "analyze_evolution",
                "entities": ["user", "authentication", "system"],
                "time_range": "last_month",
                "scope": "repository",
                "keywords": ["authentication", "system", "evolved"],
                "requires_history": true,
                "requires_semantic_search": true,
                "complexity": "high"
            }
            '''
        else:
            return '''
            {
                "required_agents": ["historian", "analyst", "synthesizer", "verifier"],
                "execution_plan": [
                    {"agent": "historian", "priority": 1, "focus": "temporal_analysis"},
                    {"agent": "analyst", "priority": 2, "focus": "structural_analysis"},
                    {"agent": "synthesizer", "priority": 3, "focus": "result_synthesis"},
                    {"agent": "verifier", "priority": 4, "focus": "validation"}
                ],
                "parallel_execution": false,
                "estimated_duration": "5-8 minutes",
                "complexity_score": 0.8
            }
            '''
    
    # Use the async mock function directly
    agents["orchestrator"]._call_llm = mock_llm_orchestrator
    
    try:
        state = await agents["orchestrator"].execute(state)
        orchestrator_findings = state.get_findings_by_agent("orchestrator")
        
        if orchestrator_findings:
            print(f"✅ Query parsed successfully!")
            parsed_query = state.query.get("parsed", {})
            print(f"   Intent: {parsed_query.get('intent', 'unknown')}")
            print(f"   Entities: {', '.join(parsed_query.get('entities', []))}")
            print(f"   Scope: {parsed_query.get('scope', 'unknown')}")
            print(f"   Complexity: {parsed_query.get('complexity', 'unknown')}")
        else:
            print("⚠️  No orchestrator findings generated")
            
    except Exception as e:
        print(f"❌ Orchestrator failed: {str(e)}")
    
    print()
    
    # Step 2: Historian analyzes temporal data
    print("📚 Step 2: Historical Analysis")
    print("-" * 30)
    
    async def mock_llm_historian(prompt, system_prompt=None, **kwargs):
        return '''
        {
            "timeline": [
                {
                    "commit_sha": "abc123",
                    "timestamp": "2024-10-01T10:00:00Z",
                    "author": "dev@example.com",
                    "message": "Implement OAuth2 authentication",
                    "intent": "feature_addition",
                    "changes": [
                        {
                            "element": "AuthService",
                            "change_type": "added",
                            "impact": "high",
                            "description": "Added OAuth2 support"
                        }
                    ]
                },
                {
                    "commit_sha": "def456",
                    "timestamp": "2024-10-15T14:30:00Z",
                    "author": "dev@example.com",
                    "message": "Fix authentication token expiry bug",
                    "intent": "bug_fix",
                    "changes": [
                        {
                            "element": "TokenManager",
                            "change_type": "modified",
                            "impact": "medium",
                            "description": "Fixed token expiry logic"
                        }
                    ]
                }
            ],
            "patterns": [
                {
                    "pattern_type": "feature_addition",
                    "description": "Major authentication features added",
                    "commits": ["abc123"],
                    "confidence": 0.9
                },
                {
                    "pattern_type": "bug_fix",
                    "description": "Authentication bug fixes",
                    "commits": ["def456"],
                    "confidence": 0.8
                }
            ],
            "evolution_summary": "Authentication system underwent significant evolution with OAuth2 addition and bug fixes"
        }
        '''
    
    # Use the async mock function directly
    agents["historian"]._call_llm = mock_llm_historian
    
    # Mock git repository operations
    class MockGitRepo:
        def get_commits_in_range(self, since=None, until=None, limit=100):
            return []
        def get_recent_commits(self, limit=50):
            return []
        def get_changed_files(self, commit_sha):
            return ["src/auth.py", "src/token_manager.py"]
    
    # Add some target elements to analyze
    state.analysis["target_elements"] = [
        {"name": "AuthService", "type": "class", "file_path": "src/auth.py"},
        {"name": "TokenManager", "type": "class", "file_path": "src/token_manager.py"}
    ]
    
    try:
        # Mock the git repository initialization
        import unittest.mock
        with unittest.mock.patch('src.code_intelligence.agents.historian_agent.GitRepository', return_value=MockGitRepo()):
            state = await agents["historian"].execute(state)
            
        historian_findings = state.get_findings_by_agent("historian")
        print(f"✅ Found {len(historian_findings)} historical insights")
        
        for finding in historian_findings[:2]:  # Show first 2 findings
            print(f"   📊 {finding.finding_type}: {finding.content[:80]}...")
            print(f"      Confidence: {finding.confidence:.2f}")
            
    except Exception as e:
        print(f"❌ Historian failed: {str(e)}")
    
    print()
    
    # Step 3: Analyst performs structural analysis
    print("🔬 Step 3: Structural Analysis")
    print("-" * 30)
    
    # Mock database clients for analyst
    class MockNeo4jClient:
        async def execute_query(self, query, params=None):
            if "CALLS" in query:
                return [{"from_name": "AuthService", "to_name": "TokenManager", "relationship": "CALLS"}]
            elif "count" in query.lower():
                return [{"fan_in": 2, "fan_out": 3}]
            return []
    
    class MockSupabaseClient:
        pass
    
    agents["analyst"].neo4j_client = MockNeo4jClient()
    agents["analyst"].supabase_client = MockSupabaseClient()
    
    try:
        state = await agents["analyst"].execute(state)
        analyst_findings = state.get_findings_by_agent("analyst")
        print(f"✅ Generated {len(analyst_findings)} structural insights")
        
        for finding in analyst_findings[:2]:  # Show first 2 findings
            print(f"   🏗️  {finding.finding_type}: {finding.content[:80]}...")
            print(f"      Confidence: {finding.confidence:.2f}")
            
    except Exception as e:
        print(f"❌ Analyst failed: {str(e)}")
    
    print()
    
    # Step 4: Synthesizer compiles results
    print("🔄 Step 4: Result Synthesis")
    print("-" * 30)
    
    async def mock_llm_synthesizer(prompt, system_prompt=None, **kwargs):
        return '''
        ## Executive Summary
        The authentication system has undergone significant evolution over the last month, with major architectural improvements and bug fixes.
        
        ## Detailed Analysis
        Key changes include the implementation of OAuth2 authentication and resolution of token expiry issues. The system now demonstrates improved security and reliability.
        
        ## Key Insights
        - OAuth2 integration represents a major architectural enhancement
        - Token management has been stabilized through bug fixes
        - Overall system complexity has increased but with better security posture
        
        ## Evidence and Citations
        Analysis based on 2 major commits and structural dependency analysis showing improved coupling between AuthService and TokenManager.
        
        ## Recommendations
        - Continue monitoring token expiry edge cases
        - Consider implementing additional OAuth2 providers
        - Review authentication flow performance under load
        
        ## Confidence Assessment
        High confidence (0.85) in findings based on strong historical evidence and structural analysis.
        '''
    
    # Use the async mock function directly
    agents["synthesizer"]._call_llm = mock_llm_synthesizer
    
    try:
        state = await agents["synthesizer"].execute(state)
        synthesizer_findings = state.get_findings_by_agent("synthesizer")
        print(f"✅ Generated comprehensive synthesis with {len(synthesizer_findings)} findings")
        
        for finding in synthesizer_findings:
            print(f"   📋 {finding.finding_type}: Confidence {finding.confidence:.2f}")
            if finding.finding_type == "comprehensive_synthesis":
                # Show first few lines of the synthesis
                lines = finding.content.split('\n')[:3]
                for line in lines:
                    if line.strip():
                        print(f"      {line.strip()}")
                        
    except Exception as e:
        print(f"❌ Synthesizer failed: {str(e)}")
    
    print()
    
    # Step 5: Verifier validates findings
    print("✅ Step 5: Finding Verification")
    print("-" * 30)
    
    try:
        # Mock file system for verification
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock files
            auth_file = os.path.join(temp_dir, "src", "auth.py")
            os.makedirs(os.path.dirname(auth_file), exist_ok=True)
            with open(auth_file, 'w') as f:
                f.write("class AuthService:\n    def authenticate(self):\n        pass\n")
            
            # Update repository path for verification
            original_path = state.repository.get("path", "")
            state.repository["path"] = temp_dir
            
            state = await agents["verifier"].execute(state)
            
            # Restore original path
            state.repository["path"] = original_path
            
        verifier_findings = state.get_findings_by_agent("verifier")
        print(f"✅ Completed verification with {len(verifier_findings)} validation results")
        
        # Show verification summary
        verification_data = state.verification
        if "verification_summary" in verification_data:
            summary = verification_data["verification_summary"]
            print(f"   📊 Validated: {summary.get('total_validated', 0)} findings")
            print(f"   ✅ Valid: {summary.get('valid_count', 0)}")
            print(f"   ❓ Uncertain: {summary.get('uncertain_count', 0)}")
            print(f"   ❌ Invalid: {summary.get('invalid_count', 0)}")
            print(f"   🎯 Overall Confidence: {summary.get('average_confidence', 0):.2f}")
            
    except Exception as e:
        print(f"❌ Verifier failed: {str(e)}")
    
    print()
    
    # Final Summary
    print("📊 Final Analysis Summary")
    print("=" * 50)
    
    all_findings = state.get_all_findings()
    findings_by_agent = {}
    for finding in all_findings:
        if finding.agent_name not in findings_by_agent:
            findings_by_agent[finding.agent_name] = []
        findings_by_agent[finding.agent_name].append(finding)
    
    print(f"🔍 Total Findings: {len(all_findings)}")
    print(f"🤖 Active Agents: {len(findings_by_agent)}")
    print()
    
    for agent_name, findings in findings_by_agent.items():
        avg_confidence = sum(f.confidence for f in findings) / len(findings)
        print(f"  {agent_name.title()}: {len(findings)} findings (avg confidence: {avg_confidence:.2f})")
    
    print()
    print("✨ Multi-agent analysis complete!")
    print(f"📝 Session: {state.session_id}")
    print(f"⏱️  Total Processing Steps: 5")
    print(f"🎯 Overall System Confidence: {state.verification.get('overall_confidence', 0.8):.2f}")


if __name__ == "__main__":
    asyncio.run(demo_specialized_agents())