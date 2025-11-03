#!/usr/bin/env python3
"""Show confidence scores from the demo."""

import asyncio
from src.code_intelligence.agents.orchestrator_agent import OrchestratorAgent
from src.code_intelligence.agents.historian_agent import HistorianAgent
from src.code_intelligence.agents.analyst_agent import AnalystAgent
from src.code_intelligence.agents.synthesizer_agent import SynthesizerAgent
from src.code_intelligence.agents.verification_agent import VerificationAgent
from src.code_intelligence.agents.state import AgentState


async def show_confidence_scores():
    """Show confidence scores from each agent."""
    print("Confidence Scores Analysis")
    print("=" * 40)
    
    # Create initial state
    state = AgentState(
        session_id="confidence-demo",
        query={
            "original": "How has the authentication system evolved?",
            "options": {"detail_level": "high"}
        },
        repository={
            "path": "/demo/repo",
            "name": "demo-project"
        }
    )
    
    # Initialize agents
    agents = {
        "orchestrator": OrchestratorAgent(),
        "historian": HistorianAgent(),
        "analyst": AnalystAgent(),
        "synthesizer": SynthesizerAgent(),
        "verifier": VerificationAgent()
    }
    
    # Mock LLM calls
    async def mock_llm(prompt, system_prompt=None, **kwargs):
        return '{"test": "response"}'
    
    for agent in agents.values():
        agent._call_llm = mock_llm
    
    # Execute orchestrator
    print("\n1. ORCHESTRATOR AGENT")
    print("-" * 20)
    try:
        state = await agents["orchestrator"].execute(state)
        findings = state.get_findings_by_agent("orchestrator")
        for finding in findings:
            print(f"Finding: {finding.finding_type}")
            print(f"Confidence: {finding.confidence:.2f}")
            print(f"Reason: Query parsing and workflow planning")
            print()
    except Exception as e:
        print(f"Error: {e}")
    
    # Add some mock target elements
    state.analysis["target_elements"] = [
        {"name": "AuthService", "type": "class", "file_path": "src/auth.py"},
        {"name": "TokenManager", "type": "class", "file_path": "src/token.py"}
    ]
    
    # Execute historian
    print("2. HISTORIAN AGENT")
    print("-" * 20)
    try:
        # Mock git repository
        class MockGitRepo:
            def get_commits_in_range(self, since=None, until=None, limit=100):
                return []
            def get_recent_commits(self, limit=50):
                return []
            def get_changed_files(self, commit_sha):
                return ["src/auth.py"]
        
        import unittest.mock
        with unittest.mock.patch('src.code_intelligence.agents.historian_agent.GitRepository', return_value=MockGitRepo()):
            state = await agents["historian"].execute(state)
            
        findings = state.get_findings_by_agent("historian")
        for finding in findings:
            print(f"Finding: {finding.finding_type}")
            print(f"Confidence: {finding.confidence:.2f}")
            print(f"Reason: Historical analysis with limited data")
            print()
    except Exception as e:
        print(f"Error: {e}")
    
    # Execute analyst
    print("3. ANALYST AGENT")
    print("-" * 20)
    try:
        # Mock database clients
        class MockNeo4jClient:
            async def execute_query(self, query, params=None):
                return []
        
        agents["analyst"].neo4j_client = MockNeo4jClient()
        agents["analyst"].supabase_client = object()  # Mock client
        
        state = await agents["analyst"].execute(state)
        findings = state.get_findings_by_agent("analyst")
        for finding in findings:
            print(f"Finding: {finding.finding_type}")
            print(f"Confidence: {finding.confidence:.2f}")
            print(f"Reason: Structural analysis with available data")
            print()
    except Exception as e:
        print(f"Error: {e}")
    
    # Execute synthesizer
    print("4. SYNTHESIZER AGENT")
    print("-" * 20)
    try:
        state = await agents["synthesizer"].execute(state)
        findings = state.get_findings_by_agent("synthesizer")
        for finding in findings:
            print(f"Finding: {finding.finding_type}")
            print(f"Confidence: {finding.confidence:.2f}")
            print(f"Reason: Synthesis of multiple agent findings")
            print()
    except Exception as e:
        print(f"Error: {e}")
    
    # Execute verifier
    print("5. VERIFICATION AGENT")
    print("-" * 20)
    try:
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock files
            auth_file = os.path.join(temp_dir, "src", "auth.py")
            os.makedirs(os.path.dirname(auth_file), exist_ok=True)
            with open(auth_file, 'w') as f:
                f.write("class AuthService:\n    def authenticate(self):\n        pass\n")
            
            # Update repository path for verification
            state.repository["path"] = temp_dir
            
            state = await agents["verifier"].execute(state)
            
        findings = state.get_findings_by_agent("verifier")
        for finding in findings:
            print(f"Finding: {finding.finding_type}")
            print(f"Confidence: {finding.confidence:.2f}")
            print(f"Reason: Validation against actual source code")
            print()
    except Exception as e:
        print(f"Error: {e}")
    
    # Show overall summary
    print("CONFIDENCE SCORE SUMMARY")
    print("=" * 40)
    all_findings = state.get_all_findings()
    
    agent_confidences = {}
    for finding in all_findings:
        if finding.agent_name not in agent_confidences:
            agent_confidences[finding.agent_name] = []
        agent_confidences[finding.agent_name].append(finding.confidence)
    
    for agent_name, confidences in agent_confidences.items():
        avg_confidence = sum(confidences) / len(confidences)
        print(f"{agent_name.title()}: {avg_confidence:.2f} (from {len(confidences)} findings)")
    
    print("\nWHY VERIFIER HAS LOWER CONFIDENCE:")
    print("-" * 40)
    print("1. VALIDATION UNCERTAINTY: Verifier validates other agents' claims")
    print("   - Can only verify what's actually in the code")
    print("   - Many claims are difficult to validate automatically")
    print("   - Missing evidence reduces confidence")
    print()
    print("2. CONSERVATIVE APPROACH: Designed to be skeptical")
    print("   - Better to flag uncertain findings than accept false positives")
    print("   - Validation failures reduce overall confidence")
    print("   - Incomplete validation data leads to lower scores")
    print()
    print("3. EVIDENCE-BASED SCORING: Confidence based on actual validation")
    print("   - Citation validation (file exists, line numbers correct)")
    print("   - Content validation (claims match actual code)")
    print("   - Weighted scoring: citations 60%, content 40%")


if __name__ == "__main__":
    asyncio.run(show_confidence_scores())