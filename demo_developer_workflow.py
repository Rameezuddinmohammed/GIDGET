#!/usr/bin/env python3
"""Demo showing the correct developer workflow with solution verification."""

import asyncio
from src.code_intelligence.agents.developer_query_orchestrator import DeveloperQueryOrchestrator
from src.code_intelligence.agents.historian_agent import HistorianAgent
from src.code_intelligence.agents.analyst_agent import AnalystAgent
from src.code_intelligence.agents.synthesizer_agent import SynthesizerAgent
from src.code_intelligence.agents.solution_verifier import SolutionVerifier
from src.code_intelligence.agents.state import AgentState, AgentFinding, Citation


async def demo_developer_workflow():
    """Demonstrate the correct workflow for your use case."""
    print("Developer Query Workflow Demo")
    print("=" * 50)
    
    # Your exact use case
    developer_query = """
    We had the user authentication feature working in version 2.1.0, 
    but now it's broken due to a deadlock issue. Find out which version 
    had it working properly and what dependencies it needs for integrating 
    it into today's codebase. Give me the code for it.
    """
    
    print(f"Developer Query: {developer_query.strip()}")
    print()
    
    # Create initial state
    state = AgentState(
        session_id="developer-workflow-001",
        query={
            "original": developer_query.strip(),
            "options": {"urgency": "high", "deliverables": ["code", "dependencies", "integration"]}
        },
        repository={
            "path": "/project/repo",
            "name": "authentication-system"
        }
    )
    
    # Initialize agents
    orchestrator = DeveloperQueryOrchestrator()
    historian = HistorianAgent()
    analyst = AnalystAgent()
    synthesizer = SynthesizerAgent()
    solution_verifier = SolutionVerifier()
    
    print("Step 1: Developer Query Analysis")
    print("-" * 30)
    
    # Mock LLM for orchestrator
    async def mock_orchestrator_llm(prompt, system_prompt=None, **kwargs):
        return '{"analysis": "regression_analysis"}'
    
    orchestrator._call_llm = mock_orchestrator_llm
    
    # Execute orchestrator
    state = await orchestrator.execute(state)
    
    if state.get_findings_by_agent("developer_orchestrator"):
        finding = state.get_findings_by_agent("developer_orchestrator")[0]
        intent = state.query.get("developer_intent", {})
        
        print(f"✅ Problem Type: {intent.get('problem_type', 'unknown')}")
        print(f"✅ Urgency: {intent.get('urgency', 'medium')}")
        print(f"✅ Required Deliverables: {', '.join(intent.get('deliverables_needed', []))}")
        print(f"✅ Success Criteria: {intent.get('success_definition', 'N/A')}")
        print()
    
    print("Step 2: Historical Investigation (Historian)")
    print("-" * 30)
    
    # Mock historian to find working version
    async def mock_historian_llm(prompt, system_prompt=None, **kwargs):
        return '''
        {
            "timeline": [
                {
                    "commit_sha": "a1b2c3d",
                    "timestamp": "2023-06-15T10:30:00Z",
                    "author": "dev@company.com",
                    "message": "Fix authentication deadlock in UserService",
                    "intent": "bug_fix",
                    "changes": [
                        {
                            "element": "UserService.authenticate",
                            "change_type": "modified",
                            "impact": "high",
                            "description": "Fixed deadlock by removing synchronized block"
                        }
                    ]
                }
            ],
            "patterns": [
                {
                    "pattern_type": "bug_fix",
                    "description": "Authentication deadlock fixed in version 2.1.0",
                    "commits": ["a1b2c3d"],
                    "confidence": 0.9
                }
            ],
            "evolution_summary": "Authentication worked properly in version 2.1.0 before deadlock regression"
        }
        '''
    
    historian._call_llm = mock_historian_llm
    
    # Mock git operations
    class MockGitRepo:
        def get_commits_in_range(self, since=None, until=None, limit=100):
            return []
        def get_recent_commits(self, limit=50):
            return []
        def get_changed_files(self, commit_sha):
            return ["src/auth/UserService.java", "src/auth/AuthenticationManager.java"]
    
    import unittest.mock
    with unittest.mock.patch('src.code_intelligence.agents.historian_agent.GitRepository', return_value=MockGitRepo()):
        state = await historian.execute(state)
    
    historian_findings = state.get_findings_by_agent("historian")
    if historian_findings:
        print(f"✅ Found {len(historian_findings)} historical insights")
        for finding in historian_findings:
            print(f"   📊 {finding.finding_type}: Confidence {finding.confidence:.0%}")
            if "working version" in finding.content.lower() or "version 2.1.0" in finding.content:
                print(f"   🎯 FOUND WORKING VERSION: {finding.content[:100]}...")
    print()
    
    print("Step 3: Code & Dependency Analysis (Analyst)")
    print("-" * 30)
    
    # Mock analyst to find dependencies
    class MockNeo4jClient:
        async def execute_query(self, query, params=None):
            if "dependency" in query.lower():
                return [
                    {"from_name": "UserService", "to_name": "DatabaseConnection", "relationship": "DEPENDS_ON"},
                    {"from_name": "UserService", "to_name": "SecurityManager", "relationship": "USES"}
                ]
            return []
    
    analyst.neo4j_client = MockNeo4jClient()
    analyst.supabase_client = object()
    
    state = await analyst.execute(state)
    
    analyst_findings = state.get_findings_by_agent("analyst")
    if analyst_findings:
        print(f"✅ Generated {len(analyst_findings)} structural insights")
        for finding in analyst_findings:
            print(f"   🏗️  {finding.finding_type}: Confidence {finding.confidence:.0%}")
            if "dependencies" in finding.content.lower():
                print(f"   🔗 DEPENDENCIES FOUND: {finding.content[:100]}...")
    print()
    
    print("Step 4: Solution Synthesis (Synthesizer)")
    print("-" * 30)
    
    # Mock synthesizer to create integration plan
    async def mock_synthesizer_llm(prompt, system_prompt=None, **kwargs):
        return '''
        ## Working Version Identified
        Version 2.1.0 (commit a1b2c3d) contains the working authentication without deadlock.
        
        ## Code Location
        - UserService.authenticate() method in src/auth/UserService.java
        - AuthenticationManager class in src/auth/AuthenticationManager.java
        
        ## Dependencies Required
        - DatabaseConnection (version 1.2.3)
        - SecurityManager (version 2.0.1)
        - No synchronized blocks (deadlock cause)
        
        ## Integration Steps
        1. Extract UserService.authenticate() from commit a1b2c3d
        2. Update dependencies to compatible versions
        3. Remove synchronized blocks from current implementation
        4. Test authentication flow without deadlock
        
        ## Code Extract
        ```java
        public boolean authenticate(String username, String password) {
            // Non-blocking implementation from v2.1.0
            return securityManager.validateCredentials(username, password);
        }
        ```
        '''
    
    synthesizer._call_llm = mock_synthesizer_llm
    
    state = await synthesizer.execute(state)
    
    synthesizer_findings = state.get_findings_by_agent("synthesizer")
    if synthesizer_findings:
        print(f"✅ Generated comprehensive solution with {len(synthesizer_findings)} components")
        for finding in synthesizer_findings:
            if finding.finding_type == "comprehensive_synthesis":
                print(f"   📋 COMPLETE SOLUTION: Confidence {finding.confidence:.0%}")
                print(f"   📝 Includes: Working version, code, dependencies, integration steps")
    print()
    
    print("Step 5: Solution Verification (Solution Verifier)")
    print("-" * 30)
    
    # Execute solution verifier
    state = await solution_verifier.execute(state)
    
    verifier_findings = state.get_findings_by_agent("solution_verifier")
    if verifier_findings:
        verification_finding = verifier_findings[0]
        verification_data = state.verification
        
        print(f"📊 Solution Confidence: {verification_data.get('solution_confidence', 0):.0%}")
        print(f"🎯 Delivery Approved: {verification_data.get('delivery_approved', False)}")
        print(f"📋 Requirements Met: {len([k for k, v in verification_data.get('requirements_met', {}).items() if v])}")
        print(f"❌ Missing Elements: {len(verification_data.get('missing_elements', []))}")
        print()
        
        if verification_data.get('delivery_approved'):
            print("✅ SOLUTION READY FOR DELIVERY")
            print("=" * 50)
            print("The system found:")
            print("• Working version: 2.1.0 (commit a1b2c3d)")
            print("• Code location: src/auth/UserService.java")
            print("• Dependencies: DatabaseConnection 1.2.3, SecurityManager 2.0.1")
            print("• Integration steps: Remove synchronized blocks, update deps")
            print("• Actual code extract provided")
            print()
            print("Developer can now:")
            print("1. Extract the working code from the identified commit")
            print("2. Update dependencies as specified")
            print("3. Follow integration steps to avoid deadlock")
            print("4. Deploy with confidence")
        else:
            print("❌ SOLUTION NOT READY")
            print("=" * 50)
            missing = verification_data.get('missing_elements', [])
            print(f"Missing: {', '.join(missing)}")
            print("System needs more investigation before delivery")
    
    print()
    print("COMPARISON: Current vs Required Approach")
    print("=" * 50)
    print()
    print("❌ CURRENT IMPLEMENTATION (Wrong):")
    print("• Verifier validates individual micro-findings")
    print("• Low confidence (50-60%) due to incomplete validation")
    print("• No solution-level decision making")
    print("• Delivers uncertain results to developer")
    print()
    print("✅ REQUIRED IMPLEMENTATION (Correct):")
    print("• Solution verifier validates complete solution")
    print("• High confidence threshold (80-90%) for delivery")
    print("• Go/no-go decision based on developer requirements")
    print("• Only delivers when solution is actionable")
    print()
    print("🎯 YOUR USE CASE REQUIREMENTS:")
    print("• Find working version ✅")
    print("• Extract actual code ✅") 
    print("• Identify dependencies ✅")
    print("• Provide integration steps ✅")
    print("• Verify 80-90% accuracy before delivery ✅")


if __name__ == "__main__":
    asyncio.run(demo_developer_workflow())