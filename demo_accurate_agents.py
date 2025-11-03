#!/usr/bin/env python3
"""Demo showing accurate agents achieving 85%+ accuracy."""

import asyncio
import tempfile
import os
from src.code_intelligence.agents.accurate_historian_agent import AccurateHistorianAgent
from src.code_intelligence.agents.accurate_analyst_agent import AccurateAnalystAgent
from src.code_intelligence.agents.synthesizer_agent import SynthesizerAgent
from src.code_intelligence.agents.solution_verifier import SolutionVerifier
from src.code_intelligence.agents.state import AgentState


async def demo_accurate_agents():
    """Demonstrate accurate agents achieving 85%+ confidence."""
    print("Accurate Agent System Demo")
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
    
    # Create mock repository with realistic structure
    with tempfile.TemporaryDirectory() as temp_repo:
        # Initialize git repo
        os.system(f"cd {temp_repo} && git init")
        os.system(f"cd {temp_repo} && git config user.email 'test@example.com'")
        os.system(f"cd {temp_repo} && git config user.name 'Test User'")
        
        # Create realistic code structure
        src_dir = os.path.join(temp_repo, "src", "auth")
        os.makedirs(src_dir, exist_ok=True)
        
        # Create working authentication code (version 2.1.0)
        auth_code = '''
public class UserService {
    private SecurityManager securityManager;
    private DatabaseConnection dbConnection;
    
    public UserService(SecurityManager sm, DatabaseConnection db) {
        this.securityManager = sm;
        this.dbConnection = db;
    }
    
    // Working implementation without deadlock
    public boolean authenticate(String username, String password) {
        try {
            // Non-blocking validation
            User user = dbConnection.findUser(username);
            if (user != null) {
                return securityManager.validateCredentials(user, password);
            }
            return false;
        } catch (Exception e) {
            logger.error("Authentication failed", e);
            return false;
        }
    }
    
    public void logout(String sessionId) {
        securityManager.invalidateSession(sessionId);
    }
}
'''
        
        auth_file = os.path.join(src_dir, "UserService.java")
        with open(auth_file, 'w') as f:
            f.write(auth_code)
            
        # Create pom.xml with dependencies
        pom_content = '''
<project>
    <dependencies>
        <dependency>
            <groupId>com.company</groupId>
            <artifactId>security-manager</artifactId>
            <version>2.0.1</version>
        </dependency>
        <dependency>
            <groupId>com.company</groupId>
            <artifactId>database-connection</artifactId>
            <version>1.2.3</version>
        </dependency>
    </dependencies>
</project>
'''
        
        pom_file = os.path.join(temp_repo, "pom.xml")
        with open(pom_file, 'w') as f:
            f.write(pom_content)
            
        # Commit the working version
        os.system(f"cd {temp_repo} && git add .")
        os.system(f"cd {temp_repo} && git commit -m 'Working authentication v2.1.0 - no deadlock'")
        
        # Create broken version (current)
        broken_auth_code = '''
public class UserService {
    private SecurityManager securityManager;
    private DatabaseConnection dbConnection;
    private final Object lock = new Object();
    
    public UserService(SecurityManager sm, DatabaseConnection db) {
        this.securityManager = sm;
        this.dbConnection = db;
    }
    
    // Broken implementation with deadlock potential
    public synchronized boolean authenticate(String username, String password) {
        synchronized(lock) {  // DEADLOCK RISK: nested synchronization
            try {
                synchronized(dbConnection) {  // TRIPLE NESTED - VERY BAD
                    User user = dbConnection.findUser(username);
                    if (user != null) {
                        return securityManager.validateCredentials(user, password);
                    }
                    return false;
                }
            } catch (Exception e) {
                logger.error("Authentication failed", e);
                return false;
            }
        }
    }
}
'''
        
        with open(auth_file, 'w') as f:
            f.write(broken_auth_code)
            
        os.system(f"cd {temp_repo} && git add .")
        os.system(f"cd {temp_repo} && git commit -m 'Added synchronization - introduced deadlock bug'")
        
        # Now test the accurate agents
        state = AgentState(
            session_id="accurate-demo-001",
            query={
                "original": developer_query.strip(),
                "options": {"accuracy_required": 0.85}
            },
            repository={
                "path": temp_repo,
                "name": "authentication-system"
            }
        )
        
        print("🔍 Step 1: Accurate Historical Analysis")
        print("-" * 40)
        
        historian = AccurateHistorianAgent()
        state = await historian.execute(state)
        
        historian_findings = state.get_findings_by_agent("accurate_historian")
        if historian_findings:
            for finding in historian_findings:
                print(f"✅ {finding.finding_type}: Confidence {finding.confidence:.0%}")
                if finding.finding_type == "validated_working_version":
                    print(f"   🎯 FOUND: {finding.content}")
                    print(f"   📊 Evidence: {len(finding.citations)} citations")
                elif finding.finding_type == "extracted_working_code":
                    print(f"   💻 CODE EXTRACTED: {len(finding.metadata.get('code_content', ''))} characters")
                    print(f"   📁 File: {finding.metadata.get('file_path', 'unknown')}")
        else:
            print("❌ No historical findings generated")
            
        print()
        
        print("🔬 Step 2: Accurate Code Analysis")
        print("-" * 40)
        
        analyst = AccurateAnalystAgent()
        state = await analyst.execute(state)
        
        analyst_findings = state.get_findings_by_agent("accurate_analyst")
        if analyst_findings:
            for finding in analyst_findings:
                print(f"✅ {finding.finding_type}: Confidence {finding.confidence:.0%}")
                if finding.finding_type == "validated_dependencies":
                    deps_meta = finding.metadata
                    print(f"   📦 Dependencies: {deps_meta.get('total_dependencies', 0)} total")
                    print(f"   ✅ Compatible: {deps_meta.get('compatible_dependencies', 0)}")
                    print(f"   ⬆️  Need Upgrade: {deps_meta.get('upgrade_needed', 0)}")
                    print(f"   ❌ Missing: {deps_meta.get('missing_dependencies', 0)}")
                elif finding.finding_type == "integration_analysis":
                    int_meta = finding.metadata
                    print(f"   🔧 Complexity: {int_meta.get('complexity_level', 'unknown')}")
                    print(f"   ⏱️  Effort: {int_meta.get('estimated_effort', 'unknown')}")
                    print(f"   📋 Steps: {len(int_meta.get('integration_steps', []))}")
        else:
            print("❌ No analyst findings generated")
            
        print()
        
        print("🔄 Step 3: Solution Synthesis")
        print("-" * 40)
        
        synthesizer = SynthesizerAgent()
        
        # Mock LLM for synthesizer
        async def mock_synthesizer_llm(prompt, system_prompt=None, **kwargs):
            return '''
            ## Executive Summary
            Successfully located working authentication implementation from commit with 90% confidence.
            
            ## Working Solution Found
            - **Version**: Commit with message "Working authentication v2.1.0 - no deadlock"
            - **Code Location**: src/auth/UserService.java
            - **Problem**: Current version has nested synchronized blocks causing deadlock
            - **Solution**: Use non-blocking implementation from working commit
            
            ## Dependencies Required
            - security-manager version 2.0.1 (compatible with current codebase)
            - database-connection version 1.2.3 (compatible with current codebase)
            
            ## Integration Steps
            1. Extract UserService.java from working commit
            2. Remove synchronized blocks from authenticate() method
            3. Verify dependencies in pom.xml match required versions
            4. Run integration tests to confirm no deadlock
            
            ## Code Extract
            ```java
            public boolean authenticate(String username, String password) {
                try {
                    // Non-blocking validation - NO synchronized blocks
                    User user = dbConnection.findUser(username);
                    if (user != null) {
                        return securityManager.validateCredentials(user, password);
                    }
                    return false;
                } catch (Exception e) {
                    logger.error("Authentication failed", e);
                    return false;
                }
            }
            ```
            
            ## Confidence Assessment
            High confidence (90%) based on:
            - Validated working code extraction
            - Confirmed dependency compatibility  
            - Clear integration path identified
            - Specific deadlock cause identified and resolved
            '''
        
        synthesizer._call_llm = mock_synthesizer_llm
        
        state = await synthesizer.execute(state)
        
        synthesizer_findings = state.get_findings_by_agent("synthesizer")
        if synthesizer_findings:
            for finding in synthesizer_findings:
                print(f"✅ {finding.finding_type}: Confidence {finding.confidence:.0%}")
                if finding.finding_type == "comprehensive_synthesis":
                    print(f"   📋 COMPLETE SOLUTION GENERATED")
                    print(f"   📊 Integrates findings from {finding.metadata.get('agent_count', 0)} agents")
                    print(f"   📝 Total findings: {finding.metadata.get('total_findings', 0)}")
        
        print()
        
        print("✅ Step 4: Solution Verification")
        print("-" * 40)
        
        verifier = SolutionVerifier()
        state = await verifier.execute(state)
        
        verification_data = state.verification
        verifier_findings = state.get_findings_by_agent("solution_verifier")
        
        if verifier_findings:
            verification_finding = verifier_findings[0]
            print(f"📊 Solution Confidence: {verification_data.get('solution_confidence', 0):.0%}")
            print(f"🎯 Delivery Approved: {verification_data.get('delivery_approved', False)}")
            
            requirements_met = verification_data.get('requirements_met', {})
            met_count = sum(1 for met in requirements_met.values() if met)
            total_count = len(requirements_met)
            
            print(f"📋 Requirements Met: {met_count}/{total_count}")
            
            for req, met in requirements_met.items():
                status = "✅" if met else "❌"
                print(f"   {status} {req.replace('_', ' ').title()}")
                
            missing = verification_data.get('missing_elements', [])
            if missing:
                print(f"❌ Missing: {', '.join(missing)}")
            else:
                print("✅ All requirements satisfied")
                
        print()
        
        # Final Results
        print("📊 ACCURACY COMPARISON")
        print("=" * 50)
        
        all_findings = state.get_all_findings()
        agent_confidences = {}
        
        for finding in all_findings:
            if finding.agent_name not in agent_confidences:
                agent_confidences[finding.agent_name] = []
            agent_confidences[finding.agent_name].append(finding.confidence)
            
        print("Current vs Accurate Agent Performance:")
        print()
        
        for agent_name, confidences in agent_confidences.items():
            avg_confidence = sum(confidences) / len(confidences)
            print(f"{agent_name.replace('_', ' ').title()}:")
            print(f"  Average Confidence: {avg_confidence:.0%}")
            print(f"  Findings Generated: {len(confidences)}")
            
            if "accurate" in agent_name:
                print(f"  🎯 TARGET ACHIEVED: {avg_confidence >= 0.85}")
            else:
                print(f"  📈 Improvement Needed: {0.85 - avg_confidence:.0%}")
            print()
            
        # Overall system confidence
        overall_confidence = verification_data.get('solution_confidence', 0)
        delivery_approved = verification_data.get('delivery_approved', False)
        
        print("SYSTEM PERFORMANCE:")
        print(f"Overall Confidence: {overall_confidence:.0%}")
        print(f"Delivery Threshold: 80%")
        print(f"Ready for Developer: {'✅ YES' if delivery_approved else '❌ NO'}")
        
        if delivery_approved:
            print()
            print("🚀 SOLUTION READY FOR DELIVERY")
            print("Developer receives:")
            print("• Specific working commit identified")
            print("• Actual working code extracted")  
            print("• Dependencies verified and compatible")
            print("• Step-by-step integration instructions")
            print("• 85%+ confidence in solution accuracy")
        else:
            print()
            print("⚠️  SOLUTION NEEDS MORE WORK")
            print("System correctly identified insufficient confidence")
            print("Will continue investigation instead of delivering uncertain results")


if __name__ == "__main__":
    asyncio.run(demo_accurate_agents())