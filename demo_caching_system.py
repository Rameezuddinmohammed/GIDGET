#!/usr/bin/env python3
"""
Demo of the intelligent caching system for the Multi-Agent Code Intelligence platform.

This demo shows:
1. Cache manager functionality
2. Cache invalidation service
3. Integration with DeveloperQueryOrchestrator
4. Performance monitoring
"""

import asyncio
import sys
import uuid
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.code_intelligence.caching.cache_manager import cache_manager
from src.code_intelligence.caching.invalidation_service import invalidation_service
from src.code_intelligence.agents.developer_query_orchestrator import DeveloperQueryOrchestrator
from src.code_intelligence.agents.state import AgentState
from src.code_intelligence.agents.base import AgentConfig, AgentFinding
from src.code_intelligence.monitoring.agent_monitor import agent_monitor


async def demo_cache_manager():
    """Demonstrate cache manager functionality."""
    print("🗄️  Cache Manager Demo")
    print("=" * 50)
    
    # Test query hash generation
    query = "Find all functions that call getUserData"
    repo_id = "demo-repo"
    options = {"max_commits": 100}
    
    hash1 = cache_manager._generate_query_hash(query, repo_id, options)
    hash2 = cache_manager._generate_query_hash(query, repo_id, options)
    
    print(f"Query: {query}")
    print(f"Generated hash: {hash1}")
    print(f"Hash consistency: {'✅ PASS' if hash1 == hash2 else '❌ FAIL'}")
    
    # Test TTL calculation
    ttl_high = cache_manager._calculate_ttl(0.95, query, {"findings": []})
    ttl_low = cache_manager._calculate_ttl(0.6, query, {"findings": []})
    ttl_history = cache_manager._calculate_ttl(0.8, "when was this function added", {"findings": []})
    
    print(f"\nTTL Calculation:")
    print(f"  High confidence (0.95): {ttl_high}s")
    print(f"  Low confidence (0.6): {ttl_low}s")
    print(f"  History query (0.8): {ttl_history}s")
    
    # Test caching criteria
    good_result = {"findings": [{"content": "Found getUserData function"}]}
    bad_result = {"findings": [], "errors": ["Parse error"]}
    
    should_cache_good = await cache_manager.should_cache_result(0.85, good_result)
    should_cache_bad = await cache_manager.should_cache_result(0.85, bad_result)
    
    print(f"\nCaching Criteria:")
    print(f"  Good result (0.85 confidence): {'✅ CACHE' if should_cache_good else '❌ SKIP'}")
    print(f"  Bad result (has errors): {'✅ CACHE' if should_cache_bad else '❌ SKIP'}")
    
    # Show cache stats
    stats = await cache_manager.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Hits: {stats['hits']}")
    print(f"  Misses: {stats['misses']}")
    print(f"  Hit Rate: {stats['hit_rate']:.1%}")
    
    print()


async def demo_invalidation_service():
    """Demonstrate cache invalidation service."""
    print("🔄 Cache Invalidation Service Demo")
    print("=" * 50)
    
    import uuid
    repo_id = str(uuid.uuid4())  # Use proper UUID format
    
    # Simulate commit ingestion
    await invalidation_service.handle_commit_ingestion(
        repository_id=repo_id,
        commit_sha="abc123def456",
        modified_files=["src/main.py", "src/utils.py", "tests/test_main.py"],
        commit_message="Fix getUserData function bug"
    )
    
    print(f"✅ Handled commit ingestion for repo: {repo_id}")
    
    # Simulate repository update
    await invalidation_service.handle_repository_update(
        repository_id=repo_id,
        update_type="major_change",
        details={"reason": "Refactored core architecture"}
    )
    
    print(f"✅ Handled repository update for repo: {repo_id}")
    
    # Show pending invalidations
    pending_count = len(invalidation_service.pending_invalidations)
    print(f"📋 Pending invalidations: {pending_count}")
    
    # Process pending invalidations
    processed = await invalidation_service.process_pending_invalidations()
    print(f"⚡ Processed {processed} invalidation events")
    
    # Show invalidation stats
    stats = await invalidation_service.get_invalidation_stats()
    print(f"\nInvalidation Statistics:")
    print(f"  Pending: {stats['pending_invalidations']}")
    print(f"  Total processed: {stats['total_invalidations']}")
    
    print()


async def demo_orchestrator_caching():
    """Demonstrate caching integration with DeveloperQueryOrchestrator."""
    print("🤖 Orchestrator Caching Integration Demo")
    print("=" * 50)
    
    # Create orchestrator
    config = AgentConfig(
        name="demo_orchestrator",
        description="Demo orchestrator with caching"
    )
    orchestrator = DeveloperQueryOrchestrator(config)
    
    # Create sample state
    state = AgentState(
        session_id="demo-session-001",
        query={
            "original": "Find all functions that call getUserData in the last 6 months",
            "options": {"max_commits": 50}
        },
        repository={
            "id": str(uuid.uuid4()),  # Use proper UUID format
            "path": "/path/to/demo/repo"
        }
    )
    
    print(f"Query: {state.query['original']}")
    print(f"Repository: {state.repository['id']}")
    
    # Test cache check (should be miss)
    cached_result = await orchestrator._check_cache(state)
    print(f"Cache check result: {'HIT' if cached_result else 'MISS'}")
    
    # Simulate adding findings to state
    from src.code_intelligence.agents.state import Citation
    
    citation1 = Citation(
        file_path="src/main.py",
        line_number=45,
        description="getUserData function definition"
    )
    
    citation2 = Citation(
        file_path="src/auth.py", 
        line_number=23,
        description="getUserData call in login function"
    )
    
    finding1 = AgentFinding(
        agent_name="analyst",
        finding_type="function_analysis",
        content="Found getUserData function in src/main.py at line 45",
        confidence=0.92,
        metadata={"file": "src/main.py", "line": 45},
        citations=[citation1]
    )
    
    finding2 = AgentFinding(
        agent_name="analyst",
        finding_type="call_analysis", 
        content="getUserData is called by 3 functions: login(), profile(), dashboard()",
        confidence=0.88,
        metadata={"callers": ["login", "profile", "dashboard"]},
        citations=[citation2]
    )
    
    state.add_finding("analyst", finding1)
    state.add_finding("analyst", finding2)
    state.verification["overall_confidence"] = 0.90
    
    print(f"Added {len(state.get_all_findings())} findings to state")
    
    # Test storing final result
    stored = await orchestrator.store_final_result(state)
    print(f"Store result: {'✅ SUCCESS' if stored else '❌ FAILED'}")
    
    # Test cache check again (should be hit if we had real storage)
    print(f"Cache would now contain result for future queries")
    
    print()


async def demo_monitoring_integration():
    """Demonstrate monitoring integration."""
    print("📊 Performance Monitoring Demo")
    print("=" * 50)
    
    # Start some mock executions
    exec_id1 = await agent_monitor.start_execution("orchestrator", "session-001")
    exec_id2 = await agent_monitor.start_execution("analyst", "session-001")
    
    print(f"Started monitoring executions: {exec_id1[:12]}..., {exec_id2[:12]}...")
    
    # Record completions
    await agent_monitor.record_execution(
        exec_id1, success=True, findings_count=2, confidence_score=0.90
    )
    
    await agent_monitor.record_execution(
        exec_id2, success=True, findings_count=5, confidence_score=0.85
    )
    
    print("✅ Recorded execution completions")
    
    # Get system metrics
    system_metrics = await agent_monitor.get_system_metrics()
    print(f"\nSystem Metrics:")
    print(f"  Total executions: {system_metrics['total_executions']}")
    print(f"  Success rate: {system_metrics['success_rate']:.1%}")
    print(f"  Average duration: {system_metrics['avg_duration_ms']:.1f}ms")
    print(f"  Active executions: {system_metrics['active_executions']}")
    
    # Get health summary
    health = await agent_monitor.get_health_summary()
    print(f"\nHealth Summary:")
    print(f"  Status: {health['status']}")
    print(f"  Success rate: {health['success_rate']:.1%}")
    print(f"  Response time: {health['avg_response_time_ms']:.1f}ms")
    
    print()


async def demo_end_to_end_workflow():
    """Demonstrate end-to-end caching workflow."""
    print("🔄 End-to-End Caching Workflow Demo")
    print("=" * 50)
    
    # 1. Query comes in
    query = "What functions were modified in the last commit?"
    import uuid
    repo_id = str(uuid.uuid4())  # Use proper UUID format
    
    print(f"1. New query: {query}")
    
    # 2. Check cache (miss)
    cached = await cache_manager.get_cached_result(query, repo_id)
    print(f"2. Cache check: {'HIT' if cached else 'MISS'}")
    
    # 3. Execute analysis (simulated)
    print("3. Executing multi-agent analysis...")
    
    # 4. Generate results
    result_data = {
        "findings": {
            "historian": [
                {"content": "Last commit abc123 modified 3 functions", "confidence": 0.95}
            ],
            "analyst": [
                {"content": "Modified functions: getUserData, validateUser, logActivity", "confidence": 0.88}
            ]
        },
        "analysis": {"commit_sha": "abc123", "modified_count": 3},
        "verification": {"overall_confidence": 0.91}
    }
    
    print("4. Generated analysis results")
    
    # 5. Store in cache
    should_cache = await cache_manager.should_cache_result(0.91, result_data)
    if should_cache:
        stored = await cache_manager.store_result(
            query=query,
            repository_id=repo_id,
            result_data=result_data,
            confidence_score=0.91
        )
        print(f"5. Stored in cache: {'✅ SUCCESS' if stored else '❌ FAILED'}")
    else:
        print("5. Result not suitable for caching")
    
    # 6. Simulate new commit
    print("6. New commit ingested, triggering cache invalidation...")
    await invalidation_service.handle_commit_ingestion(
        repository_id=repo_id,
        commit_sha="def456",
        modified_files=["src/main.py"],
        commit_message="Update getUserData function"
    )
    
    # 7. Process invalidations
    processed = await invalidation_service.process_pending_invalidations()
    print(f"7. Processed {processed} cache invalidations")
    
    print("✅ End-to-end workflow completed!")
    print()


async def main():
    """Run all caching system demos."""
    print("🚀 Multi-Agent Code Intelligence - Caching System Demo")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        await demo_cache_manager()
        await demo_invalidation_service()
        await demo_orchestrator_caching()
        await demo_monitoring_integration()
        await demo_end_to_end_workflow()
        
        print("🎉 All demos completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main())