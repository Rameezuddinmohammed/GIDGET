#!/usr/bin/env python3
"""Test script for real semantic search implementation."""

import asyncio
import sys
import os
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from code_intelligence.semantic.models import (
    CodeElement, CodeElementType, SearchQuery
)
from code_intelligence.semantic.storage import VectorStorage
from code_intelligence.semantic.search import HybridSearchEngine
from code_intelligence.logging import get_logger

logger = get_logger(__name__)


async def test_vector_storage():
    """Test real vector storage functionality."""
    print("Testing VectorStorage with real Supabase connection...")
    
    try:
        storage = VectorStorage()
        
        # Test search with a simple query
        query_embedding = [0.1] * 768  # Mock embedding
        
        # Use a valid UUID format for repository_id
        import uuid
        test_repo_id = str(uuid.uuid4())
        
        results = await storage.search_similar(
            query_embedding=query_embedding,
            repository_id=test_repo_id,
            limit=5,
            similarity_threshold=0.5
        )
        
        print(f"✅ Vector search completed successfully. Found {len(results)} results.")
        
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.element.name} (similarity: {result.similarity_score:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector storage test failed: {e}")
        return False


async def test_hybrid_search():
    """Test real hybrid search functionality."""
    print("\nTesting HybridSearchEngine with real Neo4j connection...")
    
    try:
        engine = HybridSearchEngine()
        
        # Use a valid UUID format for repository_id
        import uuid
        test_repo_id = str(uuid.uuid4())
        
        query = SearchQuery(
            query_text="process data function",
            repository_id=test_repo_id,
            max_results=5,
            similarity_threshold=0.5
        )
        
        results = await engine.hybrid_search(query)
        
        print(f"✅ Hybrid search completed successfully. Found {len(results)} results.")
        
        for i, result in enumerate(results):
            if result.best_element:
                print(f"  {i+1}. {result.best_element.name} (combined score: {result.combined_score:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid search test failed: {e}")
        return False


async def test_structural_search():
    """Test structural search specifically."""
    print("\nTesting structural search with Neo4j...")
    
    try:
        engine = HybridSearchEngine()
        
        # Use a valid UUID format for repository_id
        import uuid
        test_repo_id = str(uuid.uuid4())
        
        query = SearchQuery(
            query_text="function process",
            repository_id=test_repo_id,
            max_results=3
        )
        
        # Test the structural search directly
        structural_results = await engine._structural_search(query)
        
        print(f"✅ Structural search completed successfully. Found {len(structural_results)} results.")
        
        for i, result in enumerate(structural_results):
            element = result.get("element", {})
            print(f"  {i+1}. {element.get('name', 'unknown')} ({element.get('type', 'unknown')})")
        
        return True
        
    except Exception as e:
        print(f"❌ Structural search test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Testing Real Semantic Search Implementation")
    print("=" * 50)
    
    tests = [
        ("Vector Storage", test_vector_storage),
        ("Structural Search", test_structural_search),
        ("Hybrid Search", test_hybrid_search),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Real semantic search is working.")
        return 0
    else:
        print("⚠️  Some tests failed. Check database connections and configuration.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)