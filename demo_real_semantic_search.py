#!/usr/bin/env python3
"""Demo script showing real semantic search functionality."""

import asyncio
import sys
import os
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from code_intelligence.semantic.models import (
    CodeElement, CodeElementType, SearchQuery, CodeEmbedding
)
from code_intelligence.semantic.storage import VectorStorage
from code_intelligence.semantic.search import SemanticSearchEngine, HybridSearchEngine
from code_intelligence.semantic.embeddings import CodeEmbeddingGenerator
from code_intelligence.logging import get_logger

logger = get_logger(__name__)

# Global demo repository ID
import uuid
DEMO_REPO_ID = str(uuid.uuid4())


async def demo_embedding_storage():
    """Demo storing real embeddings."""
    print("🔧 Demo: Storing Code Embeddings")
    print("-" * 40)
    
    try:
        print(f"Using repository ID: {DEMO_REPO_ID}")
        
        # First, create the repository record in Supabase
        from code_intelligence.database.supabase_client import SupabaseClient
        supabase_client = SupabaseClient()
        
        # Create repository record
        repo_data = {
            "id": DEMO_REPO_ID,
            "name": f"demo_repository_{DEMO_REPO_ID[:8]}",
            "url": f"https://github.com/demo/repo_{DEMO_REPO_ID[:8]}",
            "description": "Demo repository for semantic search testing",
            "language": "python",
            "created_at": "2025-11-05T00:00:00Z"
        }
        
        try:
            await supabase_client.insert_repository(repo_data)
            print("✅ Repository record created")
        except Exception as e:
            print(f"⚠️  Repository creation warning (may already exist): {e}")
        
        # Create sample code elements
        elements = [
            CodeElement(
                element_type=CodeElementType.FUNCTION,
                name="calculate_sum",
                code_snippet="def calculate_sum(a, b):\n    return a + b",
                file_path="math_utils.py",
                start_line=1,
                end_line=2,
                language="python",
                metadata={"repository_id": DEMO_REPO_ID}
            ),
            CodeElement(
                element_type=CodeElementType.FUNCTION,
                name="process_data",
                code_snippet="def process_data(data):\n    return [x * 2 for x in data]",
                file_path="data_processor.py",
                start_line=5,
                end_line=6,
                language="python",
                metadata={"repository_id": DEMO_REPO_ID}
            ),
            CodeElement(
                element_type=CodeElementType.CLASS,
                name="DataManager",
                code_snippet="class DataManager:\n    def __init__(self):\n        self.data = []",
                file_path="data_manager.py",
                start_line=1,
                end_line=3,
                language="python",
                metadata={"repository_id": DEMO_REPO_ID}
            )
        ]
        
        # Generate embeddings
        generator = CodeEmbeddingGenerator()
        storage = VectorStorage()
        
        print(f"Generating embeddings for {len(elements)} code elements...")
        
        embeddings = []
        for element in elements:
            embedding = await generator.generate_embedding(element)
            embeddings.append(embedding)
            print(f"  ✅ Generated embedding for {element.name} ({element.element_type.value})")
        
        # Store embeddings
        print(f"\nStoring {len(embeddings)} embeddings in Supabase...")
        await storage.store_batch_embeddings(embeddings, "demo_commit_123")
        
        print("✅ All embeddings stored successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to store embeddings: {e}")
        return False


async def demo_semantic_search():
    """Demo semantic search functionality."""
    print("\n🔍 Demo: Semantic Search")
    print("-" * 40)
    
    try:
        engine = SemanticSearchEngine()
        
        # Test different types of queries
        queries = [
            "function that adds numbers",
            "data processing function",
            "class for managing data",
            "calculate sum of two values"
        ]
        
        for query_text in queries:
            print(f"\nSearching for: '{query_text}'")
            
            query = SearchQuery(
                query_text=query_text,
                repository_id=DEMO_REPO_ID,
                max_results=3,
                similarity_threshold=0.3,
                include_explanation=True
            )
            
            results = await engine.search(query)
            
            if results:
                print(f"  Found {len(results)} results:")
                for i, result in enumerate(results):
                    print(f"    {i+1}. {result.element.name} (similarity: {result.similarity_score:.3f})")
                    if result.explanation:
                        print(f"       {result.explanation}")
            else:
                print("  No results found.")
        
        return True
        
    except Exception as e:
        print(f"❌ Semantic search failed: {e}")
        return False


async def demo_hybrid_search():
    """Demo hybrid search functionality."""
    print("\n🔄 Demo: Hybrid Search (Semantic + Structural)")
    print("-" * 40)
    
    try:
        engine = HybridSearchEngine()
        
        query = SearchQuery(
            query_text="function that processes data",
            repository_id=DEMO_REPO_ID,
            max_results=5,
            similarity_threshold=0.3,
            include_explanation=True
        )
        
        print(f"Performing hybrid search for: '{query.query_text}'")
        
        results = await engine.hybrid_search(query)
        
        if results:
            print(f"Found {len(results)} hybrid results:")
            for i, result in enumerate(results):
                if result.best_element:
                    print(f"  {i+1}. {result.best_element.name}")
                    print(f"     Combined Score: {result.combined_score:.3f}")
                    if result.semantic_result:
                        print(f"     Semantic Score: {result.semantic_result.similarity_score:.3f}")
                    print(f"     Structural Matches: {len(result.structural_matches)}")
                    if result.explanation:
                        print(f"     {result.explanation}")
        else:
            print("No hybrid results found.")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid search failed: {e}")
        return False


async def demo_search_statistics():
    """Demo search statistics."""
    print("\n📊 Demo: Search Statistics")
    print("-" * 40)
    
    try:
        storage = VectorStorage()
        
        stats = await storage.get_embedding_stats(DEMO_REPO_ID)
        
        print("Repository embedding statistics:")
        print(f"  Total embeddings: {stats.get('total_embeddings', 0)}")
        print(f"  By element type: {stats.get('by_element_type', {})}")
        print(f"  By language: {stats.get('by_language', {})}")
        print(f"  Average confidence: {stats.get('avg_confidence', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to get statistics: {e}")
        return False


async def main():
    """Run the complete demo."""
    print("🚀 Real Semantic Search Demo")
    print("=" * 50)
    print("This demo shows the real semantic search implementation")
    print("connecting to actual Supabase (pgvector) and Neo4j databases.")
    print()
    
    demos = [
        ("Embedding Storage", demo_embedding_storage),
        ("Semantic Search", demo_semantic_search),
        ("Hybrid Search", demo_hybrid_search),
        ("Search Statistics", demo_search_statistics),
    ]
    
    results = []
    
    for demo_name, demo_func in demos:
        try:
            success = await demo_func()
            results.append((demo_name, success))
        except Exception as e:
            print(f"❌ {demo_name} demo crashed: {e}")
            results.append((demo_name, False))
    
    print("\n" + "=" * 50)
    print("📋 Demo Results Summary:")
    
    passed = 0
    for demo_name, success in results:
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {demo_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} demos completed successfully")
    
    if passed == len(results):
        print("\n🎉 All demos completed! Real semantic search is fully functional.")
        print("\nKey Features Demonstrated:")
        print("  ✅ Real pgvector similarity search in Supabase")
        print("  ✅ Real Neo4j structural graph queries")
        print("  ✅ Hybrid search combining both approaches")
        print("  ✅ Embedding storage and retrieval")
        print("  ✅ Search statistics and analytics")
    else:
        print("\n⚠️  Some demos failed. Check database connections and configuration.")
        print("\nTroubleshooting:")
        print("  1. Ensure Supabase connection is configured in .env")
        print("  2. Ensure Neo4j connection is configured in .env")
        print("  3. Verify database schemas are properly set up")
        print("  4. Check that pgvector extension is enabled in Supabase")
    
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)