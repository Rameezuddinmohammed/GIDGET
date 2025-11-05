#!/usr/bin/env python3
"""
Demo script for the semantic search and vector embedding system.

This script demonstrates the key capabilities of the semantic search system:
1. Code embedding generation
2. Vector storage and similarity search
3. Hybrid search combining semantic and structural approaches
4. Integration with the code intelligence system
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from code_intelligence.semantic.models import (
    CodeElement, CodeElementType, EmbeddingBatch, SearchQuery
)
from code_intelligence.semantic.embeddings import CodeEmbeddingGenerator, EmbeddingModel
from code_intelligence.semantic.storage import VectorStorage
from code_intelligence.semantic.search import SemanticSearchEngine, HybridSearchEngine
from code_intelligence.semantic.integration import SemanticSearchIntegration


async def demo_embedding_generation():
    """Demonstrate code embedding generation."""
    print("🔧 Demo: Code Embedding Generation")
    print("=" * 50)
    
    # Create sample code elements
    code_elements = [
        CodeElement(
            element_type=CodeElementType.FUNCTION,
            name="calculate_sum",
            code_snippet="def calculate_sum(a, b):\n    \"\"\"Calculate sum of two numbers.\"\"\"\n    return a + b",
            file_path="math_utils.py",
            start_line=1,
            end_line=3,
            language="python",
            metadata={"repository_id": "demo_repo", "docstring": "Calculate sum of two numbers."}
        ),
        CodeElement(
            element_type=CodeElementType.FUNCTION,
            name="multiply_numbers",
            code_snippet="def multiply_numbers(x, y):\n    \"\"\"Multiply two numbers.\"\"\"\n    return x * y",
            file_path="math_utils.py",
            start_line=5,
            end_line=7,
            language="python",
            metadata={"repository_id": "demo_repo", "docstring": "Multiply two numbers."}
        ),
        CodeElement(
            element_type=CodeElementType.CLASS,
            name="Calculator",
            code_snippet="class Calculator:\n    \"\"\"A simple calculator class.\"\"\"\n    def __init__(self):\n        self.history = []",
            file_path="calculator.py",
            start_line=1,
            end_line=4,
            language="python",
            metadata={"repository_id": "demo_repo", "docstring": "A simple calculator class."}
        )
    ]
    
    # Initialize embedding generator
    generator = CodeEmbeddingGenerator(
        model=EmbeddingModel.CODEBERT,
        device="cpu"  # Use CPU for demo
    )
    
    print(f"📊 Generating embeddings for {len(code_elements)} code elements...")
    
    # Create embedding batch
    batch = EmbeddingBatch(
        elements=code_elements,
        batch_id="demo_batch",
        repository_id="demo_repo",
        commit_sha="abc123def456"
    )
    
    # Generate embeddings
    embeddings = await generator.generate_batch_embeddings(batch)
    
    print(f"✅ Generated {len(embeddings)} embeddings")
    for i, embedding in enumerate(embeddings):
        print(f"   {i+1}. {embedding.element.name} ({embedding.element.element_type.value})")
        print(f"      Dimension: {embedding.embedding_dimension}")
        print(f"      Confidence: {embedding.confidence_score:.2f}")
        print(f"      Model: {embedding.model_name}")
    
    # Validate embedding quality
    print("\n🔍 Validating embedding quality...")
    quality = await generator.validate_embedding_quality(embeddings)
    
    print(f"   Quality Score: {quality.quality_score:.3f}")
    print(f"   Average Magnitude: {quality.avg_magnitude:.3f}")
    print(f"   Sparsity Ratio: {quality.sparsity_ratio:.3f}")
    print(f"   Consistency Score: {quality.consistency_score:.3f}")
    
    return embeddings


async def demo_vector_storage(embeddings):
    """Demonstrate vector storage and similarity search."""
    print("\n🗄️  Demo: Vector Storage and Similarity Search")
    print("=" * 50)
    
    # Initialize vector storage
    storage = VectorStorage()
    
    print(f"💾 Storing {len(embeddings)} embeddings...")
    
    # Store embeddings
    await storage.store_batch_embeddings(embeddings, "abc123def456")
    print("✅ Embeddings stored successfully")
    
    # Get storage statistics
    stats = await storage.get_embedding_stats("demo_repo")
    print(f"\n📈 Storage Statistics:")
    print(f"   Total Embeddings: {stats['total_embeddings']}")
    print(f"   By Element Type: {stats['by_element_type']}")
    print(f"   By Language: {stats['by_language']}")
    
    # Perform similarity search
    print(f"\n🔍 Performing similarity search...")
    
    # Use the first embedding as query
    query_embedding = embeddings[0].embedding
    
    results = await storage.search_similar(
        query_embedding=query_embedding,
        repository_id="demo_repo",
        limit=3,
        similarity_threshold=0.5
    )
    
    print(f"✅ Found {len(results)} similar code elements:")
    for result in results:
        print(f"   {result.rank}. {result.element.name}")
        print(f"      Similarity: {result.similarity_score:.3f}")
        print(f"      File: {result.element.file_path}")
        print(f"      Explanation: {result.explanation}")


async def demo_semantic_search():
    """Demonstrate semantic search engine."""
    print("\n🔎 Demo: Semantic Search Engine")
    print("=" * 50)
    
    # Initialize semantic search engine
    engine = SemanticSearchEngine()
    
    # Create search queries
    queries = [
        "function that adds two numbers",
        "calculator class",
        "mathematical operations",
        "multiply function"
    ]
    
    print(f"🔍 Testing {len(queries)} search queries...")
    
    for i, query_text in enumerate(queries, 1):
        print(f"\n   Query {i}: '{query_text}'")
        
        query = SearchQuery(
            query_text=query_text,
            repository_id="demo_repo",
            max_results=3,
            similarity_threshold=0.6,
            include_explanation=True
        )
        
        results = await engine.search(query)
        
        if results:
            print(f"   ✅ Found {len(results)} results:")
            for result in results:
                print(f"      - {result.element.name} (score: {result.similarity_score:.3f})")
        else:
            print(f"   ❌ No results found")


async def demo_hybrid_search():
    """Demonstrate hybrid search engine."""
    print("\n🔀 Demo: Hybrid Search Engine")
    print("=" * 50)
    
    # Initialize hybrid search engine
    engine = HybridSearchEngine()
    
    # Create search query
    query = SearchQuery(
        query_text="calculator function that processes numbers",
        repository_id="demo_repo",
        max_results=5,
        similarity_threshold=0.5,
        include_explanation=True
    )
    
    print(f"🔍 Performing hybrid search: '{query.query_text}'")
    
    results = await engine.hybrid_search(
        query,
        semantic_weight=0.6,
        structural_weight=0.4
    )
    
    print(f"✅ Found {len(results)} hybrid results:")
    for result in results:
        print(f"   {result.rank}. Combined Score: {result.combined_score:.3f}")
        if result.best_element:
            if hasattr(result.best_element, 'name'):
                print(f"      Element: {result.best_element.name}")
                print(f"      Type: {result.best_element.element_type.value}")
            else:
                print(f"      Element: {result.best_element}")
        print(f"      Search Types: Semantic={result.semantic_result is not None}, "
              f"Structural={len(result.structural_matches) > 0}")
        print(f"      Explanation: {result.explanation}")


async def demo_integration():
    """Demonstrate semantic search integration."""
    print("\n🔗 Demo: Semantic Search Integration")
    print("=" * 50)
    
    # Initialize integration
    integration = SemanticSearchIntegration()
    
    # Test different search types
    search_tests = [
        ("semantic", "function that calculates sum"),
        ("hybrid", "calculator class with methods"),
    ]
    
    for search_type, query_text in search_tests:
        print(f"\n🔍 Testing {search_type} search: '{query_text}'")
        
        try:
            result = await integration.search_code(
                query_text=query_text,
                repository_id="demo_repo",
                search_type=search_type,
                max_results=3
            )
            
            print(f"   ✅ Search completed:")
            print(f"      Query Hash: {result['query']['hash']}")
            print(f"      Total Results: {result['metadata']['total_results']}")
            print(f"      Search Time: {result['metadata']['search_time']}")
            
            if result['results']:
                print(f"      Top Result: {result['results'][0]['element']['name']}")
            
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
    
    # Get search statistics
    print(f"\n📊 Getting search statistics...")
    try:
        stats = await integration.get_search_statistics("demo_repo")
        print(f"   ✅ Statistics retrieved:")
        print(f"      Repository: {stats['repository_id']}")
        print(f"      Capabilities: {list(stats['capabilities'].keys())}")
        print(f"      Search Config: {stats['search_config']['model']}")
    except Exception as e:
        print(f"   ❌ Statistics failed: {e}")


async def main():
    """Run all semantic search demos."""
    print("🚀 Semantic Search and Vector Embedding System Demo")
    print("=" * 60)
    print("This demo showcases the key capabilities of the semantic search system.")
    print("Note: Using mock embeddings for demonstration purposes.\n")
    
    try:
        # Demo 1: Embedding Generation
        embeddings = await demo_embedding_generation()
        
        # Demo 2: Vector Storage
        await demo_vector_storage(embeddings)
        
        # Demo 3: Semantic Search
        await demo_semantic_search()
        
        # Demo 4: Hybrid Search
        await demo_hybrid_search()
        
        # Demo 5: Integration
        await demo_integration()
        
        print("\n🎉 All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Code embedding generation with specialized models")
        print("✅ Vector storage and similarity search")
        print("✅ Semantic search with natural language queries")
        print("✅ Hybrid search combining semantic and structural approaches")
        print("✅ Integration with the code intelligence system")
        print("✅ Comprehensive testing and validation")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())