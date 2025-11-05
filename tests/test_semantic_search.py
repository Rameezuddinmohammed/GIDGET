"""Tests for semantic search and vector embedding system."""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from src.code_intelligence.semantic.models import (
    CodeElement, CodeElementType, CodeEmbedding, EmbeddingModel,
    EmbeddingBatch, SearchQuery, SearchResult, HybridSearchResult,
    EmbeddingQuality
)
from src.code_intelligence.semantic.embeddings import CodeEmbeddingGenerator
from src.code_intelligence.semantic.storage import VectorStorage
from src.code_intelligence.semantic.search import SemanticSearchEngine, HybridSearchEngine
from src.code_intelligence.semantic.integration import SemanticSearchIntegration
from src.code_intelligence.exceptions import EmbeddingError, VectorStorageError, SearchError


class TestCodeEmbeddingGenerator:
    """Test code embedding generation pipeline."""
    
    @pytest.fixture
    def sample_code_element(self):
        """Create a sample code element for testing."""
        return CodeElement(
            element_type=CodeElementType.FUNCTION,
            name="test_function",
            code_snippet="def test_function(x, y):\n    return x + y",
            file_path="test.py",
            start_line=1,
            end_line=2,
            language="python",
            metadata={"repository_id": "test_repo"}
        )
    
    @pytest.fixture
    def embedding_generator(self):
        """Create embedding generator with mock model."""
        # Force mock mode for testing
        generator = CodeEmbeddingGenerator()
        generator._use_mock = True
        generator.embedding_dimension = 768
        return generator
    
    @pytest.mark.asyncio
    async def test_generate_single_embedding(self, embedding_generator, sample_code_element):
        """Test generating embedding for a single code element."""
        embedding = await embedding_generator.generate_embedding(sample_code_element)
        
        assert isinstance(embedding, CodeEmbedding)
        assert embedding.element == sample_code_element
        assert len(embedding.embedding) == 768
        assert embedding.embedding_dimension == 768
        assert embedding.confidence_score > 0
        assert "mock" in embedding.model_name
    
    @pytest.mark.asyncio
    async def test_generate_batch_embeddings(self, embedding_generator):
        """Test generating embeddings for a batch of elements."""
        elements = []
        for i in range(5):
            element = CodeElement(
                element_type=CodeElementType.FUNCTION,
                name=f"function_{i}",
                code_snippet=f"def function_{i}():\n    return {i}",
                file_path=f"test_{i}.py",
                start_line=1,
                end_line=2,
                language="python",
                metadata={"repository_id": "test_repo"}
            )
            elements.append(element)
        
        batch = EmbeddingBatch(
            elements=elements,
            batch_id="test_batch",
            repository_id="test_repo",
            commit_sha="abc123"
        )
        
        embeddings = await embedding_generator.generate_batch_embeddings(batch)
        
        assert len(embeddings) == 5
        assert all(isinstance(emb, CodeEmbedding) for emb in embeddings)
        assert all(len(emb.embedding) == 768 for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_validate_embedding_quality(self, embedding_generator, sample_code_element):
        """Test embedding quality validation."""
        # Generate multiple embeddings
        embeddings = []
        for i in range(10):
            element = CodeElement(
                element_type=CodeElementType.FUNCTION,
                name=f"function_{i}",
                code_snippet=f"def function_{i}():\n    return {i}",
                file_path="test.py",
                start_line=i,
                end_line=i+1,
                language="python",
                metadata={}
            )
            embedding = await embedding_generator.generate_embedding(element)
            embeddings.append(embedding)
        
        quality = await embedding_generator.validate_embedding_quality(embeddings)
        
        assert isinstance(quality, EmbeddingQuality)
        assert quality.dimension == 768
        assert 0 <= quality.quality_score <= 1
        assert quality.avg_magnitude > 0
        assert 0 <= quality.sparsity_ratio <= 1
        assert 0 <= quality.consistency_score <= 1
    
    def test_prepare_code_text(self, embedding_generator, sample_code_element):
        """Test code text preparation for embedding."""
        text = embedding_generator._prepare_code_text(sample_code_element)
        
        assert "function: test_function" in text
        assert "Language: python" in text
        assert "def test_function(x, y):" in text
    
    @pytest.mark.asyncio
    async def test_update_embeddings_for_changes(self, embedding_generator):
        """Test updating embeddings for changed elements."""
        changed_elements = [
            CodeElement(
                element_type=CodeElementType.FUNCTION,
                name="changed_function",
                code_snippet="def changed_function():\n    return 'updated'",
                file_path="changed.py",
                start_line=1,
                end_line=2,
                language="python",
                metadata={"repository_id": "test_repo"}
            )
        ]
        
        embeddings = await embedding_generator.update_embeddings_for_changes(
            repository_id="test_repo",
            changed_elements=changed_elements,
            commit_sha="def456"
        )
        
        assert len(embeddings) == 1
        assert embeddings[0].element.name == "changed_function"


class TestVectorStorage:
    """Test vector storage and similarity search."""
    
    @pytest.fixture
    def vector_storage(self):
        """Create vector storage with mocked client."""
        storage = VectorStorage()
        storage.client = Mock()
        storage.client.store_code_embedding = AsyncMock()
        return storage
    
    @pytest.fixture
    def sample_embedding(self):
        """Create a sample code embedding."""
        element = CodeElement(
            element_type=CodeElementType.FUNCTION,
            name="test_function",
            code_snippet="def test_function():\n    pass",
            file_path="test.py",
            start_line=1,
            end_line=2,
            language="python",
            metadata={"repository_id": "test_repo"}
        )
        
        return CodeEmbedding(
            element=element,
            embedding=[0.1] * 768,
            model_name="test_model",
            embedding_dimension=768,
            confidence_score=0.9
        )
    
    @pytest.mark.asyncio
    async def test_store_single_embedding(self, vector_storage, sample_embedding):
        """Test storing a single embedding."""
        await vector_storage.store_embedding(sample_embedding, "abc123")
        
        vector_storage.client.store_code_embedding.assert_called_once()
        call_args = vector_storage.client.store_code_embedding.call_args[1]
        assert call_args["element_name"] == "test_function"
        assert call_args["element_type"] == "function"
        assert call_args["commit_sha"] == "abc123"
    
    @pytest.mark.asyncio
    async def test_store_batch_embeddings(self, vector_storage):
        """Test storing a batch of embeddings."""
        embeddings = []
        for i in range(3):
            element = CodeElement(
                element_type=CodeElementType.FUNCTION,
                name=f"function_{i}",
                code_snippet=f"def function_{i}(): pass",
                file_path="test.py",
                start_line=i,
                end_line=i+1,
                language="python",
                metadata={"repository_id": "test_repo"}
            )
            embedding = CodeEmbedding(
                element=element,
                embedding=[0.1] * 768,
                model_name="test_model",
                embedding_dimension=768
            )
            embeddings.append(embedding)
        
        await vector_storage.store_batch_embeddings(embeddings, "abc123")
        
        assert vector_storage.client.store_code_embedding.call_count == 3
    
    @pytest.mark.asyncio
    async def test_search_similar(self, vector_storage):
        """Test similarity search."""
        query_embedding = [0.1] * 768
        
        results = await vector_storage.search_similar(
            query_embedding=query_embedding,
            repository_id="test_repo",
            limit=5,
            similarity_threshold=0.7
        )
        
        assert isinstance(results, list)
        assert all(isinstance(result, SearchResult) for result in results)
        assert len(results) <= 5
    
    @pytest.mark.asyncio
    async def test_get_embedding_stats(self, vector_storage):
        """Test getting embedding statistics."""
        stats = await vector_storage.get_embedding_stats("test_repo")
        
        assert isinstance(stats, dict)
        assert "total_embeddings" in stats
        assert "by_element_type" in stats
        assert "by_language" in stats
    
    @pytest.mark.asyncio
    async def test_cleanup_old_embeddings(self, vector_storage):
        """Test cleaning up old embeddings."""
        deleted_count = await vector_storage.cleanup_old_embeddings("test_repo", keep_commits=5)
        
        assert isinstance(deleted_count, int)
        assert deleted_count >= 0


class TestSemanticSearchEngine:
    """Test semantic search engine."""
    
    @pytest.fixture
    def search_engine(self):
        """Create semantic search engine with mocked dependencies."""
        embedding_generator = Mock()
        embedding_generator.generate_embedding = AsyncMock()
        
        vector_storage = Mock()
        vector_storage.search_similar = AsyncMock()
        
        return SemanticSearchEngine(
            embedding_generator=embedding_generator,
            vector_storage=vector_storage
        )
    
    @pytest.fixture
    def sample_query(self):
        """Create a sample search query."""
        return SearchQuery(
            query_text="find function that calculates sum",
            repository_id="test_repo",
            element_types=[CodeElementType.FUNCTION],
            max_results=10,
            similarity_threshold=0.7,
            include_explanation=True
        )
    
    @pytest.mark.asyncio
    async def test_search(self, search_engine, sample_query):
        """Test semantic search."""
        # Mock embedding generation
        mock_embedding = CodeEmbedding(
            element=CodeElement(
                element_type=CodeElementType.FUNCTION,
                name="query",
                code_snippet="find function that calculates sum",
                file_path="query",
                start_line=1,
                end_line=1,
                language="natural",
                metadata={"is_query": True}
            ),
            embedding=[0.1] * 768,
            model_name="test_model",
            embedding_dimension=768
        )
        search_engine.embedding_generator.generate_embedding.return_value = mock_embedding
        
        # Mock search results
        mock_results = [
            SearchResult(
                element=CodeElement(
                    element_type=CodeElementType.FUNCTION,
                    name="calculate_sum",
                    code_snippet="def calculate_sum(a, b): return a + b",
                    file_path="math.py",
                    start_line=1,
                    end_line=1,
                    language="python",
                    metadata={}
                ),
                similarity_score=0.9,
                embedding_distance=0.1,
                rank=1
            )
        ]
        search_engine.vector_storage.search_similar.return_value = mock_results
        
        results = await search_engine.search(sample_query)
        
        assert len(results) == 1
        assert results[0].element.name == "calculate_sum"
        assert results[0].similarity_score == 0.9
        search_engine.embedding_generator.generate_embedding.assert_called_once()
        search_engine.vector_storage.search_similar.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_by_code_similarity(self, search_engine):
        """Test searching by code similarity."""
        # Mock the search method
        search_engine.search = AsyncMock()
        search_engine.search.return_value = []
        
        results = await search_engine.search_by_code_similarity(
            code_snippet="def add(a, b): return a + b",
            repository_id="test_repo",
            language="python"
        )
        
        search_engine.search.assert_called_once()
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_find_similar_functions(self, search_engine):
        """Test finding similar functions."""
        search_engine.search = AsyncMock()
        search_engine.search.return_value = []
        
        results = await search_engine.find_similar_functions(
            function_name="calculate",
            repository_id="test_repo"
        )
        
        search_engine.search.assert_called_once()
        call_args = search_engine.search.call_args[0][0]
        assert "function calculate" in call_args.query_text
        assert CodeElementType.FUNCTION in call_args.element_types


class TestHybridSearchEngine:
    """Test hybrid search engine."""
    
    @pytest.fixture
    def hybrid_engine(self):
        """Create hybrid search engine with mocked dependencies."""
        semantic_engine = Mock()
        semantic_engine.search = AsyncMock()
        
        neo4j_client = Mock()
        
        return HybridSearchEngine(
            semantic_engine=semantic_engine,
            neo4j_client=neo4j_client
        )
    
    @pytest.fixture
    def sample_query(self):
        """Create a sample search query."""
        return SearchQuery(
            query_text="process data function",
            repository_id="test_repo",
            max_results=10,
            similarity_threshold=0.7
        )
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, hybrid_engine, sample_query):
        """Test hybrid search combining semantic and structural."""
        # Mock semantic results
        semantic_results = [
            SearchResult(
                element=CodeElement(
                    element_type=CodeElementType.FUNCTION,
                    name="process_data",
                    code_snippet="def process_data(data): return data",
                    file_path="processor.py",
                    start_line=1,
                    end_line=1,
                    language="python",
                    metadata={}
                ),
                similarity_score=0.8,
                embedding_distance=0.2,
                rank=1
            )
        ]
        hybrid_engine.semantic_engine.search.return_value = semantic_results
        
        results = await hybrid_engine.hybrid_search(sample_query)
        
        assert isinstance(results, list)
        assert all(isinstance(result, HybridSearchResult) for result in results)
        hybrid_engine.semantic_engine.search.assert_called_once()
    
    def test_extract_search_terms(self, hybrid_engine):
        """Test extracting search terms from query."""
        terms = hybrid_engine._extract_search_terms("find function that processes data")
        
        assert isinstance(terms, dict)
        assert "functions" in terms
        assert "general" in terms
        assert len(terms["general"]) > 0
    
    def test_build_structural_query(self, hybrid_engine):
        """Test building Cypher query for structural search."""
        search_terms = {
            "functions": ["function"],
            "general": ["process", "data"]
        }
        
        query = hybrid_engine._build_structural_query(
            search_terms=search_terms,
            repository_id="test_repo",
            element_types=[CodeElementType.FUNCTION],
            file_patterns=None,
            limit=10
        )
        
        assert isinstance(query, str)
        assert "MATCH" in query
        assert "test_repo" in query
        assert "LIMIT 10" in query
    
    @pytest.mark.asyncio
    async def test_search_with_context(self, hybrid_engine, sample_query):
        """Test context-aware hybrid search."""
        context_elements = [
            CodeElement(
                element_type=CodeElementType.CLASS,
                name="DataProcessor",
                code_snippet="class DataProcessor: pass",
                file_path="processor.py",
                start_line=1,
                end_line=1,
                language="python",
                metadata={}
            )
        ]
        
        # Mock hybrid_search method
        hybrid_engine.hybrid_search = AsyncMock()
        hybrid_engine.hybrid_search.return_value = []
        
        results = await hybrid_engine.search_with_context(sample_query, context_elements)
        
        hybrid_engine.hybrid_search.assert_called_once()
        assert isinstance(results, list)


class TestSemanticSearchIntegration:
    """Test semantic search integration."""
    
    @pytest.fixture
    def integration(self):
        """Create integration with mocked dependencies."""
        with patch('src.code_intelligence.semantic.integration.CodeEmbeddingGenerator'), \
             patch('src.code_intelligence.semantic.integration.VectorStorage'), \
             patch('src.code_intelligence.semantic.integration.SemanticSearchEngine'), \
             patch('src.code_intelligence.semantic.integration.HybridSearchEngine'):
            
            integration = SemanticSearchIntegration()
            
            # Mock the engines
            integration.embedding_generator = Mock()
            integration.vector_storage = Mock()
            integration.semantic_engine = Mock()
            integration.hybrid_engine = Mock()
            
            return integration
    
    @pytest.mark.asyncio
    async def test_search_code_semantic(self, integration):
        """Test semantic code search."""
        # Mock semantic search
        mock_results = [
            SearchResult(
                element=CodeElement(
                    element_type=CodeElementType.FUNCTION,
                    name="test_function",
                    code_snippet="def test_function(): pass",
                    file_path="test.py",
                    start_line=1,
                    end_line=1,
                    language="python",
                    metadata={}
                ),
                similarity_score=0.9,
                embedding_distance=0.1,
                rank=1,
                explanation="High similarity match"
            )
        ]
        integration.semantic_engine.search = AsyncMock(return_value=mock_results)
        
        result = await integration.search_code(
            query_text="test function",
            repository_id="test_repo",
            search_type="semantic"
        )
        
        assert result["query"]["text"] == "test function"
        assert result["query"]["search_type"] == "semantic"
        assert len(result["results"]) == 1
        assert result["results"][0]["element"]["name"] == "test_function"
        assert result["metadata"]["total_results"] == 1
    
    @pytest.mark.asyncio
    async def test_search_code_hybrid(self, integration):
        """Test hybrid code search."""
        # Mock hybrid search
        mock_results = [
            HybridSearchResult(
                semantic_result=SearchResult(
                    element=CodeElement(
                        element_type=CodeElementType.FUNCTION,
                        name="hybrid_function",
                        code_snippet="def hybrid_function(): pass",
                        file_path="test.py",
                        start_line=1,
                        end_line=1,
                        language="python",
                        metadata={}
                    ),
                    similarity_score=0.8,
                    embedding_distance=0.2,
                    rank=1
                ),
                structural_matches=[],
                combined_score=0.8,
                rank=1,
                explanation="Semantic match"
            )
        ]
        integration.hybrid_engine.hybrid_search = AsyncMock(return_value=mock_results)
        
        result = await integration.search_code(
            query_text="hybrid function",
            repository_id="test_repo",
            search_type="hybrid"
        )
        
        assert result["query"]["search_type"] == "hybrid"
        assert len(result["results"]) == 1
        assert result["results"][0]["element"]["name"] == "hybrid_function"
        assert "combined_score" in result["results"][0]
    
    @pytest.mark.asyncio
    async def test_get_search_statistics(self, integration):
        """Test getting search statistics."""
        # Mock vector storage stats
        mock_stats = {
            "total_embeddings": 100,
            "by_element_type": {"function": 60, "class": 40},
            "by_language": {"python": 80, "javascript": 20}
        }
        integration.vector_storage.get_embedding_stats = AsyncMock(return_value=mock_stats)
        
        stats = await integration.get_search_statistics("test_repo")
        
        assert stats["repository_id"] == "test_repo"
        assert "embeddings" in stats
        assert "search_config" in stats
        assert "capabilities" in stats
        assert stats["capabilities"]["semantic_search"] is True
        assert stats["capabilities"]["hybrid_search"] is True


class TestPerformance:
    """Test performance aspects of semantic search."""
    
    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test processing large batches of embeddings."""
        generator = CodeEmbeddingGenerator()
        generator._use_mock = True
        
        # Create large batch
        elements = []
        for i in range(100):
            element = CodeElement(
                element_type=CodeElementType.FUNCTION,
                name=f"function_{i}",
                code_snippet=f"def function_{i}(): return {i}",
                file_path=f"file_{i}.py",
                start_line=1,
                end_line=2,
                language="python",
                metadata={}
            )
            elements.append(element)
        
        batch = EmbeddingBatch(
            elements=elements,
            batch_id="large_batch",
            repository_id="test_repo",
            commit_sha="abc123"
        )
        
        import time
        start_time = time.time()
        embeddings = await generator.generate_batch_embeddings(batch)
        end_time = time.time()
        
        assert len(embeddings) == 100
        assert end_time - start_time < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_searches(self):
        """Test concurrent search operations."""
        engine = SemanticSearchEngine()
        engine.embedding_generator = Mock()
        engine.vector_storage = Mock()
        
        # Mock async methods
        engine.embedding_generator.generate_embedding = AsyncMock()
        engine.vector_storage.search_similar = AsyncMock(return_value=[])
        
        # Create multiple queries
        queries = []
        for i in range(10):
            query = SearchQuery(
                query_text=f"search query {i}",
                repository_id="test_repo",
                max_results=5
            )
            queries.append(query)
        
        # Execute searches concurrently
        import time
        start_time = time.time()
        tasks = [engine.search(query) for query in queries]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        assert len(results) == 10
        assert end_time - start_time < 2.0  # Should complete concurrently
    
    def test_embedding_memory_usage(self):
        """Test memory usage of embeddings."""
        import sys
        
        # Create embedding
        element = CodeElement(
            element_type=CodeElementType.FUNCTION,
            name="test_function",
            code_snippet="def test_function(): pass",
            file_path="test.py",
            start_line=1,
            end_line=1,
            language="python",
            metadata={}
        )
        
        embedding = CodeEmbedding(
            element=element,
            embedding=[0.1] * 768,
            model_name="test_model",
            embedding_dimension=768
        )
        
        # Check memory usage is reasonable
        size = sys.getsizeof(embedding)
        assert size < 50000  # Should be less than 50KB per embedding


if __name__ == "__main__":
    pytest.main([__file__])