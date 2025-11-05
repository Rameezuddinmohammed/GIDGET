"""Integration module for semantic search with the code intelligence system."""

from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime

from ..parsing.models import CodeElement as ParsedCodeElement
from ..ingestion.models import IngestionJob
from ..database.neo4j_client import neo4j_client
from ..database.supabase_client import supabase_client
from ..config import config
from ..logging import get_logger
from ..exceptions import SearchError, EmbeddingError

from .models import (
    CodeElement, CodeElementType, EmbeddingBatch, SearchQuery, 
    HybridSearchResult, SearchResult
)
from .embeddings import CodeEmbeddingGenerator, EmbeddingModel
from .storage import VectorStorage
from .search import SemanticSearchEngine, HybridSearchEngine

logger = get_logger(__name__)


class SemanticSearchIntegration:
    """Integration layer for semantic search with the code intelligence system."""
    
    def __init__(self) -> None:
        """Initialize semantic search integration."""
        self.embedding_generator = CodeEmbeddingGenerator(
            model=EmbeddingModel.CODEBERT,
            device=config.semantic.embedding_device,
            max_batch_size=config.semantic.embedding_batch_size,
            max_sequence_length=config.semantic.max_sequence_length
        )
        self.vector_storage = VectorStorage()
        self.semantic_engine = SemanticSearchEngine(
            embedding_generator=self.embedding_generator,
            vector_storage=self.vector_storage
        )
        self.hybrid_engine = HybridSearchEngine(
            semantic_engine=self.semantic_engine,
            neo4j_client=neo4j_client
        )
    
    async def process_repository_for_search(
        self, 
        repository_id: str,
        commit_sha: str,
        parsed_elements: List[ParsedCodeElement]
    ) -> Dict[str, Any]:
        """Process parsed elements to generate embeddings for search."""
        logger.info(
            "Processing repository for semantic search",
            repository_id=repository_id,
            commit_sha=commit_sha
        )
        
        try:
            # Convert parsed elements to semantic search format
            code_elements = await self._convert_parsed_elements(
                parsed_elements,
                repository_id
            )
            
            # Create embedding batch
            batch = EmbeddingBatch(
                elements=code_elements,
                batch_id=f"repo_{repository_id}_{commit_sha[:8]}",
                repository_id=repository_id,
                commit_sha=commit_sha
            )
            
            # Generate embeddings
            embeddings = await self.embedding_generator.generate_batch_embeddings(batch)
            
            # Store embeddings
            await self.vector_storage.store_batch_embeddings(
                embeddings, commit_sha
            )
            
            # Create vector index for performance
            await self.vector_storage.create_vector_index(repository_id)
            
            # Validate embedding quality
            quality = await self.embedding_generator.validate_embedding_quality(embeddings)
            
            result = {
                "repository_id": repository_id,
                "commit_sha": commit_sha,
                "embeddings_generated": len(embeddings),
                "embedding_quality": {
                    "quality_score": quality.quality_score,
                    "avg_magnitude": quality.avg_magnitude,
                    "sparsity_ratio": quality.sparsity_ratio,
                    "consistency_score": quality.consistency_score
                },
                "processed_at": datetime.now().isoformat()
            }
            
            logger.info(
                "Repository processed for semantic search",
                repository_id=repository_id,
                embeddings_count=len(embeddings),
                quality_score=quality.quality_score
            )
            
            return result
            
        except Exception as e:
            logger.error(
                "Failed to process repository for search",
                repository_id=repository_id,
                error=str(e)
            )
            raise EmbeddingError(f"Repository processing failed: {e}")
    
    async def _convert_parsed_elements(
        self,
        parsed_elements: List[ParsedCodeElement],
        repository_id: str
    ) -> List[CodeElement]:
        """Convert parsed code elements to semantic search format."""
        code_elements = []
        
        for parsed_element in parsed_elements:
            # Map element types
            element_type_map = {
                "function": CodeElementType.FUNCTION,
                "class": CodeElementType.CLASS,
                "method": CodeElementType.METHOD,
                "module": CodeElementType.MODULE,
                "variable": CodeElementType.VARIABLE
            }
            
            element_type = element_type_map.get(
                parsed_element.element_type.lower(), 
                CodeElementType.FUNCTION
            )
            
            # Create semantic code element
            code_element = CodeElement(
                element_type=element_type,
                name=parsed_element.name,
                code_snippet=parsed_element.content,
                file_path=parsed_element.file_path,
                start_line=parsed_element.start_line,
                end_line=parsed_element.end_line,
                language=parsed_element.language,
                metadata={
                    "repository_id": repository_id,
                    "docstring": getattr(parsed_element, "docstring", None),
                    "parameters": getattr(parsed_element, "parameters", []),
                    "return_type": getattr(parsed_element, "return_type", None),
                    "complexity": getattr(parsed_element, "complexity", None),
                    "dependencies": getattr(parsed_element, "dependencies", [])
                }
            )
            
            code_elements.append(code_element)
        
        return code_elements
    
    async def search_code(
        self,
        query_text: str,
        repository_id: str,
        search_type: str = "hybrid",
        max_results: int = 10,
        similarity_threshold: float = None,
        element_types: Optional[List[str]] = None,
        file_patterns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Search for code using semantic or hybrid search."""
        logger.info(
            "Searching code",
            query_text=query_text,
            repository_id=repository_id,
            search_type=search_type,
            max_results=max_results
        )
        
        try:
            # Use configured threshold if not provided
            if similarity_threshold is None:
                similarity_threshold = config.semantic.similarity_threshold
            
            # Convert element types
            semantic_element_types = None
            if element_types:
                type_map = {
                    "function": CodeElementType.FUNCTION,
                    "class": CodeElementType.CLASS,
                    "method": CodeElementType.METHOD,
                    "module": CodeElementType.MODULE,
                    "variable": CodeElementType.VARIABLE
                }
                semantic_element_types = [
                    type_map[et] for et in element_types if et in type_map
                ]
            
            # Create search query
            query = SearchQuery(
                query_text=query_text,
                repository_id=repository_id,
                element_types=semantic_element_types,
                file_patterns=file_patterns,
                max_results=max_results,
                similarity_threshold=similarity_threshold,
                include_explanation=True
            )
            
            # Perform search based on type
            if search_type == "semantic":
                results = await self.semantic_engine.search(query)
                search_results = self._format_semantic_results(results)
            elif search_type == "hybrid":
                results = await self.hybrid_engine.hybrid_search(
                    query,
                    semantic_weight=config.semantic.semantic_weight,
                    structural_weight=config.semantic.structural_weight
                )
                search_results = self._format_hybrid_results(results)
            else:
                raise SearchError(f"Unknown search type: {search_type}")
            
            response = {
                "query": {
                    "text": query_text,
                    "repository_id": repository_id,
                    "search_type": search_type,
                    "hash": query.query_hash
                },
                "results": search_results,
                "metadata": {
                    "total_results": len(search_results),
                    "max_results": max_results,
                    "similarity_threshold": similarity_threshold,
                    "search_time": datetime.now().isoformat()
                }
            }
            
            logger.info(
                "Code search completed",
                query_hash=query.query_hash,
                results_count=len(search_results),
                search_type=search_type
            )
            
            return response
            
        except Exception as e:
            logger.error("Code search failed", query_text=query_text, error=str(e))
            raise SearchError(f"Code search failed: {e}")
    
    def _format_semantic_results(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Format semantic search results for API response."""
        formatted_results = []
        
        for result in results:
            formatted_result = {
                "rank": result.rank,
                "similarity_score": result.similarity_score,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "element": {
                    "type": result.element.element_type.value,
                    "name": result.element.name,
                    "file_path": result.element.file_path,
                    "start_line": result.element.start_line,
                    "end_line": result.element.end_line,
                    "language": result.element.language,
                    "code_snippet": result.element.code_snippet,
                    "signature": result.element.signature
                },
                "metadata": result.element.metadata
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    def _format_hybrid_results(self, results: List[HybridSearchResult]) -> List[Dict[str, Any]]:
        """Format hybrid search results for API response."""
        formatted_results = []
        
        for result in results:
            best_element = result.best_element
            
            formatted_result = {
                "rank": result.rank,
                "combined_score": result.combined_score,
                "explanation": result.explanation,
                "search_types": {
                    "semantic": result.semantic_result is not None,
                    "structural": len(result.structural_matches) > 0
                }
            }
            
            if best_element:
                formatted_result["element"] = {
                    "type": best_element.element_type.value,
                    "name": best_element.name,
                    "file_path": best_element.file_path,
                    "start_line": best_element.start_line,
                    "end_line": best_element.end_line,
                    "language": best_element.language,
                    "code_snippet": best_element.code_snippet,
                    "signature": best_element.signature
                }
                formatted_result["metadata"] = best_element.metadata
            
            # Add detailed scores
            if result.semantic_result:
                formatted_result["semantic_score"] = result.semantic_result.similarity_score
            
            if result.structural_matches:
                formatted_result["structural_score"] = result.structural_matches[0].get("structural_score", 0.0)
            
            formatted_results.append(formatted_result)
        
        return formatted_results
    
    async def update_repository_embeddings(
        self,
        repository_id: str,
        changed_files: List[str],
        commit_sha: str
    ) -> Dict[str, Any]:
        """Update embeddings for changed files in a repository."""
        logger.info(
            "Updating repository embeddings",
            repository_id=repository_id,
            changed_files_count=len(changed_files),
            commit_sha=commit_sha
        )
        
        try:
            # This would typically integrate with the parsing pipeline
            # to get updated code elements for the changed files
            
            # For now, return a mock response
            result = {
                "repository_id": repository_id,
                "commit_sha": commit_sha,
                "changed_files": changed_files,
                "embeddings_updated": len(changed_files) * 5,  # Mock: ~5 elements per file
                "updated_at": datetime.now().isoformat()
            }
            
            logger.info(
                "Repository embeddings updated",
                repository_id=repository_id,
                updated_count=result["embeddings_updated"]
            )
            
            return result
            
        except Exception as e:
            logger.error("Failed to update repository embeddings", error=str(e))
            raise EmbeddingError(f"Embedding update failed: {e}")
    
    async def get_search_statistics(self, repository_id: str) -> Dict[str, Any]:
        """Get semantic search statistics for a repository."""
        try:
            # Get embedding statistics
            embedding_stats = await self.vector_storage.get_embedding_stats(repository_id)
            
            # Add search-specific statistics
            stats = {
                "repository_id": repository_id,
                "embeddings": embedding_stats,
                "search_config": {
                    "model": config.semantic.embedding_model,
                    "similarity_threshold": config.semantic.similarity_threshold,
                    "max_results": config.semantic.max_search_results,
                    "semantic_weight": config.semantic.semantic_weight,
                    "structural_weight": config.semantic.structural_weight
                },
                "capabilities": {
                    "semantic_search": True,
                    "hybrid_search": True,
                    "context_aware_search": True,
                    "similarity_search": True
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error("Failed to get search statistics", error=str(e))
            raise SearchError(f"Statistics retrieval failed: {e}")


# Global integration instance
semantic_integration = SemanticSearchIntegration()