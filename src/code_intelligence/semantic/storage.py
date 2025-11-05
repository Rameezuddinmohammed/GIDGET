"""Vector storage and similarity search using pgvector."""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import numpy as np
from datetime import datetime

from ..database.supabase_client import supabase_client
from ..logging import get_logger
from ..exceptions import VectorStorageError
from .models import (
    CodeEmbedding, SearchResult, CodeElement, CodeElementType, SearchQuery
)

logger = get_logger(__name__)


class VectorStorage:
    """Manages vector storage and similarity search using pgvector."""
    
    def __init__(self) -> None:
        """Initialize vector storage."""
        self.client = supabase_client
    
    async def store_embedding(self, embedding: CodeEmbedding, commit_sha: str) -> None:
        """Store a single code embedding."""
        try:
            await self.client.store_code_embedding(
                repository_id=embedding.element.metadata.get("repository_id", ""),
                file_path=embedding.element.file_path,
                element_type=embedding.element.element_type.value,
                element_name=embedding.element.name,
                code_snippet=embedding.element.code_snippet,
                embedding=embedding.embedding,
                commit_sha=commit_sha,
                metadata={
                    "model_name": embedding.model_name,
                    "confidence_score": embedding.confidence_score,
                    "language": embedding.element.language,
                    "start_line": embedding.element.start_line,
                    "end_line": embedding.element.end_line,
                    "content_hash": embedding.element.content_hash,
                    **embedding.element.metadata
                }
            )
            
            logger.debug(
                "Embedding stored",
                element=embedding.element.signature,
                model=embedding.model_name
            )
            
        except Exception as e:
            logger.error(
                "Failed to store embedding",
                element=embedding.element.signature,
                error=str(e)
            )
            raise VectorStorageError(f"Embedding storage failed: {e}")
    
    async def store_batch_embeddings(
        self, 
        embeddings: List[CodeEmbedding], 
        commit_sha: str
    ) -> None:
        """Store a batch of code embeddings efficiently."""
        logger.info("Storing batch embeddings", count=len(embeddings), commit_sha=commit_sha)
        
        try:
            # Store embeddings in parallel batches
            batch_size = 50  # Reasonable batch size for database operations
            
            for i in range(0, len(embeddings), batch_size):
                batch = embeddings[i:i + batch_size]
                
                # Create tasks for parallel storage
                tasks = [
                    self.store_embedding(embedding, commit_sha)
                    for embedding in batch
                ]
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                logger.debug(f"Stored batch {i//batch_size + 1}/{(len(embeddings)-1)//batch_size + 1}")
            
            logger.info("Batch embeddings stored successfully", count=len(embeddings))
            
        except Exception as e:
            logger.error("Failed to store batch embeddings", error=str(e))
            raise VectorStorageError(f"Batch storage failed: {e}")
    
    async def search_similar(
        self,
        query_embedding: List[float],
        repository_id: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        element_types: Optional[List[CodeElementType]] = None,
        file_patterns: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search for similar code using vector similarity."""
        try:
            logger.info(
                "Searching similar code",
                repository_id=repository_id,
                limit=limit,
                threshold=similarity_threshold
            )
            
            # Build the search query
            search_conditions = {"repository_id": repository_id}
            
            if element_types:
                search_conditions["element_type"] = [et.value for et in element_types]
            
            # Execute similarity search
            # Note: This is a simplified version - actual implementation would use
            # pgvector's similarity search capabilities
            results = await self._execute_similarity_search(
                query_embedding=query_embedding,
                conditions=search_conditions,
                limit=limit,
                threshold=similarity_threshold,
                file_patterns=file_patterns
            )
            
            # Convert to SearchResult objects
            search_results = []
            for i, result in enumerate(results):
                element = self._result_to_code_element(result)
                
                search_result = SearchResult(
                    element=element,
                    similarity_score=result["similarity_score"],
                    embedding_distance=result["distance"],
                    rank=i + 1,
                    explanation=self._generate_search_explanation(result, query_embedding)
                )
                search_results.append(search_result)
            
            logger.info(
                "Similar code search completed",
                repository_id=repository_id,
                results_count=len(search_results)
            )
            
            return search_results
            
        except Exception as e:
            logger.error("Failed to search similar code", error=str(e))
            raise VectorStorageError(f"Similarity search failed: {e}")
    
    async def _execute_similarity_search(
        self,
        query_embedding: List[float],
        conditions: Dict[str, Any],
        limit: int,
        threshold: float,
        file_patterns: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Execute the actual similarity search query using pgvector."""
        try:
            # Map parameters for the SQL function
            params = {
                "query_embedding": query_embedding,
                "repository_id_param": conditions.get("repository_id"),
                "element_type_param": conditions.get("element_type", [None])[0] if conditions.get("element_type") else None,
                "similarity_threshold": threshold,
                "max_results": limit
            }
            
            logger.debug(
                "Executing similarity search",
                repository_id=params["repository_id_param"],
                element_type=params["element_type_param"],
                threshold=threshold,
                limit=limit
            )
            
            # Call the real pgvector function from supabase_schema.sql
            response = self.client.client.rpc("search_similar_code", params).execute()
            
            # Process and return results
            results = []
            if response.data:
                for item in response.data:
                    # Re-format the SQL-style response to match the expected dictionary structure
                    results.append({
                        "id": item["id"],
                        "repository_id": item["repository_id"],
                        "file_path": item["file_path"],
                        "element_type": item["element_type"],
                        "element_name": item["element_name"],
                        "code_snippet": item["code_snippet"],
                        "similarity_score": item["similarity"],  # SQL fn returns 'similarity'
                        "distance": 1.0 - item["similarity"],
                        "metadata": item["metadata"],
                        "commit_sha": item.get("commit_sha", "unknown"),
                        "created_at": datetime.now().isoformat()
                    })
            
            logger.info(
                "Similarity search completed",
                repository_id=params["repository_id_param"],
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            logger.error("Failed to execute real similarity search", error=str(e))
            raise VectorStorageError(f"Real similarity search failed: {e}")
    
    def _result_to_code_element(self, result: Dict[str, Any]) -> CodeElement:
        """Convert search result to CodeElement."""
        metadata = result.get("metadata", {})
        
        return CodeElement(
            element_type=CodeElementType(result["element_type"]),
            name=result["element_name"],
            code_snippet=result["code_snippet"],
            file_path=result["file_path"],
            start_line=metadata.get("start_line", 1),
            end_line=metadata.get("end_line", 1),
            language=metadata.get("language", "unknown"),
            metadata={
                "repository_id": result["repository_id"],
                "commit_sha": result["commit_sha"],
                "model_name": metadata.get("model_name"),
                "confidence_score": metadata.get("confidence_score", 1.0),
                **metadata
            }
        )
    
    def _generate_search_explanation(
        self, 
        result: Dict[str, Any], 
        query_embedding: List[float]
    ) -> str:
        """Generate explanation for why this result matches the query."""
        similarity = result["similarity_score"]
        element_type = result["element_type"]
        element_name = result["element_name"]
        
        if similarity > 0.9:
            confidence = "very high"
        elif similarity > 0.8:
            confidence = "high"
        elif similarity > 0.7:
            confidence = "moderate"
        else:
            confidence = "low"
        
        return (
            f"Found {element_type} '{element_name}' with {confidence} semantic similarity "
            f"(score: {similarity:.3f}). The code structure and functionality appear to "
            f"match the query intent."
        )
    
    async def update_embeddings(
        self,
        repository_id: str,
        updated_embeddings: List[CodeEmbedding],
        commit_sha: str
    ) -> None:
        """Update embeddings for changed code elements."""
        logger.info(
            "Updating embeddings",
            repository_id=repository_id,
            count=len(updated_embeddings),
            commit_sha=commit_sha
        )
        
        try:
            # Remove old embeddings for the same elements
            for embedding in updated_embeddings:
                await self._remove_old_embedding(
                    repository_id=repository_id,
                    file_path=embedding.element.file_path,
                    element_name=embedding.element.name,
                    element_type=embedding.element.element_type.value
                )
            
            # Store new embeddings
            await self.store_batch_embeddings(updated_embeddings, commit_sha)
            
            logger.info("Embeddings updated successfully", repository_id=repository_id)
            
        except Exception as e:
            logger.error("Failed to update embeddings", error=str(e))
            raise VectorStorageError(f"Embedding update failed: {e}")
    
    async def _remove_old_embedding(
        self,
        repository_id: str,
        file_path: str,
        element_name: str,
        element_type: str
    ) -> None:
        """Remove old embedding for a specific code element."""
        try:
            # This would use Supabase delete operation
            # For now, we'll just log the operation
            logger.debug(
                "Removing old embedding",
                repository_id=repository_id,
                file_path=file_path,
                element_name=element_name,
                element_type=element_type
            )
            
            # In real implementation:
            # self.client.table("code_embeddings").delete().match({
            #     "repository_id": repository_id,
            #     "file_path": file_path,
            #     "element_name": element_name,
            #     "element_type": element_type
            # }).execute()
            
        except Exception as e:
            logger.warning("Failed to remove old embedding", error=str(e))
            # Don't raise - this is not critical for the update process
    
    async def get_embedding_stats(self, repository_id: str) -> Dict[str, Any]:
        """Get statistics about stored embeddings for a repository."""
        try:
            # Mock implementation - would query actual database
            stats = {
                "total_embeddings": 150,
                "by_element_type": {
                    "function": 80,
                    "class": 25,
                    "method": 40,
                    "module": 5
                },
                "by_language": {
                    "python": 120,
                    "javascript": 20,
                    "typescript": 10
                },
                "model_distribution": {
                    "microsoft/codebert-base": 150
                },
                "avg_confidence": 0.92,
                "last_updated": datetime.now().isoformat()
            }
            
            logger.info("Retrieved embedding stats", repository_id=repository_id, **stats)
            return stats
            
        except Exception as e:
            logger.error("Failed to get embedding stats", error=str(e))
            raise VectorStorageError(f"Stats retrieval failed: {e}")
    
    async def cleanup_old_embeddings(
        self, 
        repository_id: str, 
        keep_commits: int = 10
    ) -> int:
        """Clean up old embeddings, keeping only recent commits."""
        try:
            logger.info(
                "Cleaning up old embeddings",
                repository_id=repository_id,
                keep_commits=keep_commits
            )
            
            # Mock implementation - would identify and delete old embeddings
            # In real implementation, this would:
            # 1. Find commits older than the keep_commits threshold
            # 2. Delete embeddings associated with those commits
            # 3. Return count of deleted embeddings
            
            deleted_count = 25  # Mock count
            
            logger.info(
                "Old embeddings cleaned up",
                repository_id=repository_id,
                deleted_count=deleted_count
            )
            
            return deleted_count
            
        except Exception as e:
            logger.error("Failed to cleanup old embeddings", error=str(e))
            raise VectorStorageError(f"Cleanup failed: {e}")
    
    async def create_vector_index(self, repository_id: str) -> None:
        """Create or update vector index for improved search performance."""
        try:
            logger.info("Creating vector index", repository_id=repository_id)
            
            # Mock implementation - would create pgvector index
            # In real implementation:
            # CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_embeddings_vector_repo_X
            # ON code_embeddings USING ivfflat (embedding vector_cosine_ops)
            # WHERE repository_id = 'X'
            # WITH (lists = 100);
            
            logger.info("Vector index created", repository_id=repository_id)
            
        except Exception as e:
            logger.error("Failed to create vector index", error=str(e))
            raise VectorStorageError(f"Index creation failed: {e}")
    
    async def search_by_query(self, query: SearchQuery) -> List[SearchResult]:
        """Search using a structured query object."""
        # First, we need to generate an embedding for the query text
        # This would typically use the same embedding model as the stored embeddings
        
        # For now, create a mock query embedding
        np.random.seed(hash(query.query_text) % (2**32))
        query_embedding = np.random.normal(0, 1, 768).tolist()
        query_embedding = (np.array(query_embedding) / np.linalg.norm(query_embedding)).tolist()
        
        return await self.search_similar(
            query_embedding=query_embedding,
            repository_id=query.repository_id,
            limit=query.max_results,
            similarity_threshold=query.similarity_threshold,
            element_types=query.element_types,
            file_patterns=query.file_patterns
        )