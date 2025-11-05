"""Semantic search engine with hybrid capabilities."""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import numpy as np
from datetime import datetime

from ..database.neo4j_client import neo4j_client
from ..logging import get_logger
from ..exceptions import SearchError
from .models import (
    SearchQuery, SearchResult, HybridSearchResult, CodeElement, 
    CodeElementType, EmbeddingModel
)
from .embeddings import CodeEmbeddingGenerator
from .storage import VectorStorage

logger = get_logger(__name__)


class SemanticSearchEngine:
    """Semantic search engine for code using vector embeddings."""
    
    def __init__(
        self, 
        embedding_generator: Optional[CodeEmbeddingGenerator] = None,
        vector_storage: Optional[VectorStorage] = None
    ) -> None:
        """Initialize semantic search engine."""
        self.embedding_generator = embedding_generator or CodeEmbeddingGenerator()
        self.vector_storage = vector_storage or VectorStorage()
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform semantic search for code elements."""
        logger.info(
            "Performing semantic search",
            query_text=query.query_text,
            repository_id=query.repository_id,
            max_results=query.max_results
        )
        
        try:
            # Generate embedding for the query
            query_embedding = await self._generate_query_embedding(query.query_text)
            
            # Search for similar code
            results = await self.vector_storage.search_similar(
                query_embedding=query_embedding,
                repository_id=query.repository_id,
                limit=query.max_results,
                similarity_threshold=query.similarity_threshold,
                element_types=query.element_types,
                file_patterns=query.file_patterns
            )
            
            # Add explanations if requested
            if query.include_explanation:
                for result in results:
                    if not result.explanation:
                        result.explanation = await self._generate_result_explanation(
                            query.query_text, result
                        )
            
            logger.info(
                "Semantic search completed",
                query_hash=query.query_hash,
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            logger.error("Semantic search failed", query_hash=query.query_hash, error=str(e))
            raise SearchError(f"Semantic search failed: {e}")
    
    async def _generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for search query."""
        # Create a mock code element for the query
        from .models import CodeElement
        
        query_element = CodeElement(
            element_type=CodeElementType.FUNCTION,  # Default type
            name="query",
            code_snippet=query_text,
            file_path="query",
            start_line=1,
            end_line=1,
            language="natural",
            metadata={"is_query": True}
        )
        
        # Generate embedding
        embedding_result = await self.embedding_generator.generate_embedding(query_element)
        return embedding_result.embedding
    
    async def _generate_result_explanation(
        self, 
        query_text: str, 
        result: SearchResult
    ) -> str:
        """Generate detailed explanation for search result."""
        element = result.element
        similarity = result.similarity_score
        
        # Analyze why this result matches
        explanation_parts = []
        
        # Similarity assessment
        if similarity > 0.9:
            explanation_parts.append("Very strong semantic match")
        elif similarity > 0.8:
            explanation_parts.append("Strong semantic similarity")
        elif similarity > 0.7:
            explanation_parts.append("Good semantic similarity")
        else:
            explanation_parts.append("Moderate semantic similarity")
        
        # Element type context
        explanation_parts.append(f"Found {element.element_type.value} '{element.name}'")
        
        # Language context
        explanation_parts.append(f"in {element.language}")
        
        # Location context
        explanation_parts.append(f"at {element.file_path}:{element.start_line}")
        
        # Confidence indicator
        explanation_parts.append(f"(confidence: {similarity:.1%})")
        
        return " ".join(explanation_parts) + "."
    
    async def search_by_code_similarity(
        self,
        code_snippet: str,
        repository_id: str,
        language: str = "python",
        max_results: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[SearchResult]:
        """Search for code similar to a given code snippet."""
        logger.info(
            "Searching by code similarity",
            repository_id=repository_id,
            language=language,
            snippet_length=len(code_snippet)
        )
        
        # Create a query from the code snippet
        query = SearchQuery(
            query_text=code_snippet,
            repository_id=repository_id,
            language_filter=language,
            max_results=max_results,
            similarity_threshold=similarity_threshold,
            include_explanation=True
        )
        
        return await self.search(query)
    
    async def find_similar_functions(
        self,
        function_name: str,
        repository_id: str,
        max_results: int = 10
    ) -> List[SearchResult]:
        """Find functions similar to a given function name."""
        query = SearchQuery(
            query_text=f"function {function_name}",
            repository_id=repository_id,
            element_types=[CodeElementType.FUNCTION, CodeElementType.METHOD],
            max_results=max_results,
            similarity_threshold=0.6,
            include_explanation=True
        )
        
        return await self.search(query)
    
    async def search_by_functionality(
        self,
        description: str,
        repository_id: str,
        max_results: int = 10
    ) -> List[SearchResult]:
        """Search for code by functional description."""
        query = SearchQuery(
            query_text=description,
            repository_id=repository_id,
            max_results=max_results,
            similarity_threshold=0.6,
            include_explanation=True
        )
        
        return await self.search(query)


class HybridSearchEngine:
    """Hybrid search engine combining semantic and structural search."""
    
    def __init__(
        self,
        semantic_engine: Optional[SemanticSearchEngine] = None,
        neo4j_client = None
    ) -> None:
        """Initialize hybrid search engine."""
        self.semantic_engine = semantic_engine or SemanticSearchEngine()
        # Import here to avoid circular imports
        if neo4j_client is None:
            from ..database.neo4j_client import neo4j_client as default_client
            self.neo4j_client = default_client
        else:
            self.neo4j_client = neo4j_client
    
    async def hybrid_search(
        self,
        query: SearchQuery,
        semantic_weight: float = 0.6,
        structural_weight: float = 0.4
    ) -> List[HybridSearchResult]:
        """Perform hybrid search combining semantic and structural approaches."""
        logger.info(
            "Performing hybrid search",
            query_text=query.query_text,
            repository_id=query.repository_id,
            semantic_weight=semantic_weight,
            structural_weight=structural_weight
        )
        
        try:
            # Perform semantic search
            semantic_task = asyncio.create_task(
                self.semantic_engine.search(query)
            )
            
            # Perform structural search
            structural_task = asyncio.create_task(
                self._structural_search(query)
            )
            
            # Wait for both searches to complete
            semantic_results, structural_results = await asyncio.gather(
                semantic_task, structural_task
            )
            
            # Combine and rank results
            hybrid_results = await self._combine_results(
                semantic_results=semantic_results,
                structural_results=structural_results,
                semantic_weight=semantic_weight,
                structural_weight=structural_weight
            )
            
            logger.info(
                "Hybrid search completed",
                query_hash=query.query_hash,
                semantic_count=len(semantic_results),
                structural_count=len(structural_results),
                hybrid_count=len(hybrid_results)
            )
            
            return hybrid_results
            
        except Exception as e:
            logger.error("Hybrid search failed", query_hash=query.query_hash, error=str(e))
            raise SearchError(f"Hybrid search failed: {e}")
    
    async def _structural_search(self, query: SearchQuery) -> List[Dict[str, Any]]:
        """Perform structural search using Neo4j graph queries."""
        try:
            # Parse query for structural elements
            search_terms = self._extract_search_terms(query.query_text)
            
            # Build Cypher query
            cypher_query = self._build_structural_query(
                search_terms=search_terms,
                repository_id=query.repository_id,
                element_types=query.element_types,
                file_patterns=query.file_patterns,
                limit=query.max_results
            )
            
            # Build parameters for the query
            parameters = {
                "repository_id": query.repository_id,
                "limit": query.max_results
            }
            
            # Add search term parameters
            for i, term in enumerate(search_terms["general"]):
                parameters[f"search_term_{i}"] = term
            
            # Add file pattern parameters
            if query.file_patterns:
                for i, pattern in enumerate(query.file_patterns):
                    parameters[f"file_pattern_{i}"] = f".*{pattern}.*"
            
            # Execute the real Neo4j query with parameters
            results = await self._execute_structural_query_with_params(cypher_query, parameters)
            
            return results
            
        except Exception as e:
            logger.error("Structural search failed", error=str(e))
            return []  # Return empty results rather than failing the entire search
    
    def _extract_search_terms(self, query_text: str) -> Dict[str, List[str]]:
        """Extract search terms from natural language query."""
        # Simple keyword extraction - in practice, this would be more sophisticated
        words = query_text.lower().split()
        
        # Identify different types of terms
        function_keywords = ["function", "method", "def", "async", "lambda"]
        class_keywords = ["class", "object", "type", "interface"]
        action_keywords = ["create", "update", "delete", "get", "set", "process", "handle"]
        
        terms = {
            "functions": [w for w in words if any(kw in w for kw in function_keywords)],
            "classes": [w for w in words if any(kw in w for kw in class_keywords)],
            "actions": [w for w in words if any(kw in w for kw in action_keywords)],
            "general": [w for w in words if len(w) > 3 and w.isalpha()]
        }
        
        return terms
    
    def _build_structural_query(
        self,
        search_terms: Dict[str, List[str]],
        repository_id: str,
        element_types: Optional[List[CodeElementType]],
        file_patterns: Optional[List[str]],
        limit: int
    ) -> str:
        """Build Cypher query for structural search with parameter binding."""
        # Base query structure with parameter binding
        query_parts = [
            "MATCH (repo:Repository {id: $repository_id})",
            "MATCH (repo)-[:CONTAINS]->(file:File)",
            "MATCH (file)-[:CONTAINS]->(element)"
        ]
        
        # Add element type filters
        if element_types:
            type_labels = [f":{et.value.capitalize()}" for et in element_types]
            query_parts.append(f"WHERE element{' OR element'.join(type_labels)}")
        
        # Add search conditions
        conditions = []
        
        # Name-based search with parameterized queries
        if search_terms["general"]:
            name_conditions = []
            for i, term in enumerate(search_terms["general"]):
                name_conditions.append(f"toLower(element.name) CONTAINS toLower($search_term_{i})")
            conditions.append(f"({' OR '.join(name_conditions)})")
        
        # File pattern filters
        if file_patterns:
            pattern_conditions = []
            for i, pattern in enumerate(file_patterns):
                pattern_conditions.append(f"file.path =~ $file_pattern_{i}")
            conditions.append(f"({' OR '.join(pattern_conditions)})")
        
        if conditions:
            if "WHERE" in query_parts[-1]:
                query_parts.append(f"AND ({' OR '.join(conditions)})")
            else:
                query_parts.append(f"WHERE {' OR '.join(conditions)}")
        
        # Return clause
        query_parts.extend([
            "RETURN element, file, repo",
            f"LIMIT $limit"
        ])
        
        return "\n".join(query_parts)
    
    async def _execute_structural_query_with_params(
        self, 
        cypher_query: str, 
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute structural query against Neo4j with parameters."""
        try:
            logger.debug(
                "Executing Neo4j structural query with parameters", 
                query=cypher_query,
                param_count=len(parameters)
            )
            
            # Execute the real Neo4j query with parameters
            results = await self.neo4j_client.execute_query(cypher_query, parameters)
            
            # Format results to match the expected structure
            formatted_results = []
            for record in results:
                element_data = record.get("element", {})
                file_data = record.get("file", {})
                
                # Extract element properties (Neo4j nodes have properties)
                if hasattr(element_data, 'get'):
                    # Handle Neo4j node object
                    element_props = dict(element_data)
                else:
                    # Handle dictionary
                    element_props = element_data
                
                if hasattr(file_data, 'get'):
                    # Handle Neo4j node object
                    file_props = dict(file_data)
                else:
                    # Handle dictionary
                    file_props = file_data
                
                formatted_result = {
                    "element": {
                        "name": element_props.get("name", "unknown"),
                        "type": element_props.get("type", "unknown"),
                        "file_path": element_props.get("file_path", file_props.get("path", "unknown")),
                        "start_line": element_props.get("start_line", 1),
                        "end_line": element_props.get("end_line", 1),
                        "code_snippet": element_props.get("code_snippet", "# Code not available")
                    },
                    "file": {
                        "path": file_props.get("path", element_props.get("file_path", "unknown")),
                        "language": file_props.get("language", "unknown")
                    },
                    "structural_score": 0.8  # TODO: Implement real scoring based on query match
                }
                formatted_results.append(formatted_result)
            
            logger.info(
                "Structural search completed",
                query_length=len(cypher_query),
                results_count=len(formatted_results)
            )
            
            return formatted_results
            
        except Exception as e:
            logger.error("Structural search failed", query=cypher_query, error=str(e))
            return []  # Return empty list on failure
    
    async def _combine_results(
        self,
        semantic_results: List[SearchResult],
        structural_results: List[Dict[str, Any]],
        semantic_weight: float,
        structural_weight: float
    ) -> List[HybridSearchResult]:
        """Combine semantic and structural search results."""
        combined_results = []
        
        # Create a map of semantic results by element signature
        semantic_map = {}
        for result in semantic_results:
            signature = result.element.signature
            semantic_map[signature] = result
        
        # Create a map of structural results by element signature
        structural_map = {}
        for result in structural_results:
            element_data = result["element"]
            signature = f"{element_data['file_path']}:{element_data['type']}:{element_data['name']}:{element_data['start_line']}"
            structural_map[signature] = result
        
        # Get all unique signatures
        all_signatures = set(semantic_map.keys()) | set(structural_map.keys())
        
        # Combine results
        for i, signature in enumerate(all_signatures):
            semantic_result = semantic_map.get(signature)
            structural_result = structural_map.get(signature)
            
            # Calculate combined score
            semantic_score = semantic_result.similarity_score if semantic_result else 0.0
            structural_score = structural_result.get("structural_score", 0.0) if structural_result else 0.0
            
            combined_score = (
                semantic_score * semantic_weight + 
                structural_score * structural_weight
            )
            
            # Create explanation
            explanation_parts = []
            if semantic_result and structural_result:
                explanation_parts.append("Found through both semantic and structural analysis")
            elif semantic_result:
                explanation_parts.append("Found through semantic similarity")
            elif structural_result:
                explanation_parts.append("Found through structural pattern matching")
            
            explanation_parts.append(f"Combined confidence: {combined_score:.1%}")
            
            # Create hybrid result
            hybrid_result = HybridSearchResult(
                semantic_result=semantic_result,
                structural_matches=[structural_result] if structural_result else [],
                combined_score=combined_score,
                rank=i + 1,
                explanation=" - ".join(explanation_parts)
            )
            
            combined_results.append(hybrid_result)
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(combined_results):
            result.rank = i + 1
        
        return combined_results
    
    async def search_with_context(
        self,
        query: SearchQuery,
        context_elements: List[CodeElement]
    ) -> List[HybridSearchResult]:
        """Perform context-aware hybrid search."""
        logger.info(
            "Performing context-aware search",
            query_text=query.query_text,
            context_count=len(context_elements)
        )
        
        # Enhance query with context information
        enhanced_query = await self._enhance_query_with_context(query, context_elements)
        
        # Perform hybrid search with context weighting
        results = await self.hybrid_search(
            enhanced_query,
            semantic_weight=0.5,  # Reduced semantic weight
            structural_weight=0.5  # Increased structural weight for context
        )
        
        # Apply context-based re-ranking
        context_ranked_results = await self._apply_context_ranking(results, context_elements)
        
        return context_ranked_results
    
    async def _enhance_query_with_context(
        self,
        query: SearchQuery,
        context_elements: List[CodeElement]
    ) -> SearchQuery:
        """Enhance search query with context information."""
        # Extract context terms
        context_terms = []
        for element in context_elements:
            context_terms.extend([
                element.name,
                element.element_type.value,
                element.language
            ])
        
        # Enhance query text
        enhanced_text = f"{query.query_text} context: {' '.join(set(context_terms))}"
        
        # Create enhanced query
        enhanced_query = SearchQuery(
            query_text=enhanced_text,
            repository_id=query.repository_id,
            element_types=query.element_types,
            file_patterns=query.file_patterns,
            language_filter=query.language_filter,
            max_results=query.max_results,
            similarity_threshold=query.similarity_threshold,
            include_explanation=query.include_explanation
        )
        
        return enhanced_query
    
    async def _apply_context_ranking(
        self,
        results: List[HybridSearchResult],
        context_elements: List[CodeElement]
    ) -> List[HybridSearchResult]:
        """Apply context-based re-ranking to search results."""
        # Calculate context relevance for each result
        for result in results:
            if result.best_element:
                context_score = self._calculate_context_relevance(
                    result.best_element, context_elements
                )
                
                # Adjust combined score with context
                result.combined_score = (
                    result.combined_score * 0.7 + context_score * 0.3
                )
                
                # Update explanation
                result.explanation += f" (context relevance: {context_score:.1%})"
        
        # Re-sort by adjusted scores
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _calculate_context_relevance(
        self,
        element: CodeElement,
        context_elements: List[CodeElement]
    ) -> float:
        """Calculate how relevant an element is given the context."""
        if not context_elements:
            return 0.5  # Neutral relevance
        
        relevance_factors = []
        
        # Same file relevance
        same_file_count = sum(
            1 for ctx in context_elements 
            if ctx.file_path == element.file_path
        )
        file_relevance = min(same_file_count / len(context_elements), 1.0)
        relevance_factors.append(file_relevance * 0.4)
        
        # Same language relevance
        same_language_count = sum(
            1 for ctx in context_elements 
            if ctx.language == element.language
        )
        language_relevance = same_language_count / len(context_elements)
        relevance_factors.append(language_relevance * 0.3)
        
        # Same element type relevance
        same_type_count = sum(
            1 for ctx in context_elements 
            if ctx.element_type == element.element_type
        )
        type_relevance = same_type_count / len(context_elements)
        relevance_factors.append(type_relevance * 0.3)
        
        return sum(relevance_factors)