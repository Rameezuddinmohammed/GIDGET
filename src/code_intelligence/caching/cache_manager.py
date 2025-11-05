"""Intelligent caching system for query results and analysis data."""

import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import asyncio

from ..database.supabase_client import supabase_client
from ..logging import get_logger
from ..monitoring.agent_monitor import agent_monitor

logger = get_logger(__name__)


class CacheManager:
    """Manages intelligent caching of query results and analysis data."""
    
    def __init__(self):
        """Initialize the cache manager."""
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "stores": 0,
            "invalidations": 0,
            "errors": 0
        }
        self._lock = asyncio.Lock()
        
    def _generate_query_hash(
        self, 
        query: str, 
        repository_id: str, 
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a consistent hash for a query."""
        # Normalize the query for consistent hashing
        normalized_query = query.strip().lower()
        
        # Include relevant options in the hash
        cache_key_data = {
            "query": normalized_query,
            "repository_id": repository_id,
            "options": options or {}
        }
        
        # Create hash
        cache_key_str = json.dumps(cache_key_data, sort_keys=True)
        return hashlib.sha256(cache_key_str.encode()).hexdigest()[:16]
        
    async def get_cached_result(
        self,
        query: str,
        repository_id: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached result if available and not expired."""
        start_time = datetime.now()
        
        try:
            query_hash = self._generate_query_hash(query, repository_id, options)
            
            # Check cache
            cached_result = await supabase_client.get_cached_result(
                repository_id, query_hash
            )
            
            async with self._lock:
                if cached_result:
                    self.cache_stats["hits"] += 1
                    logger.info(
                        "Cache hit",
                        query_hash=query_hash,
                        repository_id=repository_id,
                        confidence=cached_result.get("confidence_score", 0.0)
                    )
                else:
                    self.cache_stats["misses"] += 1
                    logger.debug(
                        "Cache miss",
                        query_hash=query_hash,
                        repository_id=repository_id
                    )
                    
            return cached_result
            
        except Exception as e:
            async with self._lock:
                self.cache_stats["errors"] += 1
            logger.error("Cache lookup failed", error=str(e))
            return None
        finally:
            # Record cache lookup performance
            lookup_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.debug(f"Cache lookup took {lookup_time:.2f}ms")
            
    async def store_result(
        self,
        query: str,
        repository_id: str,
        result_data: Dict[str, Any],
        confidence_score: float,
        options: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Store query result in cache."""
        try:
            query_hash = self._generate_query_hash(query, repository_id, options)
            
            # Determine TTL based on confidence and query type
            if ttl_seconds is None:
                ttl_seconds = self._calculate_ttl(confidence_score, query, result_data)
                
            # Store in cache
            await supabase_client.cache_query_result(
                repository_id=repository_id,
                query_hash=query_hash,
                query_text=query,
                result_data=result_data,
                confidence_score=confidence_score,
                ttl_seconds=ttl_seconds
            )
            
            async with self._lock:
                self.cache_stats["stores"] += 1
                
            logger.info(
                "Result cached",
                query_hash=query_hash,
                repository_id=repository_id,
                confidence=confidence_score,
                ttl_seconds=ttl_seconds
            )
            
            return True
            
        except Exception as e:
            async with self._lock:
                self.cache_stats["errors"] += 1
            logger.error("Cache store failed", error=str(e))
            return False
            
    def _calculate_ttl(
        self, 
        confidence_score: float, 
        query: str, 
        result_data: Dict[str, Any]
    ) -> int:
        """Calculate appropriate TTL based on result characteristics."""
        base_ttl = 3600  # 1 hour default
        
        # Higher confidence results can be cached longer
        confidence_multiplier = min(2.0, max(0.5, confidence_score * 2))
        
        # Historical queries can be cached longer (less likely to change)
        if any(word in query.lower() for word in ["history", "when", "before", "after", "commit"]):
            base_ttl *= 2
            
        # Code structure queries can be cached longer
        if any(word in query.lower() for word in ["function", "class", "method", "structure"]):
            base_ttl *= 1.5
            
        # Performance-related queries should have shorter TTL
        if any(word in query.lower() for word in ["performance", "slow", "deadlock", "issue"]):
            base_ttl *= 0.5
            
        # Adjust based on result complexity
        findings_count = len(result_data.get("findings", []))
        if findings_count > 10:
            base_ttl *= 1.2  # Complex results are expensive to regenerate
            
        return int(base_ttl * confidence_multiplier)
        
    async def invalidate_repository_cache(self, repository_id: str) -> int:
        """Invalidate all cached results for a repository."""
        try:
            count = await supabase_client.invalidate_repository_cache(repository_id)
            
            async with self._lock:
                self.cache_stats["invalidations"] += count
                
            logger.info("Invalidated repository cache", repository_id=repository_id, count=count)
            return count
            
        except Exception as e:
            async with self._lock:
                self.cache_stats["errors"] += 1
            logger.error("Cache invalidation failed", error=str(e))
            return 0
            
    async def invalidate_pattern_cache(self, pattern: str) -> int:
        """Invalidate cached results matching a pattern."""
        try:
            count = await supabase_client.invalidate_cache_by_pattern(pattern)
            
            async with self._lock:
                self.cache_stats["invalidations"] += count
                
            logger.info("Invalidated pattern cache", pattern=pattern, count=count)
            return count
            
        except Exception as e:
            async with self._lock:
                self.cache_stats["errors"] += 1
            logger.error("Pattern cache invalidation failed", error=str(e))
            return 0
            
    async def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries."""
        try:
            count = await supabase_client.cleanup_expired_cache()
            logger.info("Cleaned up expired cache entries", count=count)
            return count
            
        except Exception as e:
            logger.error("Cache cleanup failed", error=str(e))
            return 0
            
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        async with self._lock:
            total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
            hit_rate = (
                self.cache_stats["hits"] / total_requests 
                if total_requests > 0 else 0.0
            )
            
            return {
                "hits": self.cache_stats["hits"],
                "misses": self.cache_stats["misses"],
                "stores": self.cache_stats["stores"],
                "invalidations": self.cache_stats["invalidations"],
                "errors": self.cache_stats["errors"],
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }
            
    async def should_cache_result(
        self, 
        confidence_score: float, 
        result_data: Dict[str, Any]
    ) -> bool:
        """Determine if a result should be cached."""
        # Don't cache low-confidence results
        if confidence_score < 0.7:
            return False
            
        # Don't cache empty results
        if not result_data.get("findings"):
            return False
            
        # Don't cache error results
        if result_data.get("errors"):
            return False
            
        return True
        
    async def warm_cache(
        self, 
        repository_id: str, 
        common_queries: List[str]
    ) -> int:
        """Pre-warm cache with common queries."""
        warmed_count = 0
        
        for query in common_queries:
            try:
                # Check if already cached
                cached = await self.get_cached_result(query, repository_id)
                if not cached:
                    # This would trigger analysis to warm the cache
                    logger.info("Would warm cache for query", query=query[:50])
                    warmed_count += 1
                    
            except Exception as e:
                logger.error("Cache warming failed for query", query=query[:50], error=str(e))
                
        logger.info("Cache warming completed", warmed_count=warmed_count)
        return warmed_count


# Global cache manager instance
cache_manager = CacheManager()