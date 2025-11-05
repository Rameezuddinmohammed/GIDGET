"""Cache invalidation service for handling repository updates and changes."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

from ..logging import get_logger
from ..database.supabase_client import supabase_client
from .cache_manager import cache_manager

logger = get_logger(__name__)


@dataclass
class InvalidationEvent:
    """Represents a cache invalidation event."""
    event_type: str  # 'repository_update', 'commit_ingested', 'manual'
    repository_id: str
    details: Dict[str, Any]
    timestamp: datetime


class CacheInvalidationService:
    """Service for managing intelligent cache invalidation."""
    
    def __init__(self):
        """Initialize the cache invalidation service."""
        self.pending_invalidations: List[InvalidationEvent] = []
        self.invalidation_rules: Dict[str, List[str]] = {
            # Patterns that should be invalidated for different event types
            "commit_ingested": [
                "history", "when", "commit", "change", "evolution"
            ],
            "file_modified": [
                "function", "class", "method", "structure"
            ],
            "dependency_changed": [
                "import", "dependency", "call", "usage"
            ]
        }
        self._lock = asyncio.Lock()
        
    async def handle_repository_update(
        self, 
        repository_id: str, 
        update_type: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Handle repository update events that may require cache invalidation."""
        event = InvalidationEvent(
            event_type=update_type,
            repository_id=repository_id,
            details=details or {},
            timestamp=datetime.now()
        )
        
        async with self._lock:
            self.pending_invalidations.append(event)
            
        logger.info(
            "Repository update event queued for invalidation",
            repository_id=repository_id,
            update_type=update_type
        )
        
        # Process invalidation immediately for critical updates
        if update_type in ["commit_ingested", "major_change"]:
            await self._process_invalidation_event(event)
            
    async def handle_commit_ingestion(
        self, 
        repository_id: str, 
        commit_sha: str,
        modified_files: List[str],
        commit_message: str
    ) -> None:
        """Handle new commit ingestion that requires cache invalidation."""
        details = {
            "commit_sha": commit_sha,
            "modified_files": modified_files,
            "commit_message": commit_message,
            "file_count": len(modified_files)
        }
        
        await self.handle_repository_update(
            repository_id=repository_id,
            update_type="commit_ingested",
            details=details
        )
        
    async def _process_invalidation_event(self, event: InvalidationEvent) -> None:
        """Process a single invalidation event."""
        try:
            invalidation_count = 0
            
            if event.event_type == "commit_ingested":
                # Invalidate history-related queries
                for pattern in self.invalidation_rules["commit_ingested"]:
                    count = await cache_manager.invalidate_pattern_cache(pattern)
                    invalidation_count += count
                    
                # Invalidate queries related to modified files
                modified_files = event.details.get("modified_files", [])
                for file_path in modified_files:
                    # Extract file name for pattern matching
                    file_name = file_path.split("/")[-1].split(".")[0]
                    count = await cache_manager.invalidate_pattern_cache(file_name)
                    invalidation_count += count
                    
            elif event.event_type == "major_change":
                # Invalidate all cache for the repository
                invalidation_count = await cache_manager.invalidate_repository_cache(
                    event.repository_id
                )
                
            elif event.event_type == "file_modified":
                # Invalidate structure-related queries
                for pattern in self.invalidation_rules["file_modified"]:
                    count = await cache_manager.invalidate_pattern_cache(pattern)
                    invalidation_count += count
                    
            logger.info(
                "Processed invalidation event",
                event_type=event.event_type,
                repository_id=event.repository_id,
                invalidated_count=invalidation_count
            )
            
        except Exception as e:
            logger.error(
                "Failed to process invalidation event",
                event_type=event.event_type,
                repository_id=event.repository_id,
                error=str(e)
            )
            
    async def process_pending_invalidations(self) -> int:
        """Process all pending invalidation events."""
        processed_count = 0
        
        async with self._lock:
            pending = self.pending_invalidations.copy()
            self.pending_invalidations.clear()
            
        for event in pending:
            try:
                await self._process_invalidation_event(event)
                processed_count += 1
            except Exception as e:
                logger.error(
                    "Failed to process pending invalidation",
                    event_type=event.event_type,
                    error=str(e)
                )
                
        if processed_count > 0:
            logger.info(f"Processed {processed_count} pending invalidation events")
            
        return processed_count
        
    async def schedule_periodic_cleanup(self, interval_hours: int = 6) -> None:
        """Schedule periodic cache cleanup and invalidation processing."""
        while True:
            try:
                # Process pending invalidations
                await self.process_pending_invalidations()
                
                # Clean up expired cache entries
                expired_count = await cache_manager.cleanup_expired_cache()
                
                # Log cleanup results
                if expired_count > 0:
                    logger.info(f"Periodic cleanup removed {expired_count} expired cache entries")
                    
                # Wait for next cleanup cycle
                await asyncio.sleep(interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Periodic cleanup failed: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
                
    async def invalidate_by_query_similarity(
        self, 
        reference_query: str, 
        similarity_threshold: float = 0.8
    ) -> int:
        """Invalidate cache entries similar to a reference query."""
        try:
            # This would use semantic similarity to find related queries
            # For now, we'll use simple keyword matching
            keywords = reference_query.lower().split()
            invalidation_count = 0
            
            for keyword in keywords:
                if len(keyword) > 3:  # Skip short words
                    count = await cache_manager.invalidate_pattern_cache(keyword)
                    invalidation_count += count
                    
            logger.info(
                "Invalidated similar queries",
                reference_query=reference_query[:50],
                count=invalidation_count
            )
            
            return invalidation_count
            
        except Exception as e:
            logger.error(
                "Failed to invalidate similar queries",
                reference_query=reference_query[:50],
                error=str(e)
            )
            return 0
            
    async def get_invalidation_stats(self) -> Dict[str, Any]:
        """Get statistics about cache invalidation activities."""
        async with self._lock:
            pending_count = len(self.pending_invalidations)
            
        cache_stats = await cache_manager.get_cache_stats()
        
        return {
            "pending_invalidations": pending_count,
            "total_invalidations": cache_stats["invalidations"],
            "invalidation_errors": cache_stats["errors"],
            "cache_hit_rate": cache_stats["hit_rate"],
            "last_updated": datetime.now().isoformat()
        }


# Global invalidation service instance
invalidation_service = CacheInvalidationService()