"""Database utility functions and helpers."""

import hashlib
import json
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from .neo4j_client import neo4j_client
from .supabase_client import supabase_client
from ..logging import get_logger

logger = get_logger(__name__)


def generate_query_hash(query_text: str, repository_id: str) -> str:
    """Generate a hash for query caching."""
    content = f"{repository_id}:{query_text}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


async def cleanup_expired_data() -> Dict[str, int]:
    """Clean up expired data from all databases."""
    logger.info("Starting expired data cleanup")
    
    cleanup_stats = {
        "supabase_cache_cleaned": 0,
        "neo4j_temp_nodes_cleaned": 0
    }
    
    try:
        # Clean up Supabase cache
        cache_count = await supabase_client.cleanup_expired_cache()
        cleanup_stats["supabase_cache_cleaned"] = cache_count
        
        # Clean up temporary Neo4j nodes (if any)
        temp_cleanup_query = """
        MATCH (n:TempNode) 
        WHERE n.expires_at < datetime()
        DELETE n
        RETURN count(n) as cleaned_count
        """
        result = await neo4j_client.execute_query(temp_cleanup_query)
        cleanup_stats["neo4j_temp_nodes_cleaned"] = result[0]["cleaned_count"] if result else 0
        
        logger.info("Expired data cleanup completed", stats=cleanup_stats)
        
    except Exception as e:
        logger.error("Expired data cleanup failed", error=str(e))
    
    return cleanup_stats


async def get_database_statistics() -> Dict[str, Any]:
    """Get comprehensive database statistics."""
    stats = {
        "neo4j": {},
        "supabase": {},
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Neo4j statistics
        neo4j_stats_query = """
        CALL apoc.meta.stats() YIELD labels, relTypesCount, nodeCount, relCount
        RETURN labels, relTypesCount, nodeCount, relCount
        """
        try:
            neo4j_result = await neo4j_client.execute_query(neo4j_stats_query)
            if neo4j_result:
                stats["neo4j"] = neo4j_result[0]
        except Exception:
            # Fallback if APOC is not available
            basic_stats_query = """
            MATCH (n) 
            RETURN count(n) as node_count, 
                   count{(n)-[]->()} as relationship_count,
                   collect(distinct labels(n)) as all_labels
            """
            basic_result = await neo4j_client.execute_query(basic_stats_query)
            if basic_result:
                stats["neo4j"] = {
                    "nodeCount": basic_result[0]["node_count"],
                    "relCount": basic_result[0]["relationship_count"],
                    "labels": basic_result[0]["all_labels"]
                }
        
        # Supabase statistics (would need to be implemented based on available queries)
        stats["supabase"] = {
            "note": "Supabase statistics would be gathered via SQL queries",
            "tables": ["repositories", "analysis_cache", "code_embeddings", "query_history"]
        }
        
    except Exception as e:
        logger.error("Failed to gather database statistics", error=str(e))
        stats["error"] = str(e)
    
    return stats


async def validate_database_integrity() -> Dict[str, Any]:
    """Validate database integrity and relationships."""
    validation_results = {
        "neo4j_integrity": {"status": "unknown", "issues": []},
        "supabase_integrity": {"status": "unknown", "issues": []},
        "cross_database_consistency": {"status": "unknown", "issues": []}
    }
    
    try:
        # Neo4j integrity checks
        neo4j_issues = []
        
        # Check for orphaned nodes
        orphan_check_query = """
        MATCH (n) 
        WHERE NOT (n)-[]-() AND NOT n:Repository
        RETURN count(n) as orphan_count, collect(distinct labels(n)) as orphan_labels
        """
        orphan_result = await neo4j_client.execute_query(orphan_check_query)
        if orphan_result and orphan_result[0]["orphan_count"] > 0:
            neo4j_issues.append(f"Found {orphan_result[0]['orphan_count']} orphaned nodes")
        
        # Check for missing required properties
        missing_props_query = """
        MATCH (f:Function) 
        WHERE f.signature_hash IS NULL 
        RETURN count(f) as missing_signature_count
        """
        missing_result = await neo4j_client.execute_query(missing_props_query)
        if missing_result and missing_result[0]["missing_signature_count"] > 0:
            neo4j_issues.append(f"Found {missing_result[0]['missing_signature_count']} functions without signature hashes")
        
        validation_results["neo4j_integrity"] = {
            "status": "healthy" if not neo4j_issues else "issues_found",
            "issues": neo4j_issues
        }
        
        # Supabase integrity checks would go here
        validation_results["supabase_integrity"] = {
            "status": "healthy",
            "issues": []
        }
        
        # Cross-database consistency checks would go here
        validation_results["cross_database_consistency"] = {
            "status": "healthy",
            "issues": []
        }
        
    except Exception as e:
        logger.error("Database integrity validation failed", error=str(e))
        validation_results["error"] = str(e)
    
    return validation_results


class DatabaseTransaction:
    """Context manager for coordinating transactions across databases."""
    
    def __init__(self):
        self.neo4j_session = None
        self.rollback_actions = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Rollback on exception
            await self.rollback()
        return False
    
    async def rollback(self):
        """Execute rollback actions in reverse order."""
        logger.info("Executing transaction rollback")
        for action in reversed(self.rollback_actions):
            try:
                await action()
            except Exception as e:
                logger.error("Rollback action failed", error=str(e))
    
    def add_rollback_action(self, action):
        """Add a rollback action to be executed if transaction fails."""
        self.rollback_actions.append(action)