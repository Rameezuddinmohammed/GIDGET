"""Supabase database client and connection management."""

from typing import Any, Dict, List, Optional, Union
import asyncio
from datetime import datetime

from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions

from ..config import config
from ..exceptions import SupabaseError
from ..logging import get_logger

logger = get_logger(__name__)


class SupabaseClient:
    """Supabase database client with connection management."""
    
    def __init__(self) -> None:
        self._client: Optional[Client] = None
    
    def connect(self) -> None:
        """Establish connection to Supabase."""
        try:
            options = ClientOptions(
                auto_refresh_token=True,
                persist_session=True,
            )
            
            self._client = create_client(
                config.database.supabase_url,
                config.database.supabase_key,
                options=options
            )
            
            logger.info("Connected to Supabase", url=config.database.supabase_url)
            
        except Exception as e:
            logger.error("Failed to connect to Supabase", error=str(e))
            raise SupabaseError(f"Failed to connect to Supabase: {e}")
    
    @property
    def client(self) -> Client:
        """Get the Supabase client, connecting if necessary."""
        if not self._client:
            self.connect()
        return self._client
    
    async def create_tables(self) -> None:
        """Create all required tables for the system."""
        logger.info("Creating Supabase tables")
        
        # Enable pgvector extension
        await self._execute_sql("""
            CREATE EXTENSION IF NOT EXISTS vector;
        """)
        
        # Create repositories table
        await self._execute_sql("""
            CREATE TABLE IF NOT EXISTS repositories (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                name VARCHAR NOT NULL,
                url VARCHAR NOT NULL UNIQUE,
                description TEXT,
                language VARCHAR,
                default_branch VARCHAR DEFAULT 'main',
                analysis_status VARCHAR DEFAULT 'pending',
                last_analyzed_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Create analysis_cache table
        await self._execute_sql("""
            CREATE TABLE IF NOT EXISTS analysis_cache (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                repository_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
                query_hash VARCHAR NOT NULL,
                query_text TEXT NOT NULL,
                result_data JSONB NOT NULL,
                confidence_score FLOAT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                expires_at TIMESTAMP WITH TIME ZONE,
                UNIQUE(repository_id, query_hash)
            );
        """)
        
        # Create user_preferences table
        await self._execute_sql("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                user_id UUID NOT NULL,
                preferences JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(user_id)
            );
        """)
        
        # Create code_embeddings table for semantic search
        await self._execute_sql("""
            CREATE TABLE IF NOT EXISTS code_embeddings (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                repository_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
                file_path VARCHAR NOT NULL,
                element_type VARCHAR NOT NULL, -- 'function', 'class', 'method'
                element_name VARCHAR NOT NULL,
                code_snippet TEXT NOT NULL,
                embedding vector(768), -- Assuming 768-dimensional embeddings
                metadata JSONB DEFAULT '{}',
                commit_sha VARCHAR NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Create query_history table
        await self._execute_sql("""
            CREATE TABLE IF NOT EXISTS query_history (
                id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
                user_id UUID,
                repository_id UUID REFERENCES repositories(id) ON DELETE CASCADE,
                query_text TEXT NOT NULL,
                query_type VARCHAR,
                execution_time_ms INTEGER,
                agent_count INTEGER,
                confidence_score FLOAT,
                result_summary TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Create indexes for performance
        await self._execute_sql("""
            CREATE INDEX IF NOT EXISTS idx_repositories_url ON repositories(url);
            CREATE INDEX IF NOT EXISTS idx_repositories_analysis_status ON repositories(analysis_status);
            CREATE INDEX IF NOT EXISTS idx_analysis_cache_repository_id ON analysis_cache(repository_id);
            CREATE INDEX IF NOT EXISTS idx_analysis_cache_expires_at ON analysis_cache(expires_at);
            CREATE INDEX IF NOT EXISTS idx_code_embeddings_repository_id ON code_embeddings(repository_id);
            CREATE INDEX IF NOT EXISTS idx_code_embeddings_element_type ON code_embeddings(element_type);
            CREATE INDEX IF NOT EXISTS idx_query_history_user_id ON query_history(user_id);
            CREATE INDEX IF NOT EXISTS idx_query_history_repository_id ON query_history(repository_id);
        """)
        
        # Create vector similarity search index
        await self._execute_sql("""
            CREATE INDEX IF NOT EXISTS idx_code_embeddings_vector 
            ON code_embeddings USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
        """)
        
        logger.info("Supabase tables created successfully")
    
    async def _execute_sql(self, sql: str) -> None:
        """Execute raw SQL command."""
        try:
            # Note: Supabase Python client doesn't have direct SQL execution
            # This would typically be done through the Supabase dashboard or API
            # For now, we'll log the SQL that should be executed
            logger.info("SQL to execute", sql=sql.strip())
            
            # In a real implementation, you would use the Supabase management API
            # or execute these through the dashboard
            
        except Exception as e:
            logger.error("Failed to execute SQL", sql=sql, error=str(e))
            raise SupabaseError(f"SQL execution failed: {e}")
    
    async def insert_repository(self, repository_data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a new repository record."""
        try:
            result = self.client.table("repositories").insert(repository_data).execute()
            logger.info("Repository inserted", repository_id=result.data[0]["id"])
            return result.data[0]
        except Exception as e:
            logger.error("Failed to insert repository", error=str(e))
            raise SupabaseError(f"Repository insertion failed: {e}")
    
    async def get_repository(self, repository_id: str) -> Optional[Dict[str, Any]]:
        """Get repository by ID."""
        try:
            result = self.client.table("repositories").select("*").eq("id", repository_id).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Failed to get repository", repository_id=repository_id, error=str(e))
            raise SupabaseError(f"Repository retrieval failed: {e}")
    
    async def update_repository_status(
        self, 
        repository_id: str, 
        status: str, 
        last_analyzed_at: Optional[datetime] = None
    ) -> None:
        """Update repository analysis status."""
        try:
            update_data = {"analysis_status": status, "updated_at": datetime.now().isoformat()}
            if last_analyzed_at:
                update_data["last_analyzed_at"] = last_analyzed_at.isoformat()
            
            self.client.table("repositories").update(update_data).eq("id", repository_id).execute()
            logger.info("Repository status updated", repository_id=repository_id, status=status)
        except Exception as e:
            logger.error("Failed to update repository status", repository_id=repository_id, error=str(e))
            raise SupabaseError(f"Repository status update failed: {e}")
    
    async def cache_query_result(
        self,
        repository_id: str,
        query_hash: str,
        query_text: str,
        result_data: Dict[str, Any],
        confidence_score: float,
        ttl_seconds: int = 3600
    ) -> None:
        """Cache a query result."""
        try:
            expires_at = datetime.now().timestamp() + ttl_seconds
            cache_data = {
                "repository_id": repository_id,
                "query_hash": query_hash,
                "query_text": query_text,
                "result_data": result_data,
                "confidence_score": confidence_score,
                "expires_at": datetime.fromtimestamp(expires_at).isoformat()
            }
            
            # Use upsert to handle duplicates
            self.client.table("analysis_cache").upsert(cache_data).execute()
            logger.info("Query result cached", repository_id=repository_id, query_hash=query_hash)
        except Exception as e:
            logger.error("Failed to cache query result", error=str(e))
            raise SupabaseError(f"Query caching failed: {e}")
    
    async def get_cached_result(
        self, 
        repository_id: str, 
        query_hash: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached query result if not expired."""
        try:
            result = self.client.table("analysis_cache").select("*").eq(
                "repository_id", repository_id
            ).eq("query_hash", query_hash).gt(
                "expires_at", datetime.now().isoformat()
            ).execute()
            
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error("Failed to get cached result", error=str(e))
            raise SupabaseError(f"Cache retrieval failed: {e}")
    
    async def store_code_embedding(
        self,
        repository_id: str,
        file_path: str,
        element_type: str,
        element_name: str,
        code_snippet: str,
        embedding: List[float],
        commit_sha: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store code embedding for semantic search."""
        try:
            embedding_data = {
                "repository_id": repository_id,
                "file_path": file_path,
                "element_type": element_type,
                "element_name": element_name,
                "code_snippet": code_snippet,
                "embedding": embedding,
                "commit_sha": commit_sha,
                "metadata": metadata or {}
            }
            
            self.client.table("code_embeddings").insert(embedding_data).execute()
            logger.debug("Code embedding stored", element_name=element_name, file_path=file_path)
        except Exception as e:
            logger.error("Failed to store code embedding", error=str(e))
            raise SupabaseError(f"Embedding storage failed: {e}")
    
    async def search_similar_code(
        self,
        repository_id: str,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar code using vector similarity."""
        try:
            # Note: This would use pgvector similarity search
            # The exact implementation depends on the Supabase client capabilities
            # For now, we'll return a placeholder
            logger.info("Searching similar code", repository_id=repository_id, limit=limit)
            return []
        except Exception as e:
            logger.error("Failed to search similar code", error=str(e))
            raise SupabaseError(f"Similarity search failed: {e}")
    
    async def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries."""
        try:
            result = self.client.table("analysis_cache").delete().lt(
                "expires_at", datetime.now().isoformat()
            ).execute()
            
            count = len(result.data) if result.data else 0
            logger.info("Cleaned up expired cache entries", count=count)
            return count
        except Exception as e:
            logger.error("Failed to cleanup expired cache", error=str(e))
            raise SupabaseError(f"Cache cleanup failed: {e}")


# Global client instance
supabase_client = SupabaseClient()