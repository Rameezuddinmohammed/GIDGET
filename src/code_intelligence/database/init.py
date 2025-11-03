"""Database initialization utilities."""

import asyncio
from typing import Optional

from .neo4j_client import neo4j_client
from .supabase_client import supabase_client
from .schema import cpg_schema
from .migrations import migration_manager
from ..logging import get_logger
from ..exceptions import DatabaseError

logger = get_logger(__name__)


class DatabaseInitializer:
    """Handles initialization of all database components."""
    
    async def initialize_all(self, reset: bool = False) -> None:
        """Initialize both Neo4j and Supabase databases."""
        logger.info("Initializing database infrastructure", reset=reset)
        
        try:
            # Initialize Neo4j
            await self.initialize_neo4j(reset=reset)
            
            # Initialize Supabase
            await self.initialize_supabase()
            
            logger.info("Database initialization completed successfully")
            
        except Exception as e:
            logger.error("Database initialization failed", error=str(e))
            raise DatabaseError(f"Database initialization failed: {e}")
    
    async def initialize_neo4j(self, reset: bool = False) -> None:
        """Initialize Neo4j database with schema and migrations."""
        logger.info("Initializing Neo4j database", reset=reset)
        
        try:
            # Connect to Neo4j
            await neo4j_client.connect()
            
            # Reset schema if requested
            if reset:
                logger.warning("Resetting Neo4j schema")
                await cpg_schema.drop_schema()
                await self._reset_migrations()
            
            # Run migrations
            await migration_manager.migrate_up()
            
            # Verify schema
            schema_info = await cpg_schema.get_schema_info()
            logger.info("Neo4j schema verified", 
                       constraints=schema_info["constraint_count"],
                       indexes=schema_info["index_count"])
            
        except Exception as e:
            logger.error("Neo4j initialization failed", error=str(e))
            raise DatabaseError(f"Neo4j initialization failed: {e}")
    
    async def initialize_supabase(self) -> None:
        """Initialize Supabase database with tables."""
        logger.info("Initializing Supabase database")
        
        try:
            # Connect to Supabase
            supabase_client.connect()
            
            # Create tables (note: in practice, this would be done via SQL editor)
            await supabase_client.create_tables()
            
            logger.info("Supabase initialization completed")
            
        except Exception as e:
            logger.error("Supabase initialization failed", error=str(e))
            raise DatabaseError(f"Supabase initialization failed: {e}")
    
    async def _reset_migrations(self) -> None:
        """Reset all migrations (for testing/development)."""
        try:
            await neo4j_client.execute_query("MATCH (m:Migration) DELETE m")
            logger.info("Migration history reset")
        except Exception as e:
            logger.warning("Failed to reset migration history", error=str(e))
    
    async def health_check(self) -> dict:
        """Perform health check on all database connections."""
        health_status = {
            "neo4j": {"status": "unknown", "details": {}},
            "supabase": {"status": "unknown", "details": {}}
        }
        
        # Check Neo4j
        try:
            await neo4j_client.connect()
            result = await neo4j_client.execute_query("RETURN 1 as test")
            schema_info = await cpg_schema.get_schema_info()
            migration_status = await migration_manager.get_migration_status()
            
            health_status["neo4j"] = {
                "status": "healthy",
                "details": {
                    "connection": "ok",
                    "constraints": schema_info["constraint_count"],
                    "indexes": schema_info["index_count"],
                    "migrations_applied": migration_status["applied_migrations"],
                    "migrations_pending": migration_status["pending_migrations"]
                }
            }
        except Exception as e:
            health_status["neo4j"] = {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
        
        # Check Supabase
        try:
            supabase_client.connect()
            # Simple test query (would need to be adapted based on actual client capabilities)
            health_status["supabase"] = {
                "status": "healthy",
                "details": {
                    "connection": "ok",
                    "url": supabase_client.client.supabase_url
                }
            }
        except Exception as e:
            health_status["supabase"] = {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
        
        return health_status
    
    def initialize_sync(self, reset: bool = False) -> None:
        """Synchronous version of database initialization."""
        asyncio.run(self.initialize_all(reset=reset))


# Global initializer instance
db_initializer = DatabaseInitializer()