"""Database migration management for Neo4j."""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .neo4j_client import neo4j_client
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class Migration:
    """Represents a database migration."""
    version: str
    name: str
    description: str
    up_queries: List[str]
    down_queries: List[str]
    created_at: datetime


class MigrationManager:
    """Manages database migrations for Neo4j."""
    
    def __init__(self) -> None:
        self.migrations: List[Migration] = []
        self._initialize_migrations()
    
    def _initialize_migrations(self) -> None:
        """Initialize the list of available migrations."""
        
        # Migration 001: Create migration tracking
        self.migrations.append(Migration(
            version="001",
            name="create_migration_tracking",
            description="Create migration tracking infrastructure",
            up_queries=[
                """
                CREATE CONSTRAINT migration_version_unique IF NOT EXISTS 
                FOR (m:Migration) REQUIRE m.version IS UNIQUE
                """,
                """
                CREATE INDEX migration_applied_at_index IF NOT EXISTS 
                FOR (m:Migration) ON (m.applied_at)
                """
            ],
            down_queries=[
                "DROP INDEX migration_applied_at_index IF EXISTS",
                "DROP CONSTRAINT migration_version_unique IF EXISTS",
                "MATCH (m:Migration) DELETE m"
            ],
            created_at=datetime(2024, 1, 1, 0, 0, 0)
        ))
        
        # Migration 002: Create Code Property Graph base schema
        self.migrations.append(Migration(
            version="002",
            name="create_cpg_base_schema",
            description="Create base Code Property Graph node labels and relationships",
            up_queries=[
                # Repository nodes
                """
                CREATE CONSTRAINT unique_repository_name IF NOT EXISTS 
                FOR (r:Repository) REQUIRE r.name IS UNIQUE
                """,
                """
                CREATE INDEX repository_url_index IF NOT EXISTS 
                FOR (r:Repository) ON (r.url)
                """,
                
                # Commit nodes
                """
                CREATE CONSTRAINT unique_commit_sha IF NOT EXISTS 
                FOR (c:Commit) REQUIRE (c.repository_id, c.sha) IS UNIQUE
                """,
                """
                CREATE INDEX commit_timestamp_index IF NOT EXISTS 
                FOR (c:Commit) ON (c.timestamp)
                """,
                
                # File nodes
                """
                CREATE CONSTRAINT unique_file_path IF NOT EXISTS 
                FOR (f:File) REQUIRE (f.repository_id, f.commit_sha, f.path) IS UNIQUE
                """,
                """
                CREATE INDEX file_language_index IF NOT EXISTS 
                FOR (f:File) ON (f.language)
                """
            ],
            down_queries=[
                "DROP INDEX file_language_index IF EXISTS",
                "DROP CONSTRAINT unique_file_path IF EXISTS",
                "DROP INDEX commit_timestamp_index IF EXISTS",
                "DROP CONSTRAINT unique_commit_sha IF EXISTS",
                "DROP INDEX repository_url_index IF EXISTS",
                "DROP CONSTRAINT unique_repository_name IF EXISTS"
            ],
            created_at=datetime(2024, 1, 2, 0, 0, 0)
        ))
        
        # Migration 003: Create code element schema
        self.migrations.append(Migration(
            version="003",
            name="create_code_element_schema",
            description="Create schema for functions, classes, and code elements",
            up_queries=[
                # Function nodes
                """
                CREATE CONSTRAINT function_signature_exists IF NOT EXISTS 
                FOR (f:Function) REQUIRE f.signature_hash IS NOT NULL
                """,
                """
                CREATE INDEX function_name_index IF NOT EXISTS 
                FOR (f:Function) ON (f.name)
                """,
                
                # Class nodes
                """
                CREATE CONSTRAINT class_name_exists IF NOT EXISTS 
                FOR (c:Class) REQUIRE c.name IS NOT NULL
                """,
                """
                CREATE INDEX class_name_index IF NOT EXISTS 
                FOR (c:Class) ON (c.name)
                """,
                
                # Relationship indexes
                """
                CREATE INDEX changed_in_relationship_index IF NOT EXISTS 
                FOR ()-[r:CHANGED_IN]-() ON (r.change_type)
                """
            ],
            down_queries=[
                "DROP INDEX changed_in_relationship_index IF EXISTS",
                "DROP INDEX class_name_index IF EXISTS",
                "DROP CONSTRAINT class_name_exists IF EXISTS",
                "DROP INDEX function_name_index IF EXISTS",
                "DROP CONSTRAINT function_signature_exists IF EXISTS"
            ],
            created_at=datetime(2024, 1, 3, 0, 0, 0)
        ))
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        try:
            result = await neo4j_client.execute_query(
                "MATCH (m:Migration) RETURN m.version as version ORDER BY m.version"
            )
            return [record["version"] for record in result]
        except Exception:
            # Migration table doesn't exist yet
            return []
    
    async def apply_migration(self, migration: Migration) -> None:
        """Apply a single migration."""
        logger.info("Applying migration", version=migration.version, name=migration.name)
        
        try:
            # Execute up queries
            for query in migration.up_queries:
                await neo4j_client.execute_query(query.strip())
            
            # Record migration as applied
            await neo4j_client.execute_write_query(
                """
                CREATE (m:Migration {
                    version: $version,
                    name: $name,
                    description: $description,
                    applied_at: datetime()
                })
                """,
                {
                    "version": migration.version,
                    "name": migration.name,
                    "description": migration.description
                }
            )
            
            logger.info("Migration applied successfully", version=migration.version)
            
        except Exception as e:
            logger.error("Migration failed", version=migration.version, error=str(e))
            raise
    
    async def rollback_migration(self, migration: Migration) -> None:
        """Rollback a single migration."""
        logger.info("Rolling back migration", version=migration.version, name=migration.name)
        
        try:
            # Execute down queries
            for query in migration.down_queries:
                await neo4j_client.execute_query(query.strip())
            
            # Remove migration record
            await neo4j_client.execute_write_query(
                "MATCH (m:Migration {version: $version}) DELETE m",
                {"version": migration.version}
            )
            
            logger.info("Migration rolled back successfully", version=migration.version)
            
        except Exception as e:
            logger.error("Migration rollback failed", version=migration.version, error=str(e))
            raise
    
    async def migrate_up(self, target_version: Optional[str] = None) -> None:
        """Apply all pending migrations up to target version."""
        applied_versions = await self.get_applied_migrations()
        
        for migration in self.migrations:
            if migration.version in applied_versions:
                continue
            
            if target_version and migration.version > target_version:
                break
            
            await self.apply_migration(migration)
    
    async def migrate_down(self, target_version: str) -> None:
        """Rollback migrations down to target version."""
        applied_versions = await self.get_applied_migrations()
        
        # Rollback in reverse order
        for migration in reversed(self.migrations):
            if migration.version not in applied_versions:
                continue
            
            if migration.version <= target_version:
                break
            
            await self.rollback_migration(migration)
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status."""
        applied_versions = await self.get_applied_migrations()
        
        status = {
            "total_migrations": len(self.migrations),
            "applied_migrations": len(applied_versions),
            "pending_migrations": len(self.migrations) - len(applied_versions),
            "migrations": []
        }
        
        for migration in self.migrations:
            status["migrations"].append({
                "version": migration.version,
                "name": migration.name,
                "description": migration.description,
                "applied": migration.version in applied_versions
            })
        
        return status


# Global migration manager instance
migration_manager = MigrationManager()