"""Neo4j database schema management for Code Property Graph."""

from typing import List, Dict, Any
from dataclasses import dataclass

from .neo4j_client import neo4j_client
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class SchemaConstraint:
    """Represents a database constraint."""
    name: str
    query: str
    description: str


@dataclass
class SchemaIndex:
    """Represents a database index."""
    name: str
    query: str
    description: str


class CodePropertyGraphSchema:
    """Manages the Code Property Graph schema in Neo4j."""
    
    # Node label constraints
    CONSTRAINTS = [
        SchemaConstraint(
            name="unique_repository_name",
            query="CREATE CONSTRAINT unique_repository_name IF NOT EXISTS FOR (r:Repository) REQUIRE r.name IS UNIQUE",
            description="Ensure repository names are unique"
        ),
        SchemaConstraint(
            name="unique_commit_sha",
            query="CREATE CONSTRAINT unique_commit_sha IF NOT EXISTS FOR (c:Commit) REQUIRE (c.repository_id, c.sha) IS UNIQUE",
            description="Ensure commit SHAs are unique within repository"
        ),
        SchemaConstraint(
            name="unique_file_path",
            query="CREATE CONSTRAINT unique_file_path IF NOT EXISTS FOR (f:File) REQUIRE (f.repository_id, f.commit_sha, f.path) IS UNIQUE",
            description="Ensure file paths are unique within commit"
        ),
        SchemaConstraint(
            name="function_signature_exists",
            query="CREATE CONSTRAINT function_signature_exists IF NOT EXISTS FOR (f:Function) REQUIRE f.signature_hash IS NOT NULL",
            description="Ensure functions have signature hashes"
        ),
        SchemaConstraint(
            name="class_name_exists",
            query="CREATE CONSTRAINT class_name_exists IF NOT EXISTS FOR (c:Class) REQUIRE c.name IS NOT NULL",
            description="Ensure classes have names"
        ),
    ]
    
    # Performance indexes
    INDEXES = [
        SchemaIndex(
            name="repository_url_index",
            query="CREATE INDEX repository_url_index IF NOT EXISTS FOR (r:Repository) ON (r.url)",
            description="Index on repository URL for fast lookups"
        ),
        SchemaIndex(
            name="commit_timestamp_index",
            query="CREATE INDEX commit_timestamp_index IF NOT EXISTS FOR (c:Commit) ON (c.timestamp)",
            description="Index on commit timestamp for temporal queries"
        ),
        SchemaIndex(
            name="file_language_index",
            query="CREATE INDEX file_language_index IF NOT EXISTS FOR (f:File) ON (f.language)",
            description="Index on file language for filtering"
        ),
        SchemaIndex(
            name="function_name_index",
            query="CREATE INDEX function_name_index IF NOT EXISTS FOR (f:Function) ON (f.name)",
            description="Index on function name for search"
        ),
        SchemaIndex(
            name="class_name_index",
            query="CREATE INDEX class_name_index IF NOT EXISTS FOR (c:Class) ON (c.name)",
            description="Index on class name for search"
        ),
        SchemaIndex(
            name="changed_in_relationship_index",
            query="CREATE INDEX changed_in_relationship_index IF NOT EXISTS FOR ()-[r:CHANGED_IN]-() ON (r.change_type)",
            description="Index on change type for evolution queries"
        ),
    ]
    
    async def create_schema(self) -> None:
        """Create the complete Code Property Graph schema."""
        logger.info("Creating Code Property Graph schema")
        
        # Create constraints
        for constraint in self.CONSTRAINTS:
            try:
                await neo4j_client.execute_query(constraint.query)
                logger.info("Created constraint", name=constraint.name)
            except Exception as e:
                logger.warning("Failed to create constraint", name=constraint.name, error=str(e))
        
        # Create indexes
        for index in self.INDEXES:
            try:
                await neo4j_client.execute_query(index.query)
                logger.info("Created index", name=index.name)
            except Exception as e:
                logger.warning("Failed to create index", name=index.name, error=str(e))
        
        logger.info("Code Property Graph schema creation completed")
    
    def create_schema_sync(self) -> None:
        """Create the complete Code Property Graph schema synchronously."""
        logger.info("Creating Code Property Graph schema (sync)")
        
        # Create constraints
        for constraint in self.CONSTRAINTS:
            try:
                neo4j_client.execute_query_sync(constraint.query)
                logger.info("Created constraint", name=constraint.name)
            except Exception as e:
                logger.warning("Failed to create constraint", name=constraint.name, error=str(e))
        
        # Create indexes
        for index in self.INDEXES:
            try:
                neo4j_client.execute_query_sync(index.query)
                logger.info("Created index", name=index.name)
            except Exception as e:
                logger.warning("Failed to create index", name=index.name, error=str(e))
        
        logger.info("Code Property Graph schema creation completed (sync)")
    
    async def drop_schema(self) -> None:
        """Drop all schema elements (for testing/reset)."""
        logger.warning("Dropping Code Property Graph schema")
        
        # Drop all constraints
        constraints_query = "SHOW CONSTRAINTS YIELD name"
        constraints = await neo4j_client.execute_query(constraints_query)
        
        for constraint in constraints:
            drop_query = f"DROP CONSTRAINT {constraint['name']} IF EXISTS"
            try:
                await neo4j_client.execute_query(drop_query)
                logger.info("Dropped constraint", name=constraint['name'])
            except Exception as e:
                logger.warning("Failed to drop constraint", name=constraint['name'], error=str(e))
        
        # Drop all indexes
        indexes_query = "SHOW INDEXES YIELD name WHERE name <> 'system'"
        indexes = await neo4j_client.execute_query(indexes_query)
        
        for index in indexes:
            drop_query = f"DROP INDEX {index['name']} IF EXISTS"
            try:
                await neo4j_client.execute_query(drop_query)
                logger.info("Dropped index", name=index['name'])
            except Exception as e:
                logger.warning("Failed to drop index", name=index['name'], error=str(e))
    
    async def get_schema_info(self) -> Dict[str, Any]:
        """Get current schema information."""
        constraints = await neo4j_client.execute_query("SHOW CONSTRAINTS")
        indexes = await neo4j_client.execute_query("SHOW INDEXES")
        
        return {
            "constraints": constraints,
            "indexes": indexes,
            "constraint_count": len(constraints),
            "index_count": len(indexes)
        }


# Global schema manager instance
cpg_schema = CodePropertyGraphSchema()