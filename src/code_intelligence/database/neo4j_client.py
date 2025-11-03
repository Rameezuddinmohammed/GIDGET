"""Neo4j database client and connection management."""

from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

from neo4j import GraphDatabase, Driver, AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable, AuthError

from ..config import config
from ..exceptions import Neo4jError
from ..logging import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    """Neo4j database client with connection management."""
    
    def __init__(self) -> None:
        self._driver: Optional[AsyncDriver] = None
        self._sync_driver: Optional[Driver] = None
    
    async def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self._driver = AsyncGraphDatabase.driver(
                config.database.neo4j_uri,
                auth=(config.database.neo4j_user, config.database.neo4j_password)
            )
            
            # Test connection
            await self._driver.verify_connectivity()
            logger.info("Connected to Neo4j database", uri=config.database.neo4j_uri)
            
        except (ServiceUnavailable, AuthError) as e:
            logger.error("Failed to connect to Neo4j", error=str(e))
            raise Neo4jError(f"Failed to connect to Neo4j: {e}")
    
    def connect_sync(self) -> None:
        """Establish synchronous connection to Neo4j database."""
        try:
            self._sync_driver = GraphDatabase.driver(
                config.database.neo4j_uri,
                auth=(config.database.neo4j_user, config.database.neo4j_password)
            )
            
            # Test connection
            self._sync_driver.verify_connectivity()
            logger.info("Connected to Neo4j database (sync)", uri=config.database.neo4j_uri)
            
        except (ServiceUnavailable, AuthError) as e:
            logger.error("Failed to connect to Neo4j (sync)", error=str(e))
            raise Neo4jError(f"Failed to connect to Neo4j: {e}")
    
    async def close(self) -> None:
        """Close database connection."""
        if self._driver:
            await self._driver.close()
            logger.info("Closed Neo4j connection")
    
    def close_sync(self) -> None:
        """Close synchronous database connection."""
        if self._sync_driver:
            self._sync_driver.close()
            logger.info("Closed Neo4j connection (sync)")
    
    @asynccontextmanager
    async def session(self, database: Optional[str] = None):
        """Get an async database session."""
        if not self._driver:
            await self.connect()
        
        db_name = database or config.database.neo4j_database
        async with self._driver.session(database=db_name) as session:
            yield session
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        try:
            async with self.session(database) as session:
                result = await session.run(query, parameters or {})
                records = await result.data()
                logger.debug("Executed Neo4j query", query=query, record_count=len(records))
                return records
        except Exception as e:
            logger.error("Neo4j query failed", query=query, error=str(e))
            raise Neo4jError(f"Query execution failed: {e}")
    
    async def execute_write_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute a write Cypher query and return results."""
        try:
            async with self.session(database) as session:
                result = await session.execute_write(
                    lambda tx: tx.run(query, parameters or {}).data()
                )
                logger.debug("Executed Neo4j write query", query=query, record_count=len(result))
                return result
        except Exception as e:
            logger.error("Neo4j write query failed", query=query, error=str(e))
            raise Neo4jError(f"Write query execution failed: {e}")
    
    def execute_query_sync(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Execute a synchronous Cypher query and return results."""
        if not self._sync_driver:
            self.connect_sync()
        
        try:
            db_name = database or config.database.neo4j_database
            with self._sync_driver.session(database=db_name) as session:
                result = session.run(query, parameters or {})
                records = result.data()
                logger.debug("Executed Neo4j query (sync)", query=query, record_count=len(records))
                return records
        except Exception as e:
            logger.error("Neo4j query failed (sync)", query=query, error=str(e))
            raise Neo4jError(f"Query execution failed: {e}")


# Global client instance
neo4j_client = Neo4jClient()