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
            logger.info("Attempting to connect to Neo4j", uri=config.database.neo4j_uri)
            
            # Create driver with SSL configuration for cloud instances
            driver_config = {
                "max_connection_lifetime": 30 * 60,  # 30 minutes
                "max_connection_pool_size": 50,
                "connection_acquisition_timeout": 60,  # 60 seconds
            }
            
            # For neo4j+s:// URIs, the SSL is handled automatically by the driver
            if config.database.neo4j_uri.startswith(('neo4j+s://', 'bolt+s://')):
                logger.info("Using secure connection with automatic SSL handling")
            
            self._driver = AsyncGraphDatabase.driver(
                config.database.neo4j_uri,
                auth=(config.database.neo4j_user, config.database.neo4j_password),
                **driver_config
            )
            
            # Test connection
            await self._driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database", uri=config.database.neo4j_uri)
            
        except ServiceUnavailable as e:
            # Try fallback connection with relaxed SSL verification
            logger.warning("Primary connection failed, trying fallback SSL configuration", error=str(e))
            try:
                await self._connect_with_fallback()
            except Exception as fallback_error:
                error_msg = f"Neo4j service unavailable. Please check if the database is running and accessible. Primary error: {e}, Fallback error: {fallback_error}"
                logger.error("Neo4j service unavailable", error=str(e), uri=config.database.neo4j_uri)
                raise Neo4jError(error_msg)
        except AuthError as e:
            error_msg = f"Neo4j authentication failed. Please check username and password. Error: {e}"
            logger.error("Neo4j authentication failed", error=str(e), uri=config.database.neo4j_uri)
            raise Neo4jError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error connecting to Neo4j: {e}"
            logger.error("Unexpected Neo4j connection error", error=str(e), uri=config.database.neo4j_uri)
            raise Neo4jError(error_msg)
    
    async def _connect_with_fallback(self) -> None:
        """Fallback connection method with SSL bypass for development."""
        logger.info("Attempting fallback connection with SSL bypass for development")
        
        # For development, try connecting without SSL verification
        # Convert secure URI to non-secure for testing
        fallback_uri = config.database.neo4j_uri.replace('+s://', '://')
        logger.warning(f"Using non-secure connection for development: {fallback_uri}")
        
        fallback_config = {
            "max_connection_lifetime": 30 * 60,
            "max_connection_pool_size": 10,
            "connection_acquisition_timeout": 30,
        }
        
        self._driver = AsyncGraphDatabase.driver(
            fallback_uri,
            auth=(config.database.neo4j_user, config.database.neo4j_password),
            **fallback_config
        )
        
        # Test connection
        await self._driver.verify_connectivity()
        logger.warning("Connected using non-secure fallback (development only)")
        logger.warning("For production, ensure proper SSL certificates are configured")
    
    def connect_sync(self) -> None:
        """Establish synchronous connection to Neo4j database."""
        try:
            logger.info("Attempting to connect to Neo4j (sync)", uri=config.database.neo4j_uri)
            
            # Create driver with SSL configuration for cloud instances
            driver_config = {
                "max_connection_lifetime": 30 * 60,  # 30 minutes
                "max_connection_pool_size": 50,
                "connection_acquisition_timeout": 60,  # 60 seconds
            }
            
            # For neo4j+s:// URIs, the SSL is handled automatically by the driver
            if config.database.neo4j_uri.startswith(('neo4j+s://', 'bolt+s://')):
                logger.info("Using secure connection with automatic SSL handling (sync)")
            
            self._sync_driver = GraphDatabase.driver(
                config.database.neo4j_uri,
                auth=(config.database.neo4j_user, config.database.neo4j_password),
                **driver_config
            )
            
            # Test connection
            self._sync_driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j database (sync)", uri=config.database.neo4j_uri)
            
        except ServiceUnavailable as e:
            # Try fallback connection with relaxed SSL verification
            logger.warning("Primary connection failed, trying fallback SSL configuration (sync)", error=str(e))
            try:
                self._connect_sync_with_fallback()
            except Exception as fallback_error:
                error_msg = f"Neo4j service unavailable. Please check if the database is running and accessible. Primary error: {e}, Fallback error: {fallback_error}"
                logger.error("Neo4j service unavailable (sync)", error=str(e), uri=config.database.neo4j_uri)
                raise Neo4jError(error_msg)
        except AuthError as e:
            error_msg = f"Neo4j authentication failed. Please check username and password. Error: {e}"
            logger.error("Neo4j authentication failed (sync)", error=str(e), uri=config.database.neo4j_uri)
            raise Neo4jError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error connecting to Neo4j: {e}"
            logger.error("Unexpected Neo4j connection error (sync)", error=str(e), uri=config.database.neo4j_uri)
            raise Neo4jError(error_msg)
    
    def _connect_sync_with_fallback(self) -> None:
        """Fallback synchronous connection method with SSL bypass for development."""
        logger.info("Attempting fallback connection with SSL bypass for development (sync)")
        
        # For development, try connecting without SSL verification
        # Convert secure URI to non-secure for testing
        fallback_uri = config.database.neo4j_uri.replace('+s://', '://')
        logger.warning(f"Using non-secure connection for development (sync): {fallback_uri}")
        
        fallback_config = {
            "max_connection_lifetime": 30 * 60,
            "max_connection_pool_size": 10,
            "connection_acquisition_timeout": 30,
        }
        
        self._sync_driver = GraphDatabase.driver(
            fallback_uri,
            auth=(config.database.neo4j_user, config.database.neo4j_password),
            **fallback_config
        )
        
        # Test connection
        self._sync_driver.verify_connectivity()
        logger.warning("Connected using non-secure fallback (development only, sync)")
        logger.warning("For production, ensure proper SSL certificates are configured")
    
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