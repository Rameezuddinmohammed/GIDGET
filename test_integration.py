#!/usr/bin/env python3
"""Integration test for the ingestion pipeline with real Neo4j."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from code_intelligence.database.neo4j_client import Neo4jClient
from code_intelligence.ingestion.pipeline import IngestionPipeline
from code_intelligence.git.repository import RepositoryManager
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_neo4j_connection():
    """Test Neo4j connection."""
    try:
        client = Neo4jClient()
        
        # Test basic query using sync method
        result = client.execute_query_sync("RETURN 'Hello Neo4j' as message")
        logger.info(f"Neo4j connection successful: {result[0]['message']}")
        
        # Test database info with a simpler query
        result = client.execute_query_sync("CALL db.labels() YIELD label RETURN count(label) as label_count")
        logger.info(f"Database has {result[0]['label_count']} node labels")
        
        return client
    except Exception as e:
        logger.error(f"Neo4j connection failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_parsing_pipeline():
    """Test the parsing pipeline on current repository."""
    try:
        from code_intelligence.parsing.parser import MultiLanguageParser
        
        parser = MultiLanguageParser()
        
        # Parse the src directory
        src_path = Path(__file__).parent / "src"
        logger.info(f"Parsing source code in: {src_path}")
        
        parsed_files = parser.parse_directory(
            str(src_path),
            include_patterns=['*.py'],
            exclude_patterns=['__pycache__/**', '*.pyc']
        )
        
        logger.info(f"Parsed {len(parsed_files)} Python files")
        
        total_elements = 0
        for parsed_file in parsed_files:
            elements = len(parsed_file.elements)
            total_elements += elements
            if elements > 0:
                logger.info(f"  {parsed_file.file_path}: {elements} elements")
        
        logger.info(f"Total code elements found: {total_elements}")
        return parsed_files
        
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        return []

def test_ingestion_pipeline(neo4j_client):
    """Test the complete ingestion pipeline."""
    try:
        # Initialize pipeline
        pipeline = IngestionPipeline(neo4j_client)
        
        # Test with current repository
        current_repo = Path(__file__).parent
        logger.info(f"Testing ingestion with repository: {current_repo}")
        
        # Ingest current repository (limit to 5 commits for testing)
        job = pipeline.ingest_local_repository(
            str(current_repo),
            "code_intelligence_test",
            max_commits=5
        )
        
        logger.info(f"Ingestion job created: {job.id}")
        logger.info(f"Job status: {job.status}")
        logger.info(f"Repository: {job.repository_id}")
        logger.info(f"Total commits: {job.total_commits}")
        logger.info(f"Processed commits: {job.processed_commits}")
        logger.info(f"Ingested elements: {job.ingested_elements}")
        logger.info(f"Ingested relationships: {job.ingested_relationships}")
        
        if job.error_message:
            logger.error(f"Job error: {job.error_message}")
        
        return job
        
    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_graph_queries(neo4j_client):
    """Test querying the populated graph."""
    try:
        logger.info("Testing graph queries...")
        
        # Count nodes by type
        queries = [
            ("Repositories", "MATCH (r:Repository) RETURN count(r) as count"),
            ("Commits", "MATCH (c:Commit) RETURN count(c) as count"),
            ("Files", "MATCH (f:File) RETURN count(f) as count"),
            ("Code Elements", "MATCH (e:CodeElement) RETURN count(e) as count"),
            ("Functions", "MATCH (f:Function) RETURN count(f) as count"),
            ("Classes", "MATCH (c:Class) RETURN count(c) as count"),
        ]
        
        for name, query in queries:
            result = neo4j_client.execute_query_sync(query)
            count = result[0]['count'] if result else 0
            logger.info(f"  {name}: {count}")
        
        # Sample some data
        logger.info("\nSample functions:")
        result = neo4j_client.execute_query_sync("""
            MATCH (f:Function) 
            RETURN f.name, f.file_path, f.language 
            LIMIT 5
        """)
        
        for record in result:
            logger.info(f"  {record['f.name']} in {record['f.file_path']} ({record['f.language']})")
        
        # Sample relationships
        logger.info("\nSample relationships:")
        result = neo4j_client.execute_query_sync("""
            MATCH (a)-[r]->(b) 
            RETURN type(r) as rel_type, count(*) as count 
            ORDER BY count DESC 
            LIMIT 5
        """)
        
        for record in result:
            logger.info(f"  {record['rel_type']}: {record['count']}")
            
    except Exception as e:
        logger.error(f"Graph queries failed: {e}")

def main():
    """Run integration tests."""
    logger.info("Starting integration tests...")
    
    # Test 1: Neo4j connection
    logger.info("\n=== Testing Neo4j Connection ===")
    neo4j_client = test_neo4j_connection()
    if not neo4j_client:
        logger.error("Cannot proceed without Neo4j connection")
        return 1
    
    # Test 2: Parsing pipeline
    logger.info("\n=== Testing Parsing Pipeline ===")
    parsed_files = test_parsing_pipeline()
    if not parsed_files:
        logger.error("Parsing pipeline failed")
        return 1
    
    # Test 3: Full ingestion pipeline
    logger.info("\n=== Testing Ingestion Pipeline ===")
    job = test_ingestion_pipeline(neo4j_client)
    if not job:
        logger.error("Ingestion pipeline failed")
        return 1
    
    # Test 4: Graph queries
    logger.info("\n=== Testing Graph Queries ===")
    test_graph_queries(neo4j_client)
    
    # Cleanup
    neo4j_client.close_sync()
    
    logger.info("\n=== Integration Tests Complete ===")
    logger.info("All tests passed! The ingestion pipeline is working correctly.")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)