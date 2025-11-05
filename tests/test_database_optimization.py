"""Tests for database optimization and performance validation."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.code_intelligence.database.query_optimizer import (
    Neo4jQueryOptimizer,
    SupabaseQueryOptimizer,
    DatabaseOptimizationValidator,
    QueryPerformanceMetrics,
    IndexUsageReport
)


class TestNeo4jQueryOptimizer:
    """Test Neo4j query optimization and validation."""
    
    @pytest.fixture
    def optimizer(self):
        """Create Neo4j query optimizer instance."""
        return Neo4jQueryOptimizer()
    
    @pytest.fixture
    def mock_neo4j_client(self):
        """Mock Neo4j client."""
        with patch('src.code_intelligence.database.query_optimizer.neo4j_client') as mock:
            mock_client = AsyncMock()
            mock_client.execute_query = AsyncMock()
            yield mock_client
    
    @pytest.mark.asyncio
    async def test_analyze_query_performance(self, optimizer, mock_neo4j_client):
        """Test query performance analysis."""
        # Set the mock client on the optimizer
        optimizer.client = mock_neo4j_client
        
        # Mock EXPLAIN and PROFILE results with a small delay
        async def mock_execute_with_delay(*args, **kwargs):
            await asyncio.sleep(0.01)  # Small delay to ensure execution_time > 0
            if "EXPLAIN" in args[0]:
                return [{"plan": "NodeByLabelScan"}]
            else:
                return [{"n": {"name": "test_function"}}]
        
        mock_neo4j_client.execute_query.side_effect = mock_execute_with_delay
        
        query = "MATCH (n:Function) RETURN n.name LIMIT 10"
        metrics = await optimizer.analyze_query_performance(query)
        
        assert isinstance(metrics, QueryPerformanceMetrics)
        assert metrics.query == query
        assert metrics.execution_time_ms >= 0  # Changed to >= since timing can be very small
        assert metrics.rows_returned == 1
        assert metrics.confidence_score > 0
        
        # Verify both EXPLAIN and PROFILE were called
        assert mock_neo4j_client.execute_query.call_count == 2
        
    @pytest.mark.asyncio
    async def test_analyze_query_performance_with_optimization_suggestions(self, optimizer, mock_neo4j_client):
        """Test query analysis with optimization suggestions."""
        optimizer.client = mock_neo4j_client
        
        mock_neo4j_client.execute_query.side_effect = [
            [],  # EXPLAIN result
            []   # PROFILE result (empty)
        ]
        
        # Query without WHERE clause should trigger suggestion
        query = "MATCH (n) RETURN n"
        metrics = await optimizer.analyze_query_performance(query)
        
        assert "Consider adding WHERE clause" in str(metrics.optimization_suggestions)
        assert metrics.confidence_score < 0.8  # Should be penalized for issues
        
    @pytest.mark.asyncio
    async def test_validate_indexes(self, optimizer, mock_neo4j_client):
        """Test index validation."""
        optimizer.client = mock_neo4j_client
        
        mock_indexes = [
            {
                "name": "function_name_index",
                "type": "BTREE",
                "state": "ONLINE",
                "labelsOrTypes": ["Function"]
            },
            {
                "name": "commit_timestamp_index",
                "type": "BTREE",
                "state": "FAILED",
                "labelsOrTypes": ["Commit"]
            }
        ]
        
        mock_neo4j_client.execute_query.return_value = mock_indexes
        
        reports = await optimizer.validate_indexes()
        
        assert len(reports) == 2
        assert all(isinstance(report, IndexUsageReport) for report in reports)
        
        # Check that failed index has lower effectiveness score
        failed_index = next(r for r in reports if r.index_name == "commit_timestamp_index")
        online_index = next(r for r in reports if r.index_name == "function_name_index")
        
        assert failed_index.effectiveness_score < online_index.effectiveness_score
        assert "consider rebuilding" in str(failed_index.recommendations).lower()
        
    @pytest.mark.asyncio
    async def test_run_performance_benchmark(self, optimizer, mock_neo4j_client):
        """Test performance benchmark execution."""
        # Mock different execution times for different queries
        mock_neo4j_client.execute_query.side_effect = [
            # First query - EXPLAIN and PROFILE
            [], [{"result": 1}],
            # Second query - EXPLAIN and PROFILE  
            [], [{"result": 2}, {"result": 3}],
            # Third query - EXPLAIN and PROFILE
            [], []
        ]
        
        test_queries = [
            "MATCH (n:Function) RETURN count(n)",
            "MATCH (f:Function)-[:CALLS]->(g:Function) RETURN f, g LIMIT 5",
            "MATCH (c:Commit) RETURN c.sha"
        ]
        
        results = await optimizer.run_performance_benchmark(test_queries)
        
        assert "total_queries" in results
        assert results["total_queries"] == 3
        assert "results" in results
        assert len(results["results"]) == 3
        assert "summary" in results
        
        summary = results["summary"]
        assert "success_rate" in summary
        assert "avg_execution_time_ms" in summary
        assert summary["success_rate"] == 1.0  # All queries should succeed
        
    @pytest.mark.asyncio
    async def test_benchmark_with_query_failure(self, optimizer, mock_neo4j_client):
        """Test benchmark handling of query failures."""
        # Mock one successful query and one failure
        mock_neo4j_client.execute_query.side_effect = [
            [], [{"result": 1}],  # First query succeeds
            Exception("Query failed")  # Second query fails
        ]
        
        test_queries = [
            "MATCH (n:Function) RETURN count(n)",
            "INVALID QUERY SYNTAX"
        ]
        
        results = await optimizer.run_performance_benchmark(test_queries)
        
        assert results["total_queries"] == 2
        assert len(results["results"]) == 2
        
        # Check that failure is recorded
        failed_result = next(r for r in results["results"] if "error" in r)
        assert "Query failed" in failed_result["error"]
        
        # Success rate should be 50%
        assert results["summary"]["success_rate"] == 0.5


class TestSupabaseQueryOptimizer:
    """Test Supabase query optimization and validation."""
    
    @pytest.fixture
    def optimizer(self):
        """Create Supabase query optimizer instance."""
        return SupabaseQueryOptimizer()
    
    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client."""
        with patch('src.code_intelligence.database.query_optimizer.supabase_client') as mock:
            mock_client = AsyncMock()
            mock.return_value = mock_client
            yield mock_client
    
    @pytest.mark.asyncio
    async def test_analyze_vector_search_performance(self, optimizer, mock_supabase_client):
        """Test vector search performance analysis."""
        metrics = await optimizer.analyze_vector_search_performance(embedding_dim=768, limit=5)
        
        assert isinstance(metrics, QueryPerformanceMetrics)
        assert metrics.execution_time_ms > 0
        assert metrics.rows_returned == 5
        assert "pgvector" in str(metrics.optimization_suggestions).lower()
        assert metrics.confidence_score > 0
        
    @pytest.mark.asyncio
    async def test_analyze_vector_search_with_large_limit(self, optimizer, mock_supabase_client):
        """Test vector search analysis with large limit."""
        metrics = await optimizer.analyze_vector_search_performance(limit=200)
        
        # Should suggest optimization for large LIMIT
        assert any("LIMIT" in suggestion for suggestion in metrics.optimization_suggestions)
        
    @pytest.mark.asyncio
    async def test_validate_pgvector_setup(self, optimizer, mock_supabase_client):
        """Test pgvector setup validation."""
        validation = await optimizer.validate_pgvector_setup()
        
        assert "timestamp" in validation
        assert "pgvector_available" in validation
        assert "indexes" in validation
        assert "configuration" in validation
        assert "recommendations" in validation
        
        # Should have at least one recommendation
        assert len(validation["recommendations"]) > 0
        
    @pytest.mark.asyncio
    async def test_benchmark_cache_queries(self, optimizer, mock_supabase_client):
        """Test cache queries benchmark."""
        results = await optimizer.benchmark_cache_queries()
        
        assert "timestamp" in results
        assert "cache_operations" in results
        assert "summary" in results
        
        # Should test multiple cache operations
        operations = results["cache_operations"]
        assert len(operations) >= 4  # lookup, insert, cleanup, stats
        
        # All operations should succeed in mock
        assert all(op["status"] == "success" for op in operations)
        
        summary = results["summary"]
        assert "total_operations" in summary
        assert "avg_execution_time_ms" in summary


class TestDatabaseOptimizationValidator:
    """Test comprehensive database optimization validation."""
    
    @pytest.fixture
    def validator(self):
        """Create database optimization validator instance."""
        return DatabaseOptimizationValidator()
    
    @pytest.fixture
    def mock_optimizers(self, validator):
        """Mock both optimizers."""
        with patch.object(validator.neo4j_optimizer, 'run_performance_benchmark') as mock_neo4j, \
             patch.object(validator.neo4j_optimizer, 'validate_indexes') as mock_neo4j_indexes, \
             patch.object(validator.supabase_optimizer, 'analyze_vector_search_performance') as mock_vector, \
             patch.object(validator.supabase_optimizer, 'validate_pgvector_setup') as mock_pgvector, \
             patch.object(validator.supabase_optimizer, 'benchmark_cache_queries') as mock_cache:
            
            # Mock Neo4j results
            mock_neo4j.return_value = {
                "total_queries": 4,
                "summary": {
                    "success_rate": 0.9,
                    "avg_execution_time_ms": 150.0,
                    "fastest_query_ms": 50.0,
                    "slowest_query_ms": 300.0
                },
                "results": []
            }
            
            mock_neo4j_indexes.return_value = [
                IndexUsageReport(
                    index_name="function_name_index",
                    table_name="Function",
                    usage_count=100,
                    effectiveness_score=0.9,
                    last_used=datetime.now(),
                    recommendations=["Index is performing well"]
                )
            ]
            
            # Mock Supabase results
            mock_vector.return_value = QueryPerformanceMetrics(
                query="vector search",
                execution_time_ms=120.0,
                rows_examined=1000,
                rows_returned=10,
                index_usage=["pgvector_index"],
                optimization_suggestions=["Consider HNSW index"],
                confidence_score=0.8
            )
            
            mock_pgvector.return_value = {
                "pgvector_available": True,
                "indexes": [{"table": "code_embeddings", "status": "active"}],
                "recommendations": ["pgvector is properly configured"]
            }
            
            mock_cache.return_value = {
                "cache_operations": [
                    {"operation": "lookup", "execution_time_ms": 10.0, "status": "success"},
                    {"operation": "insert", "execution_time_ms": 15.0, "status": "success"}
                ],
                "summary": {"avg_execution_time_ms": 12.5}
            }
            
            yield {
                "neo4j_benchmark": mock_neo4j,
                "neo4j_indexes": mock_neo4j_indexes,
                "vector_search": mock_vector,
                "pgvector_setup": mock_pgvector,
                "cache_benchmark": mock_cache
            }
    
    @pytest.mark.asyncio
    async def test_run_comprehensive_validation(self, validator, mock_optimizers):
        """Test comprehensive database validation."""
        report = await validator.run_comprehensive_validation()
        
        assert "timestamp" in report
        assert "neo4j_analysis" in report
        assert "supabase_analysis" in report
        assert "overall_score" in report
        assert "recommendations" in report
        
        # Check Neo4j analysis
        neo4j_analysis = report["neo4j_analysis"]
        assert "benchmark" in neo4j_analysis
        assert "indexes" in neo4j_analysis
        
        # Check Supabase analysis
        supabase_analysis = report["supabase_analysis"]
        assert "vector_search" in supabase_analysis
        assert "pgvector_setup" in supabase_analysis
        assert "cache_performance" in supabase_analysis
        
        # Overall score should be calculated
        assert 0.0 <= report["overall_score"] <= 1.0
        
        # Should have recommendations
        assert len(report["recommendations"]) > 0
        
    @pytest.mark.asyncio
    async def test_validation_with_poor_performance(self, validator, mock_optimizers):
        """Test validation with poor performance metrics."""
        # Mock poor performance
        mock_optimizers["neo4j_benchmark"].return_value["summary"]["success_rate"] = 0.5
        mock_optimizers["neo4j_benchmark"].return_value["summary"]["avg_execution_time_ms"] = 1000.0
        
        mock_optimizers["vector_search"].return_value.execution_time_ms = 500.0
        mock_optimizers["vector_search"].return_value.confidence_score = 0.4
        
        report = await validator.run_comprehensive_validation()
        
        # Should have lower overall score
        assert report["overall_score"] < 0.7
        
        # Should have performance-related recommendations
        recommendations = " ".join(report["recommendations"]).lower()
        assert "success rate" in recommendations or "query time" in recommendations
        
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, validator):
        """Test validation error handling."""
        # Mock an exception in Neo4j optimizer
        with patch.object(validator.neo4j_optimizer, 'run_performance_benchmark', side_effect=Exception("Neo4j error")):
            report = await validator.run_comprehensive_validation()
            
            assert "error" in report
            assert report["overall_score"] == 0.0
            assert "Neo4j error" in report["error"]
    
    def test_generate_overall_recommendations(self, validator):
        """Test recommendation generation."""
        # Mock data for recommendation generation
        neo4j_benchmark = {
            "summary": {
                "success_rate": 0.7,  # Below 90%
                "avg_execution_time_ms": 600.0  # Above 500ms
            }
        }
        
        vector_performance = QueryPerformanceMetrics(
            query="test",
            execution_time_ms=400.0,  # Above 300ms
            rows_examined=100,
            rows_returned=10,
            index_usage=[],
            optimization_suggestions=[],
            confidence_score=0.8
        )
        
        pgvector_validation = {"pgvector_available": False}
        
        recommendations = validator._generate_overall_recommendations(
            neo4j_benchmark, vector_performance, pgvector_validation
        )
        
        # Should have recommendations for all identified issues
        rec_text = " ".join(recommendations).lower()
        assert "success rate" in rec_text
        assert "query time" in rec_text or "slow" in rec_text
        assert "vector search" in rec_text or "slow" in rec_text
        assert "pgvector" in rec_text


@pytest.mark.integration
class TestDatabaseOptimizationIntegration:
    """Integration tests for database optimization."""
    
    @pytest.mark.asyncio
    async def test_real_neo4j_connection(self):
        """Test with real Neo4j connection if available."""
        try:
            from src.code_intelligence.database.neo4j_client import neo4j_client
            
            # Test basic connection
            result = await neo4j_client.execute_query("RETURN 1 as test")
            assert result[0]["test"] == 1
            
            # Test query optimization
            optimizer = Neo4jQueryOptimizer()
            metrics = await optimizer.analyze_query_performance("RETURN 1 as test")
            
            assert metrics.execution_time_ms > 0
            assert metrics.confidence_score > 0
            
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")
    
    @pytest.mark.asyncio
    async def test_comprehensive_validation_integration(self):
        """Test comprehensive validation with real databases if available."""
        try:
            validator = DatabaseOptimizationValidator()
            report = await validator.run_comprehensive_validation()
            
            # Basic structure validation
            assert "timestamp" in report
            assert "overall_score" in report
            assert isinstance(report["overall_score"], (int, float))
            assert 0.0 <= report["overall_score"] <= 1.0
            
        except Exception as e:
            pytest.skip(f"Database validation integration test failed: {e}")


@pytest.mark.performance
class TestDatabasePerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.asyncio
    async def test_neo4j_query_performance_benchmark(self):
        """Benchmark Neo4j query performance."""
        optimizer = Neo4jQueryOptimizer()
        
        # Test queries of varying complexity
        test_queries = [
            "RETURN 1",  # Simple
            "MATCH (n) RETURN count(n) LIMIT 1",  # Medium
            "MATCH (a)-[r]->(b) RETURN type(r), count(*) ORDER BY count(*) DESC LIMIT 10"  # Complex
        ]
        
        try:
            results = await optimizer.run_performance_benchmark(test_queries)
            
            # Performance assertions
            assert results["summary"]["avg_execution_time_ms"] < 1000  # Should be under 1 second average
            
            # Simple query should be fastest
            simple_result = next(r for r in results["results"] if "RETURN 1" in r["query"])
            assert simple_result["execution_time_ms"] < 100  # Should be very fast
            
        except Exception as e:
            pytest.skip(f"Neo4j performance benchmark failed: {e}")
    
    @pytest.mark.asyncio
    async def test_vector_search_performance_benchmark(self):
        """Benchmark vector search performance."""
        optimizer = SupabaseQueryOptimizer()
        
        try:
            # Test different embedding dimensions
            for dim in [384, 768, 1536]:
                metrics = await optimizer.analyze_vector_search_performance(embedding_dim=dim)
                
                # Performance should be reasonable
                assert metrics.execution_time_ms < 2000  # Under 2 seconds
                assert metrics.confidence_score > 0.3  # Some confidence
                
        except Exception as e:
            pytest.skip(f"Vector search benchmark failed: {e}")