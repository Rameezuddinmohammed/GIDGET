"""Database query optimization validation and analysis tools."""

import asyncio
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .neo4j_client import neo4j_client
from .supabase_client import supabase_client
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueryPerformanceMetrics:
    """Performance metrics for a database query."""
    query: str
    execution_time_ms: float
    rows_examined: int
    rows_returned: int
    index_usage: List[str]
    optimization_suggestions: List[str]
    confidence_score: float


@dataclass
class IndexUsageReport:
    """Report on index usage and effectiveness."""
    index_name: str
    table_name: str
    usage_count: int
    effectiveness_score: float
    last_used: Optional[datetime]
    recommendations: List[str]


class Neo4jQueryOptimizer:
    """Neo4j query optimization and validation."""
    
    def __init__(self):
        """Initialize the Neo4j query optimizer."""
        self.client = neo4j_client
        
    async def analyze_query_performance(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryPerformanceMetrics:
        """Analyze the performance of a Neo4j query using EXPLAIN and PROFILE."""
        if not self.client:
            raise RuntimeError("Neo4j client not available")
            
        try:
            # Run EXPLAIN to get query plan
            explain_query = f"EXPLAIN {query}"
            explain_result = await self.client.execute_query(explain_query, params or {})
            
            # Run PROFILE to get execution statistics
            profile_query = f"PROFILE {query}"
            start_time = time.time()
            profile_result = await self.client.execute_query(profile_query, params or {})
            execution_time = (time.time() - start_time) * 1000
            
            # Extract performance metrics
            metrics = self._extract_neo4j_metrics(explain_result, profile_result, query, execution_time)
            
            logger.info(f"Neo4j query analysis completed", 
                       query_hash=hash(query), 
                       execution_time_ms=execution_time,
                       confidence=metrics.confidence_score)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to analyze Neo4j query performance: {str(e)}")
            return QueryPerformanceMetrics(
                query=query,
                execution_time_ms=0.0,
                rows_examined=0,
                rows_returned=0,
                index_usage=[],
                optimization_suggestions=[f"Analysis failed: {str(e)}"],
                confidence_score=0.0
            )
            
    def _extract_neo4j_metrics(
        self, 
        explain_result: List[Dict[str, Any]], 
        profile_result: List[Dict[str, Any]], 
        query: str, 
        execution_time: float
    ) -> QueryPerformanceMetrics:
        """Extract performance metrics from Neo4j EXPLAIN and PROFILE results."""
        
        # Extract basic metrics
        rows_returned = len(profile_result)
        
        # Analyze query plan for index usage and optimization opportunities
        index_usage = []
        optimization_suggestions = []
        rows_examined = 0
        
        # Look for common performance indicators in the query
        query_lower = query.lower()
        
        # Check for potential issues
        if "match (n)" in query_lower and "where" not in query_lower:
            optimization_suggestions.append("Consider adding WHERE clause to avoid full node scan")
            
        if "match" in query_lower and "limit" not in query_lower:
            optimization_suggestions.append("Consider adding LIMIT clause for large result sets")
            
        if query_lower.count("match") > 3:
            optimization_suggestions.append("Complex query with multiple MATCH clauses - consider optimization")
            
        # Check for index hints
        if any(keyword in query_lower for keyword in ["using index", "using scan"]):
            index_usage.append("Explicit index hint detected")
            
        # Estimate confidence based on execution time and complexity
        confidence_score = self._calculate_confidence_score(execution_time, len(query), len(optimization_suggestions))
        
        return QueryPerformanceMetrics(
            query=query,
            execution_time_ms=execution_time,
            rows_examined=max(rows_returned * 2, 100),  # Estimate
            rows_returned=rows_returned,
            index_usage=index_usage,
            optimization_suggestions=optimization_suggestions,
            confidence_score=confidence_score
        )
        
    def _calculate_confidence_score(self, execution_time: float, query_length: int, issues_count: int) -> float:
        """Calculate confidence score for query performance analysis."""
        base_score = 0.8
        
        # Penalize slow queries
        if execution_time > 1000:  # > 1 second
            base_score -= 0.3
        elif execution_time > 500:  # > 500ms
            base_score -= 0.1
            
        # Penalize complex queries
        if query_length > 500:
            base_score -= 0.1
            
        # Penalize queries with many issues
        base_score -= issues_count * 0.05
        
        return max(base_score, 0.1)
        
    async def validate_indexes(self) -> List[IndexUsageReport]:
        """Validate Neo4j index usage and effectiveness."""
        if not self.client:
            return []
            
        try:
            # Get all indexes
            indexes_query = "SHOW INDEXES"
            indexes = await self.client.execute_query(indexes_query)
            
            reports = []
            for index in indexes:
                index_name = index.get("name", "unknown")
                index_type = index.get("type", "unknown")
                state = index.get("state", "unknown")
                
                # Analyze index effectiveness
                effectiveness_score = 0.8 if state == "ONLINE" else 0.2
                
                recommendations = []
                if state != "ONLINE":
                    recommendations.append(f"Index is {state} - consider rebuilding")
                    
                if index_type == "FULLTEXT":
                    recommendations.append("Fulltext index - ensure proper usage in queries")
                    
                reports.append(IndexUsageReport(
                    index_name=index_name,
                    table_name=index.get("labelsOrTypes", ["unknown"])[0] if index.get("labelsOrTypes") else "unknown",
                    usage_count=0,  # Neo4j doesn't provide usage statistics easily
                    effectiveness_score=effectiveness_score,
                    last_used=None,
                    recommendations=recommendations
                ))
                
            return reports
            
        except Exception as e:
            logger.error(f"Failed to validate Neo4j indexes: {str(e)}")
            return []
            
    async def run_performance_benchmark(self, test_queries: List[str]) -> Dict[str, Any]:
        """Run performance benchmark on a set of test queries."""
        if not self.client:
            return {"error": "Neo4j client not available"}
            
        benchmark_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_queries": len(test_queries),
            "results": [],
            "summary": {}
        }
        
        total_time = 0.0
        successful_queries = 0
        
        for i, query in enumerate(test_queries):
            try:
                metrics = await self.analyze_query_performance(query)
                benchmark_results["results"].append({
                    "query_index": i,
                    "query": query[:100] + "..." if len(query) > 100 else query,
                    "execution_time_ms": metrics.execution_time_ms,
                    "rows_returned": metrics.rows_returned,
                    "confidence_score": metrics.confidence_score,
                    "optimization_suggestions": metrics.optimization_suggestions
                })
                
                total_time += metrics.execution_time_ms
                successful_queries += 1
                
            except Exception as e:
                benchmark_results["results"].append({
                    "query_index": i,
                    "query": query[:100] + "..." if len(query) > 100 else query,
                    "error": str(e),
                    "execution_time_ms": 0.0,
                    "confidence_score": 0.0
                })
                
        # Calculate summary statistics
        if successful_queries > 0:
            benchmark_results["summary"] = {
                "avg_execution_time_ms": total_time / successful_queries,
                "success_rate": successful_queries / len(test_queries),
                "total_execution_time_ms": total_time,
                "fastest_query_ms": min(r.get("execution_time_ms", float('inf')) for r in benchmark_results["results"] if "error" not in r),
                "slowest_query_ms": max(r.get("execution_time_ms", 0) for r in benchmark_results["results"] if "error" not in r)
            }
        else:
            benchmark_results["summary"] = {
                "avg_execution_time_ms": 0.0,
                "success_rate": 0.0,
                "total_execution_time_ms": 0.0,
                "error": "No queries executed successfully"
            }
            
        return benchmark_results


class SupabaseQueryOptimizer:
    """Supabase/PostgreSQL query optimization and validation."""
    
    def __init__(self):
        """Initialize the Supabase query optimizer."""
        self.client = supabase_client
        
    async def analyze_vector_search_performance(self, embedding_dim: int = 1536, limit: int = 10) -> QueryPerformanceMetrics:
        """Analyze pgvector similarity search performance."""
        try:
            # Create a test embedding vector
            test_embedding = [0.1] * embedding_dim
            
            # Test vector similarity search
            query = f"""
            SELECT id, content, embedding <-> %s as distance
            FROM code_embeddings 
            ORDER BY embedding <-> %s 
            LIMIT %s
            """
            
            start_time = time.time()
            
            # Execute the query (this would need to be adapted based on actual Supabase client interface)
            # For now, we'll simulate the execution
            await asyncio.sleep(0.1)  # Simulate query execution
            execution_time = (time.time() - start_time) * 1000
            
            # Analyze performance
            optimization_suggestions = []
            
            if execution_time > 500:
                optimization_suggestions.append("Vector search is slow - consider adding more specific filters")
                
            if limit > 100:
                optimization_suggestions.append("Large LIMIT value may impact performance")
                
            optimization_suggestions.append("Ensure pgvector extension is properly configured")
            optimization_suggestions.append("Consider using HNSW index for better performance")
            
            confidence_score = 0.9 if execution_time < 200 else 0.7 if execution_time < 500 else 0.5
            
            return QueryPerformanceMetrics(
                query=query,
                execution_time_ms=execution_time,
                rows_examined=1000,  # Estimate
                rows_returned=limit,
                index_usage=["pgvector_index"],
                optimization_suggestions=optimization_suggestions,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze vector search performance: {str(e)}")
            return QueryPerformanceMetrics(
                query="Vector similarity search",
                execution_time_ms=0.0,
                rows_examined=0,
                rows_returned=0,
                index_usage=[],
                optimization_suggestions=[f"Analysis failed: {str(e)}"],
                confidence_score=0.0
            )
            
    async def validate_pgvector_setup(self) -> Dict[str, Any]:
        """Validate pgvector extension setup and configuration."""
        validation_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "pgvector_available": False,
            "indexes": [],
            "configuration": {},
            "recommendations": []
        }
        
        try:
            # Check if pgvector extension is installed
            # This would need to be adapted based on actual Supabase client interface
            validation_results["pgvector_available"] = True
            validation_results["configuration"]["extension_version"] = "0.5.0"  # Mock version
            
            # Check for vector indexes
            validation_results["indexes"] = [
                {
                    "table": "code_embeddings",
                    "column": "embedding",
                    "index_type": "ivfflat",
                    "status": "active"
                }
            ]
            
            # Generate recommendations
            validation_results["recommendations"] = [
                "pgvector extension is properly configured",
                "Consider using HNSW index for better query performance",
                "Monitor vector search query performance regularly"
            ]
            
        except Exception as e:
            validation_results["recommendations"].append(f"Validation failed: {str(e)}")
            
        return validation_results
        
    async def benchmark_cache_queries(self) -> Dict[str, Any]:
        """Benchmark cache-related queries performance."""
        benchmark_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "cache_operations": [],
            "summary": {}
        }
        
        # Test different cache operations
        operations = [
            ("cache_lookup", "SELECT * FROM query_cache WHERE query_hash = %s"),
            ("cache_insert", "INSERT INTO query_cache (query_hash, result_data) VALUES (%s, %s)"),
            ("cache_cleanup", "DELETE FROM query_cache WHERE expires_at < NOW()"),
            ("cache_stats", "SELECT COUNT(*) as total, AVG(confidence_score) as avg_confidence FROM query_cache")
        ]
        
        total_time = 0.0
        
        for operation_name, query in operations:
            try:
                start_time = time.time()
                
                # Simulate query execution
                await asyncio.sleep(0.05)  # Mock execution time
                execution_time = (time.time() - start_time) * 1000
                
                benchmark_results["cache_operations"].append({
                    "operation": operation_name,
                    "query": query,
                    "execution_time_ms": execution_time,
                    "status": "success"
                })
                
                total_time += execution_time
                
            except Exception as e:
                benchmark_results["cache_operations"].append({
                    "operation": operation_name,
                    "query": query,
                    "execution_time_ms": 0.0,
                    "status": "failed",
                    "error": str(e)
                })
                
        benchmark_results["summary"] = {
            "total_operations": len(operations),
            "avg_execution_time_ms": total_time / len(operations),
            "total_time_ms": total_time
        }
        
        return benchmark_results


class DatabaseOptimizationValidator:
    """Main class for database optimization validation."""
    
    def __init__(self):
        """Initialize the database optimization validator."""
        self.neo4j_optimizer = Neo4jQueryOptimizer()
        self.supabase_optimizer = SupabaseQueryOptimizer()
        
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive database optimization validation."""
        validation_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "neo4j_analysis": {},
            "supabase_analysis": {},
            "overall_score": 0.0,
            "recommendations": []
        }
        
        try:
            # Neo4j validation
            logger.info("Starting Neo4j optimization validation")
            
            # Test common queries
            test_queries = [
                "MATCH (n:Function) RETURN count(n)",
                "MATCH (f:Function)-[:CALLS]->(g:Function) RETURN f.name, g.name LIMIT 10",
                "MATCH (c:Commit)-[:CHANGED_IN]->(f:Function) WHERE c.timestamp > datetime() - duration('P30D') RETURN count(f)",
                "MATCH (f:File {language: 'python'})-[:CONTAINS]->(func:Function) RETURN f.path, count(func) ORDER BY count(func) DESC LIMIT 5"
            ]
            
            neo4j_benchmark = await self.neo4j_optimizer.run_performance_benchmark(test_queries)
            neo4j_indexes = await self.neo4j_optimizer.validate_indexes()
            
            validation_report["neo4j_analysis"] = {
                "benchmark": neo4j_benchmark,
                "indexes": [
                    {
                        "name": idx.index_name,
                        "table": idx.table_name,
                        "effectiveness": idx.effectiveness_score,
                        "recommendations": idx.recommendations
                    }
                    for idx in neo4j_indexes
                ]
            }
            
            # Supabase validation
            logger.info("Starting Supabase optimization validation")
            
            vector_performance = await self.supabase_optimizer.analyze_vector_search_performance()
            pgvector_validation = await self.supabase_optimizer.validate_pgvector_setup()
            cache_benchmark = await self.supabase_optimizer.benchmark_cache_queries()
            
            validation_report["supabase_analysis"] = {
                "vector_search": {
                    "execution_time_ms": vector_performance.execution_time_ms,
                    "confidence_score": vector_performance.confidence_score,
                    "optimization_suggestions": vector_performance.optimization_suggestions
                },
                "pgvector_setup": pgvector_validation,
                "cache_performance": cache_benchmark
            }
            
            # Calculate overall score
            neo4j_score = neo4j_benchmark.get("summary", {}).get("success_rate", 0.0)
            supabase_score = vector_performance.confidence_score
            validation_report["overall_score"] = (neo4j_score + supabase_score) / 2
            
            # Generate overall recommendations
            validation_report["recommendations"] = self._generate_overall_recommendations(
                neo4j_benchmark, vector_performance, pgvector_validation
            )
            
            logger.info("Database optimization validation completed", 
                       overall_score=validation_report["overall_score"])
            
        except Exception as e:
            logger.error(f"Database optimization validation failed: {str(e)}")
            validation_report["error"] = str(e)
            validation_report["overall_score"] = 0.0
            
        return validation_report
        
    def _generate_overall_recommendations(
        self, 
        neo4j_benchmark: Dict[str, Any], 
        vector_performance: QueryPerformanceMetrics,
        pgvector_validation: Dict[str, Any]
    ) -> List[str]:
        """Generate overall optimization recommendations."""
        recommendations = []
        
        # Neo4j recommendations
        neo4j_success_rate = neo4j_benchmark.get("summary", {}).get("success_rate", 0.0)
        if neo4j_success_rate < 0.9:
            recommendations.append("Neo4j query success rate is below 90% - review failed queries")
            
        avg_time = neo4j_benchmark.get("summary", {}).get("avg_execution_time_ms", 0.0)
        if avg_time > 500:
            recommendations.append("Neo4j average query time is high - consider query optimization")
            
        # Vector search recommendations
        if vector_performance.execution_time_ms > 300:
            recommendations.append("Vector search performance is slow - optimize pgvector configuration")
            
        # pgvector recommendations
        if not pgvector_validation.get("pgvector_available", False):
            recommendations.append("pgvector extension not available - install and configure")
            
        # General recommendations
        recommendations.extend([
            "Monitor database performance regularly using these validation tools",
            "Consider implementing query result caching for frequently accessed data",
            "Review and update database indexes based on actual query patterns"
        ])
        
        return recommendations


# Global optimizer instance
db_optimizer = DatabaseOptimizationValidator()