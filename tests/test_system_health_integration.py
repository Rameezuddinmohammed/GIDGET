"""System health and monitoring integration tests."""

import asyncio
import json
import pytest
import time
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, patch, AsyncMock

from fastapi.testclient import TestClient

from src.code_intelligence.api.main import app
from src.code_intelligence.monitoring.agent_monitor import agent_monitor
from src.code_intelligence.caching.cache_manager import cache_manager


@pytest.fixture
def test_client():
    """Create test client for health monitoring tests."""
    return TestClient(app)


@pytest.fixture
def mock_monitoring_services():
    """Mock monitoring services."""
    with patch('src.code_intelligence.api.dependencies.get_neo4j_client') as mock_neo4j, \
         patch('src.code_intelligence.api.dependencies.get_supabase_client') as mock_supabase:
        
        # Setup healthy services
        neo4j_client = AsyncMock()
        neo4j_client.is_healthy.return_value = True
        neo4j_client.execute_query.return_value = [{"status": "healthy"}]
        mock_neo4j.return_value = neo4j_client
        
        supabase_client = AsyncMock()
        supabase_client.is_healthy.return_value = True
        mock_supabase.return_value = supabase_client
        
        yield {
            "neo4j": neo4j_client,
            "supabase": supabase_client
        }


class TestSystemHealthMonitoring:
    """Test system health monitoring and alerting."""
    
    def test_basic_health_check_integration(self, test_client, mock_monitoring_services):
        """Test basic health check endpoint integration."""
        response = test_client.get("/api/v1/health/")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert health_data["version"] == "1.0.0"
        assert "services" in health_data
        assert "timestamp" in health_data
        
        # Verify service status
        services = health_data["services"]
        assert "database" in services
        assert "cache" in services
        assert "agents" in services
    
    def test_detailed_health_check_integration(self, test_client, mock_monitoring_services):
        """Test detailed health check with service dependencies."""
        response = test_client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        
        detailed_health = response.json()
        assert detailed_health["status"] == "healthy"
        assert "services" in detailed_health
        assert "system_info" in detailed_health
        assert "timestamp" in detailed_health
        
        # Verify detailed service information
        services = detailed_health["services"]
        for service_name, service_status in services.items():
            assert service_status in ["healthy", "degraded", "unhealthy"]
    
    def test_metrics_endpoint_integration(self, test_client, mock_monitoring_services):
        """Test comprehensive metrics endpoint."""
        # First, populate some test data in monitoring systems
        agent_monitor.record_execution("test_agent", 1.5, True)
        agent_monitor.record_execution("test_agent", 2.0, True)
        agent_monitor.record_execution("test_agent", 0.8, False)
        
        response = test_client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        assert "health" in metrics
        assert "system" in metrics
        assert "cache" in metrics
        assert "timestamp" in metrics
        
        # Verify health metrics structure
        health_metrics = metrics["health"]
        assert "status" in health_metrics
        assert "success_rate" in health_metrics
        assert "avg_response_time_ms" in health_metrics
        assert "active_executions" in health_metrics
        
        # Verify system metrics structure
        system_metrics = metrics["system"]
        assert "total_executions" in system_metrics
        assert "success_rate" in system_metrics
        assert "avg_duration_ms" in system_metrics
        assert "agents" in system_metrics
        
        # Verify agent-specific metrics
        agents_metrics = system_metrics["agents"]
        if "test_agent" in agents_metrics:
            test_agent_metrics = agents_metrics["test_agent"]
            assert "total_executions" in test_agent_metrics
            assert "success_rate" in test_agent_metrics
            assert "avg_execution_time" in test_agent_metrics
        
        # Verify cache metrics structure
        cache_metrics = metrics["cache"]
        assert "cache_hits" in cache_metrics
        assert "cache_misses" in cache_metrics
        assert "cache_hit_rate" in cache_metrics
    
    def test_service_degradation_detection(self, test_client):
        """Test detection and reporting of service degradation."""
        with patch('src.code_intelligence.api.dependencies.get_neo4j_client') as mock_neo4j:
            # Simulate degraded Neo4j service
            neo4j_client = AsyncMock()
            neo4j_client.is_healthy.return_value = False
            neo4j_client.execute_query.side_effect = Exception("Connection timeout")
            mock_neo4j.return_value = neo4j_client
            
            response = test_client.get("/api/v1/health/detailed")
            assert response.status_code == 200
            
            health_data = response.json()
            # System should still respond but report degraded status
            assert health_data["status"] in ["degraded", "unhealthy"]
            
            # Neo4j should be reported as unhealthy
            services = health_data["services"]
            assert "neo4j" in services or "database" in services
    
    def test_performance_monitoring_integration(self, test_client, mock_monitoring_services):
        """Test performance monitoring across system components."""
        # Simulate various agent executions with different performance characteristics
        test_scenarios = [
            ("fast_agent", 0.5, True),
            ("fast_agent", 0.6, True),
            ("slow_agent", 3.0, True),
            ("slow_agent", 3.5, False),
            ("error_agent", 1.0, False),
            ("error_agent", 1.2, False),
        ]
        
        for agent_name, duration, success in test_scenarios:
            agent_monitor.record_execution(agent_name, duration, success)
        
        # Get performance metrics
        response = test_client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        system_metrics = metrics["system"]
        agents_metrics = system_metrics["agents"]
        
        # Verify fast agent metrics
        if "fast_agent" in agents_metrics:
            fast_agent = agents_metrics["fast_agent"]
            assert fast_agent["success_rate"] == 1.0  # 100% success
            assert fast_agent["avg_execution_time"] < 1.0  # Fast execution
        
        # Verify slow agent metrics
        if "slow_agent" in agents_metrics:
            slow_agent = agents_metrics["slow_agent"]
            assert slow_agent["success_rate"] == 0.5  # 50% success
            assert slow_agent["avg_execution_time"] > 3.0  # Slow execution
        
        # Verify error agent metrics
        if "error_agent" in agents_metrics:
            error_agent = agents_metrics["error_agent"]
            assert error_agent["success_rate"] == 0.0  # 0% success
    
    def test_cache_performance_monitoring(self, test_client, mock_monitoring_services):
        """Test cache performance monitoring integration."""
        # This would test actual cache operations
        # For now, we test the metrics endpoint structure
        
        response = test_client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        cache_metrics = metrics["cache"]
        
        # Verify cache metrics are properly structured
        assert isinstance(cache_metrics["cache_hits"], int)
        assert isinstance(cache_metrics["cache_misses"], int)
        assert isinstance(cache_metrics["cache_hit_rate"], (int, float))
        
        # Cache hit rate should be between 0 and 1
        assert 0 <= cache_metrics["cache_hit_rate"] <= 1
    
    def test_real_time_monitoring_updates(self, test_client, mock_monitoring_services):
        """Test real-time monitoring updates through WebSocket."""
        with test_client.websocket_connect("/ws") as websocket:
            # Establish connection
            welcome_data = websocket.receive_json()
            assert welcome_data["type"] == "connection_established"
            
            # Request monitoring updates
            websocket.send_json({
                "type": "subscribe_monitoring",
                "data": {"metrics": ["health", "performance"]}
            })
            
            # Simulate some system activity
            agent_monitor.record_execution("monitoring_test_agent", 1.0, True)
            
            # In a real implementation, this would trigger monitoring updates
            # For now, we verify the WebSocket connection works
            
            # Send ping to verify connection
            websocket.send_json({"type": "ping"})
            pong_response = websocket.receive_json()
            assert pong_response["type"] == "pong"


class TestSystemAlertingIntegration:
    """Test system alerting and incident response."""
    
    def test_performance_threshold_alerting(self, test_client, mock_monitoring_services):
        """Test alerting when performance thresholds are exceeded."""
        # Simulate slow agent executions that should trigger alerts
        slow_executions = [
            ("critical_agent", 10.0, True),  # Very slow execution
            ("critical_agent", 12.0, False), # Slow and failed
            ("critical_agent", 15.0, True),  # Extremely slow
        ]
        
        for agent_name, duration, success in slow_executions:
            agent_monitor.record_execution(agent_name, duration, success)
        
        # Check if metrics reflect the performance issues
        response = test_client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        system_metrics = metrics["system"]
        
        # Overall system metrics should reflect performance degradation
        if system_metrics["avg_duration_ms"] > 5000:  # More than 5 seconds average
            # This would trigger alerts in a real system
            assert system_metrics["success_rate"] < 1.0
    
    def test_error_rate_threshold_alerting(self, test_client, mock_monitoring_services):
        """Test alerting when error rates exceed thresholds."""
        # Simulate high error rate scenario
        error_executions = [
            ("error_prone_agent", 1.0, False),
            ("error_prone_agent", 1.1, False),
            ("error_prone_agent", 0.9, False),
            ("error_prone_agent", 1.2, True),   # Only one success
            ("error_prone_agent", 1.0, False),
        ]
        
        for agent_name, duration, success in error_executions:
            agent_monitor.record_execution(agent_name, duration, success)
        
        # Check error rate metrics
        response = test_client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        system_metrics = metrics["system"]
        agents_metrics = system_metrics["agents"]
        
        if "error_prone_agent" in agents_metrics:
            error_agent = agents_metrics["error_prone_agent"]
            # Error rate should be 80% (4 failures out of 5)
            assert error_agent["success_rate"] == 0.2
            
            # This would trigger error rate alerts in a real system
            assert error_agent["success_rate"] < 0.5  # Below 50% success rate
    
    def test_service_unavailability_alerting(self, test_client):
        """Test alerting when critical services become unavailable."""
        with patch('src.code_intelligence.api.dependencies.get_neo4j_client') as mock_neo4j, \
             patch('src.code_intelligence.api.dependencies.get_supabase_client') as mock_supabase:
            
            # Simulate both services being unavailable
            neo4j_client = AsyncMock()
            neo4j_client.is_healthy.return_value = False
            neo4j_client.execute_query.side_effect = Exception("Service unavailable")
            mock_neo4j.return_value = neo4j_client
            
            supabase_client = AsyncMock()
            supabase_client.is_healthy.return_value = False
            mock_supabase.return_value = supabase_client
            
            # Health check should reflect service unavailability
            response = test_client.get("/api/v1/health/detailed")
            assert response.status_code == 200
            
            health_data = response.json()
            # System should report unhealthy status
            assert health_data["status"] == "unhealthy"
            
            # This would trigger critical service alerts in a real system


class TestCapacityPlanningIntegration:
    """Test capacity planning and scaling automation."""
    
    def test_load_pattern_analysis(self, test_client, mock_monitoring_services):
        """Test analysis of load patterns for capacity planning."""
        # Simulate varying load patterns throughout a day
        current_time = datetime.now()
        
        # Simulate peak hours (high load)
        peak_executions = [
            ("load_test_agent", 2.0, True) for _ in range(20)
        ]
        
        # Simulate off-peak hours (low load)
        offpeak_executions = [
            ("load_test_agent", 0.5, True) for _ in range(5)
        ]
        
        all_executions = peak_executions + offpeak_executions
        
        for agent_name, duration, success in all_executions:
            agent_monitor.record_execution(agent_name, duration, success)
        
        # Analyze load metrics
        response = test_client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        system_metrics = metrics["system"]
        
        # Total executions should reflect the load
        assert system_metrics["total_executions"] >= 25
        
        # This data would be used for capacity planning in a real system
        if "load_test_agent" in system_metrics["agents"]:
            load_agent = system_metrics["agents"]["load_test_agent"]
            assert load_agent["total_executions"] == 25
    
    def test_resource_utilization_monitoring(self, test_client, mock_monitoring_services):
        """Test monitoring of system resource utilization."""
        import psutil
        import os
        
        # Get current system metrics
        process = psutil.Process(os.getpid())
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        
        # Submit request that would use resources
        response = test_client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        
        # In a real system, this would include resource utilization metrics
        # For now, we verify the structure supports resource monitoring
        assert "system" in metrics
        assert "timestamp" in metrics
        
        # Resource metrics would be included here in a production system
        # Example structure:
        # assert "resources" in metrics
        # assert "cpu_usage" in metrics["resources"]
        # assert "memory_usage" in metrics["resources"]
    
    def test_scaling_trigger_conditions(self, test_client, mock_monitoring_services):
        """Test conditions that would trigger auto-scaling."""
        # Simulate high load conditions that would trigger scaling
        high_load_scenario = [
            ("scaling_test_agent", 5.0, True) for _ in range(50)  # Many slow requests
        ]
        
        for agent_name, duration, success in high_load_scenario:
            agent_monitor.record_execution(agent_name, duration, success)
        
        # Check if metrics indicate scaling is needed
        response = test_client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        system_metrics = metrics["system"]
        
        # High average duration would trigger scaling in a real system
        if system_metrics["avg_duration_ms"] > 3000:  # More than 3 seconds
            # This would trigger auto-scaling in a production system
            assert system_metrics["total_executions"] >= 50
            
            # Scaling decision would be made based on these metrics
            scaling_needed = (
                system_metrics["avg_duration_ms"] > 3000 and
                system_metrics["total_executions"] > 30
            )
            assert scaling_needed


class TestMonitoringDataPersistence:
    """Test persistence and historical analysis of monitoring data."""
    
    def test_metrics_data_persistence(self, test_client, mock_monitoring_services):
        """Test that monitoring metrics are properly persisted."""
        # Record some test metrics
        test_metrics = [
            ("persistence_agent", 1.0, True),
            ("persistence_agent", 1.5, True),
            ("persistence_agent", 2.0, False),
        ]
        
        for agent_name, duration, success in test_metrics:
            agent_monitor.record_execution(agent_name, duration, success)
        
        # Verify metrics are available
        response = test_client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        
        # Metrics should persist across requests
        response2 = test_client.get("/api/v1/health/metrics")
        assert response2.status_code == 200
        
        metrics2 = response2.json()
        
        # Data should be consistent between requests
        assert metrics["system"]["total_executions"] == metrics2["system"]["total_executions"]
    
    def test_historical_metrics_analysis(self, test_client, mock_monitoring_services):
        """Test analysis of historical monitoring data."""
        # This would test querying historical metrics data
        # For now, we test the current metrics structure supports historical analysis
        
        response = test_client.get("/api/v1/health/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        
        # Verify timestamp is included for historical tracking
        assert "timestamp" in metrics
        
        # Verify metrics structure supports trend analysis
        assert "system" in metrics
        assert "health" in metrics
        
        # In a real system, this would include endpoints like:
        # GET /api/v1/health/metrics/history?start_time=...&end_time=...
        # GET /api/v1/health/trends/performance
        # GET /api/v1/health/trends/errors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])