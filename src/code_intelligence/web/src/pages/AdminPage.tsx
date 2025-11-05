import React, { useState, useEffect } from 'react';

interface HealthSummary {
  status: string;
  success_rate: number;
  avg_response_time_ms: number;
  active_executions: number;
  total_executions_24h: number;
  timestamp: string;
}

interface AgentMetrics {
  total_executions: number;
  success_rate: number;
  avg_duration_ms: number;
  avg_confidence: number;
  error_rate: number;
}

interface SystemMetrics {
  total_executions: number;
  success_rate: number;
  avg_duration_ms: number;
  active_executions: number;
  agents: Record<string, AgentMetrics>;
  last_updated: string;
}

interface CacheMetrics {
  cache_hits: number;
  cache_misses: number;
  cache_hit_rate: number;
  avg_cache_lookup_ms: number;
  cache_size: number;
  cache_evictions: number;
}

interface MetricsData {
  health: HealthSummary;
  system: SystemMetrics;
  cache: CacheMetrics;
  timestamp: string;
}

const AdminPage: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchMetrics = async () => {
    try {
      const response = await fetch('/api/v1/health/metrics');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setMetrics(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch metrics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, []);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(fetchMetrics, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [autoRefresh]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-600';
      case 'degraded': return 'text-yellow-600';
      case 'slow': return 'text-orange-600';
      case 'unhealthy': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${Math.round(ms)}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading metrics...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-600 text-xl mb-4">⚠️ Error</div>
          <p className="text-gray-600 mb-4">{error}</p>
          <button
            onClick={fetchMetrics}
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!metrics) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <p className="text-gray-600">No metrics data available</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex justify-between items-center">
            <h1 className="text-3xl font-bold text-gray-900">System Metrics</h1>
            <div className="flex items-center space-x-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                  className="mr-2"
                />
                Auto-refresh (30s)
              </label>
              <button
                onClick={fetchMetrics}
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
              >
                Refresh Now
              </button>
            </div>
          </div>
          <p className="text-gray-600 mt-2">
            Last updated: {new Date(metrics.timestamp).toLocaleString()}
          </p>
        </div>

        {/* Health Overview */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">System Status</h3>
            <div className={`text-2xl font-bold ${getStatusColor(metrics.health.status)}`}>
              {metrics.health.status.toUpperCase()}
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Success Rate</h3>
            <div className="text-2xl font-bold text-blue-600">
              {formatPercentage(metrics.health.success_rate)}
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Avg Response Time</h3>
            <div className="text-2xl font-bold text-purple-600">
              {formatDuration(metrics.health.avg_response_time_ms)}
            </div>
          </div>
          
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Active Executions</h3>
            <div className="text-2xl font-bold text-green-600">
              {metrics.health.active_executions}
            </div>
          </div>
        </div>

        {/* System Metrics */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">System Overview</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Total Executions:</span>
                <span className="font-semibold">{metrics.system.total_executions}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Success Rate:</span>
                <span className="font-semibold">{formatPercentage(metrics.system.success_rate)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Avg Duration:</span>
                <span className="font-semibold">{formatDuration(metrics.system.avg_duration_ms)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">24h Executions:</span>
                <span className="font-semibold">{metrics.health.total_executions_24h}</span>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">Cache Performance</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Hit Rate:</span>
                <span className="font-semibold">{formatPercentage(metrics.cache.cache_hit_rate)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Cache Hits:</span>
                <span className="font-semibold">{metrics.cache.cache_hits}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Cache Misses:</span>
                <span className="font-semibold">{metrics.cache.cache_misses}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Cache Size:</span>
                <span className="font-semibold">{metrics.cache.cache_size}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Avg Lookup Time:</span>
                <span className="font-semibold">{formatDuration(metrics.cache.avg_cache_lookup_ms)}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Agent Performance */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-4">Agent Performance</h3>
          {Object.keys(metrics.system.agents).length === 0 ? (
            <p className="text-gray-600">No agent metrics available</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Agent
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Executions
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Success Rate
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Avg Duration
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Avg Confidence
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Error Rate
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {Object.entries(metrics.system.agents).map(([agentName, agentMetrics]) => (
                    <tr key={agentName}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {agentName}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {agentMetrics.total_executions}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatPercentage(agentMetrics.success_rate)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatDuration(agentMetrics.avg_duration_ms)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatPercentage(agentMetrics.avg_confidence)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <span className={agentMetrics.error_rate > 0.1 ? 'text-red-600' : 'text-green-600'}>
                          {formatPercentage(agentMetrics.error_rate)}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AdminPage;