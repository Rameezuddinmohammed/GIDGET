import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  Link,
  Button,
  CircularProgress,
} from '@mui/material';
import {
  ExpandMore,
  Download,
  Share,
  Refresh,
} from '@mui/icons-material';
import AgentVisualization from '../components/AgentVisualization';
import { useWebSocket } from '../contexts/WebSocketContext';

interface QueryResult {
  query_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: {
    current_agent: string;
    completed_steps: string[];
    total_steps: number;
    progress_percentage: number;
    estimated_remaining_seconds?: number;
    current_step: string;
  };
  results?: {
    summary: string;
    findings: Array<{
      agent_name: string;
      finding_type: string;
      content: string;
      confidence: number;
      citations: Array<{
        file_path: string;
        line_number?: number;
        commit_sha?: string;
        url?: string;
        description: string;
      }>;
      metadata: any;
      timestamp: string;
    }>;
    confidence_score: number;
    citations: any[];
    metadata: any;
    processing_time_seconds: number;
  };
  error?: string;
  created_at: string;
  updated_at: string;
  completed_at?: string;
}

const QueryResultsPage: React.FC = () => {
  const { queryId } = useParams<{ queryId: string }>();
  const { subscribe, unsubscribe, messages } = useWebSocket();
  const [queryResult, setQueryResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!queryId) return;

    // Subscribe to WebSocket updates for this query
    subscribe(queryId);

    // Fetch initial query status
    fetchQueryStatus();

    return () => {
      unsubscribe(queryId);
    };
  }, [queryId, subscribe, unsubscribe]);

  // Listen for WebSocket messages
  useEffect(() => {
    const relevantMessages = messages.filter(msg => msg.query_id === queryId);
    const latestMessage = relevantMessages[relevantMessages.length - 1];

    if (latestMessage) {
      switch (latestMessage.type) {
        case 'query_progress':
          setQueryResult(prev => prev ? {
            ...prev,
            progress: latestMessage.data
          } : null);
          break;
        case 'query_completed':
          setQueryResult(prev => prev ? {
            ...prev,
            status: 'completed',
            results: latestMessage.data,
            completed_at: new Date().toISOString()
          } : null);
          break;
        case 'query_failed':
          setQueryResult(prev => prev ? {
            ...prev,
            status: 'failed',
            error: latestMessage.data.error
          } : null);
          break;
      }
    }
  }, [messages, queryId]);

  const fetchQueryStatus = async () => {
    if (!queryId) return;

    try {
      const response = await fetch(`/api/v1/queries/${queryId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch query status');
      }
      const data = await response.json();
      setQueryResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async (format: string) => {
    if (!queryId) return;

    try {
      const response = await fetch(`/api/v1/queries/${queryId}/export`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query_id: queryId,
          format: format,
          include_citations: true,
          include_metadata: false
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to export results');
      }

      const exportData = await response.json();
      // In a real app, you would handle the download URL
      console.log('Export created:', exportData);
    } catch (err) {
      console.error('Export failed:', err);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error || !queryResult) {
    return (
      <Alert severity="error">
        {error || 'Query not found'}
      </Alert>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 3 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Query Results
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Query ID: {queryId}
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Button
            startIcon={<Refresh />}
            onClick={fetchQueryStatus}
            disabled={queryResult.status === 'processing'}
          >
            Refresh
          </Button>
          {queryResult.status === 'completed' && (
            <>
              <Button
                startIcon={<Download />}
                onClick={() => handleExport('json')}
              >
                Export
              </Button>
              <Button
                startIcon={<Share />}
              >
                Share
              </Button>
            </>
          )}
        </Box>
      </Box>

      {/* Agent Visualization */}
      <AgentVisualization
        progress={queryResult.progress}
        status={queryResult.status}
      />

      {/* Query Status */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              Query Status
            </Typography>
            <Chip
              label={queryResult.status}
              color={
                queryResult.status === 'completed' ? 'success' :
                queryResult.status === 'processing' ? 'primary' :
                queryResult.status === 'failed' ? 'error' : 'default'
              }
            />
          </Box>
          <Typography variant="body2" color="text.secondary">
            Created: {new Date(queryResult.created_at).toLocaleString()}
          </Typography>
          {queryResult.completed_at && (
            <Typography variant="body2" color="text.secondary">
              Completed: {new Date(queryResult.completed_at).toLocaleString()}
            </Typography>
          )}
        </CardContent>
      </Card>

      {/* Error Display */}
      {queryResult.error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {queryResult.error}
        </Alert>
      )}

      {/* Results */}
      {queryResult.results && (
        <>
          {/* Summary */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Summary
              </Typography>
              <Typography variant="body1" paragraph>
                {queryResult.results.summary}
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
                <Chip
                  label={`${Math.round(queryResult.results.confidence_score * 100)}% confidence`}
                  color="success"
                  variant="outlined"
                />
                <Typography variant="caption" color="text.secondary">
                  Processing time: {Math.round(queryResult.results.processing_time_seconds)}s
                </Typography>
              </Box>
            </CardContent>
          </Card>

          {/* Detailed Findings */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Detailed Findings
              </Typography>
              {queryResult.results.findings.map((finding, index) => (
                <Accordion key={index}>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                      <Typography variant="subtitle1">
                        {finding.agent_name} - {finding.finding_type}
                      </Typography>
                      <Chip
                        label={`${Math.round(finding.confidence * 100)}% confidence`}
                        size="small"
                        color={finding.confidence > 0.8 ? 'success' : 'warning'}
                        variant="outlined"
                      />
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Typography variant="body1" paragraph>
                      {finding.content}
                    </Typography>
                    
                    {finding.citations.length > 0 && (
                      <Box sx={{ mt: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Citations:
                        </Typography>
                        <List dense>
                          {finding.citations.map((citation, citIndex) => (
                            <ListItem key={citIndex}>
                              <ListItemText
                                primary={
                                  <Link href={citation.url} target="_blank" rel="noopener">
                                    {citation.file_path}
                                    {citation.line_number && `:${citation.line_number}`}
                                  </Link>
                                }
                                secondary={citation.description}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </Box>
                    )}
                    
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                      Generated: {new Date(finding.timestamp).toLocaleString()}
                    </Typography>
                  </AccordionDetails>
                </Accordion>
              ))}
            </CardContent>
          </Card>
        </>
      )}
    </Box>
  );
};

export default QueryResultsPage;