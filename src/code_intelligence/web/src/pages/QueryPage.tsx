import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Alert,
  CircularProgress,
} from '@mui/material';
import { Send, GitHub } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const QueryPage: React.FC = () => {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [repositoryUrl, setRepositoryUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Example queries for inspiration
  const exampleQueries = [
    'What changed in the authentication system since last week?',
    'Find all functions that call the database connection',
    'How did the UserService class evolve over time?',
    'What are the dependencies of the payment module?',
    'Show me recent changes to the API endpoints',
    'Which functions were modified in the last 5 commits?'
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !repositoryUrl.trim()) {
      setError('Please provide both a query and repository URL');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/queries/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          repository_url: repositoryUrl,
          query: query,
          options: {
            max_commits: 100,
            include_tests: false
          }
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to submit query');
      }

      const result = await response.json();
      navigate(`/query/${result.query_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (exampleQuery: string) => {
    setQuery(exampleQuery);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Ask a Question
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Ask natural language questions about your code evolution. Our AI agents will analyze your repository and provide detailed insights.
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <form onSubmit={handleSubmit}>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              <TextField
                label="Repository URL"
                value={repositoryUrl}
                onChange={(e) => setRepositoryUrl(e.target.value)}
                placeholder="https://github.com/user/repository.git"
                fullWidth
                required
                InputProps={{
                  startAdornment: <GitHub sx={{ mr: 1, color: 'text.secondary' }} />
                }}
                helperText="Enter the Git repository URL you want to analyze"
              />

              <TextField
                label="Your Question"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="What changed in the authentication system since last week?"
                multiline
                rows={4}
                fullWidth
                required
                helperText="Ask about code changes, dependencies, evolution, or any aspect of your codebase"
              />

              {error && (
                <Alert severity="error">
                  {error}
                </Alert>
              )}

              <Button
                type="submit"
                variant="contained"
                size="large"
                startIcon={loading ? <CircularProgress size={20} /> : <Send />}
                disabled={loading || !query.trim() || !repositoryUrl.trim()}
                sx={{ alignSelf: 'flex-start' }}
              >
                {loading ? 'Analyzing...' : 'Analyze Code'}
              </Button>
            </Box>
          </form>
        </CardContent>
      </Card>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Example Questions
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Click on any example to use it as a starting point:
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {exampleQueries.map((example, index) => (
              <Chip
                key={index}
                label={example}
                variant="outlined"
                clickable
                onClick={() => handleExampleClick(example)}
                sx={{ 
                  height: 'auto',
                  py: 1,
                  '& .MuiChip-label': {
                    whiteSpace: 'normal',
                    textAlign: 'left'
                  }
                }}
              />
            ))}
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default QueryPage;