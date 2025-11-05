import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Alert,
} from '@mui/material';
import {
  Add,
  Delete,
  Refresh,
  GitHub,
  Storage,
  Code,
} from '@mui/icons-material';

interface Repository {
  id: string;
  name: string;
  url: string;
  status: 'not_analyzed' | 'analyzing' | 'analyzed' | 'analysis_failed';
  last_analyzed?: string;
  commit_count: number;
  supported_languages: string[];
  file_count: number;
  lines_of_code: number;
}

const RepositoriesPage: React.FC = () => {
  const [repositories, setRepositories] = useState<Repository[]>([
    {
      id: '1',
      name: 'my-web-app',
      url: 'https://github.com/user/my-web-app.git',
      status: 'analyzed',
      last_analyzed: '2 hours ago',
      commit_count: 245,
      supported_languages: ['TypeScript', 'JavaScript', 'Python'],
      file_count: 156,
      lines_of_code: 12450
    },
    {
      id: '2',
      name: 'api-service',
      url: 'https://github.com/user/api-service.git',
      status: 'analyzing',
      commit_count: 0,
      supported_languages: [],
      file_count: 0,
      lines_of_code: 0
    }
  ]);

  const [dialogOpen, setDialogOpen] = useState(false);
  const [newRepoUrl, setNewRepoUrl] = useState('');
  const [newRepoName, setNewRepoName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAddRepository = async () => {
    if (!newRepoUrl.trim()) {
      setError('Repository URL is required');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/v1/repositories/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: newRepoUrl,
          name: newRepoName || undefined,
          auto_sync: true
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to add repository');
      }

      const newRepo = await response.json();
      setRepositories(prev => [...prev, newRepo]);
      setDialogOpen(false);
      setNewRepoUrl('');
      setNewRepoName('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteRepository = async (repoId: string) => {
    try {
      const response = await fetch(`/api/v1/repositories/${repoId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        throw new Error('Failed to delete repository');
      }

      setRepositories(prev => prev.filter(repo => repo.id !== repoId));
    } catch (err) {
      console.error('Failed to delete repository:', err);
    }
  };

  const handleAnalyzeRepository = async (repoId: string) => {
    try {
      const response = await fetch(`/api/v1/repositories/${repoId}/analyze`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('Failed to trigger analysis');
      }

      // Update repository status
      setRepositories(prev => 
        prev.map(repo => 
          repo.id === repoId 
            ? { ...repo, status: 'analyzing' as const }
            : repo
        )
      );
    } catch (err) {
      console.error('Failed to trigger analysis:', err);
    }
  };

  const getStatusColor = (status: Repository['status']) => {
    switch (status) {
      case 'analyzed': return 'success';
      case 'analyzing': return 'primary';
      case 'analysis_failed': return 'error';
      default: return 'default';
    }
  };

  const getStatusLabel = (status: Repository['status']) => {
    switch (status) {
      case 'analyzed': return 'Analyzed';
      case 'analyzing': return 'Analyzing...';
      case 'analysis_failed': return 'Failed';
      default: return 'Not Analyzed';
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Repositories
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Manage your code repositories for analysis
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<Add />}
          onClick={() => setDialogOpen(true)}
        >
          Add Repository
        </Button>
      </Box>

      {repositories.length === 0 ? (
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 6 }}>
            <Storage sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              No repositories added yet
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              Add your first repository to start analyzing code evolution
            </Typography>
            <Button
              variant="contained"
              startIcon={<Add />}
              onClick={() => setDialogOpen(true)}
            >
              Add Repository
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <List>
            {repositories.map((repo, index) => (
              <ListItem
                key={repo.id}
                divider={index < repositories.length - 1}
                sx={{ py: 2 }}
              >
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                      <GitHub />
                      <Typography variant="h6">{repo.name}</Typography>
                      <Chip
                        label={getStatusLabel(repo.status)}
                        color={getStatusColor(repo.status)}
                        size="small"
                      />
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        {repo.url}
                      </Typography>
                      {repo.status === 'analyzed' && (
                        <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mt: 1 }}>
                          <Chip
                            icon={<Code />}
                            label={`${repo.commit_count} commits`}
                            size="small"
                            variant="outlined"
                          />
                          <Chip
                            label={`${repo.file_count} files`}
                            size="small"
                            variant="outlined"
                          />
                          <Chip
                            label={`${repo.lines_of_code.toLocaleString()} LOC`}
                            size="small"
                            variant="outlined"
                          />
                          {repo.supported_languages.map(lang => (
                            <Chip
                              key={lang}
                              label={lang}
                              size="small"
                              variant="outlined"
                            />
                          ))}
                        </Box>
                      )}
                      {repo.last_analyzed && (
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                          Last analyzed: {repo.last_analyzed}
                        </Typography>
                      )}
                    </Box>
                  }
                />
                <ListItemSecondaryAction>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    {repo.status !== 'analyzing' && (
                      <IconButton
                        onClick={() => handleAnalyzeRepository(repo.id)}
                        title="Analyze Repository"
                      >
                        <Refresh />
                      </IconButton>
                    )}
                    <IconButton
                      onClick={() => handleDeleteRepository(repo.id)}
                      title="Delete Repository"
                      color="error"
                    >
                      <Delete />
                    </IconButton>
                  </Box>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
        </Card>
      )}

      {/* Add Repository Dialog */}
      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Add Repository</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
            <TextField
              label="Repository URL"
              value={newRepoUrl}
              onChange={(e) => setNewRepoUrl(e.target.value)}
              placeholder="https://github.com/user/repository.git"
              fullWidth
              required
            />
            <TextField
              label="Display Name (optional)"
              value={newRepoName}
              onChange={(e) => setNewRepoName(e.target.value)}
              placeholder="my-awesome-project"
              fullWidth
            />
            {error && (
              <Alert severity="error">
                {error}
              </Alert>
            )}
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleAddRepository}
            variant="contained"
            disabled={loading || !newRepoUrl.trim()}
          >
            {loading ? 'Adding...' : 'Add Repository'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default RepositoriesPage;