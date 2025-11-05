import React from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  Chip,
  Avatar,
} from '@mui/material';
import { Link } from 'react-router-dom';
import {
  Search,
  Storage,
  TrendingUp,
  Speed,
  History,
  Code,
} from '@mui/icons-material';

const Dashboard: React.FC = () => {
  // Mock data - in real app, fetch from API
  const recentQueries = [
    {
      id: '1',
      query: 'What changed in the authentication system since last week?',
      status: 'completed',
      confidence: 0.92,
      timestamp: '2 hours ago'
    },
    {
      id: '2', 
      query: 'Find all functions that call the database connection',
      status: 'processing',
      confidence: null,
      timestamp: '5 minutes ago'
    },
    {
      id: '3',
      query: 'How did the UserService class evolve over time?',
      status: 'completed',
      confidence: 0.87,
      timestamp: '1 day ago'
    }
  ];

  const stats = {
    totalQueries: 156,
    successfulQueries: 142,
    repositoriesAnalyzed: 12,
    averageConfidence: 0.89
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Code Intelligence Dashboard
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Welcome to the Multi-Agent Code Intelligence System. Ask natural language questions about your code evolution.
      </Typography>

      <Grid container spacing={3}>
        {/* Quick Actions */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Quick Actions
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Button
                  component={Link}
                  to="/query"
                  variant="contained"
                  startIcon={<Search />}
                  size="large"
                  fullWidth
                >
                  Ask a Question
                </Button>
                <Button
                  component={Link}
                  to="/repositories"
                  variant="outlined"
                  startIcon={<Storage />}
                  size="large"
                  fullWidth
                >
                  Manage Repositories
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Statistics */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Usage Statistics
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={6}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Avatar sx={{ bgcolor: 'primary.main', mx: 'auto', mb: 1 }}>
                      <TrendingUp />
                    </Avatar>
                    <Typography variant="h4">{stats.totalQueries}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Total Queries
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Avatar sx={{ bgcolor: 'success.main', mx: 'auto', mb: 1 }}>
                      <Speed />
                    </Avatar>
                    <Typography variant="h4">{Math.round(stats.averageConfidence * 100)}%</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Avg Confidence
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Avatar sx={{ bgcolor: 'info.main', mx: 'auto', mb: 1 }}>
                      <Storage />
                    </Avatar>
                    <Typography variant="h4">{stats.repositoriesAnalyzed}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Repositories
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={6}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Avatar sx={{ bgcolor: 'warning.main', mx: 'auto', mb: 1 }}>
                      <Code />
                    </Avatar>
                    <Typography variant="h4">{stats.successfulQueries}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      Successful
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Queries */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Recent Queries
                </Typography>
                <Button
                  component={Link}
                  to="/query"
                  startIcon={<History />}
                  size="small"
                >
                  View All
                </Button>
              </Box>
              <List>
                {recentQueries.map((query) => (
                  <ListItem
                    key={query.id}
                    component={Link}
                    to={`/query/${query.id}`}
                    sx={{
                      border: 1,
                      borderColor: 'divider',
                      borderRadius: 1,
                      mb: 1,
                      textDecoration: 'none',
                      color: 'inherit',
                      '&:hover': {
                        bgcolor: 'action.hover'
                      }
                    }}
                  >
                    <ListItemText
                      primary={query.query}
                      secondary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mt: 1 }}>
                          <Chip
                            label={query.status}
                            size="small"
                            color={query.status === 'completed' ? 'success' : 'primary'}
                            variant="outlined"
                          />
                          {query.confidence && (
                            <Chip
                              label={`${Math.round(query.confidence * 100)}% confidence`}
                              size="small"
                              variant="outlined"
                            />
                          )}
                          <Typography variant="caption" color="text.secondary">
                            {query.timestamp}
                          </Typography>
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;