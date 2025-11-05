import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box, Chip } from '@mui/material';
import { Link, useLocation } from 'react-router-dom';
import { Code, Dashboard, Search, Storage, AdminPanelSettings } from '@mui/icons-material';
import { useWebSocket } from '../contexts/WebSocketContext';

const Navbar: React.FC = () => {
  const location = useLocation();
  const { isConnected } = useWebSocket();

  const navItems = [
    { path: '/', label: 'Dashboard', icon: <Dashboard /> },
    { path: '/query', label: 'Query', icon: <Search /> },
    { path: '/repositories', label: 'Repositories', icon: <Storage /> },
    { path: '/admin', label: 'Admin', icon: <AdminPanelSettings /> },
  ];

  return (
    <AppBar position="static" sx={{ bgcolor: 'background.paper', borderBottom: 1, borderColor: 'divider' }}>
      <Toolbar>
        <Box sx={{ display: 'flex', alignItems: 'center', mr: 4 }}>
          <Code sx={{ mr: 1 }} />
          <Typography variant="h6" component="div" sx={{ fontWeight: 'bold' }}>
            Code Intelligence
          </Typography>
        </Box>

        <Box sx={{ flexGrow: 1, display: 'flex', gap: 1 }}>
          {navItems.map((item) => (
            <Button
              key={item.path}
              component={Link}
              to={item.path}
              startIcon={item.icon}
              variant={location.pathname === item.path ? 'contained' : 'text'}
              sx={{ textTransform: 'none' }}
            >
              {item.label}
            </Button>
          ))}
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Chip
            label={isConnected ? 'Connected' : 'Disconnected'}
            color={isConnected ? 'success' : 'error'}
            size="small"
            variant="outlined"
          />
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;