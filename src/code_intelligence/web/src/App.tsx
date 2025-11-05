import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box } from '@mui/material';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import QueryPage from './pages/QueryPage';
import RepositoriesPage from './pages/RepositoriesPage';
import QueryResultsPage from './pages/QueryResultsPage';
import { WebSocketProvider } from './contexts/WebSocketContext';

function App() {
  return (
    <WebSocketProvider>
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <Navbar />
        <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/query" element={<QueryPage />} />
            <Route path="/repositories" element={<RepositoriesPage />} />
            <Route path="/query/:queryId" element={<QueryResultsPage />} />
          </Routes>
        </Box>
      </Box>
    </WebSocketProvider>
  );
}

export default App;