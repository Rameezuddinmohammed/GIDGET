import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  Stepper,
  Step,
  StepLabel,
  StepContent,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
} from '@mui/material';
import {
  Psychology,
  History,
  Analytics,
  AutoFixHigh,
  VerifiedUser,
  CheckCircle,
  RadioButtonUnchecked,
  Error,
} from '@mui/icons-material';

interface AgentProgress {
  current_agent: string;
  completed_steps: string[];
  total_steps: number;
  progress_percentage: number;
  estimated_remaining_seconds?: number;
  current_step: string;
}

interface AgentVisualizationProps {
  progress?: AgentProgress;
  status: 'pending' | 'processing' | 'completed' | 'failed';
}

const agentConfig = {
  orchestrator: {
    name: 'Orchestrator',
    icon: <Psychology />,
    color: '#2196f3',
    description: 'Parsing query and coordinating workflow'
  },
  historian: {
    name: 'Historian',
    icon: <History />,
    color: '#ff9800',
    description: 'Analyzing git history and temporal changes'
  },
  analyst: {
    name: 'Analyst',
    icon: <Analytics />,
    color: '#4caf50',
    description: 'Performing deep code analysis'
  },
  synthesizer: {
    name: 'Synthesizer',
    icon: <AutoFixHigh />,
    color: '#9c27b0',
    description: 'Compiling and synthesizing results'
  },
  verifier: {
    name: 'Verifier',
    icon: <VerifiedUser />,
    color: '#f44336',
    description: 'Validating findings and ensuring accuracy'
  }
};

const AgentVisualization: React.FC<AgentVisualizationProps> = ({ progress, status }) => {
  const agents = Object.keys(agentConfig);
  const currentAgentIndex = progress ? agents.indexOf(progress.current_agent) : -1;

  const getStepIcon = (agentName: string) => {
    if (!progress) {
      return <RadioButtonUnchecked />;
    }

    if (progress.completed_steps.includes(agentName)) {
      return <CheckCircle color="success" />;
    } else if (progress.current_agent === agentName) {
      return agentConfig[agentName as keyof typeof agentConfig]?.icon || <RadioButtonUnchecked />;
    } else if (status === 'failed') {
      return <Error color="error" />;
    } else {
      return <RadioButtonUnchecked />;
    }
  };

  const getStepStatus = (agentName: string) => {
    if (!progress) return 'pending';
    
    if (progress.completed_steps.includes(agentName)) {
      return 'completed';
    } else if (progress.current_agent === agentName) {
      return 'active';
    } else if (status === 'failed') {
      return 'error';
    } else {
      return 'pending';
    }
  };

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Agent Execution Progress
        </Typography>

        {progress && (
          <Box sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
              <Typography variant="body2" color="text.secondary">
                Overall Progress
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {Math.round(progress.progress_percentage)}%
              </Typography>
            </Box>
            <LinearProgress 
              variant="determinate" 
              value={progress.progress_percentage} 
              sx={{ height: 8, borderRadius: 4 }}
            />
            {progress.estimated_remaining_seconds && (
              <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                Estimated time remaining: {Math.round(progress.estimated_remaining_seconds / 60)} minutes
              </Typography>
            )}
          </Box>
        )}

        <List>
          {agents.map((agentName) => {
            const agent = agentConfig[agentName as keyof typeof agentConfig];
            const stepStatus = getStepStatus(agentName);
            
            return (
              <ListItem key={agentName} sx={{ py: 1 }}>
                <ListItemAvatar>
                  <Avatar 
                    sx={{ 
                      bgcolor: stepStatus === 'completed' ? 'success.main' : 
                               stepStatus === 'active' ? agent.color : 
                               stepStatus === 'error' ? 'error.main' : 'grey.500',
                      width: 40,
                      height: 40
                    }}
                  >
                    {getStepIcon(agentName)}
                  </Avatar>
                </ListItemAvatar>
                <ListItemText
                  primary={
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Typography variant="subtitle1">
                        {agent.name}
                      </Typography>
                      <Chip
                        label={stepStatus}
                        size="small"
                        color={
                          stepStatus === 'completed' ? 'success' :
                          stepStatus === 'active' ? 'primary' :
                          stepStatus === 'error' ? 'error' : 'default'
                        }
                        variant="outlined"
                      />
                    </Box>
                  }
                  secondary={
                    <Box>
                      <Typography variant="body2" color="text.secondary">
                        {agent.description}
                      </Typography>
                      {progress && progress.current_agent === agentName && (
                        <Typography variant="caption" color="primary" sx={{ fontStyle: 'italic' }}>
                          Current step: {progress.current_step}
                        </Typography>
                      )}
                    </Box>
                  }
                />
              </ListItem>
            );
          })}
        </List>

        {status === 'completed' && (
          <Box sx={{ mt: 2, p: 2, bgcolor: 'success.dark', borderRadius: 1 }}>
            <Typography variant="body2" color="success.contrastText">
              ✅ All agents completed successfully! Results are ready for review.
            </Typography>
          </Box>
        )}

        {status === 'failed' && (
          <Box sx={{ mt: 2, p: 2, bgcolor: 'error.dark', borderRadius: 1 }}>
            <Typography variant="body2" color="error.contrastText">
              ❌ Analysis failed. Please check the error details and try again.
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default AgentVisualization;