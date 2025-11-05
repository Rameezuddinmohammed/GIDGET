# Code Intelligence System Documentation

Welcome to the Code Intelligence System - a next-generation AI-powered platform that enables developers to ask natural language questions about code evolution across multiple versions of a codebase.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [User Guides](#user-guides)
- [API Documentation](#api-documentation)
- [Deployment Guide](#deployment-guide)
- [Developer Guide](#developer-guide)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Git
- Neo4j 5.15+
- PostgreSQL 15+ with pgvector extension

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/code-intelligence.git
   cd code-intelligence
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

4. **Verify installation:**
   ```bash
   curl http://localhost:8000/api/v1/health/
   ```

### First Query

1. **Register a repository:**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/repositories/" \
        -H "Content-Type: application/json" \
        -d '{
          "url": "https://github.com/your-org/your-repo.git",
          "name": "my-repo"
        }'
   ```

2. **Submit a query:**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/queries/" \
        -H "Content-Type: application/json" \
        -d '{
          "repository_url": "https://github.com/your-org/your-repo.git",
          "query": "What changed in the authentication system between versions?",
          "options": {
            "max_commits": 50,
            "include_tests": false
          }
        }'
   ```

3. **Check results:**
   ```bash
   curl "http://localhost:8000/api/v1/queries/{query_id}"
   ```

## Architecture Overview

The Code Intelligence System is built on a three-layer architecture:

### Data Plane
- **Neo4j Graph Database**: Stores temporal Code Property Graphs
- **PostgreSQL with pgvector**: User data and semantic search
- **Redis**: Caching layer for performance optimization

### Intelligence Plane
- **Multi-Agent System**: Specialized AI agents for different analysis tasks
- **LangGraph Orchestration**: Coordinates agent workflows
- **Verification System**: Ensures accuracy of all findings

### Experience Plane
- **REST API**: Programmatic access to all functionality
- **WebSocket API**: Real-time updates during processing
- **Web Interface**: Interactive dashboard for query management
- **CLI Tool**: Command-line interface for developer workflows

## User Guides

### For Developers

- [Getting Started Guide](./user-guides/getting-started.md)
- [Query Examples](./user-guides/query-examples.md)
- [CLI Usage Guide](./user-guides/cli-guide.md)
- [Web Interface Guide](./user-guides/web-interface.md)

### For Administrators

- [Installation Guide](./admin-guides/installation.md)
- [Configuration Guide](./admin-guides/configuration.md)
- [Monitoring Guide](./admin-guides/monitoring.md)
- [Backup and Recovery](./admin-guides/backup-recovery.md)

## API Documentation

### REST API

The system provides a comprehensive REST API for all operations:

- **Base URL**: `http://localhost:8000/api/v1`
- **Authentication**: Bearer token (when enabled)
- **Rate Limiting**: 1000 requests per hour per user

#### Key Endpoints

- `POST /queries/` - Submit a new query
- `GET /queries/{id}` - Get query status and results
- `POST /repositories/` - Register a repository
- `GET /health/` - System health check
- `GET /health/metrics` - Performance metrics

[Full API Documentation](./api/README.md)

### WebSocket API

Real-time updates during query processing:

- **Endpoint**: `ws://localhost:8000/ws`
- **Authentication**: Query parameter or header
- **Message Types**: `query_progress`, `partial_results`, `query_completed`

[WebSocket Documentation](./api/websocket.md)

## Deployment Guide

### Development Environment

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Run tests
python -m pytest tests/ -v

# Start development server
python -m uvicorn src.code_intelligence.api.main:app --reload
```

### Production Deployment

#### Docker Compose
```bash
# Production deployment
docker-compose up -d

# Scale API instances
docker-compose up -d --scale api=3
```

#### Kubernetes
```bash
# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/

# Check deployment status
kubectl get pods -n code-intelligence
```

#### Cloud Deployment (AWS)
```bash
# Deploy infrastructure with Terraform
cd deployment/terraform
terraform init
terraform plan -var="environment=production"
terraform apply
```

[Detailed Deployment Guide](./deployment/README.md)

## Developer Guide

### Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes and add tests**
4. **Run the test suite**: `python -m pytest`
5. **Submit a pull request**

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Run linting
black src/ tests/
isort src/ tests/
mypy src/

# Run tests with coverage
pytest --cov=src tests/
```

### Architecture Deep Dive

- [Agent System Architecture](./developer/agents.md)
- [Database Schema](./developer/database.md)
- [Caching Strategy](./developer/caching.md)
- [Monitoring and Metrics](./developer/monitoring.md)

## Troubleshooting

### Common Issues

#### Query Processing Fails
```bash
# Check agent system health
curl http://localhost:8000/api/v1/health/components/agents

# Check logs
docker-compose logs api

# Restart services
docker-compose restart
```

#### Database Connection Issues
```bash
# Check database health
curl http://localhost:8000/api/v1/health/components/neo4j
curl http://localhost:8000/api/v1/health/components/supabase

# Reset databases
docker-compose down -v
docker-compose up -d
```

#### Performance Issues
```bash
# Check system metrics
curl http://localhost:8000/api/v1/health/metrics

# Monitor resource usage
docker stats

# Scale services
docker-compose up -d --scale api=3
```

### Getting Help

- **Documentation**: Check the relevant guide in this documentation
- **Issues**: Create an issue on GitHub with detailed information
- **Discussions**: Join our community discussions
- **Support**: Contact support@code-intelligence.example.com

### Monitoring and Alerting

The system includes comprehensive monitoring:

- **Prometheus Metrics**: Available at `/api/v1/health/prometheus`
- **Grafana Dashboards**: Pre-configured dashboards for system monitoring
- **Alertmanager**: Automated alerting for critical issues
- **Health Checks**: Kubernetes-compatible health endpoints

[Monitoring Guide](./admin-guides/monitoring.md)

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Changelog

See [CHANGELOG.md](../CHANGELOG.md) for a list of changes and version history.

## Support

For support, please:

1. Check this documentation
2. Search existing GitHub issues
3. Create a new issue with detailed information
4. Contact our support team

---

**Next Steps:**
- [Quick Start Tutorial](./user-guides/getting-started.md)
- [API Reference](./api/README.md)
- [Deployment Guide](./deployment/README.md)