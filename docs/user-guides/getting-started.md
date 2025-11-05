# Getting Started with Code Intelligence System

This guide will help you get up and running with the Code Intelligence System in just a few minutes.

## What is Code Intelligence System?

The Code Intelligence System is an AI-powered platform that helps developers understand code evolution by asking natural language questions. Instead of manually digging through git history and code changes, you can ask questions like:

- "What changed in the authentication system between versions?"
- "Find the working version of the user login feature"
- "Show me all API changes in the last month"
- "What caused the performance regression in the payment module?"

## Prerequisites

Before you begin, ensure you have:

- **Docker** and **Docker Compose** installed
- **Git** for repository access
- **curl** or **Postman** for API testing (optional)
- A code repository you want to analyze

## Step 1: Installation

### Option A: Docker Compose (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/code-intelligence.git
   cd code-intelligence
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file with your settings:
   ```env
   # Database Configuration
   NEO4J_PASSWORD=your-secure-password
   POSTGRES_PASSWORD=your-secure-password
   
   # API Configuration
   API_HOST=0.0.0.0
   API_PORT=8000
   
   # Environment
   ENVIRONMENT=development
   LOG_LEVEL=INFO
   ```

3. **Start the system:**
   ```bash
   docker-compose -f docker-compose.dev.yml up -d
   ```

4. **Verify installation:**
   ```bash
   curl http://localhost:8000/api/v1/health/
   ```
   
   You should see:
   ```json
   {
     "status": "healthy",
     "version": "1.0.0",
     "services": {
       "database": "healthy",
       "cache": "healthy",
       "agents": "healthy"
     }
   }
   ```

### Option B: Local Development

1. **Install Python dependencies:**
   ```bash
   pip install -e .
   ```

2. **Set up databases:**
   ```bash
   # Start only databases with Docker
   docker-compose -f docker-compose.dev.yml up -d neo4j postgres redis
   ```

3. **Initialize databases:**
   ```bash
   python -m src.code_intelligence.cli setup
   ```

4. **Start the API server:**
   ```bash
   python -m uvicorn src.code_intelligence.api.main:app --reload --host 0.0.0.0 --port 8000
   ```

## Step 2: Register Your First Repository

Before you can ask questions about code, you need to register a repository for analysis.

### Using the API

```bash
curl -X POST "http://localhost:8000/api/v1/repositories/" \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://github.com/your-username/your-repo.git",
       "name": "my-first-repo",
       "auto_sync": true
     }'
```

### Using the CLI

```bash
python -m src.code_intelligence.cli repositories add \
  --url "https://github.com/your-username/your-repo.git" \
  --name "my-first-repo"
```

### Response

You'll receive a response like:
```json
{
  "id": "repo-123e4567-e89b-12d3-a456-426614174000",
  "name": "my-first-repo",
  "url": "https://github.com/your-username/your-repo.git",
  "status": "not_analyzed",
  "created_at": "2024-01-15T10:30:00Z"
}
```

## Step 3: Trigger Repository Analysis

The system needs to analyze your repository before you can query it.

### Using the API

```bash
curl -X POST "http://localhost:8000/api/v1/repositories/{repo-id}/analyze"
```

### Using the CLI

```bash
python -m src.code_intelligence.cli repositories analyze --id {repo-id}
```

### Monitor Analysis Progress

```bash
# Check repository status
curl "http://localhost:8000/api/v1/repositories/{repo-id}/status"
```

The analysis process will:
1. Clone the repository
2. Parse code files (Python, JavaScript, TypeScript)
3. Build a temporal code property graph
4. Generate semantic embeddings
5. Process git history

This may take several minutes depending on repository size.

## Step 4: Submit Your First Query

Once analysis is complete, you can start asking questions!

### Example Query: Code Evolution

```bash
curl -X POST "http://localhost:8000/api/v1/queries/" \
     -H "Content-Type: application/json" \
     -d '{
       "repository_url": "https://github.com/your-username/your-repo.git",
       "query": "What changed in the authentication system between versions?",
       "options": {
         "max_commits": 50,
         "include_tests": false
       }
     }'
```

### Response

```json
{
  "query_id": "query-123e4567-e89b-12d3-a456-426614174000",
  "status": "pending",
  "message": "Query submitted successfully",
  "estimated_completion": "2024-01-15T10:35:00Z"
}
```

## Step 5: Monitor Query Progress

### Check Query Status

```bash
curl "http://localhost:8000/api/v1/queries/{query-id}"
```

### Real-time Updates (WebSocket)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function() {
    // Subscribe to query updates
    ws.send(JSON.stringify({
        type: 'subscribe_query',
        data: { query_id: 'your-query-id' }
    }));
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Progress update:', message);
};
```

### Using the CLI

```bash
python -m src.code_intelligence.cli status {query-id}
```

## Step 6: View Results

Once processing is complete, you'll receive detailed results:

```json
{
  "query_id": "query-123...",
  "status": "completed",
  "results": {
    "summary": "Analysis found significant changes in authentication system between v1.2 and v1.3",
    "confidence_score": 0.92,
    "processing_time_seconds": 15.5,
    "findings": [
      {
        "agent_name": "historian",
        "finding_type": "version_comparison",
        "content": "Authentication module was refactored in commit abc123...",
        "confidence": 0.95,
        "citations": [
          {
            "file_path": "src/auth/login.py",
            "line_number": 42,
            "description": "Updated password validation logic"
          }
        ]
      }
    ]
  }
}
```

## Common Query Examples

### 1. Find Working Code
```json
{
  "query": "Find the working version of the user registration feature",
  "options": { "max_commits": 100 }
}
```

### 2. Regression Analysis
```json
{
  "query": "What broke the payment processing between last week and today?",
  "options": { "include_tests": true }
}
```

### 3. API Changes
```json
{
  "query": "Show me all API endpoint changes in the last month",
  "options": { "max_commits": 200 }
}
```

### 4. Performance Issues
```json
{
  "query": "What caused the performance regression in the database queries?",
  "options": { "include_tests": false }
}
```

## Using the Web Interface

1. **Open your browser** to `http://localhost:3000`
2. **Register repositories** through the UI
3. **Submit queries** using the query form
4. **Monitor progress** with real-time updates
5. **Explore results** with interactive visualizations

## Using the CLI Tool

The CLI provides a convenient way to interact with the system:

```bash
# Configure CLI
python -m src.code_intelligence.cli config api_url http://localhost:8000/api/v1

# List repositories
python -m src.code_intelligence.cli repositories list

# Submit query
python -m src.code_intelligence.cli query \
  "What changed in the API layer?" \
  --repo "https://github.com/your-username/your-repo.git" \
  --wait

# View query history
python -m src.code_intelligence.cli history --limit 10

# Export results
python -m src.code_intelligence.cli export {query-id} --format json
```

## Next Steps

Now that you have the system running:

1. **Explore More Queries**: Try different types of questions about your code
2. **Learn the API**: Read the [API Documentation](../api/README.md)
3. **Understand Results**: Learn how to interpret agent findings
4. **Optimize Performance**: Configure caching and scaling options
5. **Set Up Monitoring**: Enable metrics and alerting

## Troubleshooting

### Common Issues

**Repository Analysis Fails**
```bash
# Check repository access
git clone https://github.com/your-username/your-repo.git

# Check system logs
docker-compose logs api

# Retry analysis
curl -X POST "http://localhost:8000/api/v1/repositories/{repo-id}/analyze"
```

**Query Processing Hangs**
```bash
# Check agent system health
curl http://localhost:8000/api/v1/health/components/agents

# Check system resources
docker stats

# Restart services if needed
docker-compose restart
```

**Database Connection Issues**
```bash
# Check database health
curl http://localhost:8000/api/v1/health/detailed

# Reset databases
docker-compose down -v
docker-compose up -d
```

### Getting Help

- **Documentation**: Check the [troubleshooting guide](../troubleshooting.md)
- **Logs**: Use `docker-compose logs` to check service logs
- **Health Checks**: Monitor `/api/v1/health/` endpoints
- **Community**: Join our discussions for help and tips

## What's Next?

- [Query Examples and Best Practices](./query-examples.md)
- [Web Interface Guide](./web-interface.md)
- [CLI Reference](./cli-guide.md)
- [API Documentation](../api/README.md)
- [Advanced Configuration](../admin-guides/configuration.md)