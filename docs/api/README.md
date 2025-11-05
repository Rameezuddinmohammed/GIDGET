# Code Intelligence System API Documentation

The Code Intelligence System provides a comprehensive REST API for all operations, along with WebSocket support for real-time updates.

## Base Information

- **Base URL**: `http://localhost:8000/api/v1`
- **Content Type**: `application/json`
- **Authentication**: Bearer token (when enabled)
- **Rate Limiting**: 1000 requests per hour per user
- **API Version**: v1

## Authentication

When authentication is enabled, include the Bearer token in the Authorization header:

```bash
curl -H "Authorization: Bearer your-token-here" \
     "http://localhost:8000/api/v1/queries/"
```

## Core Endpoints

### Health and Status

#### GET /health/
Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "database": "healthy",
    "cache": "healthy",
    "agents": "healthy"
  }
}
```

#### GET /health/detailed
Detailed health check with service dependencies.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "neo4j": "healthy",
    "supabase": "healthy",
    "agents": "healthy"
  },
  "system_info": {
    "python_version": "3.11",
    "environment": "production"
  }
}
```

#### GET /health/metrics
Performance metrics for monitoring.

**Response:**
```json
{
  "health": {
    "status": "healthy",
    "success_rate": 0.95,
    "avg_response_time_ms": 150.0,
    "active_executions": 3
  },
  "system": {
    "total_executions": 1250,
    "success_rate": 0.94,
    "avg_duration_ms": 2500.0,
    "agents": {
      "historian": {
        "total_executions": 400,
        "success_rate": 0.96,
        "avg_execution_time": 2.1
      }
    }
  },
  "cache": {
    "cache_hits": 850,
    "cache_misses": 150,
    "cache_hit_rate": 0.85,
    "cache_size_bytes": 104857600
  }
}
```

### Repository Management

#### POST /repositories/
Register a new repository for analysis.

**Request Body:**
```json
{
  "url": "https://github.com/username/repo.git",
  "name": "my-repo",
  "auto_sync": true
}
```

**Response:**
```json
{
  "id": "repo-123e4567-e89b-12d3-a456-426614174000",
  "name": "my-repo",
  "url": "https://github.com/username/repo.git",
  "status": "not_analyzed",
  "auto_sync": true,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

#### GET /repositories/
List all registered repositories.

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `page_size` (int): Items per page (default: 20)
- `status` (string): Filter by status

**Response:**
```json
[
  {
    "id": "repo-123...",
    "name": "my-repo",
    "url": "https://github.com/username/repo.git",
    "status": "analyzed",
    "commit_count": 150,
    "supported_languages": ["Python", "JavaScript"],
    "last_analyzed": "2024-01-15T10:30:00Z"
  }
]
```

#### GET /repositories/{id}
Get details for a specific repository.

**Response:**
```json
{
  "id": "repo-123...",
  "name": "my-repo",
  "url": "https://github.com/username/repo.git",
  "status": "analyzed",
  "commit_count": 150,
  "file_count": 45,
  "supported_languages": ["Python", "JavaScript"],
  "analysis_stats": {
    "functions_analyzed": 120,
    "classes_analyzed": 25,
    "dependencies_mapped": 80
  },
  "created_at": "2024-01-15T10:30:00Z",
  "last_analyzed": "2024-01-15T11:00:00Z"
}
```

#### POST /repositories/{id}/analyze
Trigger analysis for a repository.

**Response:**
```json
{
  "message": "Analysis started",
  "analysis_id": "analysis-456...",
  "estimated_completion": "2024-01-15T10:45:00Z"
}
```

#### GET /repositories/{id}/status
Get analysis status for a repository.

**Response:**
```json
{
  "analysis_status": "in_progress",
  "progress_percentage": 65.0,
  "current_step": "Processing git history",
  "commit_count": 150,
  "files_processed": 30,
  "total_files": 45,
  "supported_languages": ["Python", "JavaScript"],
  "started_at": "2024-01-15T10:30:00Z",
  "estimated_completion": "2024-01-15T10:45:00Z"
}
```

#### DELETE /repositories/{id}
Delete a repository and all associated data.

**Response:**
```json
{
  "message": "Repository deleted successfully"
}
```

### Query Management

#### POST /queries/
Submit a new query for analysis.

**Request Body:**
```json
{
  "repository_url": "https://github.com/username/repo.git",
  "query": "What changed in the authentication system between versions?",
  "options": {
    "max_commits": 50,
    "include_tests": false,
    "languages": ["Python", "JavaScript"]
  }
}
```

**Response:**
```json
{
  "query_id": "query-123e4567-e89b-12d3-a456-426614174000",
  "status": "pending",
  "message": "Query submitted successfully",
  "estimated_completion": "2024-01-15T10:35:00Z"
}
```

#### GET /queries/{id}
Get query status and results.

**Response (Processing):**
```json
{
  "query_id": "query-123...",
  "status": "processing",
  "progress": {
    "current_agent": "historian",
    "completed_steps": ["orchestrator"],
    "total_steps": 5,
    "progress_percentage": 40.0,
    "current_step": "Analyzing git history",
    "estimated_completion": "2024-01-15T10:35:00Z"
  },
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:32:00Z"
}
```

**Response (Completed):**
```json
{
  "query_id": "query-123...",
  "status": "completed",
  "query": "What changed in the authentication system between versions?",
  "repository_name": "my-repo",
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
            "commit_sha": "abc123...",
            "description": "Updated password validation logic"
          }
        ],
        "metadata": {
          "commits_analyzed": 25,
          "files_changed": 8
        }
      }
    ]
  },
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:30:45Z"
}
```

#### GET /queries/
List query history.

**Query Parameters:**
- `page` (int): Page number (default: 1)
- `page_size` (int): Items per page (default: 20)
- `status` (string): Filter by status
- `repository_id` (string): Filter by repository

**Response:**
```json
{
  "queries": [
    {
      "query_id": "query-123...",
      "query": "What changed in the authentication system?",
      "repository_name": "my-repo",
      "status": "completed",
      "confidence_score": 0.92,
      "created_at": "2024-01-15T10:30:00Z",
      "completed_at": "2024-01-15T10:30:45Z"
    }
  ],
  "total_count": 1,
  "page": 1,
  "page_size": 20
}
```

#### DELETE /queries/{id}
Cancel a query (if still processing).

**Response:**
```json
{
  "message": "Query cancelled successfully"
}
```

#### POST /queries/{id}/export
Export query results in various formats.

**Request Body:**
```json
{
  "format": "json",
  "include_citations": true,
  "include_metadata": false
}
```

**Response:**
```json
{
  "export_id": "export-123...",
  "download_url": "http://localhost:8000/api/v1/exports/export-123.json",
  "expires_at": "2024-01-16T10:30:00Z",
  "format": "json",
  "size_bytes": 15420
}
```

### User Management

#### GET /users/me
Get current user profile (when authentication is enabled).

**Response:**
```json
{
  "id": "user-123...",
  "email": "user@example.com",
  "name": "John Doe",
  "created_at": "2024-01-01T00:00:00Z",
  "preferences": {
    "theme": "dark",
    "notifications_enabled": true
  }
}
```

#### PUT /users/me/preferences
Update user preferences.

**Request Body:**
```json
{
  "theme": "dark",
  "notifications_enabled": true,
  "default_max_commits": 100
}
```

#### GET /users/me/stats
Get user statistics.

**Response:**
```json
{
  "total_queries": 25,
  "successful_queries": 23,
  "total_repositories": 5,
  "avg_query_time_seconds": 18.5,
  "most_used_query_types": ["code_evolution", "regression_analysis"]
}
```

## Error Responses

All endpoints return consistent error responses:

```json
{
  "error": "ValidationError",
  "message": "Invalid request data",
  "details": {
    "field": "repository_url",
    "issue": "Invalid URL format"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Common HTTP Status Codes

- `200 OK`: Request successful
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Default Limit**: 1000 requests per hour per user
- **Headers**: Rate limit information is included in response headers
  - `X-RateLimit-Limit`: Maximum requests per window
  - `X-RateLimit-Remaining`: Remaining requests in current window
  - `X-RateLimit-Reset`: Time when the rate limit resets

When rate limit is exceeded:
```json
{
  "error": "RateLimitExceeded",
  "message": "Too many requests",
  "details": {
    "limit": 1000,
    "window": "1 hour",
    "reset_at": "2024-01-15T11:00:00Z"
  }
}
```

## Pagination

List endpoints support pagination:

**Query Parameters:**
- `page`: Page number (1-based, default: 1)
- `page_size`: Items per page (default: 20, max: 100)

**Response Format:**
```json
{
  "items": [...],
  "total_count": 150,
  "page": 1,
  "page_size": 20,
  "total_pages": 8,
  "has_next": true,
  "has_previous": false
}
```

## Filtering and Sorting

Many endpoints support filtering and sorting:

**Query Parameters:**
- `status`: Filter by status
- `created_after`: Filter by creation date
- `sort_by`: Sort field (e.g., `created_at`, `name`)
- `sort_order`: Sort order (`asc` or `desc`)

Example:
```bash
GET /api/v1/queries/?status=completed&sort_by=created_at&sort_order=desc
```

## WebSocket API

Real-time updates are available via WebSocket connection.

**Endpoint**: `ws://localhost:8000/ws`

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = function() {
    console.log('Connected to WebSocket');
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
};
```

### Message Types

#### Subscribe to Query Updates
```json
{
  "type": "subscribe_query",
  "data": {
    "query_id": "query-123..."
  }
}
```

#### Query Progress Update
```json
{
  "type": "query_progress",
  "query_id": "query-123...",
  "data": {
    "current_agent": "historian",
    "progress_percentage": 40.0,
    "current_step": "Analyzing git history",
    "estimated_completion": "2024-01-15T10:35:00Z"
  }
}
```

#### Partial Results
```json
{
  "type": "partial_results",
  "query_id": "query-123...",
  "data": {
    "agent": "historian",
    "findings": [
      {
        "content": "Found 15 relevant commits",
        "confidence": 0.9
      }
    ]
  }
}
```

#### Query Completed
```json
{
  "type": "query_completed",
  "query_id": "query-123...",
  "data": {
    "summary": "Analysis completed successfully",
    "confidence": 0.92,
    "total_findings": 5
  }
}
```

## SDK and Client Libraries

### Python SDK

```python
from code_intelligence_sdk import CodeIntelligenceClient

client = CodeIntelligenceClient(
    base_url="http://localhost:8000/api/v1",
    api_key="your-api-key"  # if authentication enabled
)

# Register repository
repo = client.repositories.create(
    url="https://github.com/username/repo.git",
    name="my-repo"
)

# Submit query
query = client.queries.create(
    repository_url="https://github.com/username/repo.git",
    query="What changed in the authentication system?",
    options={"max_commits": 50}
)

# Wait for completion
result = client.queries.wait_for_completion(query.id)
print(result.summary)
```

### JavaScript SDK

```javascript
import { CodeIntelligenceClient } from 'code-intelligence-sdk';

const client = new CodeIntelligenceClient({
  baseUrl: 'http://localhost:8000/api/v1',
  apiKey: 'your-api-key'  // if authentication enabled
});

// Register repository
const repo = await client.repositories.create({
  url: 'https://github.com/username/repo.git',
  name: 'my-repo'
});

// Submit query with real-time updates
const query = await client.queries.create({
  repositoryUrl: 'https://github.com/username/repo.git',
  query: 'What changed in the authentication system?',
  options: { maxCommits: 50 }
});

// Listen for updates
client.queries.onProgress(query.id, (progress) => {
  console.log(`Progress: ${progress.percentage}%`);
});

const result = await client.queries.waitForCompletion(query.id);
console.log(result.summary);
```

## Examples

### Complete Workflow Example

```bash
#!/bin/bash

# 1. Register repository
REPO_RESPONSE=$(curl -s -X POST "http://localhost:8000/api/v1/repositories/" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://github.com/username/repo.git",
    "name": "example-repo"
  }')

REPO_ID=$(echo $REPO_RESPONSE | jq -r '.id')
echo "Repository registered: $REPO_ID"

# 2. Trigger analysis
curl -X POST "http://localhost:8000/api/v1/repositories/$REPO_ID/analyze"

# 3. Wait for analysis to complete
while true; do
  STATUS=$(curl -s "http://localhost:8000/api/v1/repositories/$REPO_ID/status" | jq -r '.analysis_status')
  if [ "$STATUS" = "completed" ]; then
    break
  fi
  echo "Analysis status: $STATUS"
  sleep 10
done

# 4. Submit query
QUERY_RESPONSE=$(curl -s -X POST "http://localhost:8000/api/v1/queries/" \
  -H "Content-Type: application/json" \
  -d '{
    "repository_url": "https://github.com/username/repo.git",
    "query": "What are the main changes in the last 10 commits?",
    "options": {"max_commits": 10}
  }')

QUERY_ID=$(echo $QUERY_RESPONSE | jq -r '.query_id')
echo "Query submitted: $QUERY_ID"

# 5. Wait for results
while true; do
  RESULT=$(curl -s "http://localhost:8000/api/v1/queries/$QUERY_ID")
  STATUS=$(echo $RESULT | jq -r '.status')
  
  if [ "$STATUS" = "completed" ]; then
    echo "Query completed!"
    echo $RESULT | jq '.results.summary'
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "Query failed!"
    echo $RESULT | jq '.error'
    break
  fi
  
  echo "Query status: $STATUS"
  sleep 5
done
```

## Next Steps

- [WebSocket API Documentation](./websocket.md)
- [Authentication Guide](./authentication.md)
- [SDK Documentation](./sdks.md)
- [API Examples](./examples.md)