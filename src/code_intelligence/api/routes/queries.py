"""Query management API routes."""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query

from ..models import (
    QueryRequest, QueryResponse, QueryResult, QueryHistory, 
    QueryHistoryItem, ExportRequest, ExportResponse, QueryStatus
)
from ..dependencies import get_current_user, get_supabase_client, get_orchestrator
from ...database.supabase_client import SupabaseClient
from ...agents.orchestrator_agent import OrchestratorAgent
from ...agents.state import AgentState
from ...logging import get_logger


router = APIRouter(prefix="/queries", tags=["queries"])
logger = get_logger(__name__)


# In-memory storage for demo purposes - in production, use database
query_storage = {}


@router.post("/", response_model=QueryResponse)
async def submit_query(
    request: QueryRequest,
    user_id: Optional[str] = Depends(get_current_user),
    supabase: SupabaseClient = Depends(get_supabase_client),
    orchestrator: OrchestratorAgent = Depends(get_orchestrator)
) -> QueryResponse:
    """Submit a new query for analysis."""
    try:
        # Generate unique query ID
        query_id = str(uuid.uuid4())
        
        # Create initial state
        state = AgentState(
            session_id=query_id,
            query={
                "original": request.query,
                "repository_url": request.repository_url,
                "options": request.options or {}
            },
            repository={
                "url": request.repository_url,
                "path": f"/tmp/repos/{query_id}"  # Temporary path
            }
        )
        
        # Store query in memory (in production, store in database)
        query_storage[query_id] = {
            "id": query_id,
            "status": QueryStatus.PENDING,
            "state": state,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "user_id": user_id
        }
        
        # TODO: Start async processing with orchestrator
        # For now, just return pending status
        
        logger.info(f"Query submitted: {query_id}", extra={
            "query_id": query_id,
            "user_id": user_id,
            "repository_url": request.repository_url
        })
        
        return QueryResponse(
            query_id=query_id,
            status=QueryStatus.PENDING,
            message="Query submitted successfully",
            estimated_duration_seconds=180
        )
        
    except Exception as e:
        logger.error(f"Failed to submit query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit query: {str(e)}"
        )


@router.get("/{query_id}", response_model=QueryResult)
async def get_query_status(
    query_id: str,
    user_id: Optional[str] = Depends(get_current_user)
) -> QueryResult:
    """Get query status and results."""
    if query_id not in query_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Query not found"
        )
    
    query_data = query_storage[query_id]
    
    # Check user access (in production, implement proper authorization)
    if user_id and query_data.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return QueryResult(
        query_id=query_id,
        status=query_data["status"],
        created_at=query_data["created_at"],
        updated_at=query_data["updated_at"],
        # TODO: Add progress and results when available
    )


@router.delete("/{query_id}")
async def cancel_query(
    query_id: str,
    user_id: Optional[str] = Depends(get_current_user)
) -> dict:
    """Cancel a running query."""
    if query_id not in query_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Query not found"
        )
    
    query_data = query_storage[query_id]
    
    # Check user access
    if user_id and query_data.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Update status to cancelled
    query_data["status"] = QueryStatus.CANCELLED
    query_data["updated_at"] = datetime.utcnow()
    
    logger.info(f"Query cancelled: {query_id}", extra={
        "query_id": query_id,
        "user_id": user_id
    })
    
    return {"message": "Query cancelled successfully"}


@router.get("/", response_model=QueryHistory)
async def get_query_history(
    user_id: Optional[str] = Depends(get_current_user),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page")
) -> QueryHistory:
    """Get query history for the current user."""
    # Filter queries by user
    user_queries = [
        q for q in query_storage.values() 
        if q.get("user_id") == user_id
    ]
    
    # Sort by creation time (newest first)
    user_queries.sort(key=lambda x: x["created_at"], reverse=True)
    
    # Paginate
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_queries = user_queries[start_idx:end_idx]
    
    # Convert to history items
    history_items = []
    for q in page_queries:
        state = q["state"]
        history_items.append(QueryHistoryItem(
            query_id=q["id"],
            query=state.query.get("original", ""),
            repository_name=state.repository.get("url", "").split("/")[-1],
            status=q["status"],
            created_at=q["created_at"],
            completed_at=q.get("completed_at")
        ))
    
    return QueryHistory(
        queries=history_items,
        total_count=len(user_queries),
        page=page,
        page_size=page_size
    )


@router.post("/{query_id}/export", response_model=ExportResponse)
async def export_query_results(
    query_id: str,
    request: ExportRequest,
    user_id: Optional[str] = Depends(get_current_user)
) -> ExportResponse:
    """Export query results in specified format."""
    if query_id not in query_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Query not found"
        )
    
    query_data = query_storage[query_id]
    
    # Check user access
    if user_id and query_data.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Check if query is completed
    if query_data["status"] != QueryStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query must be completed before export"
        )
    
    # Generate export ID and mock download URL
    export_id = str(uuid.uuid4())
    download_url = f"/api/v1/exports/{export_id}/download"
    
    logger.info(f"Export requested: {export_id}", extra={
        "query_id": query_id,
        "export_format": request.format,
        "user_id": user_id
    })
    
    return ExportResponse(
        export_id=export_id,
        download_url=download_url,
        expires_at=datetime.utcnow().replace(hour=23, minute=59, second=59),
        format=request.format,
        file_size_bytes=1024  # Mock size
    )