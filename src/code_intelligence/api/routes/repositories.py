"""Repository management API routes."""

import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query

from ..models import RepositoryRequest, RepositoryInfo, RepositoryStatus
from ..dependencies import get_current_user, get_supabase_client
from ...database.supabase_client import SupabaseClient
from ...logging import get_logger


router = APIRouter(prefix="/repositories", tags=["repositories"])
logger = get_logger(__name__)


# In-memory storage for demo purposes - in production, use database
repository_storage = {}


@router.post("/", response_model=RepositoryInfo)
async def register_repository(
    request: RepositoryRequest,
    user_id: Optional[str] = Depends(get_current_user),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> RepositoryInfo:
    """Register a new repository for analysis."""
    try:
        # Generate unique repository ID
        repo_id = str(uuid.uuid4())
        
        # Extract repository name from URL if not provided
        repo_name = request.name or request.url.split("/")[-1].replace(".git", "")
        
        # Create repository record
        repo_info = RepositoryInfo(
            id=repo_id,
            name=repo_name,
            url=request.url,
            status=RepositoryStatus.NOT_ANALYZED,
            commit_count=0,
            supported_languages=[],
            file_count=0,
            lines_of_code=0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Store repository (in production, store in database)
        repository_storage[repo_id] = {
            "info": repo_info,
            "user_id": user_id,
            "auto_sync": request.auto_sync,
            "analysis_options": request.analysis_options
        }
        
        logger.info(f"Repository registered: {repo_id}", extra={
            "repository_id": repo_id,
            "repository_url": request.url,
            "user_id": user_id
        })
        
        return repo_info
        
    except Exception as e:
        logger.error(f"Failed to register repository: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register repository: {str(e)}"
        )


@router.get("/", response_model=List[RepositoryInfo])
async def list_repositories(
    user_id: Optional[str] = Depends(get_current_user),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page")
) -> List[RepositoryInfo]:
    """List repositories for the current user."""
    # Filter repositories by user
    user_repos = [
        data["info"] for data in repository_storage.values()
        if data.get("user_id") == user_id
    ]
    
    # Sort by creation time (newest first)
    user_repos.sort(key=lambda x: x.created_at, reverse=True)
    
    # Paginate
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    return user_repos[start_idx:end_idx]


@router.get("/{repo_id}", response_model=RepositoryInfo)
async def get_repository(
    repo_id: str,
    user_id: Optional[str] = Depends(get_current_user)
) -> RepositoryInfo:
    """Get repository information."""
    if repo_id not in repository_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found"
        )
    
    repo_data = repository_storage[repo_id]
    
    # Check user access
    if user_id and repo_data.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    return repo_data["info"]


@router.post("/{repo_id}/analyze")
async def trigger_analysis(
    repo_id: str,
    user_id: Optional[str] = Depends(get_current_user)
) -> dict:
    """Trigger repository analysis."""
    if repo_id not in repository_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found"
        )
    
    repo_data = repository_storage[repo_id]
    
    # Check user access
    if user_id and repo_data.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Update status to analyzing
    repo_data["info"].status = RepositoryStatus.ANALYZING
    repo_data["info"].updated_at = datetime.utcnow()
    
    logger.info(f"Repository analysis triggered: {repo_id}", extra={
        "repository_id": repo_id,
        "user_id": user_id
    })
    
    return {"message": "Repository analysis started"}


@router.get("/{repo_id}/status")
async def get_repository_status(
    repo_id: str,
    user_id: Optional[str] = Depends(get_current_user)
) -> dict:
    """Get detailed repository analysis status."""
    if repo_id not in repository_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found"
        )
    
    repo_data = repository_storage[repo_id]
    
    # Check user access
    if user_id and repo_data.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    repo_info = repo_data["info"]
    
    return {
        "repository_id": repo_id,
        "status": repo_info.status,
        "last_analyzed": repo_info.last_analyzed,
        "commit_count": repo_info.commit_count,
        "supported_languages": repo_info.supported_languages,
        "file_count": repo_info.file_count,
        "lines_of_code": repo_info.lines_of_code,
        "analysis_progress": {
            "current_step": "idle",
            "progress_percentage": 100 if repo_info.status == RepositoryStatus.ANALYZED else 0
        }
    }


@router.delete("/{repo_id}")
async def delete_repository(
    repo_id: str,
    user_id: Optional[str] = Depends(get_current_user)
) -> dict:
    """Delete a repository and its analysis data."""
    if repo_id not in repository_storage:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository not found"
        )
    
    repo_data = repository_storage[repo_id]
    
    # Check user access
    if user_id and repo_data.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )
    
    # Delete repository
    del repository_storage[repo_id]
    
    logger.info(f"Repository deleted: {repo_id}", extra={
        "repository_id": repo_id,
        "user_id": user_id
    })
    
    return {"message": "Repository deleted successfully"}