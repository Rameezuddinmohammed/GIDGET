"""User management API routes."""

from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status

from ..models import UserProfile
from ..dependencies import require_auth, get_supabase_client
from ...database.supabase_client import SupabaseClient
from ...logging import get_logger


router = APIRouter(prefix="/users", tags=["users"])
logger = get_logger(__name__)


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(
    user_id: str = Depends(require_auth),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> UserProfile:
    """Get current user profile."""
    try:
        # In a real implementation, fetch from database
        # For now, return mock data
        return UserProfile(
            user_id=user_id,
            email="user@example.com",
            preferences={
                "preferred_detail_level": "medium",
                "include_citations": True,
                "max_results": 50,
                "confidence_threshold": 0.7,
                "theme": "dark",
                "notifications_enabled": True
            },
            created_at=datetime(2024, 1, 1),
            last_active=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get user profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )


@router.put("/me/preferences")
async def update_user_preferences(
    preferences: Dict[str, Any],
    user_id: str = Depends(require_auth),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> dict:
    """Update user preferences."""
    try:
        # Validate preferences
        allowed_keys = {
            "preferred_detail_level", "include_citations", "max_results",
            "confidence_threshold", "theme", "notifications_enabled"
        }
        
        invalid_keys = set(preferences.keys()) - allowed_keys
        if invalid_keys:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid preference keys: {list(invalid_keys)}"
            )
        
        # In a real implementation, update in database
        logger.info(f"User preferences updated: {user_id}", extra={
            "user_id": user_id,
            "preferences": preferences
        })
        
        return {"message": "Preferences updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update preferences: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferences"
        )


@router.get("/me/stats")
async def get_user_stats(
    user_id: str = Depends(require_auth)
) -> dict:
    """Get user usage statistics."""
    try:
        # In a real implementation, calculate from database
        return {
            "total_queries": 42,
            "successful_queries": 38,
            "repositories_analyzed": 5,
            "average_confidence_score": 0.85,
            "total_processing_time_minutes": 127,
            "queries_this_month": 15,
            "most_analyzed_language": "python",
            "favorite_query_type": "find_changes"
        }
        
    except Exception as e:
        logger.error(f"Failed to get user stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user statistics"
        )