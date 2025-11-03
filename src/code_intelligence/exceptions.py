"""Custom exceptions for the Code Intelligence System."""

from typing import Any, Dict, Optional


class CodeIntelligenceError(Exception):
    """Base exception for all Code Intelligence System errors."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class DatabaseError(CodeIntelligenceError):
    """Database-related errors."""
    pass


class Neo4jError(DatabaseError):
    """Neo4j-specific errors."""
    pass


class SupabaseError(DatabaseError):
    """Supabase-specific errors."""
    pass


class AgentError(CodeIntelligenceError):
    """Agent execution errors."""
    pass


class AgentTimeoutError(AgentError):
    """Agent execution timeout errors."""
    pass


class VerificationError(CodeIntelligenceError):
    """Verification and validation errors."""
    pass


class RepositoryError(CodeIntelligenceError):
    """Repository processing errors."""
    pass


class ConfigurationError(CodeIntelligenceError):
    """Configuration-related errors."""
    pass