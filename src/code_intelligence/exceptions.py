"""Custom exceptions for the code intelligence system."""

from typing import Any, Dict, Optional


class CodeIntelligenceError(Exception):
    """Base exception for code intelligence errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, cause: Optional[Exception] = None):
        """Initialize with message, optional details, and cause."""
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause
        
        # Set the cause for proper exception chaining
        if cause is not None:
            self.__cause__ = cause
        
    def __str__(self) -> str:
        """String representation with details."""
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message
    
    @classmethod
    def from_exception(cls, message: str, cause: Exception, details: Optional[Dict[str, Any]] = None):
        """Create exception with proper chaining from another exception."""
        return cls(message, details, cause)


class RepositoryError(CodeIntelligenceError):
    """Exception raised for repository-related errors."""
    pass


class ParsingError(CodeIntelligenceError):
    """Exception raised for parsing-related errors."""
    pass


class DatabaseError(CodeIntelligenceError):
    """Exception raised for database-related errors."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Exception raised for database connection errors."""
    pass


class DatabaseQueryError(DatabaseError):
    """Exception raised for database query errors."""
    pass


class Neo4jError(DatabaseError):
    """Neo4j-specific errors."""
    pass


class SupabaseError(DatabaseError):
    """Supabase-specific errors."""
    pass


class ConfigurationError(CodeIntelligenceError):
    """Exception raised for configuration-related errors."""
    pass


class AgentError(CodeIntelligenceError):
    """Base exception for agent-related errors."""
    pass


class AgentExecutionError(AgentError):
    """Exception raised when agent execution fails."""
    pass


class AgentTimeoutError(AgentError):
    """Exception raised when agent execution times out."""
    pass


class AgentCommunicationError(AgentError):
    """Exception raised for agent communication errors."""
    pass


class LLMError(CodeIntelligenceError):
    """Base exception for LLM-related errors."""
    pass


class LLMConnectionError(LLMError):
    """Exception raised for LLM connection errors."""
    pass


class LLMRateLimitError(LLMError):
    """Exception raised when LLM rate limit is exceeded."""
    pass


class LLMTokenLimitError(LLMError):
    """Exception raised when LLM token limit is exceeded."""
    pass


class ValidationError(CodeIntelligenceError):
    """Exception raised for validation errors."""
    pass


class VerificationError(CodeIntelligenceError):
    """Verification and validation errors."""
    pass


class EmbeddingError(CodeIntelligenceError):
    """Exception raised for embedding generation errors."""
    pass


class VectorStorageError(CodeIntelligenceError):
    """Exception raised for vector storage operations."""
    pass


class SearchError(CodeIntelligenceError):
    """Exception raised for search operations."""
    pass