"""System-wide constants and configuration values."""

from typing import Final

# Timeout Constants (in seconds)
DEFAULT_AGENT_TIMEOUT: Final[int] = 60
DEFAULT_WORKFLOW_TIMEOUT: Final[int] = 300
DEFAULT_LLM_TIMEOUT: Final[int] = 30
DEFAULT_DATABASE_TIMEOUT: Final[int] = 10

# Retry Constants
DEFAULT_MAX_RETRIES: Final[int] = 2
DEFAULT_RETRY_BACKOFF_BASE: Final[float] = 2.0
DEFAULT_RETRY_BACKOFF_MAX: Final[float] = 60.0

# Connection Pool Constants
DEFAULT_MAX_CONNECTIONS: Final[int] = 10
DEFAULT_CONNECTION_POOL_TIMEOUT: Final[int] = 30

# LLM Constants
DEFAULT_LLM_TEMPERATURE: Final[float] = 0.1
DEFAULT_LLM_MAX_TOKENS: Final[int] = 2000
MAX_LLM_CONTEXT_LENGTH: Final[int] = 128000

# Agent Constants
DEFAULT_AGENT_CONFIDENCE_THRESHOLD: Final[float] = 0.7
MIN_CONFIDENCE_SCORE: Final[float] = 0.0
MAX_CONFIDENCE_SCORE: Final[float] = 1.0

# Database Constants
DEFAULT_BATCH_SIZE: Final[int] = 100
MAX_QUERY_RESULTS: Final[int] = 10000

# Cache Constants
DEFAULT_CACHE_TTL: Final[int] = 3600  # 1 hour
MAX_CACHE_SIZE: Final[int] = 1000

# File System Constants
MAX_FILE_SIZE_MB: Final[int] = 100
MAX_REPOSITORY_SIZE_MB: Final[int] = 1000

# Logging Constants
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
MAX_LOG_MESSAGE_LENGTH: Final[int] = 10000

# Monitoring Constants
METRICS_COLLECTION_INTERVAL: Final[int] = 60  # seconds
HEALTH_CHECK_INTERVAL: Final[int] = 30  # seconds

# API Constants
DEFAULT_API_RATE_LIMIT: Final[int] = 100  # requests per minute
DEFAULT_API_TIMEOUT: Final[int] = 30  # seconds

# Configuration Validation Constants
MIN_AZURE_API_KEY_LENGTH: Final[int] = 32
NEO4J_AURA_DOMAIN: Final[str] = "databases.neo4j.io"
AZURE_COGNITIVE_SERVICES_DOMAIN: Final[str] = ".cognitiveservices.azure.com"