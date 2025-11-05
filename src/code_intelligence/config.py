"""Configuration management for the Code Intelligence System."""

from typing import Optional
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from urllib.parse import urlparse

from .exceptions import ConfigurationError
from .core.constants import MIN_AZURE_API_KEY_LENGTH, AZURE_COGNITIVE_SERVICES_DOMAIN


class DatabaseConfig(BaseSettings):
    """Database configuration settings."""
    
    # Neo4j Configuration
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="password", alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
    
    # Supabase Configuration
    supabase_url: Optional[str] = Field(default=None, alias="SUPABASE_URL")
    supabase_key: Optional[str] = Field(default=None, alias="SUPABASE_ANON_KEY")
    supabase_service_key: Optional[str] = Field(default=None, alias="SUPABASE_SERVICE_KEY")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )


class LLMConfig(BaseSettings):
    """LLM configuration settings with validation."""
    
    # Azure OpenAI Configuration
    azure_openai_endpoint: Optional[str] = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: Optional[str] = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_deployment_name: Optional[str] = Field(default=None, alias="AZURE_OPENAI_DEPLOYMENT_NAME")
    azure_openai_api_version: str = Field(default="2025-01-01-preview", alias="AZURE_OPENAI_API_VERSION")
    
    # OpenAI Configuration (fallback)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", alias="OPENAI_MODEL")
    
    # LLM Parameters
    temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE", ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, alias="LLM_MAX_TOKENS", gt=0, le=128000)
    timeout_seconds: int = Field(default=30, alias="LLM_TIMEOUT_SECONDS", gt=0, le=300)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )
    
    @field_validator('azure_openai_endpoint')
    @classmethod
    def validate_azure_endpoint(cls, v):
        """Validate Azure OpenAI endpoint URL."""
        if v is not None:
            try:
                parsed = urlparse(v)
                if not parsed.scheme or not parsed.netloc:
                    raise ConfigurationError("Azure OpenAI endpoint must be a valid URL with scheme and host")
                if not parsed.netloc.endswith(AZURE_COGNITIVE_SERVICES_DOMAIN):
                    raise ConfigurationError(
                        "Azure OpenAI endpoint must be an Azure Cognitive Services endpoint "
                        "(*.cognitiveservices.azure.com)"
                    )
            except ConfigurationError:
                raise  # Re-raise our specific errors
            except Exception as e:
                raise ConfigurationError(f"Failed to parse Azure OpenAI endpoint URL: {str(e)}")
        return v
        
    @field_validator('azure_openai_api_key')
    @classmethod
    def validate_azure_api_key(cls, v):
        """Validate Azure OpenAI API key format."""
        if v is not None and len(v) < MIN_AZURE_API_KEY_LENGTH:
            raise ConfigurationError("Azure OpenAI API key appears to be too short")
        return v
        
    @model_validator(mode='after')
    def validate_llm_config(self):
        """Validate that at least one LLM configuration is provided."""
        azure_configured = all([
            self.azure_openai_endpoint,
            self.azure_openai_api_key,
            self.azure_openai_deployment_name
        ])
        openai_configured = self.openai_api_key is not None
        
        if not azure_configured and not openai_configured:
            raise ConfigurationError(
                "Either Azure OpenAI or OpenAI configuration must be provided"
            )
        return self


class SemanticSearchConfig(BaseSettings):
    """Semantic search configuration settings."""
    
    # Embedding Model Configuration
    embedding_model: str = Field(default="microsoft/codebert-base", alias="EMBEDDING_MODEL")
    embedding_device: str = Field(default="auto", alias="EMBEDDING_DEVICE")  # auto, cpu, cuda
    max_sequence_length: int = Field(default=512, alias="MAX_SEQUENCE_LENGTH")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    
    # Vector Search Configuration
    similarity_threshold: float = Field(default=0.7, alias="SIMILARITY_THRESHOLD", ge=0.0, le=1.0)
    max_search_results: int = Field(default=10, alias="MAX_SEARCH_RESULTS", gt=0, le=100)
    vector_index_lists: int = Field(default=100, alias="VECTOR_INDEX_LISTS", gt=0)
    
    # Hybrid Search Configuration
    semantic_weight: float = Field(default=0.6, alias="SEMANTIC_WEIGHT", ge=0.0, le=1.0)
    structural_weight: float = Field(default=0.4, alias="STRUCTURAL_WEIGHT", ge=0.0, le=1.0)
    
    # Performance Configuration
    embedding_cache_ttl: int = Field(default=7200, alias="EMBEDDING_CACHE_TTL", gt=0)
    max_embedding_workers: int = Field(default=4, alias="MAX_EMBEDDING_WORKERS", gt=0, le=16)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )
    
    @model_validator(mode='after')
    def validate_weights(self):
        """Validate that semantic and structural weights sum to 1.0."""
        total_weight = self.semantic_weight + self.structural_weight
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError("Semantic and structural weights must sum to 1.0")
        return self


class AppConfig(BaseSettings):
    """Application configuration settings."""
    
    # Environment
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=True, alias="DEBUG")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    
    # Agent Configuration
    max_concurrent_agents: int = Field(default=5, alias="MAX_CONCURRENT_AGENTS")
    agent_timeout_seconds: int = Field(default=300, alias="AGENT_TIMEOUT_SECONDS")
    
    # Performance
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")
    max_repository_size_mb: int = Field(default=1000, alias="MAX_REPOSITORY_SIZE_MB")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )


class Config:
    """Main configuration class."""
    
    def __init__(self) -> None:
        self.database = DatabaseConfig()
        self.app = AppConfig()
        self.llm = LLMConfig()
        self.semantic = SemanticSearchConfig()
    
    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment and .env file."""
        return cls()


# Global configuration instance
config = Config.load()