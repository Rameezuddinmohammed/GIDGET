"""Configuration management for the Code Intelligence System."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    
    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment and .env file."""
        return cls()


# Global configuration instance
config = Config.load()