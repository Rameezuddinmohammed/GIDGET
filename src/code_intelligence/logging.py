"""Logging configuration for the Code Intelligence System."""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.types import FilteringBoundLogger

from .config import config


def configure_logging() -> FilteringBoundLogger:
    """Configure structured logging for the application."""
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.app.log_level.upper()),
    )
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    
    if config.app.log_format == "json":
        processors.extend([
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ])
    else:
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, config.app.log_level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()


def get_logger(name: str) -> FilteringBoundLogger:
    """Get a logger instance for the given name."""
    return structlog.get_logger(name)


# Initialize logging
logger = configure_logging()