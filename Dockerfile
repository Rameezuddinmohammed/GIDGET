# Multi-stage Docker build for Code Intelligence System
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY requirements.txt* ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e .

# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-asyncio pytest-cov black isort mypy

# Copy source code
COPY . .

# Change ownership to app user
RUN chown -R appuser:appuser /app

USER appuser

# Expose ports
EXPOSE 8000 8001

# Default command for development
CMD ["python", "-m", "src.code_intelligence.api.main"]

# Production stage
FROM base as production

# Copy only necessary files
COPY src/ ./src/
COPY pyproject.toml ./
COPY README.md ./

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health/ || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["python", "-m", "uvicorn", "src.code_intelligence.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]