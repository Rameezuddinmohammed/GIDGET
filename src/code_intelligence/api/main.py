"""Main FastAPI application for the Code Intelligence System."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog

from .routes import queries_router, repositories_router, users_router, health_router
from .websocket import router as websocket_router
from .models import ErrorResponse
from ..config import config
from ..logging import get_logger


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Code Intelligence API", extra={
        "version": "1.0.0",
        "environment": config.app.environment
    })
    
    try:
        # Initialize database connections
        from ..database.supabase_client import SupabaseClient
        from ..database.neo4j_client import Neo4jClient
        
        supabase = SupabaseClient()
        await supabase.initialize()
        logger.info("Supabase connection initialized")
        
        neo4j = Neo4jClient()
        await neo4j.initialize()
        logger.info("Neo4j connection initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        # Continue startup even if some services fail
    
    yield
    
    # Shutdown
    logger.info("Shutting down Code Intelligence API")


# Create FastAPI application
app = FastAPI(
    title="Code Intelligence API",
    description="Multi-Agent Code Intelligence System for analyzing code evolution",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # In production, specify allowed hosts
)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", extra={
        "path": request.url.path,
        "method": request.method,
        "exception_type": exc.__class__.__name__
    })
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An internal server error occurred",
            details={"exception_type": exc.__class__.__name__}
        ).model_dump()
    )


# Include routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(queries_router, prefix="/api/v1")
app.include_router(repositories_router, prefix="/api/v1")
app.include_router(users_router, prefix="/api/v1")
app.include_router(websocket_router)  # WebSocket routes don't need prefix


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Code Intelligence API",
        "version": "1.0.0",
        "description": "Multi-Agent Code Intelligence System",
        "docs_url": "/docs",
        "health_url": "/api/v1/health"
    }


# API info endpoint
@app.get("/api/v1/info")
async def api_info():
    """API information endpoint."""
    return {
        "name": "Code Intelligence API",
        "version": "1.0.0",
        "environment": config.app.environment,
        "features": [
            "Natural language code queries",
            "Multi-agent analysis",
            "Temporal code evolution tracking",
            "Semantic code search",
            "Repository management",
            "Real-time progress updates"
        ],
        "endpoints": {
            "queries": "/api/v1/queries",
            "repositories": "/api/v1/repositories",
            "users": "/api/v1/users",
            "health": "/api/v1/health",
            "websocket": "/ws"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.code_intelligence.api.main:app",
        host=config.app.api_host,
        port=config.app.api_port,
        reload=config.app.debug,
        log_level=config.app.log_level.lower()
    )