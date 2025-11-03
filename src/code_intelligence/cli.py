"""Command-line interface for the Code Intelligence System."""

import typer
from typing import Optional

from .logging import get_logger

app = typer.Typer(
    name="code-intel",
    help="Multi-Agent Code Intelligence System CLI",
    add_completion=False,
)

logger = get_logger(__name__)


@app.command()
def version() -> None:
    """Show version information."""
    from . import __version__
    typer.echo(f"Multi-Agent Code Intelligence System v{__version__}")


@app.command()
def setup(
    reset: bool = typer.Option(False, "--reset", help="Reset existing schema and data"),
    neo4j_only: bool = typer.Option(False, "--neo4j-only", help="Initialize only Neo4j"),
    supabase_only: bool = typer.Option(False, "--supabase-only", help="Initialize only Supabase"),
) -> None:
    """Set up databases and initial configuration."""
    from .database.init import db_initializer
    
    typer.echo("Setting up Code Intelligence System...")
    
    if reset:
        typer.echo("⚠️  Warning: This will reset all existing data!")
        if not typer.confirm("Are you sure you want to continue?"):
            typer.echo("Setup cancelled.")
            return
    
    try:
        if neo4j_only:
            import asyncio
            asyncio.run(db_initializer.initialize_neo4j(reset=reset))
            typer.echo("✅ Neo4j setup completed!")
        elif supabase_only:
            import asyncio
            asyncio.run(db_initializer.initialize_supabase())
            typer.echo("✅ Supabase setup completed!")
        else:
            db_initializer.initialize_sync(reset=reset)
            typer.echo("✅ Database setup completed!")
            
    except Exception as e:
        typer.echo(f"❌ Setup failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def health() -> None:
    """Check database health and connectivity."""
    import asyncio
    from .database.init import db_initializer
    
    typer.echo("Checking database health...")
    
    try:
        health_status = asyncio.run(db_initializer.health_check())
        
        for db_name, status in health_status.items():
            if status["status"] == "healthy":
                typer.echo(f"✅ {db_name.upper()}: {status['status']}")
                for key, value in status["details"].items():
                    typer.echo(f"   {key}: {value}")
            else:
                typer.echo(f"❌ {db_name.upper()}: {status['status']}")
                typer.echo(f"   Error: {status['details'].get('error', 'Unknown error')}")
                
    except Exception as e:
        typer.echo(f"❌ Health check failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def query(
    repository: str = typer.Argument(..., help="Repository path or URL"),
    question: str = typer.Argument(..., help="Natural language query"),
    output_format: str = typer.Option("text", "--format", "-f", help="Output format (text, json)"),
) -> None:
    """Ask a question about code evolution."""
    logger.info("Processing query", repository=repository, question=question)
    typer.echo(f"Analyzing repository: {repository}")
    typer.echo(f"Question: {question}")
    # Query processing will be implemented in subsequent tasks
    typer.echo("Query processing not yet implemented.")


if __name__ == "__main__":
    app()