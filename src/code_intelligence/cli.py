"""Command-line interface for the Code Intelligence System."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import typer
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax

from .config import config
from .logging import get_logger


app = typer.Typer(
    name="code-intel",
    help="Multi-Agent Code Intelligence System CLI",
    add_completion=False
)
console = Console()
logger = get_logger(__name__)

# Configuration file path
CONFIG_FILE = Path.home() / ".code-intelligence" / "config.json"

# Default API base URL
DEFAULT_API_URL = f"http://{config.app.api_host}:{config.app.api_port}/api/v1"


class CLIConfig:
    """CLI configuration management."""
    
    def __init__(self):
        self.config_file = CONFIG_FILE
        self.config_file.parent.mkdir(exist_ok=True)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                return json.loads(self.config_file.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "api_url": DEFAULT_API_URL,
            "output_format": "table",
            "timeout": 300,
            "auto_open_results": False
        }
    
    def save_config(self):
        """Save configuration to file."""
        try:
            self.config_file.write_text(json.dumps(self._config, indent=2))
        except OSError as e:
            console.print(f"[red]Failed to save config: {e}[/red]")
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self._config[key] = value
        self.save_config()


cli_config = CLIConfig()


def get_api_client() -> httpx.Client:
    """Get HTTP client for API requests."""
    return httpx.Client(
        base_url=cli_config.get("api_url"),
        timeout=cli_config.get("timeout", 300)
    )


@app.command()
def version() -> None:
    """Show version information."""
    console.print("[green]Multi-Agent Code Intelligence System v1.0.0[/green]")


@app.command()
def setup(
    reset: bool = typer.Option(False, "--reset", help="Reset existing schema and data"),
    neo4j_only: bool = typer.Option(False, "--neo4j-only", help="Initialize only Neo4j"),
    supabase_only: bool = typer.Option(False, "--supabase-only", help="Initialize only Supabase"),
) -> None:
    """Set up databases and initial configuration."""
    from .database.init import db_initializer
    
    console.print("Setting up Code Intelligence System...")
    
    if reset:
        console.print("⚠️  Warning: This will reset all existing data!")
        if not typer.confirm("Are you sure you want to continue?"):
            console.print("Setup cancelled.")
            return
    
    try:
        if neo4j_only:
            import asyncio
            asyncio.run(db_initializer.initialize_neo4j(reset=reset))
            console.print("✅ Neo4j setup completed!")
        elif supabase_only:
            import asyncio
            asyncio.run(db_initializer.initialize_supabase())
            console.print("✅ Supabase setup completed!")
        else:
            db_initializer.initialize_sync(reset=reset)
            console.print("✅ Database setup completed!")
            
    except Exception as e:
        console.print(f"❌ Setup failed: {e}")
        raise typer.Exit(1)


@app.command()
def config_cmd(
    key: Optional[str] = typer.Argument(None, help="Configuration key to get/set"),
    value: Optional[str] = typer.Argument(None, help="Configuration value to set"),
    list_all: bool = typer.Option(False, "--list", "-l", help="List all configuration")
):
    """Manage CLI configuration."""
    if list_all:
        table = Table(title="CLI Configuration")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        
        for k, v in cli_config._config.items():
            table.add_row(k, str(v))
        
        console.print(table)
        return
    
    if key is None:
        console.print("[red]Please specify a configuration key[/red]")
        raise typer.Exit(1)
    
    if value is None:
        # Get configuration value
        val = cli_config.get(key)
        if val is not None:
            console.print(f"{key}: {val}")
        else:
            console.print(f"[red]Configuration key '{key}' not found[/red]")
            raise typer.Exit(1)
    else:
        # Set configuration value
        cli_config.set(key, value)
        console.print(f"[green]Set {key} = {value}[/green]")


@app.command()
def query(
    question: str = typer.Argument(..., help="Natural language question about code"),
    repository: str = typer.Option(..., "--repo", "-r", help="Repository URL"),
    output: str = typer.Option("table", "--output", "-o", help="Output format (table, json, markdown)"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for query completion"),
    max_commits: int = typer.Option(100, "--max-commits", help="Maximum commits to analyze"),
    include_tests: bool = typer.Option(False, "--include-tests", help="Include test files in analysis")
):
    """Submit a query for code analysis."""
    console.print(f"[blue]Submitting query:[/blue] {question}")
    console.print(f"[blue]Repository:[/blue] {repository}")
    
    try:
        with get_api_client() as client:
            # Submit query
            response = client.post("/queries/", json={
                "repository_url": repository,
                "query": question,
                "options": {
                    "max_commits": max_commits,
                    "include_tests": include_tests
                }
            })
            response.raise_for_status()
            
            query_data = response.json()
            query_id = query_data["query_id"]
            
            console.print(f"[green]Query submitted successfully![/green]")
            console.print(f"[blue]Query ID:[/blue] {query_id}")
            
            if not wait:
                console.print(f"[yellow]Use 'code-intel status {query_id}' to check progress[/yellow]")
                return
            
            # Wait for completion with progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Processing query...", total=100)
                
                while True:
                    status_response = client.get(f"/queries/{query_id}")
                    status_response.raise_for_status()
                    status_data = status_response.json()
                    
                    if status_data["status"] == "completed":
                        progress.update(task, completed=100)
                        break
                    elif status_data["status"] == "failed":
                        console.print(f"[red]Query failed: {status_data.get('error', 'Unknown error')}[/red]")
                        raise typer.Exit(1)
                    elif status_data["status"] == "processing" and status_data.get("progress"):
                        prog = status_data["progress"]
                        progress.update(
                            task, 
                            completed=prog["progress_percentage"],
                            description=f"Processing query... ({prog['current_agent']})"
                        )
                    
                    time.sleep(2)
            
            # Display results
            display_query_results(status_data, output)
            
    except httpx.HTTPError as e:
        console.print(f"[red]API request failed: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status(
    query_id: str = typer.Argument(..., help="Query ID to check"),
    output: str = typer.Option("table", "--output", "-o", help="Output format (table, json)")
):
    """Check the status of a query."""
    try:
        with get_api_client() as client:
            response = client.get(f"/queries/{query_id}")
            response.raise_for_status()
            
            data = response.json()
            
            if output == "json":
                console.print_json(json.dumps(data, indent=2))
                return
            
            # Display status in table format
            table = Table(title=f"Query Status: {query_id}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Status", data["status"])
            table.add_row("Created", data["created_at"])
            table.add_row("Updated", data["updated_at"])
            
            if data.get("progress"):
                prog = data["progress"]
                table.add_row("Current Agent", prog["current_agent"])
                table.add_row("Progress", f"{prog['progress_percentage']:.1f}%")
                table.add_row("Current Step", prog["current_step"])
            
            if data.get("completed_at"):
                table.add_row("Completed", data["completed_at"])
            
            if data.get("error"):
                table.add_row("Error", data["error"])
            
            console.print(table)
            
            # Display results if completed
            if data["status"] == "completed" and data.get("results"):
                display_query_results(data, "table")
                
    except httpx.HTTPError as e:
        console.print(f"[red]API request failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def repositories(
    action: str = typer.Argument("list", help="Action: list, add, delete, analyze"),
    url: Optional[str] = typer.Option(None, "--url", help="Repository URL (for add action)"),
    name: Optional[str] = typer.Option(None, "--name", help="Repository name (for add action)"),
    repo_id: Optional[str] = typer.Option(None, "--id", help="Repository ID (for delete/analyze actions)")
):
    """Manage repositories."""
    try:
        with get_api_client() as client:
            if action == "list":
                response = client.get("/repositories/")
                response.raise_for_status()
                
                repos = response.json()
                
                if not repos:
                    console.print("[yellow]No repositories found[/yellow]")
                    return
                
                table = Table(title="Repositories")
                table.add_column("ID", style="cyan")
                table.add_column("Name", style="green")
                table.add_column("Status", style="yellow")
                table.add_column("Commits", style="blue")
                table.add_column("Languages", style="magenta")
                
                for repo in repos:
                    table.add_row(
                        repo["id"][:8] + "...",
                        repo["name"],
                        repo["status"],
                        str(repo["commit_count"]),
                        ", ".join(repo["supported_languages"][:3])
                    )
                
                console.print(table)
                
            elif action == "add":
                if not url:
                    console.print("[red]Repository URL is required for add action[/red]")
                    raise typer.Exit(1)
                
                response = client.post("/repositories/", json={
                    "url": url,
                    "name": name,
                    "auto_sync": True
                })
                response.raise_for_status()
                
                repo = response.json()
                console.print(f"[green]Repository added successfully![/green]")
                console.print(f"[blue]ID:[/blue] {repo['id']}")
                console.print(f"[blue]Name:[/blue] {repo['name']}")
                
            elif action == "delete":
                if not repo_id:
                    console.print("[red]Repository ID is required for delete action[/red]")
                    raise typer.Exit(1)
                
                response = client.delete(f"/repositories/{repo_id}")
                response.raise_for_status()
                
                console.print("[green]Repository deleted successfully![/green]")
                
            elif action == "analyze":
                if not repo_id:
                    console.print("[red]Repository ID is required for analyze action[/red]")
                    raise typer.Exit(1)
                
                response = client.post(f"/repositories/{repo_id}/analyze")
                response.raise_for_status()
                
                console.print("[green]Repository analysis started![/green]")
                
            else:
                console.print(f"[red]Unknown action: {action}[/red]")
                raise typer.Exit(1)
                
    except httpx.HTTPError as e:
        console.print(f"[red]API request failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def history(
    limit: int = typer.Option(10, "--limit", "-l", help="Number of queries to show"),
    output: str = typer.Option("table", "--output", "-o", help="Output format (table, json)")
):
    """Show query history."""
    try:
        with get_api_client() as client:
            response = client.get(f"/queries/?page_size={limit}")
            response.raise_for_status()
            
            data = response.json()
            queries = data["queries"]
            
            if output == "json":
                console.print_json(json.dumps(queries, indent=2))
                return
            
            if not queries:
                console.print("[yellow]No queries found[/yellow]")
                return
            
            table = Table(title="Query History")
            table.add_column("ID", style="cyan")
            table.add_column("Query", style="green", max_width=50)
            table.add_column("Repository", style="blue")
            table.add_column("Status", style="yellow")
            table.add_column("Created", style="magenta")
            
            for query in queries:
                table.add_row(
                    query["query_id"][:8] + "...",
                    query["query"][:47] + "..." if len(query["query"]) > 50 else query["query"],
                    query["repository_name"],
                    query["status"],
                    query["created_at"][:16]
                )
            
            console.print(table)
            
    except httpx.HTTPError as e:
        console.print(f"[red]API request failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def export(
    query_id: str = typer.Argument(..., help="Query ID to export"),
    format: str = typer.Option("json", "--format", "-f", help="Export format (json, markdown)"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path")
):
    """Export query results."""
    try:
        with get_api_client() as client:
            response = client.post(f"/queries/{query_id}/export", json={
                "query_id": query_id,
                "format": format,
                "include_citations": True,
                "include_metadata": False
            })
            response.raise_for_status()
            
            export_data = response.json()
            console.print(f"[green]Export created successfully![/green]")
            console.print(f"[blue]Export ID:[/blue] {export_data['export_id']}")
            console.print(f"[blue]Download URL:[/blue] {export_data['download_url']}")
            console.print(f"[blue]Expires:[/blue] {export_data['expires_at']}")
            
    except httpx.HTTPError as e:
        console.print(f"[red]API request failed: {e}[/red]")
        raise typer.Exit(1)


def display_query_results(data: Dict[str, Any], output_format: str):
    """Display query results in the specified format."""
    if not data.get("results"):
        return
    
    results = data["results"]
    
    if output_format == "json":
        console.print_json(json.dumps(results, indent=2))
        return
    
    # Display summary
    console.print("\n")
    console.print(Panel(
        results["summary"],
        title="Analysis Summary",
        border_style="green"
    ))
    
    # Display confidence and timing
    console.print(f"\n[green]Confidence Score:[/green] {results['confidence_score']:.1%}")
    console.print(f"[blue]Processing Time:[/blue] {results['processing_time_seconds']:.1f}s")
    
    # Display findings
    if results.get("findings"):
        console.print("\n[bold]Detailed Findings:[/bold]")
        
        for i, finding in enumerate(results["findings"], 1):
            console.print(f"\n[cyan]{i}. {finding['agent_name']} - {finding['finding_type']}[/cyan]")
            console.print(f"[green]Confidence:[/green] {finding['confidence']:.1%}")
            console.print(finding["content"])
            
            if finding.get("citations"):
                console.print("[yellow]Citations:[/yellow]")
                for citation in finding["citations"]:
                    console.print(f"  • {citation['file_path']}")
                    if citation.get("line_number"):
                        console.print(f"    Line {citation['line_number']}")
                    if citation.get("description"):
                        console.print(f"    {citation['description']}")


@app.command()
def health():
    """Check API health status."""
    try:
        with get_api_client() as client:
            response = client.get("/health/detailed")
            response.raise_for_status()
            
            data = response.json()
            
            console.print(f"[green]API Status:[/green] {data['status']}")
            console.print(f"[blue]Version:[/blue] {data['version']}")
            
            table = Table(title="Service Health")
            table.add_column("Service", style="cyan")
            table.add_column("Status", style="green")
            
            for service, status in data["services"].items():
                table.add_row(service, status)
            
            console.print(table)
            
    except httpx.HTTPError as e:
        console.print(f"[red]API health check failed: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()