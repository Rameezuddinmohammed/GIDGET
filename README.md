# Multi-Agent Code Intelligence System

A next-generation AI-powered platform that enables developers to ask natural language questions about code evolution across multiple versions of a codebase.

## Features

- **Multi-Agent Architecture**: Specialized AI agents working in parallel for comprehensive analysis
- **Temporal Code Analysis**: Track code evolution across git history
- **Semantic Search**: Find code by concept rather than exact text matching
- **Verification-First Design**: All findings verified against actual code and git history
- **Multi-Language Support**: Python, JavaScript, and TypeScript codebases

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- Neo4j database
- Supabase account

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multi-agent-code-intelligence
```

2. Install dependencies:
```bash
poetry install
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your database credentials
```

4. Set up databases (see Database Setup section)

5. Run the system:
```bash
poetry run code-intel --help
```

## Database Setup

### Neo4j Setup
1. Install Neo4j Desktop or use Neo4j Cloud
2. Create a new database
3. Update NEO4J_* variables in .env

### Supabase Setup
1. Create a new Supabase project
2. Enable pgvector extension
3. Update SUPABASE_* variables in .env

## Development

### Running Tests
```bash
poetry run pytest
```

### Code Formatting
```bash
poetry run black .
poetry run isort .
```

### Type Checking
```bash
poetry run mypy src/
```

## Architecture

The system is built on three main planes:
- **Data Plane**: Neo4j graph database + Supabase/PostgreSQL
- **Intelligence Plane**: Multi-agent orchestration with LangGraph
- **Experience Plane**: REST API, WebSocket, Web UI, and CLI interfaces

- ![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/Rameezuddinmohammed/GIDGET?utm_source=oss&utm_medium=github&utm_campaign=Rameezuddinmohammed%2FGIDGET&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)

## License

[License information]
