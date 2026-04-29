# code-search

An MCP (Model Context Protocol) server that provides semantic code search across microservices codebases.
It indexes code symbols from GitHub repositories and makes them searchable via natural language queries or
symbol name lookups.

## How it works

1. Fetches source files from configured GitHub repositories
2. Parses code symbols (functions, classes, methods, components) using Tree-sitter
3. Generates embeddings via Jina Code V2
4. Stores vectors in Qdrant for fast semantic search
5. Exposes search tools through the MCP protocol

## Supported languages

Language is detected automatically from file extension or filename — no configuration needed.

- Go
- Java (including Spring annotations)
- Python
- TypeScript / React (including hooks and components)
- Dockerfile
- Docker Compose
- Markdown

## Setup

**Prerequisites:** Python 3.12+, Docker, GitHub token

```bash
# Install dependencies
uv sync

# Copy and configure environment
cp .env.example .env

# Copy and configure services
cp config.example.yaml config.yaml
```

Configure which repositories to index in `config.yaml`:

```yaml
services:
  - name: my-service
    github_repo: owner/repo
    github_ref: main              # optional, defaults to "main"
    root: src/main/java           # optional — limit indexing to this subdirectory (useful for monorepos)
    exclude:                      # optional — skip matching paths
      - "**/vendor/**"
      - "**/node_modules/**"
```

The indexer automatically discovers and indexes all files with recognised extensions. Use `root` to scope a service to a subdirectory within a shared repo, and `exclude` to skip paths you don't want indexed (tests, build artifacts, generated code, etc.).

## Running

```bash
docker-compose up
```

This starts:
- **Qdrant** (vector DB) on port `6333`
- **Jina Embeddings** (TEI) on port `8087`
- **code-search MCP server** on port `8090`

## Tests

```bash
uv sync --group dev
uv run pytest
```

Tests live under `tests/` and snapshot current parser behavior against canonical
fixtures in `tests/fixtures/<language>/`.

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_code` | Semantic search by query, with optional filters for language, service, symbol type |
| `find_symbol` | Look up a symbol by name (exact or fuzzy) |
| `reindex` | Trigger indexing of one or all services |
| `list_indexed_services` | List all indexed services |
| `index_stats` | View collection statistics |

## Environment variables

| Variable | Description |
|----------|-------------|
| `GITHUB_TOKEN` | GitHub token with repo read access |
| `QDRANT_URL` | Qdrant connection URL (default: `http://localhost:6333`) |
| `EMBEDDINGS_URL` | Jina TEI URL (default: `http://localhost:8087`) |
| `MCP_TRANSPORT` | `streamable-http` or `stdio` |
| `MCP_HOST` / `MCP_PORT` | Server bind address (default: `0.0.0.0:8090`) |

## Project structure

```
server/
├── main.py          # MCP server entry point
├── config.py        # Settings and service configuration
├── parser/          # Tree-sitter parsers (Go, Java, Python, TypeScript)
├── embeddings/      # Jina Code V2 embedding client
├── indexer/         # Indexing pipeline and GitHub file fetcher
├── store/           # Qdrant vector store abstraction
└── tools/           # MCP tool implementations
```
