from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from server.embeddings.jina import get_embedding_provider
from server.indexer.git_history import GitHistoryPipeline
from server.state import get_commit_store


def register_history_tools(mcp: FastMCP) -> None:

    @mcp.tool()
    async def search_commits(
        query: str,
        service: str | None = None,
        limit: int = 10,
    ) -> str:
        """Search git commit history using natural language.

        Args:
            query: Natural language description of what you're looking for in commit history.
            service: Filter by service name
            limit: Maximum number of results (default 10)
        """
        embedder = get_embedding_provider()
        store = get_commit_store()

        query_vector = await embedder.embed_query(query)
        results = await store.search(query_vector=query_vector, service=service, limit=limit)

        if not results:
            return "No commits found."

        lines = [f"Found {len(results)} commit(s) for: {query!r}\n"]
        for i, hit in enumerate(results, 1):
            p = hit.payload
            sha_short = (p.get("sha") or "")[:8]
            lines.append(f"### {i}. `{sha_short}` — score {hit.score:.3f}")
            lines.append(f"**Service**: {p.get('service')} | **Author**: {p.get('author_name')}")
            lines.append(f"**Date**: {p.get('committed_at')}")
            lines.append("")
            lines.append(p.get("message") or "")
            lines.append("")

        return "\n".join(lines)

    @mcp.tool()
    async def index_history(
        service: str | None = None,
        force: bool = False,
    ) -> str:
        """Index git commit history for one or all services.

        Args:
            service: Name of the service to index. If omitted, all configured services are indexed.
            force: If true, re-index all commits even if already indexed. Defaults to false (incremental).
        """
        store = get_commit_store()
        pipeline = GitHistoryPipeline(store)

        if service:
            result = await pipeline.index_service(service, force=force)
            if "error" in result:
                return f"Service `{service}` not found in config.yaml."
            return (
                f"Git history indexed for `{service}`:\n"
                f"- New commits: {result['new']}\n"
                f"- Skipped (already indexed): {result.get('skipped', 0)}"
            )

        results = await pipeline.index_all(force=force)
        lines = ["Git history indexed for all services:\n"]
        total_new = 0
        for svc_name, r in results.items():
            lines.append(
                f"- **{svc_name}**: {r.get('new', 0)} new commits "
                f"({r.get('skipped', 0)} skipped)"
            )
            total_new += r.get("new", 0)
        lines.append(f"\n**Total**: {total_new} new commits")
        return "\n".join(lines)
