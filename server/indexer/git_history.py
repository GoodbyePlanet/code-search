from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from server.config import settings
from server.embeddings.base import EmbeddingProvider
from server.embeddings.jina import get_embedding_provider
from server.indexer.github_source import GitHubCommit, list_commits
from server.store.commit_store import CommitStore

logger = logging.getLogger(__name__)


def _build_embedding_text(commit: GitHubCommit, service_name: str) -> str:
    return "\n".join([
        f"Commit to {service_name} by {commit.author_name}",
        f"Date: {commit.committed_at}",
        "",
        commit.message,
    ])


def _commit_to_payload(commit: GitHubCommit, service_name: str) -> dict[str, Any]:
    return {
        "sha": commit.sha,
        "service": service_name,
        "message": commit.message,
        "author_name": commit.author_name,
        "author_email": commit.author_email,
        "committed_at": commit.committed_at,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }


class GitHistoryPipeline:
    def __init__(self, store: CommitStore) -> None:
        self._store = store
        self._embedder: EmbeddingProvider = get_embedding_provider()

    async def index_service(self, service_name: str, force: bool = False) -> dict[str, int]:
        services = settings.load_services()
        svc = next((s for s in services if s.name == service_name), None)
        if svc is None:
            return {"error": 1, "new": 0, "skipped": 0}

        commits = await list_commits(
            settings.github_token, svc.github_repo, svc.github_ref,
            root=svc.root, max_commits=settings.git_history_max_commits,
        )

        existing_shas = set() if force else await self._store.get_indexed_shas(svc.name)
        new_commits = [c for c in commits if c.sha not in existing_shas]
        skipped = len(commits) - len(new_commits)

        if not new_commits:
            return {"new": 0, "skipped": skipped}

        texts = [_build_embedding_text(c, svc.name) for c in new_commits]
        try:
            vectors = await self._embedder.embed_batch(texts)
        except Exception as exc:
            logger.error("Embedding failed for %s git history: %s", service_name, exc)
            return {"error": 1, "new": 0, "skipped": skipped}

        payloads = [_commit_to_payload(c, svc.name) for c in new_commits]
        await self._store.upsert_commits(svc.name, payloads, vectors)

        logger.info("Indexed %d new commits for %s", len(new_commits), service_name)
        return {"new": len(new_commits), "skipped": skipped}

    async def index_all(self, force: bool = False) -> dict[str, Any]:
        services = settings.load_services()
        results: dict[str, Any] = {}
        for svc in services:
            logger.info("Indexing git history for: %s", svc.name)
            results[svc.name] = await self.index_service(svc.name, force=force)
        return results
