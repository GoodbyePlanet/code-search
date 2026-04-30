from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from server.config import settings
from server.embeddings.base import EmbeddingProvider
from server.embeddings.jina import get_embedding_provider
from server.indexer.github_source import GitHubCommit, fetch_commits_with_diffs, list_commits
from server.store.commit_store import CommitStore

logger = logging.getLogger(__name__)


def _build_embedding_text(commit: GitHubCommit, service_name: str) -> str:
    parts = [
        f"Commit to {service_name} by {commit.author_name}",
        f"Date: {commit.committed_at}",
        "",
        commit.message,
    ]
    if commit.files:
        file_list = ", ".join(f.filename for f in commit.files[:20])
        parts.append(f"\nFiles changed: {file_list}")
    return "\n".join(parts)


_MAX_FILES_IN_PAYLOAD = 50
_MAX_PATCH_CHARS = 2000


def _commit_to_payload(commit: GitHubCommit, service_name: str) -> dict[str, Any]:
    files_payload = [
        {
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "patch": (f.patch or "")[:_MAX_PATCH_CHARS],
        }
        for f in commit.files[:_MAX_FILES_IN_PAYLOAD]
    ]
    return {
        "sha": commit.sha,
        "service": service_name,
        "message": commit.message,
        "author_name": commit.author_name,
        "author_email": commit.author_email,
        "committed_at": commit.committed_at,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "files": files_payload,
        "has_diff": len(files_payload) > 0,
        "diff_truncated": len(commit.files) > _MAX_FILES_IN_PAYLOAD,
    }


class GitHistoryPipeline:
    def __init__(self, store: CommitStore) -> None:
        self._store = store
        self._embedder: EmbeddingProvider = get_embedding_provider()

    async def index_service(self, service_name: str, force: bool = False) -> dict[str, int]:
        await self._store.ensure_collection()
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

        shas_without_diffs = set(await self._store.get_commits_without_diffs(svc.name)) if not force else set()
        commits_needing_diffs = [c for c in commits if c.sha in shas_without_diffs]

        if not new_commits and not commits_needing_diffs:
            return {"new": 0, "skipped": skipped, "diff_updated": 0}

        diff_updated = 0

        if new_commits:
            logger.info("Fetching diffs for %d new commits in %s", len(new_commits), service_name)
            new_commits = await fetch_commits_with_diffs(
                settings.github_token, svc.github_repo, new_commits
            )
            texts = [_build_embedding_text(c, svc.name) for c in new_commits]
            try:
                vectors = await self._embedder.embed_batch(texts)
            except Exception as exc:
                logger.error("Embedding failed for %s git history: %s", service_name, exc)
                return {"error": 1, "new": 0, "skipped": skipped, "diff_updated": 0}
            payloads = [_commit_to_payload(c, svc.name) for c in new_commits]
            await self._store.upsert_commits(svc.name, payloads, vectors)
            logger.info("Indexed %d new commits for %s", len(new_commits), service_name)

        if commits_needing_diffs:
            logger.info("Fetching diffs for %d existing commits in %s", len(commits_needing_diffs), service_name)
            commits_needing_diffs = await fetch_commits_with_diffs(
                settings.github_token, svc.github_repo, commits_needing_diffs
            )
            payloads = [_commit_to_payload(c, svc.name) for c in commits_needing_diffs]
            await self._store.update_commit_diffs(svc.name, payloads)
            diff_updated = len(commits_needing_diffs)

        return {"new": len(new_commits), "skipped": skipped, "diff_updated": diff_updated}

    async def index_all(self, force: bool = False) -> dict[str, Any]:
        services = settings.load_services()
        results: dict[str, Any] = {}
        for svc in services:
            logger.info("Indexing git history for: %s", svc.name)
            results[svc.name] = await self.index_service(svc.name, force=force)
        return results
