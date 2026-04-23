from __future__ import annotations

from server.store.qdrant import QdrantStore

_store: QdrantStore | None = None


def get_store() -> QdrantStore:
    if _store is None:
        raise RuntimeError("Store not initialized")
    return _store


def set_store(store: QdrantStore) -> None:
    global _store
    _store = store
