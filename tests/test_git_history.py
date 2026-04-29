from __future__ import annotations

from server.indexer.git_history import _build_embedding_text, _commit_to_payload
from server.indexer.github_source import GitHubCommit


def _commit(message: str = "Fix bug in auth") -> GitHubCommit:
    return GitHubCommit(
        sha="abc123def456",
        message=message,
        author_name="Jane Doe",
        author_email="jane@example.com",
        committed_at="2024-03-15T10:00:00Z",
    )


def test_embedding_text_contains_service_and_author():
    text = _build_embedding_text(_commit(), "auth-server")
    assert "auth-server" in text
    assert "Jane Doe" in text


def test_embedding_text_contains_message():
    text = _build_embedding_text(_commit("Refactor payment processing"), "payments")
    assert "Refactor payment processing" in text


def test_embedding_text_contains_date():
    text = _build_embedding_text(_commit(), "svc")
    assert "2024-03-15T10:00:00Z" in text


def test_payload_fields():
    payload = _commit_to_payload(_commit(), "auth-server")
    assert payload["sha"] == "abc123def456"
    assert payload["service"] == "auth-server"
    assert payload["message"] == "Fix bug in auth"
    assert payload["author_name"] == "Jane Doe"
    assert payload["author_email"] == "jane@example.com"
    assert payload["committed_at"] == "2024-03-15T10:00:00Z"
    assert "indexed_at" in payload
