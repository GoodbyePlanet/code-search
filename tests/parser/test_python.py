from __future__ import annotations

from server.parser.python import PythonParser


def test_empty_file_returns_no_symbols():
    assert PythonParser().parse_file(b"", "svc/empty.py") == []


def test_canonical_api_fixture(read_fixture):
    src = read_fixture("python/api.py")
    syms = PythonParser().parse_file(src, "svc/api.py")

    by_name = {s.name: s for s in syms}
    assert set(by_name) == {"ChatRequest", "chat", "helper"}

    assert by_name["ChatRequest"].symbol_type == "pydantic_model"
    assert "BaseModel" in by_name["ChatRequest"].extras["bases"]

    chat = by_name["chat"]
    assert chat.symbol_type == "api_route"
    assert chat.extras["is_async"] is True
    assert chat.extras["http_method"] == "POST"
    assert chat.extras["http_route"] == "/api/chat"

    helper = by_name["helper"]
    assert helper.symbol_type == "function"
    assert helper.extras["is_async"] is False

    for s in syms:
        assert s.language == "python"
        assert s.package == "svc.api"
