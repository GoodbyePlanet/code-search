from __future__ import annotations

from server.parser.typescript import TypeScriptParser


def test_empty_file_returns_no_symbols():
    assert TypeScriptParser().parse_file(b"", "svc/empty.ts") == []


def test_canonical_react_component_fixture(read_fixture):
    src = read_fixture("typescript/Counter.tsx")
    syms = TypeScriptParser().parse_file(src, "svc/Counter.tsx")

    by_name = {s.name: s for s in syms}
    assert set(by_name) == {"CounterProps", "useCounter", "Counter"}

    assert by_name["CounterProps"].symbol_type == "interface"
    assert by_name["useCounter"].symbol_type == "react_hook"
    assert by_name["Counter"].symbol_type == "react_component"

    for s in syms:
        assert s.language == "typescript"
        assert s.file_path == "svc/Counter.tsx"
