#!/usr/bin/env python3
"""
Validate that hybrid (BM25 + dense + RRF) search outperforms dense-only baseline.

Runs every entry in TEST_CASES through both search modes, measures ranking
positions, and checks two success criteria:

  CRITERION 1  Exact / tokenized / snake_case / prefix queries rank on average
               ≥2 positions higher (lower rank number) in hybrid mode.
  CRITERION 2  No semantic query drops more than 1 position in hybrid mode
               compared with dense-only.

Populate TEST_CASES before running:
    uv run scripts/seed_test_cases.py --per-bucket 5
    # paste the output block below, replacing the empty list

Usage:
    uv run scripts/validate_hybrid.py
    uv run scripts/validate_hybrid.py --embeddings-url http://localhost:8087
    uv run scripts/validate_hybrid.py --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass

import httpx
from fastembed.sparse.bm25 import Bm25
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
    SparseVector,
)

# ---------------------------------------------------------------------------
# TEST_CASES  — populate via: uv run scripts/seed_test_cases.py
# Each entry: (query, expected_symbol_name, kind)
# kind: "exact" | "tokenized" | "snake_case" | "prefix" | "semantic"
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[str, str, str]] = [
    # Paste seed_test_cases.py output here.
    # Example:
    #   ("PlaceOrderRequest",  "PlaceOrderRequest",  "exact"),
    #   ("place order request","PlaceOrderRequest",  "tokenized"),
    #   ("place order",        "PlaceOrderRequest",  "prefix"),
]

# Kinds that should benefit from BM25 (identifier-based queries)
IDENTIFIER_KINDS = {"exact", "tokenized", "snake_case", "prefix"}
SEMANTIC_KIND = "semantic"

DEFAULT_LIMIT = 10


# ---------------------------------------------------------------------------
# BM25 helper  (mirrors server/embeddings/bm25.py without importing server/)
# ---------------------------------------------------------------------------

def _split_code_identifiers(text: str) -> str:
    """Inline of server/embeddings/code_tokenizer.py:split_code_identifiers."""
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    expanded = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", expanded)
    expanded = expanded.replace("_", " ").replace("-", " ")
    return text + "\n" + expanded


async def get_sparse_vector(model: Bm25, text: str) -> SparseVector:
    loop = asyncio.get_running_loop()
    prepared = _split_code_identifiers(text)
    [embedding] = await loop.run_in_executor(None, lambda: list(model.query_embed(prepared)))
    return SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())


# ---------------------------------------------------------------------------
# Dense embedding helper  (mirrors server/embeddings/jina.py)
# ---------------------------------------------------------------------------

async def get_dense_vector(client: httpx.AsyncClient, embeddings_url: str, text: str) -> list[float]:
    response = await client.post(
        f"{embeddings_url}/embed",
        json={"inputs": [text]},
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, list):
        return data[0]
    return data["data"][0]["embedding"]


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

async def search_hybrid(
    client: AsyncQdrantClient,
    collection: str,
    dense: list[float],
    sparse: SparseVector,
    limit: int,
) -> list[str]:
    result = await client.query_points(
        collection_name=collection,
        prefetch=[
            Prefetch(query=dense, using="text-dense", limit=limit * 2),
            Prefetch(query=sparse, using="text-sparse", limit=limit * 2),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=["symbol_name"],
    )
    return [p.payload.get("symbol_name", "") for p in result.points]


async def search_dense(
    client: AsyncQdrantClient,
    collection: str,
    dense: list[float],
    limit: int,
) -> list[str]:
    result = await client.query_points(
        collection_name=collection,
        query=dense,
        using="text-dense",
        limit=limit,
        with_payload=["symbol_name"],
    )
    return [p.payload.get("symbol_name", "") for p in result.points]


def get_rank(names: list[str], expected: str, fallback: int) -> int:
    try:
        return names.index(expected) + 1
    except ValueError:
        return fallback


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class Result:
    query: str
    expected: str
    kind: str
    dense_rank: int
    hybrid_rank: int

    @property
    def delta(self) -> int:
        return self.dense_rank - self.hybrid_rank


def print_report(results: list[Result], limit: int) -> bool:
    col_q = max(len(r.query) for r in results)
    col_q = min(col_q, 48)
    col_e = max(len(r.expected) for r in results)
    col_e = min(col_e, 40)

    header = (
        f"{'Query':<{col_q}}  {'Expected':<{col_e}}  {'Kind':<10}  "
        f"{'Dense':>5}  {'Hybrid':>6}  {'Delta':>5}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        q = r.query if len(r.query) <= col_q else r.query[: col_q - 1] + "…"
        e = r.expected if len(r.expected) <= col_e else r.expected[: col_e - 1] + "…"
        d_str = str(r.dense_rank) if r.dense_rank <= limit else f">{limit}"
        h_str = str(r.hybrid_rank) if r.hybrid_rank <= limit else f">{limit}"
        sign = "+" if r.delta > 0 else ""
        print(
            f"{q:<{col_q}}  {e:<{col_e}}  {r.kind:<10}  "
            f"{d_str:>5}  {h_str:>6}  {sign}{r.delta:>4}"
        )

    print()

    # Per-kind summary
    from collections import defaultdict

    by_kind: dict[str, list[int]] = defaultdict(list)
    for r in results:
        by_kind[r.kind].append(r.delta)

    print(f"{'Kind':<12}  {'Count':>5}  {'Avg delta':>9}  {'Min delta':>9}  {'Max delta':>9}")
    print("-" * 52)
    for kind in ["exact", "tokenized", "snake_case", "prefix", "semantic"]:
        deltas = by_kind.get(kind, [])
        if not deltas:
            continue
        avg = sum(deltas) / len(deltas)
        print(
            f"{kind:<12}  {len(deltas):>5}  {avg:>+9.2f}  "
            f"{min(deltas):>+9}  {max(deltas):>+9}"
        )
    print()

    # Success criteria
    identifier_deltas = [r.delta for r in results if r.kind in IDENTIFIER_KINDS]
    semantic_deltas = [r.delta for r in results if r.kind == SEMANTIC_KIND]

    passed = True

    if identifier_deltas:
        avg_id = sum(identifier_deltas) / len(identifier_deltas)
        c1_pass = avg_id >= 2.0
        status = "PASS" if c1_pass else "FAIL"
        print(f"[{status}] CRITERION 1 — identifier avg delta: {avg_id:+.2f} (threshold: ≥+2.0)")
        if not c1_pass:
            passed = False
    else:
        print("[SKIP] CRITERION 1 — no identifier-kind test cases found")

    if semantic_deltas:
        worst = min(semantic_deltas)
        c2_pass = worst >= -1
        status = "PASS" if c2_pass else "FAIL"
        print(f"[{status}] CRITERION 2 — semantic worst delta: {worst:+d} (threshold: ≥-1)")
        if not c2_pass:
            passed = False
    else:
        print("[SKIP] CRITERION 2 — no semantic-kind test cases found")

    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_test_cases(path: str) -> list[tuple[str, str, str]]:
    with open(path) as f:
        return [(q, e, k) for q, e, k in json.load(f)]


async def main(
    url: str,
    collection: str,
    embeddings_url: str,
    limit: int,
    test_cases_file: str | None,
    kinds: set[str] | None,
) -> None:
    cases: list[tuple[str, str, str]] = (
        _load_test_cases(test_cases_file) if test_cases_file else list(TEST_CASES)
    )
    if kinds:
        cases = [(q, e, k) for q, e, k in cases if k in kinds]
    if not cases:
        print(
            "TEST_CASES is empty.\n"
            "Run:  uv run scripts/seed_test_cases.py\n"
            "then paste the output into this file, or use --test-cases-file.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading BM25 model…", file=sys.stderr)
    bm25 = Bm25("Qdrant/bm25")

    qdrant = AsyncQdrantClient(url=url)
    fallback = limit + 1

    async with httpx.AsyncClient() as http:
        results: list[Result] = []

        for i, (query, expected, kind) in enumerate(cases, 1):
            print(f"  [{i}/{len(cases)}] {query!r}…", file=sys.stderr, end="\r")
            dense = await get_dense_vector(http, embeddings_url, query)
            sparse = await get_sparse_vector(bm25, query)

            hybrid_names = await search_hybrid(qdrant, collection, dense, sparse, limit)
            dense_names = await search_dense(qdrant, collection, dense, limit)

            results.append(
                Result(
                    query=query,
                    expected=expected,
                    kind=kind,
                    dense_rank=get_rank(dense_names, expected, fallback),
                    hybrid_rank=get_rank(hybrid_names, expected, fallback),
                )
            )

    print(" " * 60, file=sys.stderr, end="\r")
    print(f"Results for {len(results)} queries (search limit={limit})\n", flush=True)
    passed = print_report(results, limit)

    await qdrant.close()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant URL (default: $QDRANT_URL or http://localhost:6333)",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "code_symbols"),
        help="Collection name (default: $QDRANT_COLLECTION or code_symbols)",
    )
    parser.add_argument(
        "--embeddings-url",
        default=os.getenv("EMBEDDINGS_URL", "http://localhost:8087"),
        help="Jina TEI server URL (default: $EMBEDDINGS_URL or http://localhost:8087)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        metavar="N",
        help=f"Number of results to fetch per search (default: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--test-cases-file",
        default=None,
        metavar="FILE",
        help="JSON file produced by seed_test_cases.py --output-json. "
             "Overrides the hardcoded TEST_CASES list.",
    )
    parser.add_argument(
        "--kinds",
        nargs="+",
        choices=["exact", "tokenized", "snake_case", "prefix", "semantic"],
        default=None,
        metavar="KIND",
        help="Only validate test cases of these kinds (default: all).",
    )
    args = parser.parse_args()
    asyncio.run(main(
        args.url, args.collection, args.embeddings_url, args.limit,
        args.test_cases_file,
        set(args.kinds) if args.kinds else None,
    ))
