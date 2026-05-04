#!/usr/bin/env python3
"""
Validate dense (embedding-only) semantic search quality.

Runs every entry in TEST_CASES through dense vector search and measures:
  - Hit@1, Hit@3, Hit@5, Hit@10  (% of queries where expected symbol ranks ≤K)
  - MRR  (Mean Reciprocal Rank: mean of 1/rank, higher is better)
  - Average rank

Success criteria (configurable via --mrr-threshold / --hit-threshold):
  MRR    ≥ 0.5   — expected symbol ranks roughly 2nd on average
  Hit@10 ≥ 0.8   — expected symbol found in top-10 for 80% of queries

Populate TEST_CASES from semantic entries only:
    uv run scripts/seed_test_cases.py --kinds semantic --per-bucket 10
    # paste the output block below, replacing the empty list

Usage:
    uv run scripts/validate_dense.py
    uv run scripts/validate_dense.py --embeddings-url http://localhost:8087
    uv run scripts/validate_dense.py --limit 10 --mrr-threshold 0.6 --hit-threshold 0.9
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass

import httpx
from qdrant_client import AsyncQdrantClient

# ---------------------------------------------------------------------------
# TEST_CASES  — populate via: uv run scripts/seed_test_cases.py --kinds semantic
# Each entry: (query, expected_symbol_name, kind)
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[str, str, str]] = [
    # From actual docstrings
    ("BeginRegistration POST /api/register/begin",                           "BeginRegistration",           "semantic"),
    ("FinishRegistration POST /api/register/finish",                         "FinishRegistration",          "semantic"),
    ("BeginLogin POST /api/authenticate/begin",                              "BeginLogin",                  "semantic"),
    ("FinishLogin POST /api/authenticate/finish",                            "FinishLogin",                 "semantic"),
    ("GetRegisteredPasskeys GET /api/users/:username/registered-passkeys",   "GetRegisteredPasskeys",       "semantic"),

    # Manual — natural language queries
    ("Spring Security web security filter chain setup",                      "SecurityConfiguration",       "semantic"),
    ("OAuth2 authorization server beans and configuration",                  "AuthorizationServerConfiguration", "semantic"),
    ("JPA service for persisting OAuth2 authorization consents",             "JpaAuthorizationConsentService", "semantic"),
    ("JPA-backed storage and retrieval of OAuth2 authorizations",            "JpaAuthorizationService",     "semantic"),
    ("authentication provider that checks for leaked or breached passwords", "LeakedPasswordsAuthenticationProvider", "semantic"),
    ("authentication provider for WebAuthn passkey login",                   "WebAuthnAuthenticationProvider", "semantic"),
    ("controller for WebAuthn passkey registration and authentication",      "WebAuthnController",          "semantic"),
    ("initializes OAuth2 registered clients at application startup",         "RegisteredClientInitializer", "semantic"),
    ("seeds initial user accounts in the database on startup",               "UserInitializer",             "semantic"),
    ("JPA implementation of the OAuth2 registered client repository",        "JpaClientRepository",        "semantic"),
    ("exception thrown when WebAuthn operation fails",                       "WebAuthnException",           "semantic"),
    ("Spring Cloud Gateway route definitions with token relay",              "RouteConfiguration",          "semantic"),
    ("bean that customizes JWT token claims before issuing",                 "tokenCustomizer",             "semantic"),
    ("find an OAuth2 authorization by any token or code value",              "findByStateOrAuthorizationCodeValueOrAccessTokenValueOrRefreshTokenValueOrOidcIdTokenValueOrUserCodeValueOrDeviceCodeValue", "semantic"),
    ("fetch a user record by their username",                                "GetUserByUsername",           "semantic"),
    ("retrieve a user together with their associated passkey credentials",   "GetUserWithCredentials",      "semantic"),
    ("initialize the WebAuthn relying party configuration",                  "InitWebAuthn",                "semantic"),
    ("HTTP handler to check if a password appears in breach data",           "CheckPasswordHandler",        "semantic"),
    ("HTTP handler to look up a password hash in the leaked passwords store","GetByHashHandler",            "semantic"),
    ("open the database connection for the service",                         "ConnectDatabase",             "semantic"),
    ("JPA repository for managing authorization consent records",            "AuthorizationConsentRepository", "semantic"),
    ("repository interface for looking up registered OAuth2 clients",        "ClientRepository",           "semantic"),
    ("client interface for calling the leaked passwords API",                "LeakedPasswordsClient",       "semantic"),
    ("request body for checking if a password has been compromised",         "CheckPasswordRequest",        "semantic"),
    ("response indicating whether a password hash was found in breach data", "CheckPasswordResponse",       "semantic"),

    # Ambiguous — query could reasonably match several symbols
    ("security configuration",                                               "SecurityConfiguration",       "semantic"),
    ("service that checks whether a password has been exposed in a breach",  "LeakedPasswordsService",      "semantic"),
    ("repository for storing authorization records",                         "AuthorizationRepository",     "semantic"),
    ("authentication provider for hardware security key login",              "WebAuthnAuthenticationProvider", "semantic"),
    ("persists OAuth2 tokens, codes and authorization state",                "JpaAuthorizationService",     "semantic"),
    ("HTTP client interface for the breached passwords backend",             "LeakedPasswordsClient",       "semantic"),
    ("starts a passkey authentication ceremony",                             "BeginLogin",                  "semantic"),
    ("error raised during a passkey or WebAuthn operation",                  "WebAuthnException",           "semantic"),
    ("gateway route definitions",                                            "RouteConfiguration",          "semantic"),
    ("creates default application users on startup",                         "UserInitializer",             "semantic"),
    ("find authorization by token value",                                    "findByStateOrAuthorizationCodeValueOrAccessTokenValueOrRefreshTokenValueOrOidcIdTokenValueOrUserCodeValueOrDeviceCodeValue", "semantic"),
    ("manages user consent grants for OAuth2 scopes",                        "JpaAuthorizationConsentService", "semantic"),
    ("represents a registered passkey device credential",                    "RegisteredPasskey",           "semantic"),
    ("connect to the database",                                              "ConnectDatabase",             "semantic"),
    ("endpoint to begin passkey registration",                               "BeginRegistration",           "semantic"),
    ("validates identity using a hardware authenticator",                    "WebAuthnAuthenticationProvider", "semantic"),
    ("exposes endpoints for querying the leaked passwords dataset",          "LeakedPasswordsApi",          "semantic"),
    ("pre-loads OAuth2 clients into the authorization server",               "RegisteredClientInitializer", "semantic"),
    ("retrieves and deserializes a stored WebAuthn session",                 "getAndParseWebAuthnSession",  "semantic"),
    ("adds custom claims to JWT access tokens",                              "JwtTokenCustomizerConfig",    "semantic"),
]

DEFAULT_LIMIT = 10
DEFAULT_MRR_THRESHOLD = 0.5
DEFAULT_HIT_THRESHOLD = 0.8
HIT_AT_KS = [1, 3, 5, 10]


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
# Search helper
# ---------------------------------------------------------------------------

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
    rank: int
    limit: int

    @property
    def found(self) -> bool:
        return self.rank <= self.limit

    def hit_at(self, k: int) -> bool:
        return self.rank <= k

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank if self.found else 0.0


def print_report(results: list[Result], limit: int, mrr_threshold: float, hit_threshold: float) -> bool:
    col_q = min(max(len(r.query) for r in results), 52)
    col_e = min(max(len(r.expected) for r in results), 36)

    header = f"{'Query':<{col_q}}  {'Expected':<{col_e}}  {'Rank':>6}  {'RR':>6}"
    print(header)
    print("-" * len(header))

    for r in results:
        q = r.query if len(r.query) <= col_q else r.query[: col_q - 1] + "…"
        e = r.expected if len(r.expected) <= col_e else r.expected[: col_e - 1] + "…"
        rank_str = str(r.rank) if r.found else f">{limit}"
        rr_str = f"{r.reciprocal_rank:.3f}"
        print(f"{q:<{col_q}}  {e:<{col_e}}  {rank_str:>6}  {rr_str:>6}")

    print()

    n = len(results)
    mrr = sum(r.reciprocal_rank for r in results) / n
    avg_rank = sum(r.rank for r in results) / n

    # Hit@K table
    print(f"{'Metric':<12}  {'Value':>8}")
    print("-" * 24)
    for k in HIT_AT_KS:
        if k > limit:
            continue
        hit = sum(1 for r in results if r.hit_at(k)) / n
        print(f"Hit@{k:<8}  {hit:>7.1%}")
    print(f"{'MRR':<12}  {mrr:>8.4f}")
    print(f"{'Avg rank':<12}  {avg_rank:>8.2f}")
    print()

    # Success criteria
    hit10 = sum(1 for r in results if r.hit_at(min(10, limit))) / n
    passed = True

    c1_pass = mrr >= mrr_threshold
    status = "PASS" if c1_pass else "FAIL"
    print(f"[{status}] CRITERION 1 — MRR: {mrr:.4f} (threshold: ≥{mrr_threshold})")
    if not c1_pass:
        passed = False

    effective_k = min(10, limit)
    c2_pass = hit10 >= hit_threshold
    status = "PASS" if c2_pass else "FAIL"
    print(f"[{status}] CRITERION 2 — Hit@{effective_k}: {hit10:.1%} (threshold: ≥{hit_threshold:.0%})")
    if not c2_pass:
        passed = False

    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_test_cases(path: str, kinds: set[str] | None) -> list[tuple[str, str, str]]:
    with open(path) as f:
        rows = json.load(f)
    cases = [(q, e, k) for q, e, k in rows]
    if kinds:
        cases = [(q, e, k) for q, e, k in cases if k in kinds]
    return cases


async def main(
    url: str,
    collection: str,
    embeddings_url: str,
    limit: int,
    mrr_threshold: float,
    hit_threshold: float,
    test_cases_file: str | None,
    kinds: set[str] | None,
) -> None:
    cases: list[tuple[str, str, str]] = (
        _load_test_cases(test_cases_file, kinds) if test_cases_file else list(TEST_CASES)
    )
    if kinds and not test_cases_file:
        cases = [(q, e, k) for q, e, k in cases if k in kinds]
    if not cases:
        print(
            "TEST_CASES is empty.\n"
            "Run:  uv run scripts/seed_test_cases.py --kinds semantic\n"
            "then paste the output into this file, or use --test-cases-file.",
            file=sys.stderr,
        )
        sys.exit(1)

    qdrant = AsyncQdrantClient(url=url)
    fallback = limit + 1

    async with httpx.AsyncClient() as http:
        results: list[Result] = []

        for i, entry in enumerate(cases, 1):
            query, expected = entry[0], entry[1]
            print(f"  [{i}/{len(TEST_CASES)}] {query!r}…", file=sys.stderr, end="\r")
            dense = await get_dense_vector(http, embeddings_url, query)
            names = await search_dense(qdrant, collection, dense, limit)
            results.append(Result(
                query=query,
                expected=expected,
                rank=get_rank(names, expected, fallback),
                limit=limit,
            ))

    print(" " * 60, file=sys.stderr, end="\r")
    print(f"Dense semantic search — {len(results)} queries, search limit={limit}\n", flush=True)
    passed = print_report(results, limit, mrr_threshold, hit_threshold)

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
        "--mrr-threshold",
        type=float,
        default=DEFAULT_MRR_THRESHOLD,
        metavar="F",
        help=f"Minimum MRR to pass criterion 1 (default: {DEFAULT_MRR_THRESHOLD})",
    )
    parser.add_argument(
        "--hit-threshold",
        type=float,
        default=DEFAULT_HIT_THRESHOLD,
        metavar="F",
        help=f"Minimum Hit@10 to pass criterion 2 (default: {DEFAULT_HIT_THRESHOLD})",
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
        args.url, args.collection, args.embeddings_url,
        args.limit, args.mrr_threshold, args.hit_threshold,
        args.test_cases_file,
        set(args.kinds) if args.kinds else None,
    ))
