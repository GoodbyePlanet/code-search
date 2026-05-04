#!/usr/bin/env python3
"""
Run the full search-quality validation pipeline in one command.

Steps:
  1. seed          — scroll Qdrant, pick diverse multi-word symbols, write
                     identifier test cases to a temporary JSON file
  2. hybrid        — validate BM25 + dense + RRF beats dense-only on identifier
                     queries (exact / tokenized / snake_case / prefix)
  3. hybrid-sem    — run the same 30 curated semantic queries through hybrid vs
                     dense to check for regressions (test_cases_semantic.json)
  4. dense         — validate dense-only semantic search quality (MRR / Hit@K)

Overall exit code is 0 only when all three validators pass.

Usage:
    uv run scripts/run_validation.py
    uv run scripts/run_validation.py --per-bucket 10 --limit 20
    uv run scripts/run_validation.py --url http://localhost:6333 \\
        --embeddings-url http://localhost:8087
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

SCRIPTS = {
    "seed":   "scripts/seed_test_cases.py",
    "hybrid": "scripts/validate_hybrid.py",
    "dense":  "scripts/validate_dense.py",
}

SEMANTIC_CASES_FILE = "scripts/test_cases_semantic.json"

SEPARATOR = "=" * 72


def run(label: str, cmd: list[str]) -> int:
    print(f"\n{SEPARATOR}", flush=True)
    print(f"  {label}", flush=True)
    print(SEPARATOR, flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
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
        "--per-bucket",
        type=int,
        default=10,
        metavar="N",
        help="Symbols to pick per symbol type during seeding (default: 10)",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=500,
        metavar="N",
        help="Max points to scan per symbol type during seeding (default: 500)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="Search result limit passed to both validators (default: 10)",
    )
    parser.add_argument(
        "--mrr-threshold",
        type=float,
        default=0.5,
        metavar="F",
        help="Minimum MRR for the dense validator to pass (default: 0.5)",
    )
    parser.add_argument(
        "--hit-threshold",
        type=float,
        default=0.8,
        metavar="F",
        help="Minimum Hit@10 for the dense validator to pass (default: 0.8)",
    )
    args = parser.parse_args()

    py = sys.executable
    qdrant_args = ["--url", args.url, "--collection", args.collection]
    embed_args  = ["--embeddings-url", args.embeddings_url]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        cases_file = tmp.name

    try:
        # Step 1 — seed
        seed_rc = run("STEP 1 / 3 — seed_test_cases.py", [
            py, SCRIPTS["seed"],
            *qdrant_args,
            "--per-bucket", str(args.per_bucket),
            "--scan-limit",  str(args.scan_limit),
            "--output-json", cases_file,
        ])
        if seed_rc != 0:
            print("\n[ABORT] Seeding failed — check Qdrant connection and collection.", file=sys.stderr)
            sys.exit(seed_rc)

        # Step 2 — hybrid validator (identifier kinds from seed)
        hybrid_rc = run("STEP 2 / 4 — validate_hybrid.py  (identifier queries: hybrid vs dense)", [
            py, SCRIPTS["hybrid"],
            *qdrant_args,
            *embed_args,
            "--limit",           str(args.limit),
            "--test-cases-file", cases_file,
        ])

        # Step 3 — hybrid validator (semantic queries from curated file)
        hybrid_sem_rc = run("STEP 3 / 4 — validate_hybrid.py  (semantic queries: hybrid vs dense)", [
            py, SCRIPTS["hybrid"],
            *qdrant_args,
            *embed_args,
            "--limit",           str(args.limit),
            "--test-cases-file", SEMANTIC_CASES_FILE,
            "--kinds",           "semantic",
        ])

        # Step 4 — dense validator (absolute quality on semantic queries)
        dense_rc = run("STEP 4 / 4 — validate_dense.py  (dense semantic search quality)", [
            py, SCRIPTS["dense"],
            *qdrant_args,
            *embed_args,
            "--limit",           str(args.limit),
            "--mrr-threshold",   str(args.mrr_threshold),
            "--hit-threshold",   str(args.hit_threshold),
        ])

    finally:
        os.unlink(cases_file)

    # Combined summary
    print(f"\n{SEPARATOR}")
    print("  SUMMARY")
    print(SEPARATOR)
    print(f"  validate_hybrid (identifiers)  : {'PASS' if hybrid_rc     == 0 else 'FAIL'}")
    print(f"  validate_hybrid (semantic)     : {'PASS' if hybrid_sem_rc == 0 else 'FAIL'}")
    print(f"  validate_dense  (semantic)     : {'PASS' if dense_rc      == 0 else 'FAIL'}")
    overall = hybrid_rc == 0 and hybrid_sem_rc == 0 and dense_rc == 0
    print(f"\n  Overall                        : {'PASS' if overall else 'FAIL'}")
    print(SEPARATOR)

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
