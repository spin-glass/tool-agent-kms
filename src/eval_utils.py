"""Reusable retrieval evaluation primitives.

All metrics operate on a uniform shape:
    retrieved: list[dict] returned by ``agent_kms.retrieve.retrieve()``,
               each dict has at minimum ``source`` and ``heading``.
    gold:      set[tuple[str, str]] of (source_basename, heading) deemed
               relevant for the query.

We identify a chunk by (basename(source), heading) so eval queries can
reference docs without the absolute filesystem path the embedding pipeline
records in payloads.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml


def resolve_eval_paths(repo_root: Path) -> tuple[Path, Path, str]:
    """Resolve corpus dir, queries file, and Qdrant collection for the
    current evaluation mode.

    Two modes:

    - ``EVAL_MODE=public`` (default): everything under ``data/corpus/`` +
      ``data/eval_queries.yaml`` + collection ``eval_corpus``. This is
      the corpus that ships with the repository (React JP + JS Primer JP,
      both CC-BY-4.0).

    - ``EVAL_MODE=private``: switch to ``data/private/corpus/`` +
      ``data/private/eval_queries.yaml`` + collection
      ``eval_corpus_private``. The ``data/private/`` tree is gitignored so
      proprietary corpora can be evaluated with the same notebooks
      without leaking into the published portfolio. Aggregate metrics
      from a private run are publishable (see README "Validation on
      private corpora"); raw chunks and queries are not.

    Override the collection name explicitly via ``AGENT_KMS_COLLECTION``
    when you want to keep multiple private experiments side-by-side
    (e.g. ``eval_corpus_private_2026q2``).
    """
    mode = os.environ.get("EVAL_MODE", "public").lower()
    if mode == "private":
        root = repo_root / "data" / "private"
        return (
            root / "corpus",
            root / "eval_queries.yaml",
            os.environ.get("AGENT_KMS_COLLECTION", "eval_corpus_private"),
        )
    return (
        repo_root / "data" / "corpus",
        repo_root / "data" / "eval_queries.yaml",
        os.environ.get("AGENT_KMS_COLLECTION", "eval_corpus"),
    )


def chunk_key(item: dict) -> tuple[str, str]:
    """Canonical identity used to match retrieved chunks against gold."""
    source = item.get("source", "")
    heading = item.get("heading", "")
    return (Path(source).name, heading)


# ──────────────────────────────────────────────────────────────────────────
# Threshold-based metrics (agent-kms style: every chunk above T is returned)
# ──────────────────────────────────────────────────────────────────────────


def precision_at_threshold(retrieved: list[dict], gold: set[tuple[str, str]]) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if chunk_key(r) in gold)
    return hits / len(retrieved)


def recall_at_threshold(retrieved: list[dict], gold: set[tuple[str, str]]) -> float:
    if not gold:
        return 0.0
    hits = sum(1 for r in retrieved if chunk_key(r) in gold)
    return hits / len(gold)


def f1_at_threshold(retrieved: list[dict], gold: set[tuple[str, str]]) -> float:
    p = precision_at_threshold(retrieved, gold)
    r = recall_at_threshold(retrieved, gold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────
# Ranking metrics (work on ordered retrieved list)
# ──────────────────────────────────────────────────────────────────────────


def reciprocal_rank(retrieved: list[dict], gold: set[tuple[str, str]]) -> float:
    """1 / position of first relevant chunk, or 0 if none."""
    for i, r in enumerate(retrieved, start=1):
        if chunk_key(r) in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[dict], gold: set[tuple[str, str]], k: int = 10) -> float:
    """Binary-relevance NDCG@k. DCG = sum(rel_i / log2(i+1))."""
    dcg = 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        if chunk_key(r) in gold:
            dcg += 1.0 / math.log2(i + 1)
    # Ideal DCG: relevant items packed at top
    n_rel = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────
# Eval-query I/O
# ──────────────────────────────────────────────────────────────────────────


def load_queries(path: str | Path) -> list[dict[str, Any]]:
    """Load eval queries from YAML.

    Schema per item:
        id: str
        query: str
        gold:
          - source: <filename or basename>
            heading: <H2 title>
        notes: str  (optional)
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or []
    if not isinstance(data, list):
        raise ValueError(f"{path}: top-level must be a list of query objects")
    return data


def gold_set(query: dict) -> set[tuple[str, str]]:
    return {(g["source"], g["heading"]) for g in query.get("gold", [])}


# ──────────────────────────────────────────────────────────────────────────
# Sweep helpers
# ──────────────────────────────────────────────────────────────────────────


def sweep_threshold(
    queries: list[dict],
    retrieve_fn,
    thresholds: Iterable[float],
) -> list[dict]:
    """Run each query at each threshold; return rows for a DataFrame.

    Row schema: threshold, query_id, precision, recall, f1, mrr, ndcg10,
                num_returned.
    """
    rows: list[dict] = []
    for t in thresholds:
        for q in queries:
            retrieved = retrieve_fn(q["query"], score_threshold=t)
            g = gold_set(q)
            rows.append(
                {
                    "threshold": t,
                    "query_id": q["id"],
                    "precision": precision_at_threshold(retrieved, g),
                    "recall": recall_at_threshold(retrieved, g),
                    "f1": f1_at_threshold(retrieved, g),
                    "mrr": reciprocal_rank(retrieved, g),
                    "ndcg10": ndcg_at_k(retrieved, g, k=10),
                    "num_returned": len(retrieved),
                }
            )
    return rows
