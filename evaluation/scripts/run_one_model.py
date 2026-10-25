"""Run a threshold sweep against one embedding model in an isolated process.

Why a subprocess? `agent_kms.store` reads `AGENT_KMS_MODEL` / `AGENT_KMS_VECTOR_SIZE`
at import time and caches the SentenceTransformer + Qdrant client behind
`@lru_cache`. Swapping models inside one Python process therefore requires
fragile cache-clearing dances. A subprocess gives each model a clean slate
and a clean failure mode.

Output: a single CSV at ``--out`` with one row per (query, threshold):
    threshold, query_id, precision, recall, f1, mrr, ndcg10,
    num_returned, model, ingest_time_s
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id (e.g. intfloat/multilingual-e5-base)")
    ap.add_argument("--dim", type=int, required=True, help="vector dimensionality for this model")
    ap.add_argument("--collection", required=True, help="Qdrant collection to use (per-model isolation)")
    ap.add_argument(
        "--thresholds",
        default="0.65,0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84,0.86,0.88,0.90,0.92",
        help="comma-separated thresholds to sweep",
    )
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    # Env must be set BEFORE importing agent_kms.store
    os.environ["QDRANT_URL"] = os.environ.get("QDRANT_URL", "http://localhost:6433")
    os.environ["AGENT_KMS_MODEL"] = args.model
    os.environ["AGENT_KMS_VECTOR_SIZE"] = str(args.dim)
    os.environ["AGENT_KMS_COLLECTION"] = args.collection

    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("AGENT_KMS_CONFIG", str(root / ".agent-kms" / "kms.toml"))
    os.environ.setdefault("AGENT_KMS_KNOWLEDGE_ROOT", str(root / "data" / "corpus"))

    sys.path.insert(0, str(root))

    from agent_kms.config import load_config, reset_cache
    from agent_kms.ingest import collect_chunks, upsert_chunks
    from agent_kms.retrieve import retrieve
    from agent_kms.store import ensure_collection, get_model

    from src.eval_utils import load_queries, sweep_threshold

    reset_cache()
    cfg = load_config()

    print(f"model:      {args.model}")
    print(f"dim:        {args.dim}")
    print(f"collection: {cfg.collection}")

    # ModernBERT-based models (ruri-v3 line) default to max_seq_length=8192,
    # which makes SDPA attention allocate tens of GiB on CPU during batched
    # encoding. Cap the context to a reasonable size — our chunks are short.
    m = get_model()
    if getattr(m, "max_seq_length", 0) and m.max_seq_length > 512:
        m.max_seq_length = 512
        print(f"capped max_seq_length to 512 (was {m.max_seq_length} default)")

    ensure_collection(reset=True)

    t0 = time.time()
    chunks = collect_chunks(cfg)
    n = upsert_chunks(cfg, chunks)
    ingest_time = time.time() - t0
    print(f"ingested {n} chunks in {ingest_time:.1f}s")

    queries = load_queries(root / "data" / "eval_queries.yaml")
    in_domain = [q for q in queries if q.get("gold")]

    thresholds = [float(t) for t in args.thresholds.split(",")]
    rows = sweep_threshold(
        in_domain,
        lambda q, score_threshold: retrieve(q, score_threshold=score_threshold),
        thresholds,
    )

    import pandas as pd

    df = pd.DataFrame(rows)
    df["model"] = args.model
    df["ingest_time_s"] = ingest_time

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
