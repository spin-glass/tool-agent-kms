"""For each coding task, generate two outputs from Ollama qwen3:8b:
condition A (prompt only) and condition B (prompt + agent-kms retrieved
context). Results land in ``results/pairwise/<task_id>__<cond>.txt``
and a single summary CSV with per-task metadata.

Calling pattern:
    python scripts/run_pairwise_generation.py \\
        --target-model qwen3:8b \\
        --retrieve-threshold 0.81 \\
        --out-dir results/pairwise

We use **Ollama's HTTP API directly** (no extra Python deps). The Ollama
daemon must be running on localhost:11434.

This script writes outputs only — it does NOT decide winners. See
``run_pairwise_judge.py`` for the comparison step.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ollama_generate(model: str, prompt: str, *, host: str = "http://localhost:11434",
                    timeout: float = 600.0, options: dict | None = None,
                    think: bool = False) -> dict:
    """Single-shot generation via Ollama /api/generate. Returns parsed JSON.

    For qwen3 models, ``think=False`` disables the chain-of-thought
    "thinking" trace which is enabled by default and roughly doubles
    latency without materially improving short-code-generation quality.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": think,
        "keep_alive": "20m",  # keep model resident across the run
        "options": options or {"temperature": 0.2, "num_ctx": 6144},
    }
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def build_prompt(task_prompt: str, context_chunks: list[dict] | None) -> str:
    """Assemble the prompt for qwen3:8b.

    Condition A: ``context_chunks is None`` → bare prompt.
    Condition B: chunks present → small system-style preamble that
    encourages use of the context without slavishly copying it.
    """
    if not context_chunks:
        return (
            "あなたは熟練したフロントエンドエンジニアです。"
            "以下の指示に対して、要件を満たす最小のコードを書いてください。"
            "コードブロック (```) で囲んでください。\n\n"
            f"指示:\n{task_prompt.strip()}\n"
        )
    ctx_md = "\n\n---\n\n".join(
        f"## {c['heading']}  ({Path(c['source']).name})\n\n{c['text']}"
        for c in context_chunks
    )
    return (
        "あなたは熟練したフロントエンドエンジニアです。"
        "以下の指示に対して、要件を満たす最小のコードを書いてください。"
        "コードブロック (```) で囲んでください。\n\n"
        "参考になりそうな関連ドキュメント抜粋を、必要に応じて参照してください "
        "(全てを使う必要はありません)。\n\n"
        "=== 関連ドキュメント抜粋 ===\n"
        f"{ctx_md}\n"
        "=== ここまで ===\n\n"
        f"指示:\n{task_prompt.strip()}\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", default="qwen3:8b")
    ap.add_argument("--tasks", default="data/coding_tasks.yaml")
    ap.add_argument("--retrieve-threshold", type=float, default=0.81,
                    help="score_threshold for agent-kms retrieve()")
    ap.add_argument("--max-chunks", type=int, default=8,
                    help="cap on number of chunks injected as context")
    ap.add_argument("--out-dir", default="results/pairwise")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--qdrant-url", default="http://localhost:6433")
    ap.add_argument("--collection", default="eval_corpus")
    ap.add_argument("--embed-model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--embed-dim", type=int, default=768)
    ap.add_argument("--only", default=None,
                    help="comma-separated task ids to run (default: all)")
    args = ap.parse_args()

    os.environ["QDRANT_URL"] = args.qdrant_url
    os.environ["AGENT_KMS_COLLECTION"] = args.collection
    os.environ["AGENT_KMS_MODEL"] = args.embed_model
    os.environ["AGENT_KMS_VECTOR_SIZE"] = str(args.embed_dim)
    os.environ["AGENT_KMS_CONFIG"] = str(ROOT / ".agent-kms" / "kms.toml")
    os.environ["AGENT_KMS_KNOWLEDGE_ROOT"] = str(ROOT / "data" / "corpus")

    # Imports deferred so the env vars above take effect at module init.
    from agent_kms.retrieve import retrieve

    tasks_path = ROOT / args.tasks
    with tasks_path.open("r", encoding="utf-8") as f:
        tasks = yaml.safe_load(f) or []

    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        tasks = [t for t in tasks if t["id"] in wanted]

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    for task in tasks:
        tid = task["id"]
        print(f"\n──── {tid}  ({task.get('difficulty', '?')})", flush=True)

        # Retrieve once per task; reuse for condition B.
        retrieved = retrieve(
            task["retrieve_query"],
            score_threshold=args.retrieve_threshold,
        )[: args.max_chunks]
        print(f"  retrieved {len(retrieved)} chunks at T={args.retrieve_threshold}", flush=True)

        for cond, chunks in [("A_noctx", None), ("B_ctx", retrieved)]:
            prompt = build_prompt(task["prompt"], chunks)
            t0 = time.time()
            try:
                resp = ollama_generate(args.target_model, prompt, host=args.ollama_host)
                text = resp.get("response", "")
                tokens = resp.get("eval_count", 0)
                ok = True
                err = ""
            except (urllib.error.URLError, TimeoutError) as e:
                text, tokens, ok, err = "", 0, False, str(e)
            elapsed = time.time() - t0

            out_path = out_dir / f"{tid}__{cond}.txt"
            out_path.write_text(text, encoding="utf-8")
            meta_rows.append({
                "task_id": tid,
                "condition": cond,
                "ok": ok,
                "tokens_out": tokens,
                "elapsed_s": round(elapsed, 2),
                "n_chunks": len(chunks) if chunks is not None else 0,
                "prompt_chars": len(prompt),
                "out_chars": len(text),
                "error": err,
            })
            print(f"  {cond:10s}  ok={ok}  tokens={tokens}  {elapsed:5.1f}s  out={len(text)}c",
                  flush=True)

    meta_csv = out_dir / "_generation_meta.csv"
    with meta_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        w.writeheader()
        w.writerows(meta_rows)
    print(f"\nwrote {meta_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
