"""Pairwise judge: for each task, decide which of A_noctx / B_ctx is better.

Run AFTER ``run_pairwise_generation.py`` has produced
``results/pairwise/<id>__A_noctx.txt`` and ``__B_ctx.txt`` files.

To control positional bias we ask the judge **twice per task**: once with
A=noctx,B=ctx and once with A=ctx,B=noctx (swap). A condition only
"wins" a task if it wins **both** orderings; otherwise the task is
recorded as a tie.

Judge model defaults to ``qwen2.5:14b`` (a stronger local model than
the target ``qwen3:8b``). For larger studies, swap to a Claude API
judge by setting ``--judge-mode anthropic`` (not implemented here;
extension hook only).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

JUDGE_SYSTEM = (
    "あなたは厳格なコードレビュアです。"
    "同じ指示に対する 2 つのコード回答を比較し、"
    "rubric (採点項目) を踏まえてどちらが優れているかを判定してください。"
    "コードの正しさ、簡潔さ、要件充足度を重視します。"
)

JUDGE_TEMPLATE = """以下のコーディング指示に対して、2 つの回答 (Answer X / Answer Y) が提出されました。
rubric を参考に、どちらが優れているか判定してください。

=== 指示 ===
{task_prompt}

=== rubric (採点基準) ===
{rubric}

=== Answer X ===
{answer_x}

=== Answer Y ===
{answer_y}

=== 判定 ===
最終行に必ず `VERDICT: X` `VERDICT: Y` `VERDICT: TIE` のいずれか 1 行のみを出力してください。
その前に 1〜3 行で簡潔な理由を書いてください。
"""


def ollama_chat(model: str, system: str, user: str, *,
                host: str = "http://localhost:11434",
                timeout: float = 600.0, think: bool = False) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": think,
        "keep_alive": "20m",
        "options": {"temperature": 0.0, "num_ctx": 8192},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data.get("message", {}).get("content", "")


def parse_verdict(text: str) -> str:
    """Extract X / Y / TIE from the judge's last VERDICT line; fall back to TIE."""
    for line in reversed(text.strip().splitlines()):
        ll = line.strip().upper()
        if ll.startswith("VERDICT:"):
            v = ll.split(":", 1)[1].strip()
            if v in ("X", "Y", "TIE"):
                return v
    return "TIE"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="qwen2.5:14b")
    ap.add_argument("--tasks", default="data/coding_tasks.yaml")
    ap.add_argument("--gen-dir", default="results/pairwise")
    ap.add_argument("--out", default="results/pairwise/_judge_results.csv")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    random.seed(args.seed)

    tasks_path = ROOT / args.tasks
    with tasks_path.open("r", encoding="utf-8") as f:
        tasks = yaml.safe_load(f) or []
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        tasks = [t for t in tasks if t["id"] in wanted]

    gen_dir = ROOT / args.gen_dir
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for task in tasks:
        tid = task["id"]
        rubric = "\n".join(f"- {r}" for r in task.get("rubric", []))
        a = (gen_dir / f"{tid}__A_noctx.txt").read_text(encoding="utf-8").strip()
        b = (gen_dir / f"{tid}__B_ctx.txt").read_text(encoding="utf-8").strip()
        if not a or not b:
            print(f"skip {tid}: missing output")
            continue

        # Two orderings to neutralize positional bias.
        for order in ["AB", "BA"]:
            if order == "AB":
                x, y = a, b   # X = noctx, Y = ctx
                label_x, label_y = "noctx", "ctx"
            else:
                x, y = b, a   # X = ctx,   Y = noctx
                label_x, label_y = "ctx", "noctx"

            prompt = JUDGE_TEMPLATE.format(
                task_prompt=task["prompt"].strip(),
                rubric=rubric,
                answer_x=x,
                answer_y=y,
            )
            t0 = time.time()
            resp = ollama_chat(args.judge_model, JUDGE_SYSTEM, prompt,
                               host=args.ollama_host)
            elapsed = time.time() - t0
            verdict = parse_verdict(resp)
            winner_label = (
                "tie" if verdict == "TIE"
                else (label_x if verdict == "X" else label_y)
            )
            rows.append({
                "task_id": tid,
                "order": order,
                "verdict_raw": verdict,
                "winner": winner_label,
                "elapsed_s": round(elapsed, 1),
                "judge_response": resp.replace("\n", " ⏎ ")[:500],
            })
            print(f"  {tid} {order}: VERDICT={verdict}  →  winner={winner_label}  ({elapsed:.1f}s)",
                  flush=True)

    # Aggregate: a task is a "ctx win" only if ctx beats noctx in BOTH orderings
    by_task: dict[str, dict] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], {})[r["order"]] = r["winner"]
    agg_rows = []
    for tid, orders in by_task.items():
        winners = list(orders.values())
        if all(w == "ctx" for w in winners):
            outcome = "ctx_win"
        elif all(w == "noctx" for w in winners):
            outcome = "noctx_win"
        elif winners.count("ctx") > winners.count("noctx"):
            outcome = "ctx_lean"
        elif winners.count("noctx") > winners.count("ctx"):
            outcome = "noctx_lean"
        else:
            outcome = "tie"
        agg_rows.append({"task_id": tid, "outcome": outcome,
                         "order_AB_winner": orders.get("AB", ""),
                         "order_BA_winner": orders.get("BA", "")})

    out_raw = out_path
    out_agg = out_path.with_name(out_path.stem + "_agg.csv")
    with out_raw.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with out_agg.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader(); w.writerows(agg_rows)
    print(f"\nwrote {out_raw}\nwrote {out_agg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
