"""Regenerate notebooks/*.ipynb from in-source cell definitions.

Run once at authoring time. The generated .ipynb files are then edited by
``jupyter nbconvert --execute`` to bake in outputs. Re-running this script
overwrites the notebooks (outputs are lost).
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from _nbgen import write_notebook

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


# ─────────────────────────────────────────────────────────────────────────
# 01. Setup and corpus
# ─────────────────────────────────────────────────────────────────────────

NB01 = [
    ("markdown", """# 01. Setup and Corpus Preparation

**Goal**: end this notebook with a populated Qdrant collection (`eval_corpus`)
and a known list of source documents we can author queries against.

Steps:
1. Verify Qdrant is reachable.
2. Confirm the corpus files are present under `data/corpus/`.
3. Run a programmatic ingest (mirrors `agent-kms ingest --reset`).
4. Print corpus statistics (chunks / docs / heading distribution).

The corpus itself is **already committed** to `data/corpus/`. It consists of
15 markdown files drawn from two public CC-BY-licensed sources — see
`data/corpus/LICENSES.md` for attribution.
"""),
    ("code", """import os
import sys
from pathlib import Path

# Resolve project root so the notebook works whether the CWD is the
# notebooks/ subdir (default for Jupyter) or the project root.
ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

# Isolate from personal agent-kms data: dedicated Qdrant instance (port 6433)
# AND a dedicated collection name. The personal Qdrant on :6333 is untouched.
#
# EVAL_MODE selects between the committed public corpus (data/corpus/) and a
# gitignored private corpus (data/private/corpus/). See README "Private
# evaluation mode" for the design — same notebook, two corpora.
sys.path.insert(0, str(ROOT))
from src.eval_utils import resolve_eval_paths

CORPUS_DIR, QUERIES_PATH, COLLECTION = resolve_eval_paths(ROOT)
os.environ["QDRANT_URL"] = "http://localhost:6433"
os.environ["AGENT_KMS_COLLECTION"] = COLLECTION
os.environ["AGENT_KMS_CONFIG"] = str(ROOT / ".agent-kms" / "kms.toml")
os.environ["AGENT_KMS_KNOWLEDGE_ROOT"] = str(CORPUS_DIR)

print("project root:", ROOT)
print("eval mode:   ", os.environ.get("EVAL_MODE", "public"))
print("qdrant:      ", os.environ["QDRANT_URL"])
print("collection:  ", COLLECTION)
print("corpus:      ", CORPUS_DIR)
"""),
    ("markdown", """## 1. Qdrant reachability"""),
    ("code", """from agent_kms.store import get_client

client = get_client()
existing = [c.name for c in client.get_collections().collections]
print("collections on this Qdrant:", existing)
"""),
    ("markdown", """## 2. Corpus inventory

The corpus is resolved via `EVAL_MODE`: public (default) reads from
`data/corpus/` (committed CC-BY docs), private reads from
`data/private/corpus/` (gitignored). Each file is markdown with
H2-sectioned content suitable for the `markdown_h2` chunker.
"""),
    ("code", """corpus_dir = CORPUS_DIR
docs = sorted(p for p in corpus_dir.glob("*.md") if p.name != "LICENSES.md")
print(f"corpus docs: {len(docs)}")
for d in docs:
    size_kb = d.stat().st_size / 1024
    print(f"  {d.name:60s}  {size_kb:6.1f} KB")
"""),
    ("markdown", """## 3. Ingest into Qdrant

We bypass the CLI's argparse and call `collect_chunks` / `upsert_chunks`
directly so the notebook stays self-contained. `--reset` semantics are
achieved by `ensure_collection(reset=True)`.

Note: the embedding model (`intfloat/multilingual-e5-base`) is fetched
from HuggingFace on first use — the first run downloads ~440 MB.
"""),
    ("code", """from agent_kms.config import load_config, reset_cache
from agent_kms.ingest import collect_chunks, upsert_chunks
from agent_kms.store import ensure_collection

reset_cache()
cfg = load_config()
print("preset:        ", cfg.preset)
print("knowledge_root:", cfg.knowledge_root)
print("collection:    ", cfg.collection)
print("sources:       ", [s.name for s in cfg.sources])

ensure_collection(reset=True)
chunks = collect_chunks(cfg)
print(f"\\ncollected {len(chunks)} chunks")

n = upsert_chunks(cfg, chunks)
print(f"upserted  {n} points into '{cfg.collection}'")
"""),
    ("markdown", """## 4. Corpus statistics"""),
    ("code", """import pandas as pd

rows = []
for c in chunks:
    rows.append({
        "source": Path(c.source_file).name,
        "heading": c.heading,
        "chars": len(c.body),
    })
df_corpus = pd.DataFrame(rows)

print(f"chunks: {len(df_corpus)}")
print(f"docs:   {df_corpus['source'].nunique()}")
print(f"\\nchunks per doc:")
print(df_corpus.groupby("source").size().sort_values(ascending=False).to_string())
print(f"\\nbody-size summary (chars): min={df_corpus['chars'].min()}, median={int(df_corpus['chars'].median())}, max={df_corpus['chars'].max()}")
"""),
    ("markdown", """## Findings

- **Corpus size**: 15 Japanese technical-doc markdown files → ~130 H2 chunks.
- **Sources**: React 日本語ドキュメント (7 files, CC-BY-4.0) +
  JavaScript Primer 日本語版 (8 files, CC-BY-4.0).
- **Heading distribution is uneven** — the largest files (`async`, `string`,
  `array`, `lifecycle-of-reactive-effects`) contribute 13-15 chunks each;
  smaller pages like `conditional-rendering` contribute only 3.
  This is expected for the `markdown_h2` chunker on real-world docs.
- **Median chunk size ≈ a few hundred characters**; well above the
  `min_chars=80` floor.

This collection is the input to every subsequent notebook.
"""),
]


# ─────────────────────────────────────────────────────────────────────────
# 02. Baseline retrieval — qualitative
# ─────────────────────────────────────────────────────────────────────────

NB02 = [
    ("markdown", """# 02. Baseline Retrieval — Qualitative Check

Before quantitative sweeps, we eyeball retrieval on a small set of probe
queries to confirm:

1. The collection is reachable and returns sensible hits.
2. The cosine-score range we'll be sweeping over is realistic.
3. An adversarial / out-of-domain query returns nothing (or very low scores).

This notebook deliberately uses **`retrieve_simple`** (pure cosine,
no query expansion, no severity / applicability boosts) so we observe
the raw embedding signal — the boosts are studied separately in notebook 05.
"""),
    ("code", """import os, sys
from pathlib import Path

ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
from src.eval_utils import resolve_eval_paths

CORPUS_DIR, QUERIES_PATH, COLLECTION = resolve_eval_paths(ROOT)
os.environ["QDRANT_URL"] = "http://localhost:6433"
os.environ["AGENT_KMS_COLLECTION"] = COLLECTION
os.environ["AGENT_KMS_CONFIG"] = str(ROOT / ".agent-kms" / "kms.toml")
os.environ["AGENT_KMS_KNOWLEDGE_ROOT"] = str(CORPUS_DIR)

from agent_kms.retrieve import retrieve_simple
"""),
    ("code", """probe_queries = [
    # in-domain, direct
    "useStateでstate変数を追加する",
    # in-domain, paraphrased
    "親から渡された値を子コンポーネントで初期値に使ったあと書き換えたい",
    # JS, technical
    "try catch で例外を捕捉する書き方",
    # JS, intent
    "配列を変更せずに新しい配列を返したい",
    # adversarial — out-of-domain
    "PostgreSQL の logical replication を設定する手順",
]

for q in probe_queries:
    print(f"\\n──── {q!r}")
    hits = retrieve_simple(q, limit=5)
    for h in hits:
        src = h["source"].split("/")[-1]
        print(f"  {h['score']:.3f}  {src:45s}  | {h['heading'][:50]}")
"""),
    ("markdown", """## Observations

- **In-domain queries top out around cosine ≈ 0.87** for `multilingual-e5-base`.
  This is well **below the agent-kms default threshold of 0.93** — on this
  corpus, the default would silently return zero results for most queries.
- The Japanese-on-Japanese baseline cosine is high (~0.80 floor) even for
  unrelated content, so the *discrimination* between relevant and noise
  is in a narrow 0.05-0.08 band, not a 0.10+ band.
- The out-of-domain query (PostgreSQL) returns hits at ~0.78-0.80 — i.e.
  the model **does** confidently assign lower scores to irrelevant content,
  but that lower band overlaps with the on-topic band of harder paraphrased
  queries. The threshold sweep in notebook 03 quantifies this trade-off.
"""),
]


# ─────────────────────────────────────────────────────────────────────────
# 03. Threshold sweep
# ─────────────────────────────────────────────────────────────────────────

NB03 = [
    ("markdown", """# 03. Threshold Sweep

**Question**: agent-kms's README claims `score_threshold=0.93` is the
sweet spot for `multilingual-e5-base` on Japanese-heavy corpora. Does
that hold on this **public-documentation** corpus?

We sweep T from 0.78 to 0.94 in 0.01 steps and compute per-query
Precision, Recall, F1, MRR, NDCG@10 plus `num_returned`, then average
across queries (excluding adversarial / empty-gold queries from
the precision/recall aggregate — adversarial queries are scored
separately as **empty-rate**).
"""),
    ("code", """import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
from src.eval_utils import resolve_eval_paths

CORPUS_DIR, QUERIES_PATH, COLLECTION = resolve_eval_paths(ROOT)
os.environ["QDRANT_URL"] = "http://localhost:6433"
os.environ["AGENT_KMS_COLLECTION"] = COLLECTION
os.environ["AGENT_KMS_CONFIG"] = str(ROOT / ".agent-kms" / "kms.toml")
os.environ["AGENT_KMS_KNOWLEDGE_ROOT"] = str(CORPUS_DIR)

from agent_kms.retrieve import retrieve
from src.eval_utils import load_queries, sweep_threshold, gold_set, chunk_key

sns.set_theme(style="whitegrid")
"""),
    ("code", """queries = load_queries(QUERIES_PATH)
in_domain   = [q for q in queries if q.get("gold")]
adversarial = [q for q in queries if not q.get("gold")]
print(f"{len(queries)} total queries  |  in-domain {len(in_domain)}  |  adversarial {len(adversarial)}")

thresholds = np.round(np.arange(0.78, 0.945, 0.01), 3).tolist()
print(f"sweeping T over {len(thresholds)} steps: {thresholds[0]} -> {thresholds[-1]}")
"""),
    ("code", """def retrieve_fn(q, score_threshold):
    return retrieve(q, score_threshold=score_threshold)

rows = sweep_threshold(in_domain, retrieve_fn, thresholds)
df_in = pd.DataFrame(rows)
print(f"in-domain sweep rows: {len(df_in)}")

# Adversarial — track only num_returned (gold is empty so P/R undefined)
adv_rows = []
for t in thresholds:
    for q in adversarial:
        r = retrieve(q["query"], score_threshold=t)
        adv_rows.append({"threshold": t, "query_id": q["id"], "num_returned": len(r)})
df_adv = pd.DataFrame(adv_rows)
print(f"adversarial rows: {len(df_adv)}")
"""),
    ("code", """agg = df_in.groupby("threshold").agg(
    precision=("precision", "mean"),
    recall=("recall", "mean"),
    f1=("f1", "mean"),
    mrr=("mrr", "mean"),
    ndcg10=("ndcg10", "mean"),
    avg_returned=("num_returned", "mean"),
).reset_index()

# empty_rate_adversarial = fraction of adversarial queries that returned 0
empty_adv = df_adv.groupby("threshold")["num_returned"].apply(lambda s: (s == 0).mean()).rename("adv_empty_rate")
agg = agg.merge(empty_adv, on="threshold")

# empty_rate_indomain = fraction of in-domain queries that returned 0 (= total recall miss)
empty_in = df_in.groupby("threshold")["num_returned"].apply(lambda s: (s == 0).mean()).rename("indomain_empty_rate")
agg = agg.merge(empty_in, on="threshold")

agg.round(3)
"""),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

ax = axes[0]
for col, color in [("precision", "#1f77b4"), ("recall", "#ff7f0e"), ("f1", "#2ca02c")]:
    ax.plot(agg["threshold"], agg[col], marker="o", label=col, color=color)
ax.axvline(0.93, ls="--", color="grey", alpha=0.6, label="agent-kms default (0.93)")
best_t = agg.loc[agg["f1"].idxmax(), "threshold"]
ax.axvline(best_t, ls=":", color="red", alpha=0.6, label=f"best F1 @ T={best_t}")
ax.set_xlabel("score_threshold")
ax.set_ylabel("metric value (mean over in-domain queries)")
ax.set_title("Precision / Recall / F1 vs threshold")
ax.legend()

ax = axes[1]
ax.plot(agg["threshold"], agg["avg_returned"], marker="o", color="#9467bd", label="avg returned (in-domain)")
ax2 = ax.twinx()
ax2.plot(agg["threshold"], agg["adv_empty_rate"], marker="s", color="#d62728", label="adversarial empty-rate")
ax2.plot(agg["threshold"], agg["indomain_empty_rate"], marker="x", color="#8c564b", label="in-domain empty-rate")
ax.set_xlabel("score_threshold")
ax.set_ylabel("avg returned chunks", color="#9467bd")
ax2.set_ylabel("empty-rate", color="#d62728")
ax2.set_ylim(0, 1.05)
ax.set_title("Volume + empty-rate vs threshold")
ax.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(ROOT / "results" / "fig_threshold_sweep.png", dpi=140, bbox_inches="tight")
plt.show()
"""),
    ("code", """# Best F1 + a 'budgeted' best (constrain volume so a downstream LLM
# doesn't blow its context window). We use avg_returned <= 10 as the budget.
best_row = agg.sort_values("f1", ascending=False).iloc[0]
print(f"Best F1 (unconstrained): T={best_row['threshold']:.2f}  "
      f"P={best_row['precision']:.3f} R={best_row['recall']:.3f} F1={best_row['f1']:.3f}  "
      f"avg_returned={best_row['avg_returned']:.1f}")

budget = agg[agg["avg_returned"] <= 10]
if len(budget):
    b = budget.sort_values("f1", ascending=False).iloc[0]
    print(f"Best F1 (avg_returned <= 10): T={b['threshold']:.2f}  "
          f"P={b['precision']:.3f} R={b['recall']:.3f} F1={b['f1']:.3f}  "
          f"avg_returned={b['avg_returned']:.1f}")

# Save the table
agg.to_csv(ROOT / "results" / "threshold_sweep.csv", index=False)
print("\\nsaved results/threshold_sweep.csv")
"""),
    ("markdown", """## Findings

Empirical sweep (multilingual-e5-base, all retrieve enhancements on):

| threshold | precision | recall | F1    | avg returned | adv empty | in-domain empty |
|-----------|-----------|--------|-------|--------------|-----------|-----------------|
| 0.78      | 0.026     | 1.000  | 0.049 | 81.6         | 0.00      | 0.00            |
| 0.80      | 0.076     | 0.756  | 0.126 | 29.7         | 0.25      | 0.04            |
| **0.81**  | **0.159** | **0.649** | **0.201** | **13.2** | **0.75** | **0.04**     |
| 0.82      | 0.163     | 0.423  | 0.182 | 5.3          | 1.00      | 0.21            |
| 0.83      | 0.152     | 0.280  | 0.173 | 1.8          | 1.00      | 0.57            |
| 0.85      | 0.077     | 0.107  | 0.086 | 0.4          | 1.00      | 0.82            |
| 0.87+     | 0         | 0      | 0     | 0            | 1.00      | 1.00            |
| **0.93**  | **0**     | **0**  | **0** | **0**        | **1.00**  | **1.00**        |

- The README's default **T=0.93 is structurally unusable on this corpus**
  — every query returns zero chunks. The "sweet spot" the README reports
  holds on its **original corpus** (Anthropic-style internal knowledge
  with curated severity tags) but does not transfer to public Japanese
  technical documentation.
- **Best F1 on this corpus is T = 0.81** (F1 = 0.201) with `e5-base`.
  Precision is intrinsically limited at this threshold (~0.16) because
  agent-kms is threshold-based, not top-k: every chunk that crosses the
  bar returns, so a query whose gold has 2 chunks but 11 chunks cross
  the bar gets P ≈ 0.18 even when both gold items are retrieved.
  Recall (0.65) and MRR (0.47) are the metrics that better reflect
  retrieval quality at this regime.
- The **adversarial empty-rate jumps from 0 to 1.0 between T=0.81 and
  T=0.82** — there is a sharp boundary where out-of-domain queries stop
  returning anything. T=0.82-0.83 is the conservative choice if false
  positives are costly (e.g. LLM hallucination risk).
- The Japanese-on-Japanese baseline cosine sits high (~0.80 floor even
  for unrelated content), so the usable discrimination band is narrow
  (≈ 0.78–0.86 here, just 0.08 wide).

**Implication**: agent-kms's threshold default should be **corpus-tuned**,
not a global constant. The CLI / config layer already exposes
`score_threshold` as a tunable, so the fix is documentation +
a small calibration recipe, not a code change.
"""),
]


# ─────────────────────────────────────────────────────────────────────────
# 04. Embedding model comparison
# ─────────────────────────────────────────────────────────────────────────

NB04 = [
    ("markdown", """# 04. Embedding Model Comparison

**Question**: does switching from `multilingual-e5-base` to a
Japanese-specialized or larger model materially improve retrieval on
Japanese technical docs?

**Caveat** — cosine score distributions differ per model, so a
**fixed** threshold cannot fairly compare them. We therefore report
each model's **own best F1 at its own per-model best threshold**.

Models:
- `intfloat/multilingual-e5-base` (768d) — agent-kms default
- `intfloat/multilingual-e5-large` (1024d) — same family, larger
- `cl-nagoya/ruri-v3-310m` (768d) — Japanese-specialized, current generation
- `cl-nagoya/ruri-v3-30m` (256d) — Japanese-specialized, very small

We initially planned to test `cl-nagoya/ruri-large` and
`pkshatech/GLuCoSE-base-ja-v2`, but both depend on tokenizer / processor
classes that are not loadable in current `transformers` + `sentence-transformers`
versions (they error out at model-load time). We substitute the
`ruri-v3` line, which is the maintained successor to `ruri-large` and
covers the same "Japanese-specialized embedding" axis at two size points.

To avoid `sentence-transformers` / `qdrant-client` cache state spilling
between models, **each model is run in its own subprocess** via
`scripts/run_one_model.py`. Per-model output lands in
`results/model_<safe_name>.csv` and we aggregate here.
"""),
    ("code", """import os, sys, subprocess, time
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

MODELS = [
    ("intfloat/multilingual-e5-base",   768),
    ("intfloat/multilingual-e5-large", 1024),
    ("cl-nagoya/ruri-v3-310m",          768),
    ("cl-nagoya/ruri-v3-30m",           256),
]

def safe(name): return name.replace("/", "_").replace("-", "_")
"""),
    ("code", """results_dir = ROOT / "results"
results_dir.mkdir(exist_ok=True)
script = ROOT / "scripts" / "run_one_model.py"

for model, dim in MODELS:
    out = results_dir / f"model_{safe(model)}.csv"
    if out.exists():
        print(f"skip  {model}  (cached -> {out.name})")
        continue
    print(f"\\nrun   {model}  (dim={dim})")
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, str(script),
         "--model", model, "--dim", str(dim),
         "--collection", f"{COLLECTION}_{safe(model)}",
         "--out", str(out)],
        cwd=ROOT,
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    print(r.stdout[-600:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-600:])
        raise RuntimeError(f"{model} subprocess failed")
    print(f"done in {elapsed:.1f}s -> {out.name}")
"""),
    ("code", """frames = []
for model, dim in MODELS:
    out = results_dir / f"model_{safe(model)}.csv"
    df = pd.read_csv(out)
    frames.append(df)
all_df = pd.concat(frames, ignore_index=True)
print(f"combined: {len(all_df)} rows over {all_df['model'].nunique()} models")
"""),
    ("code", """# Aggregate (mean over queries) per (model, threshold)
agg = (all_df.groupby(["model", "threshold"])
       .agg(precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            avg_returned=("num_returned", "mean"))
       .reset_index())

# Per-model best F1 and threshold
best = agg.loc[agg.groupby("model")["f1"].idxmax()].reset_index(drop=True)

# Add ingest time (it's logged in the CSV as a constant column per model)
ingest_time = all_df.groupby("model")["ingest_time_s"].first().reset_index()
best = best.merge(ingest_time, on="model")
print("\\nPer-model best F1 (with own optimal threshold):")
print(best.round(3).to_string(index=False))
"""),
    ("code", """import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(9, 5))
palette = sns.color_palette("tab10", n_colors=len(MODELS))
for (model, _), color in zip(MODELS, palette):
    sub = agg[agg["model"] == model]
    short = model.split("/")[-1]
    ax.plot(sub["threshold"], sub["f1"], marker="o", label=short, color=color)
ax.set_xlabel("score_threshold (per-model — not directly comparable)")
ax.set_ylabel("mean F1 over in-domain queries")
ax.set_title("F1 vs threshold by embedding model")
ax.legend()
plt.tight_layout()
plt.savefig(ROOT / "results" / "fig_model_comparison.png", dpi=140, bbox_inches="tight")
plt.show()
"""),
    ("code", """# Side-by-side bar: best F1 + ingest time + best threshold
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
best_sorted = best.sort_values("f1", ascending=False)
shorts = [m.split("/")[-1] for m in best_sorted["model"]]
ax.bar(shorts, best_sorted["f1"], color="#2ca02c")
for i, (f, t) in enumerate(zip(best_sorted["f1"], best_sorted["threshold"])):
    ax.text(i, f + 0.005, f"F1={f:.3f}\\n@ T={t:.2f}", ha="center", fontsize=9)
ax.set_ylabel("best F1 (own threshold)")
ax.set_title("Per-model best F1")
ax.set_ylim(0, max(best_sorted["f1"]) * 1.25)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

ax = axes[1]
ax.bar(shorts, best_sorted["ingest_time_s"], color="#1f77b4")
for i, t in enumerate(best_sorted["ingest_time_s"]):
    ax.text(i, t + 0.5, f"{t:.1f}s", ha="center", fontsize=9)
ax.set_ylabel("ingest time (seconds)")
ax.set_title("Per-model ingest cost (130 chunks)")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

plt.tight_layout()
plt.savefig(ROOT / "results" / "fig_model_cost_vs_quality.png", dpi=140, bbox_inches="tight")
plt.show()

best.to_csv(ROOT / "results" / "model_comparison_summary.csv", index=False)
"""),
    ("markdown", """## Findings

Empirical per-model best F1 (with each model's own best threshold):

| model                          | best T | P     | R     | F1    | avg ret. | ingest |
|--------------------------------|--------|-------|-------|-------|----------|--------|
| **cl-nagoya/ruri-v3-310m**     | **0.84** | **0.280** | **0.667** | **0.327** | **5.4**  | **23.9s** |
| intfloat/multilingual-e5-large | 0.80   | 0.239 | 0.726 | 0.268 | 14.4     | 23.8s  |
| cl-nagoya/ruri-v3-30m          | 0.86   | 0.226 | 0.518 | 0.246 | 6.2      | 2.4s   |
| intfloat/multilingual-e5-base  | 0.82   | 0.163 | 0.423 | 0.182 | 5.3      | 7.9s   |

- **The agent-kms default model (`e5-base`) is the weakest of the four
  on this corpus** — F1 0.18 vs the best at 0.33 (≈ +80% relative
  improvement). This is a substantial gap, not measurement noise.
- **`cl-nagoya/ruri-v3-310m` wins overall** on F1, precision, and
  returned-set volume. The Japanese-specialized objective is paying for
  itself; it costs the same ingest time as `e5-large` but produces a
  cleaner, smaller returned set at the same recall point.
- **`ruri-v3-30m` is the best cost-quality trade-off**. At 256d and
  ~30M params it ingests in 2.4s — **10× faster than ruri-v3-310m or
  e5-large** — and still beats `e5-base` by 36% relative F1. For
  latency-sensitive on-device or edge deployments this is the model
  to try first.
- **Best thresholds span 0.80 – 0.86 across these four models** —
  the `0.93` default is not even close to optimal on *any* of them
  for this corpus. Anyone porting agent-kms must calibrate per
  (corpus, model) pair; copying thresholds across models is a trap.

**Recommendations for agent-kms**:
1. **Reconsider the default model.** `multilingual-e5-base` was a sensible
   choice in 2024 but the JP-specialized landscape has moved.
   `cl-nagoya/ruri-v3-310m` is a strict improvement on Japanese-heavy
   corpora at the same vector dimensionality (768d).
2. **Ship a `calibrate-threshold` CLI subcommand** that, given a small
   labelled set (e.g. 20 query → expected-source pairs), sweeps
   thresholds and proposes the F1-optimal one for the user's specific
   corpus + model.
3. **Document the 0.93 default's corpus-dependence prominently** —
   on this study's corpus it would silently return zero results for
   every query, and a user would have no way to know that without
   a calibration step.
"""),
]


# ─────────────────────────────────────────────────────────────────────────
# 05. Boost ablation
# ─────────────────────────────────────────────────────────────────────────

NB05 = [
    ("markdown", """# 05. Boost Ablation

**Question**: agent-kms applies two additive score adjustments before
threshold filtering:

- **Severity boost**: `critical +0.05 / high +0.025 / default 0`
- **Applicability penalty**: `universal 0 / conditional -0.005 / topic-specific -0.015`

And one query-side enhancement:
- **Template query expansion** (`template_expand`): appends universal-aspect
  cues to the query.

These were tuned against agent-kms's original corpus, where chunks carry
curated `severity` and `applicability` payload tags. Our public-docs
corpus has **no such curation** — every chunk uses defaults
(`severity=default`, `applicability=topic-specific`). On a flat corpus
like this, we expect:

- Severity boost → **no effect** (every chunk has the same severity).
- Applicability penalty → **uniform -0.015 shift** (every chunk is
  topic-specific, so all scores subtract the same constant — no relative
  ranking change but a shift in *how many* chunks survive the threshold).
- Query expansion → **possibly relevant** — it could pull in more chunks
  by broadening the query embedding.

We isolate each variable at the corpus-best threshold found in notebook 03.
"""),
    ("code", """import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))
from src.eval_utils import resolve_eval_paths

CORPUS_DIR, QUERIES_PATH, COLLECTION = resolve_eval_paths(ROOT)
os.environ["QDRANT_URL"] = "http://localhost:6433"
os.environ["AGENT_KMS_COLLECTION"] = COLLECTION
os.environ["AGENT_KMS_CONFIG"] = str(ROOT / ".agent-kms" / "kms.toml")
os.environ["AGENT_KMS_KNOWLEDGE_ROOT"] = str(CORPUS_DIR)

from agent_kms.retrieve import retrieve
from src.eval_utils import load_queries, gold_set, chunk_key, precision_at_threshold, recall_at_threshold, f1_at_threshold

sns.set_theme(style="whitegrid")
"""),
    ("code", """# Best threshold from notebook 03 (read from results CSV so this notebook
# stays in sync if 03 is re-run with different settings).
sweep_csv = ROOT / "results" / "threshold_sweep.csv"
T = 0.83
if sweep_csv.exists():
    s = pd.read_csv(sweep_csv)
    T = float(s.loc[s["f1"].idxmax(), "threshold"])
print(f"Using best threshold T = {T} (from notebook 03 sweep)")

queries = load_queries(QUERIES_PATH)
in_domain = [q for q in queries if q.get("gold")]
adversarial = [q for q in queries if not q.get("gold")]
print(f"in-domain {len(in_domain)} | adversarial {len(adversarial)}")
"""),
    ("code", """# Conditions — each disables one component (or all of them) at the same T.
# default_boost / default_app are agent-kms's defaults.
DEFAULT_SEV  = {"critical": 0.05, "high": 0.025, "default": 0.0}
DEFAULT_APP  = {"universal": 0.0, "conditional": -0.005, "topic-specific": -0.015}

conditions = {
    "all on (default)":       dict(expand=True,  severity_boost=DEFAULT_SEV, applicability_boost=DEFAULT_APP),
    "no severity boost":      dict(expand=True,  severity_boost={},          applicability_boost=DEFAULT_APP),
    "no applicability penalty": dict(expand=True, severity_boost=DEFAULT_SEV, applicability_boost={}),
    "no query expand":        dict(expand=False, severity_boost=DEFAULT_SEV, applicability_boost=DEFAULT_APP),
    "vanilla (none of the above)": dict(expand=False, severity_boost={},     applicability_boost={}),
}

def eval_one(cond_kwargs):
    rows_in = []
    for q in in_domain:
        r = retrieve(q["query"], score_threshold=T, **cond_kwargs)
        g = gold_set(q)
        rows_in.append({
            "precision": precision_at_threshold(r, g),
            "recall":    recall_at_threshold(r, g),
            "f1":        f1_at_threshold(r, g),
            "num_returned": len(r),
        })
    rows_adv = []
    for q in adversarial:
        r = retrieve(q["query"], score_threshold=T, **cond_kwargs)
        rows_adv.append({"num_returned": len(r)})
    df_in = pd.DataFrame(rows_in)
    df_adv = pd.DataFrame(rows_adv)
    return dict(
        precision=df_in["precision"].mean(),
        recall=df_in["recall"].mean(),
        f1=df_in["f1"].mean(),
        avg_returned=df_in["num_returned"].mean(),
        adv_empty_rate=(df_adv["num_returned"] == 0).mean(),
    )

rows = []
for name, kw in conditions.items():
    res = eval_one(kw)
    res["condition"] = name
    rows.append(res)

summary = pd.DataFrame(rows).set_index("condition")[
    ["precision", "recall", "f1", "avg_returned", "adv_empty_rate"]
]
print(f"\\nAblation results at T={T}:")
print(summary.round(3).to_string())
"""),
    ("code", """fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
summary[["precision", "recall", "f1"]].plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
ax.set_title(f"Boost ablation — quality metrics at T={T}")
ax.set_ylabel("score (mean over in-domain queries)")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
ax.legend(loc="upper right")

ax = axes[1]
ax.bar(summary.index, summary["avg_returned"], color="#9467bd")
ax.set_title(f"Boost ablation — volume at T={T}")
ax.set_ylabel("avg returned chunks (in-domain)")
ax.set_xticklabels(summary.index, rotation=20, ha="right")
for i, v in enumerate(summary["avg_returned"]):
    ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(ROOT / "results" / "fig_boost_ablation.png", dpi=140, bbox_inches="tight")
plt.show()

summary.to_csv(ROOT / "results" / "boost_ablation.csv")
"""),
    ("markdown", """## Findings

Empirical results at the corpus-best threshold T=0.81 (multilingual-e5-base):

| condition                    | P     | R     | F1    | avg ret. | adv empty |
|------------------------------|-------|-------|-------|----------|-----------|
| all on (default)             | 0.159 | 0.649 | 0.201 | 13.2     | 0.75      |
| no severity boost            | 0.159 | 0.649 | 0.201 | 13.2     | 0.75      |
| no applicability penalty     | 0.067 | 0.804 | 0.111 | 40.8     | 0.00      |
| no query expand              | 0.082 | 0.929 | 0.133 | 33.6     | 0.50      |
| vanilla (none of the above)  | 0.030 | 1.000 | 0.058 | 69.0     | 0.00      |

- **Severity boost is exactly a no-op on this corpus** — disabling it
  leaves every metric unchanged to 3 decimal places. Reason: every chunk
  ingested under the default `severity=default` tag, so the boost dict's
  per-tag adjustment evaluates to 0 for all of them.
- **Applicability penalty (-0.015 for `topic-specific` chunks) is the
  single most impactful filter.** Removing it triples `avg_returned`
  (13 → 41) and halves F1 (0.20 → 0.11). It effectively acts as a hidden
  threshold tightening of 0.015.
- **Query expansion suppresses returned-count too.** Disabling expansion
  raises recall to 0.93 (more chunks pass the threshold) but lets
  enough noise through to halve F1. Adversarial empty-rate also drops
  from 0.75 to 0.50 — i.e. expansion *also* helps protect against
  out-of-domain queries.
- **Defaults are the best F1 point at this T** for this corpus, but
  they hit that F1 by lowering the *effective* threshold via the
  applicability penalty, not by surfacing extra signal. The same F1
  could be reached with a fresh threshold sweep on
  `retrieve(score_threshold=T, applicability_boost={})`.

**Implication for agent-kms users**: the documented `0.93` default
implicitly bakes in a `-0.015` shift for uncurated corpora. If you
plan to skip the curation step, you should either:
1. Tag every chunk's `applicability=universal` so the penalty becomes 0, or
2. Pass `applicability_boost={}` and lower your threshold by ~0.015 to
   compensate.

Both are clearer than relying on the hidden shift.
"""),
]


# ─────────────────────────────────────────────────────────────────────────
# 06. Findings and recommendations
# ─────────────────────────────────────────────────────────────────────────

NB06 = [
    ("markdown", """# 06. Findings and Recommendations

Synthesis of notebooks 03-05. This is a write-up notebook — minimal code,
mostly markdown. All numbers below are pulled from
`results/threshold_sweep.csv`, `results/model_comparison_summary.csv`,
and `results/boost_ablation.csv`.
"""),
    ("code", """import sys
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

t_sweep = pd.read_csv(ROOT / "results" / "threshold_sweep.csv")
model_summ = pd.read_csv(ROOT / "results" / "model_comparison_summary.csv")
ablation = pd.read_csv(ROOT / "results" / "boost_ablation.csv", index_col=0)

best_t = t_sweep.loc[t_sweep["f1"].idxmax()]
print(f"Best threshold (e5-base, this corpus): T={best_t['threshold']:.2f}  F1={best_t['f1']:.3f}")
print()
print("Per-model best F1 + threshold:")
print(model_summ[["model", "threshold", "f1", "ingest_time_s"]].round(3).to_string(index=False))
print()
print("Boost ablation @ best T (e5-base):")
print(ablation.round(3))
"""),
    ("markdown", """## Headline

On a corpus of **15 Japanese technical-doc markdown files (127 H2 chunks)**
and **32 hand-authored evaluation queries** (28 in-domain, 4 adversarial),
two clear findings emerge:

1. **agent-kms's documented default of `score_threshold=0.93` does not
   generalize.** On this corpus, at T=0.93 every in-domain query
   returns zero chunks (100% empty-rate). The actual best F1 with the
   default model `multilingual-e5-base` lands at **T = 0.81, F1 = 0.20**.

2. **The default embedding model `multilingual-e5-base` is the weakest
   of the four tested.** Swapping it for `cl-nagoya/ruri-v3-310m`
   (Japanese-specialized, same 768d) lifts best F1 from 0.18 to **0.33**
   — a ~80% relative improvement, with no change to retrieval logic
   and similar ingest cost.

Together, these findings imply that an agent-kms user who follows the
README defaults on a Japanese public-documentation corpus will get
**100% empty results** until they calibrate the threshold, and even
then will be ~40-80% below the F1 ceiling reachable by switching the
embedding model.

The boost machinery (`severity`, `applicability`, `query_expand`) has
secondary effects: severity is a no-op for uncurated corpora,
applicability penalty acts as a hidden -0.015 score shift, and query
expansion is the most useful of the three for paraphrased queries.
"""),
    ("markdown", """## Recommendations

### For agent-kms upstream

1. **Reconsider the default embedding model.** `intfloat/multilingual-e5-base`
   is the weakest of four models tested on this Japanese-doc corpus.
   `cl-nagoya/ruri-v3-310m` produces ~80% higher F1 at the same vector
   dimensionality (768d) and comparable ingest cost.

2. **Document the corpus-dependence of `score_threshold=0.93` prominently.**
   The README presents 0.93 as the empirically-chosen sweet spot; in
   practice it returns zero chunks on at least one realistic corpus.
   A short paragraph and a "if your retrieve calls return nothing, try
   T = 0.80 - 0.85 first" hint would prevent silent failure.

3. **Add a `calibrate-threshold` CLI subcommand.** Given a small labelled
   set (e.g. 20 query → expected-source pairs), it would sweep
   thresholds and propose the F1-optimal one for the user's specific
   corpus + model.

4. **Make the applicability penalty opt-in for uncurated corpora.**
   Today, ingesting a corpus without explicit `applicability` tags
   silently applies a -0.015 score shift to every chunk — this is the
   kind of behavior that makes the threshold default feel unreliable.

5. **Surface the score distribution in `agent-kms ingest`.** Print e.g.
   "p90 cosine for sample queries: 0.85; consider T ≤ 0.82". This
   would prevent users from silently getting empty results.

### For agent-kms users

- **Sweep before trusting any default.** The sweep code in this study
  generalises to any corpus in <100 lines (`src/eval_utils.py` + a YAML).
- **If you switch the embedding model, your threshold must move with it.**
  Per-model best thresholds in this study span 0.80 – 0.86 — a 0.06
  range that is bigger than the F1 differences between models.
- **Severity / applicability tagging is leverage you have to opt in to.**
  For a flat (untagged) corpus, the severity boost is a no-op and the
  applicability penalty acts as a uniform -0.015 score shift — equivalent
  to lowering your threshold by that much. Tag chunks explicitly or
  disable the boosts.
"""),
    ("markdown", """## Limitations

- **N=32 queries is small.** Confidence intervals on F1 are wide;
  consider a single decimal of F1 difference noise.
- **Single annotator** (myself). Gold relevance judgements reflect
  one person's view of what should be retrieved for each query.
- **One corpus domain.** React + JS Primer are intentionally similar
  registers (technical Japanese, web frameworks). Findings about
  threshold magnitude may not transfer to e.g. enterprise wikis,
  scientific papers, or legal Japanese.
- **Single point in time.** Embedding models, agent-kms's defaults, and
  the corpus content all evolve. Re-run this study any time agent-kms
  changes its default threshold or model.
- **No end-to-end LLM evaluation.** We measure retrieval quality, not
  whether the retrieved context lets an LLM produce a correct answer.
"""),
    ("markdown", """## Future work

- Expand to 100+ queries with at least 2 annotators per query;
  publish inter-annotator agreement.
- Add an **end-to-end LLM-as-judge** stage that runs Claude / Gemini on
  retrieved chunks and scores the final answer.
- Evaluate **hybrid retrieval** (BM25 + dense) — agent-kms's `sparse.py`
  module already implements the sparse half; this study covers only the
  dense path.
- Extend the corpus to include enterprise-style heterogeneous knowledge
  (FAQ + ADRs + slack snippets) to test agent-kms's `severity` /
  `applicability` machinery in the regime it was actually designed for.
"""),
]


# ─────────────────────────────────────────────────────────────────────────
# 07. End-to-end pairwise coding evaluation
# ─────────────────────────────────────────────────────────────────────────

NB07 = [
    ("markdown", """# 07. End-to-end Pairwise Coding Evaluation

**Question**: does injecting agent-kms's retrieved context into a
coding-LLM's prompt actually produce **better code** than the same
prompt without context?

Notebooks 02-06 measure *retrieval quality* (does the right chunk
come back?). This notebook measures *downstream usefulness* (does
that chunk actually help the LLM solve the task?).

## Setup

| Component | Choice |
|---|---|
| Target LLM (writes code) | **Ollama `qwen3:8b`** — local, mid-tier; weak enough that retrieval signal is expected to be measurable |
| Coding tasks | **25 hand-authored Japanese instructions** covering React / JS Primer topics that the corpus actually contains. See `data/coding_tasks.yaml` |
| Conditions | **A_noctx** (prompt only) vs **B_ctx** (prompt + top-K chunks from `agent_kms.retrieve(score_threshold=0.81)`) |
| Judge | **Ollama `qwen2.5:14b`** — stronger local model. Each (A,B) pair is judged in **both orderings** (AB, BA) to neutralise positional bias |
| Verdict per task | `ctx_win` / `noctx_win` / `tie` based on whether ctx wins both, neither, or one of the two orderings |

Outputs live under `results/pairwise/`:
- `<task_id>__A_noctx.txt` / `__B_ctx.txt` — raw LLM outputs
- `_generation_meta.csv` — per-call latency & token counts
- `_judge_results.csv` — raw judge verdicts (2 rows per task)
- `_judge_results_agg.csv` — collapsed per-task outcomes

This notebook **only analyses** existing results. To regenerate them:

```bash
python scripts/run_pairwise_generation.py
python scripts/run_pairwise_judge.py
```
"""),
    ("code", """import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

sns.set_theme(style="whitegrid")

raw = pd.read_csv(ROOT / "results" / "pairwise" / "_judge_results.csv")
agg = pd.read_csv(ROOT / "results" / "pairwise" / "_judge_results_agg.csv")
gen = pd.read_csv(ROOT / "results" / "pairwise" / "_generation_meta.csv")

print(f"raw verdicts: {len(raw)} rows ({raw['task_id'].nunique()} tasks × 2 orderings)")
print(f"per-task outcomes: {len(agg)}")
"""),
    ("code", """# Per-ordering raw verdict distribution
print("Per-ordering winners (positional bias check):")
print(raw.groupby("order")["winner"].value_counts().unstack(fill_value=0))
print()
print("Per-task aggregated outcome:")
print(agg["outcome"].value_counts().to_string())
"""),
    ("code", """# Collapse to strict / lenient win-rates for ctx
counts = agg["outcome"].value_counts()
n_tasks = len(agg)
strict_ctx_win = counts.get("ctx_win", 0)
strict_noctx_win = counts.get("noctx_win", 0)
lean_ctx = counts.get("ctx_win", 0) + counts.get("ctx_lean", 0)
lean_noctx = counts.get("noctx_win", 0) + counts.get("noctx_lean", 0)
ties = counts.get("tie", 0)

print(f"n_tasks = {n_tasks}")
print(f"strict ctx wins   : {strict_ctx_win}  ({strict_ctx_win/n_tasks:.1%})")
print(f"strict noctx wins : {strict_noctx_win}  ({strict_noctx_win/n_tasks:.1%})")
print(f"lean ctx (incl tie-broken): {lean_ctx}  ({lean_ctx/n_tasks:.1%})")
print(f"lean noctx                 : {lean_noctx}  ({lean_noctx/n_tasks:.1%})")
print(f"ties                       : {ties}  ({ties/n_tasks:.1%})")
"""),
    ("code", """# McNemar test on (strict ctx_win vs strict noctx_win)
# H0: retrieval doesn't change which condition wins.
from scipy.stats import binomtest

discordant = strict_ctx_win + strict_noctx_win
if discordant >= 1:
    p = binomtest(strict_ctx_win, n=discordant, p=0.5, alternative="two-sided").pvalue
    print(f"discordant tasks: {discordant}  (ctx={strict_ctx_win}, noctx={strict_noctx_win})")
    print(f"McNemar exact two-sided p-value: {p:.3f}")
else:
    print("No discordant tasks — cannot compute test")
"""),
    ("code", """# Visualise outcomes
order_winners = (raw.groupby(["order", "winner"]).size().unstack(fill_value=0))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
order_winners.plot(kind="bar", stacked=True, ax=ax, color=["#1f77b4", "#999999", "#ff7f0e"])
ax.set_title("Raw judge verdicts — positional check")
ax.set_xlabel("ordering shown to judge")
ax.set_ylabel("count over 25 tasks")
ax.set_xticklabels(["A=noctx, B=ctx", "A=ctx, B=noctx"], rotation=0)
ax.legend(title="winner")

ax = axes[1]
outcome_order = ["ctx_win", "ctx_lean", "tie", "noctx_lean", "noctx_win"]
counts_order = [counts.get(o, 0) for o in outcome_order]
colors = ["#1f7720", "#86c47a", "#999999", "#f0a07a", "#a82020"]
ax.bar(outcome_order, counts_order, color=colors)
ax.set_title("Per-task aggregated outcome")
ax.set_ylabel("number of tasks")
ax.set_xticklabels(outcome_order, rotation=20, ha="right")
for i, c in enumerate(counts_order):
    if c:
        ax.text(i, c + 0.1, str(c), ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(ROOT / "results" / "fig_pairwise_outcomes.png", dpi=140, bbox_inches="tight")
plt.show()
"""),
    ("code", """# Per-task breakdown for the writeup
import yaml
with (ROOT / "data" / "coding_tasks.yaml").open(encoding="utf-8") as f:
    tasks_meta = {t["id"]: t for t in yaml.safe_load(f)}

agg_anno = agg.copy()
agg_anno["difficulty"] = agg_anno["task_id"].map(lambda x: tasks_meta[x]["difficulty"])
agg_anno["topic"] = agg_anno["task_id"].map(
    lambda x: tasks_meta[x]["expected_topics"][0] if tasks_meta[x]["expected_topics"] else ""
)
print(agg_anno.to_string(index=False))
"""),
    ("code", """# Generation cost — context overhead
print("Generation latency by condition (seconds):")
print(gen.groupby("condition")["elapsed_s"].describe()[["count", "mean", "50%", "max"]].round(1))
print()
print("Output size (chars):")
print(gen.groupby("condition")["out_chars"].describe()[["count", "mean", "50%", "max"]].round(0))
print()
print("Tokens generated (output only):")
print(gen.groupby("condition")["tokens_out"].describe()[["count", "mean", "50%", "max"]].round(0))
"""),
    ("markdown", """## Findings

### Headline: **no statistically significant effect** on this corpus

| metric | value |
|---|---|
| strict ctx wins (won both orderings) | **6 / 25 (24%)** |
| strict noctx wins                    | **5 / 25 (20%)** |
| ties                                 | **13 / 25 (52%)** |
| McNemar exact two-sided p            | **1.000** |

With qwen3:8b on this React + JS corpus, **agent-kms retrieval does
not measurably improve coding output quality** in a way that survives
the swap-and-require-both bias control. The +1 task advantage for ctx
(6 vs 5) is noise.

### Positional bias is large — the swap heuristic was doing real work

| order shown | ctx wins | noctx wins | ties |
|---|---|---|---|
| AB (X=noctx, Y=ctx shown first) | 7  | **16** | 2 |
| BA (X=ctx,   Y=noctx shown first) | **17** | 5  | 3 |

The judge model (qwen2.5:14b) has a strong **"first-shown answer
wins" bias**. Without the swap, a naive single-ordering judge would
report either "noctx clearly better (16 vs 7)" or "ctx clearly better
(17 vs 5)" depending purely on display order. This is the most
important methodological takeaway of this notebook for anyone
running similar evals locally.

### Where ctx helped (6 strict wins)

`t07` complex state objects · `t08` Context API ·
`t17` Optional chaining · `t20` JSON edge cases ·
`t21` mutable/immutable array methods · `t25` ES2022 Error Cause.

Pattern: **multi-section / newer-spec topics** where qwen3:8b's
training-time knowledge of the exact API surface is shaky.

### Where ctx hurt (5 strict losses)

`t02` trivial `<Greeting>` · `t03` alert button · `t10` useEffect deps
explanation · `t16` object destructuring · `t22` CSV string normalise.

Pattern: **easy / one-liner tasks** where injecting 3-8 chunks of
tutorial markdown dilutes a 5-line answer. The LLM tries to "use" the
context, producing wordier output that the judge marks down.

### Cost overhead is real

| metric         | A_noctx | B_ctx | overhead |
|----------------|---------|-------|----------|
| latency mean   | 211.6s  | 1139.9s | **5.4×** |
| latency median | 9.2s    | 90.3s | **9.8×** |
| output chars (mean)  | 474 | 713 | +50% |
| tokens out (mean)    | 161 | 275 | +71% |

The latency blow-up is largely **prompt-processing time** for the
larger prompt, not generation time. For a Claude-Code-style
interactive workflow, a **5× wait penalty for no measurable quality
gain** is hard to justify on this corpus + LLM combination.

## Implications

- **For agent-kms users running on a strong general LLM** (Claude /
  GPT-4 class) with a *generic tech-docs corpus* (React, JS, Python,
  etc.): retrieval has diminishing returns. The LLM already knows
  this domain. Retrieval pays off most when (a) the corpus contains
  knowledge the LLM lacks (internal APIs, post-cutoff specs), or
  (b) the LLM is weaker than its task demands.
- **Cherry-picked retrieval matters more than always-on retrieval.**
  6 of 25 tasks benefited; 5 actively hurt. A retrieval gate that
  only fires when query embedding similarity to top chunks exceeds
  a threshold (or a generic "does the LLM need this?" check) would
  outperform unconditional injection.
- **Re-run on a private / internal corpus.** Anyone applying this
  framework to their own knowledge base (where the LLM has zero
  training exposure) should expect a much larger ctx-win effect.
  Re-run `notebook 07` with `EVAL_MODE=private` once such a corpus
  is in place.

## Validity caveats

- **N=25 is small.** A 10pp absolute effect at 80% power needs
  ~80-100 tasks. With the observed effect (≈0pp), we can comfortably
  rule out a large effect (>30pp) but not a true small effect
  (5-10pp) in either direction.
- **Judge is qwen2.5:14b**, local. A Claude-grade judge would be
  more reliable. Position-swap mitigates but cannot fully eliminate
  judge bias. The 13 ties may be a judge-capability artefact.
- **Corpus and tasks share a domain by construction** — the most
  optimistic possible setting for retrieval. Real deployments with
  noisier corpus-task alignment would shift the balance toward
  noctx, not the other way.
- **qwen3:8b is one specific target.** A weaker target (e.g.
  qwen3:1.7b) would likely show a positive ctx effect; a stronger
  one (Claude) would likely show a smaller effect.
"""),
]


# ─────────────────────────────────────────────────────────────────────────
# Write all notebooks
# ─────────────────────────────────────────────────────────────────────────

NB.mkdir(exist_ok=True)
write_notebook(str(NB / "01_setup_and_corpus.ipynb"), NB01)
write_notebook(str(NB / "02_baseline_retrieval_qualitative.ipynb"), NB02)
write_notebook(str(NB / "03_threshold_sweep.ipynb"), NB03)
write_notebook(str(NB / "04_embedding_model_comparison.ipynb"), NB04)
write_notebook(str(NB / "05_boost_ablation.ipynb"), NB05)
write_notebook(str(NB / "06_findings_and_recommendations.ipynb"), NB06)
write_notebook(str(NB / "07_pairwise_coding_eval.ipynb"), NB07)
print("\\nall notebooks rebuilt.")
