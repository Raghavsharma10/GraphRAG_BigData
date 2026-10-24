#!/usr/bin/env python3
"""
dashboard.py
============
Milestone 6 — Streamlit Dashboard

Reads from local CSV files only (no HDFS, no Parquet, no pandas dependency
on pyarrow, no sentence_transformers).

Tabs:
  1. Pipeline Summary  — milestone status, dataset stats
  2. Graph Metrics     — degree/PageRank histograms, component sizes
  3. Retrieval         — MRR@10, NDCG@10 comparison table + per-query scatter
  4. Risk Passages     — top-50 passages by hallucination-risk proxy

Run:
  streamlit run src/dashboard.py
"""

import os
import csv
import math

import streamlit as st

# ── Data paths (relative to project root) ────────────────────────────────────
SUMMARY_CSV     = "data/scores/summary.csv"
RESULTS_CSV     = "data/scores/retrieval_results.csv"
TOP_CSV         = "data/scores/top_passages.csv"
GRAPHX_CSV      = "data/graphx_results.csv"
EDGES_CSV       = "data/edges.csv"
PID_MAP_CSV     = "data/pid_to_idx.csv"
SUBSET_CSV      = "data/msmarco_subset.csv"
EMBED_CSV       = "data/embeddings.csv"       # passage embeddings (1053×384)
MODEL_REPO      = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM       = 384
MAX_SEQ_LEN     = 128
LAMBDA          = 0.7    # cosine weight in combined score
FALLBACK_NORM_PR = 0.5   # used for isolated passages (not in GraphX output)



# ── Helpers ──────────────────────────────────────────────────────────────────

def load_csv_dicts(path):
    """Load CSV as list of dicts. Returns [] if file not found."""
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_summary(path):
    """Returns {metric: {cosine, graph, delta}} dict."""
    rows = load_csv_dicts(path)
    summary = {}
    for row in rows:
        summary[row["metric"]] = {
            "cosine": row.get("cosine_baseline", ""),
            "graph":  row.get("graph_reranked", ""),
            "delta":  row.get("delta", ""),
        }
    return summary


def md_table(rows: list, max_col_width: int = 60) -> None:
    """
    Render a list-of-dicts as a markdown table using st.markdown().
    Completely avoids pandas / pyarrow / altair, all of which fail under numpy 2.x.
    Each cell value is truncated to max_col_width characters.
    """
    if not rows:
        st.info("No data to display.")
        return
    headers = list(rows[0].keys())
    # Header row
    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    sep_line    = "| " + " | ".join("---" for _ in headers) + " |"
    data_lines  = []
    for row in rows:
        cells = []
        for h in headers:
            val = str(row.get(h, ""))
            if len(val) > max_col_width:
                val = val[:max_col_width - 1] + "…"
            cells.append(val)
        data_lines.append("| " + " | ".join(cells) + " |")
    st.markdown("\n".join([header_line, sep_line] + data_lines))


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRAG-Scale Dashboard",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 GraphRAG-Scale: Distributed Topological Analysis of Vector Embeddings")
st.caption("CSE412 Big Data Project | HDFS → Spark → Spark GraphX → Evaluation")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📋 Pipeline Summary", "🕸️ Graph Metrics", "📊 Retrieval Comparison",
     "⚠️ Risk Passages", "🔎 Live Retrieval"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pipeline Summary
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Pipeline Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Corpus Passages", "1,053")
        st.metric("Evaluation Queries", "1,000")
    with col2:
        graphx = load_csv_dicts(GRAPHX_CSV)
        edges  = load_csv_dicts(EDGES_CSV)
        st.metric("Graph Vertices", f"{len(graphx):,}")
        st.metric("Graph Edges", f"{len(edges):,}")
    with col3:
        summary = load_summary(SUMMARY_CSV)
        mrr = summary.get("MRR@10", {})
        ndcg = summary.get("NDCG@10", {})
        st.metric("Cosine MRR@10",  mrr.get("cosine", "—"))
        st.metric("Cosine NDCG@10", ndcg.get("cosine", "—"))

    st.divider()
    st.subheader("Milestone Status")
    milestones = [
        ("M0", "Environment Setup",    "✅ COMPLETE", "HDFS running, PySpark importable"),
        ("M1", "HDFS Ingestion",        "✅ COMPLETE", "1,053 passages → HDFS Parquet (3.4s)"),
        ("M2", "Embeddings (ONNX)",     "✅ COMPLETE", "1,053 × 384-dim, self-sim=1.000 (47.4s)"),
        ("M3", "FAISS k-NN Edges",      "✅ COMPLETE", "2,291 edges, 0 self-edges, avg deg=4.35 (10.4s)"),
        ("M4", "Spark GraphX (Scala)",  "✅ COMPLETE", "936 vertices, 21 components, PR∈[0.24, 3.16]"),
        ("M5", "Retrieval Evaluation",  "✅ COMPLETE", "MRR@10=0.9874 cosine | 0.9635 graph-reranked"),
        ("M6", "Streamlit Dashboard",   "✅ COMPLETE", "This dashboard"),
        ("M7", "Tests + Report",        "🔄 IN PROGRESS", "13 dataset tests pass"),
    ]
    for mid, name, status, notes in milestones:
        with st.expander(f"{status} **{mid}: {name}**"):
            st.write(notes)

    st.divider()
    st.subheader("Technology Stack")
    st.markdown("""
    | Layer | Technology | Purpose |
    |-------|-----------|---------|
    | Storage | **HDFS 3.3.6** | Distributed storage for all intermediate Parquet files |
    | Processing | **Apache Spark 4.2.0** | PySpark for data ingestion, embedding write, edge write |
    | Graph | **Spark GraphX (Scala 2.13)** | Native PageRank, ConnectedComponents, TriangleCount |
    | Embeddings | **all-MiniLM-L6-v2 (ONNX)** | 384-dim sentence embeddings via ONNX Runtime |
    | k-NN | **FAISS IndexFlatIP** | Cosine k-NN graph construction |
    | Evaluation | **Python (pure stdlib)** | MRR@10, NDCG@10, retrieval latency |
    | Dashboard | **Streamlit** | This interface |
    """)

    st.info(
        "**Research integrity note**: MS MARCO relevance labels indicate "
        "retrieval relevance, not hallucination ground truth. "
        "Graph-reranking is reported as an experimental hypothesis. "
        "The negative delta (−0.024 MRR) is an honest result."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Graph Metrics
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Graph Metrics (Spark GraphX Output)")

    graphx_rows = load_csv_dicts(GRAPHX_CSV)
    if not graphx_rows:
        st.error(f"Could not load {GRAPHX_CSV}")
        st.stop()

    pageranks   = [float(r["pagerank"])       for r in graphx_rows]
    norm_prs    = [float(r["norm_pagerank"])   for r in graphx_rows]
    degrees     = [int(r["degree"])            for r in graphx_rows]
    triangles   = [int(r["triangle_count"])    for r in graphx_rows]
    components  = [int(r["component_id"])      for r in graphx_rows]

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vertices in Graph", f"{len(graphx_rows):,}")
    c2.metric("Connected Components", f"{len(set(components)):,}")
    c3.metric("Max PageRank", f"{max(pageranks):.3f}")
    c4.metric("Max Degree", f"{max(degrees):,}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("PageRank Distribution")
        # Histogram using pure Python bins
        n_bins = 20
        pr_min, pr_max = min(pageranks), max(pageranks)
        bin_width = (pr_max - pr_min) / n_bins
        bins = [0] * n_bins
        for pr in pageranks:
            idx = min(int((pr - pr_min) / bin_width), n_bins - 1)
            bins[idx] += 1

        pr_rows = [
            {"PageRank bin": f"{pr_min + i*bin_width:.3f}–{pr_min + (i+1)*bin_width:.3f}",
             "Count": bins[i],
             "Bar": "█" * max(1, int(bins[i] * 20 / max(bins))) if bins[i] > 0 else ""}
            for i in range(n_bins)
        ]
        md_table(pr_rows)
        st.caption(f"Mean PageRank: {sum(pageranks)/len(pageranks):.4f} | "
                   f"Range: [{pr_min:.3f}, {pr_max:.3f}]")

    with col_right:
        st.subheader("Degree Distribution")
        max_deg = max(degrees)
        deg_bins = [0] * (max_deg + 1)
        for d in degrees:
            deg_bins[d] += 1
        deg_rows = [
            {"Degree": d,
             "Count": deg_bins[d],
             "Bar": "█" * max(1, int(deg_bins[d] * 20 / max(deg_bins))) if deg_bins[d] > 0 else ""}
            for d in range(max_deg + 1) if deg_bins[d] > 0
        ]
        md_table(deg_rows)
        avg_deg = sum(degrees) / len(degrees)
        st.caption(f"Mean degree: {avg_deg:.2f} | Max: {max_deg}")

    st.divider()
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.subheader("Triangle Count Distribution")
        max_tri = max(triangles) if triangles else 0
        tri_bins = [0] * (min(max_tri + 1, 21))
        for t in triangles:
            tri_bins[min(t, 20)] += 1
        tri_rows = [
            {"Triangle count (≥20 grouped)": "≥20" if i == 20 else str(i),
             "Count": tri_bins[i],
             "Bar": "█" * max(1, int(tri_bins[i] * 20 / max(tri_bins))) if tri_bins[i] > 0 else ""}
            for i in range(len(tri_bins)) if tri_bins[i] > 0
        ]
        md_table(tri_rows)
        n_zero_tri = sum(1 for t in triangles if t == 0)
        st.caption(f"Passages with 0 triangles: {n_zero_tri} / {len(triangles)}")

    with col_r2:
        st.subheader("Component Size Distribution")
        from collections import Counter
        comp_counts = Counter(components)
        sizes = sorted(Counter(comp_counts.values()).items())
        st.write("Component sizes:")
        for size, count in sizes:
            st.write(f"  {count} component(s) with {size} passage(s)")
        st.caption(f"Total components: {len(comp_counts)} | "
                   f"Largest: {max(comp_counts.values())} passages")

    st.divider()
    st.subheader("Top 20 Passages by PageRank")
    top_pr = sorted(graphx_rows, key=lambda r: -float(r["pagerank"]))[:20]
    md_table([{
            "idx": r["passage_idx"],
            "PageRank": f"{float(r['pagerank']):.4f}",
            "Norm PR": f"{float(r['norm_pagerank']):.4f}",
            "Degree": int(r["degree"]),
            "Triangles": int(r["triangle_count"]),
            "Component": r["component_id"],
        } for r in top_pr])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Retrieval Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Retrieval Evaluation Results")

    summary = load_summary(SUMMARY_CSV)
    if not summary:
        st.error(f"Could not load {SUMMARY_CSV}")
        st.stop()

    st.subheader("Aggregate Metrics")
    mrr_data  = summary.get("MRR@10", {})
    ndcg_data = summary.get("NDCG@10", {})
    lat_data  = summary.get("avg_latency_ms", {})
    n_data    = summary.get("queries_evaluated", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MRR@10 — Cosine",   mrr_data.get("cosine", "—"))
    c2.metric("MRR@10 — Graph",    mrr_data.get("graph", "—"),
              delta=mrr_data.get("delta", ""))
    c3.metric("NDCG@10 — Cosine",  ndcg_data.get("cosine", "—"))
    c4.metric("NDCG@10 — Graph",   ndcg_data.get("graph", "—"),
              delta=ndcg_data.get("delta", ""))

    st.info(
        f"**λ = 0.7** (cosine weight) | Queries evaluated: "
        f"{n_data.get('cosine', '1,000')} | "
        f"Avg latency — cosine: {lat_data.get('cosine', '—')} ms | "
        f"graph: {lat_data.get('graph', '—')} ms"
    )

    st.caption(
        "**Interpretation**: Graph-reranking (0.7·cosine + 0.3·norm_pagerank) "
        "slightly underperforms pure cosine on this closed-corpus task. "
        "High absolute scores (MRR≈0.99) occur because the evaluation corpus "
        "contains exactly the passages referenced in qrels — retrieval is nearly "
        "deterministic in this closed setting."
    )

    st.divider()
    st.subheader("Per-Query Metrics (first 200 queries)")
    results = load_csv_dicts(RESULTS_CSV)
    if results:
        display = [{
            "query_id":  r["query_id"],
            "query":     r["query"][:50] + "…" if len(r["query"]) > 50 else r["query"],
            "n_relevant": r["n_relevant"],
            "cos_mrr":   float(r["cos_mrr"]),
            "cos_ndcg":  float(r["cos_ndcg"]),
            "grp_mrr":   float(r["grp_mrr"]),
            "grp_ndcg":  float(r["grp_ndcg"]),
            "mrr_delta": round(float(r["grp_mrr"]) - float(r["cos_mrr"]), 4),
        } for r in results[:200]]
        md_table(display)

        # Delta histogram
        st.subheader("MRR@10 Delta Distribution (Graph − Cosine)")
        all_deltas = [round(float(r["grp_mrr"]) - float(r["cos_mrr"]), 3)
                      for r in results]
        n_pos = sum(1 for d in all_deltas if d > 0)
        n_neg = sum(1 for d in all_deltas if d < 0)
        n_zero = sum(1 for d in all_deltas if d == 0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Queries where graph helps (Δ>0)", n_pos)
        c2.metric("Queries where graph hurts (Δ<0)", n_neg)
        c3.metric("No change (Δ=0)", n_zero)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Risk Passages
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("Top-50 Passages by Hallucination-Risk Proxy")
    st.caption(
        "Risk score = 0.5·(1−norm_pagerank) + 0.3·(1/(degree+1)) + 0.2·(triangle_count==0)\n\n"
        "**Important**: This is an experimental hypothesis. Low graph centrality "
        "is used as a proxy for passages that may be less well-supported by the "
        "document network — not a validated hallucination detector."
    )

    top_rows = load_csv_dicts(TOP_CSV)
    if not top_rows:
        st.error(f"Could not load {TOP_CSV}")
        st.stop()

    # Load passage text for display
    passage_text = {}
    subset_rows = load_csv_dicts(SUBSET_CSV)
    for r in subset_rows:
        passage_text[r["passage_id"]] = r["passage"]

    display_top = []
    for r in top_rows:
        pid = r["passage_id"]
        display_top.append({
            "passage_id":    pid,
            "risk_score":    float(r["risk_score"]),
            "norm_pagerank": float(r["norm_pagerank"]),
            "degree":        int(r["degree"]),
            "triangles":     int(r["triangle_count"]),
            "component_id":  r["component_id"],
            "passage":       (passage_text.get(pid, "")[:120] + "…")
                              if len(passage_text.get(pid, "")) > 120
                              else passage_text.get(pid, "—"),
        })

    md_table(display_top, max_col_width=90)

    st.divider()
    risk_scores = [float(r["risk_score"]) for r in top_rows]
    c1, c2 = st.columns(2)
    c1.metric("Max Risk Score",  f"{max(risk_scores):.4f}")
    c2.metric("Min Risk Score (top-50)", f"{min(risk_scores):.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Live Retrieval
# ══════════════════════════════════════════════════════════════════════════════

# ── Cached data loaders (loaded once per session) ────────────────────────────

@st.cache_data(show_spinner="Loading passage embeddings…")
def _load_passage_embeddings():
    """
    Load 1,053 passage embeddings from data/embeddings.csv.
    Returns (list[str], list[list[float]]) — passage_ids, vectors.
    Already L2-normalised (written by 02_embed.py).
    """
    pids, vecs = [], []
    with open(EMBED_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)  # skip header
        for row in reader:
            pids.append(row[0])
            vecs.append([float(v) for v in row[1:]])
    return pids, vecs


@st.cache_data(show_spinner="Loading passage text…")
def _load_passage_text():
    """Returns {passage_id: passage_text} from data/msmarco_subset.csv."""
    text = {}
    with open(SUBSET_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            text[row["passage_id"]] = row["passage"]
    return text


@st.cache_data(show_spinner="Loading graph metrics…")
def _load_graph_metrics():
    """
    Returns {passage_idx (int): {norm_pagerank, pagerank, degree, triangle_count}}.
    norm_pagerank = (PR - PR_min) / (PR_max - PR_min), pre-computed by GraphXJob.scala.
    Isolated passages (117 of 1,053) are absent; callers must apply FALLBACK_NORM_PR.
    """
    gx = {}
    with open(GRAPHX_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gx[int(row["passage_idx"])] = {
                "pagerank":       float(row["pagerank"]),
                "norm_pagerank":  float(row["norm_pagerank"]),
                "degree":         int(row["degree"]),
                "triangle_count": int(row["triangle_count"]),
            }
    return gx


@st.cache_data(show_spinner="Loading passage-index map…")
def _load_pid_to_idx():
    """Returns {passage_id: int idx} from data/pid_to_idx.csv."""
    pid2idx = {}
    with open(PID_MAP_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid2idx[row["passage_id"]] = int(row["idx"])
    return pid2idx


@st.cache_resource(show_spinner="Loading ONNX embedding model…")
def _load_embedder():
    """
    Loads all-MiniLM-L6-v2 ONNX model and HuggingFace tokenizer.
    Uses the cached model from ~/.cache/huggingface/hub (no download if present).
    Same preprocessing as src/02_embed.py and src/embed_queries.py:
      tokenize → ONNX inference → mean-pool over non-padding tokens → L2-normalise.
    """
    try:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        return None, None, str(e)

    try:
        onnx_path = hf_hub_download(repo_id=MODEL_REPO, filename="onnx/model.onnx")
        tok_path  = hf_hub_download(repo_id=MODEL_REPO, filename="tokenizer.json")
        tokenizer = Tokenizer.from_file(tok_path)
        tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=MAX_SEQ_LEN)
        tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        return tokenizer, session, None
    except Exception as e:
        return None, None, str(e)


def _embed_query(query_text: str, tokenizer, session) -> list:
    """
    Embed a single query string using the same pipeline as 02_embed.py:
    1. Tokenise with HuggingFace fast tokenizer (pad/truncate to MAX_SEQ_LEN=128)
    2. Run ONNX model → last hidden state (shape: 1, seq_len, 384)
    3. Mean-pool over non-padding tokens
    4. L2-normalise
    Returns a plain Python list[float] of length 384.
    """
    import numpy as np

    encoded = tokenizer.encode_batch([query_text])
    ids  = np.array([e.ids            for e in encoded], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    tids = np.zeros_like(ids)

    out = session.run(
        None,
        {"input_ids": ids, "attention_mask": mask, "token_type_ids": tids}
    )[0]  # shape: (1, seq_len, 384)

    m    = mask[:, :, np.newaxis].astype("float32")
    vec  = (out * m).sum(axis=1) / m.sum(axis=1).clip(min=1e-9)  # (1, 384)
    vec  = vec[0]
    norm = np.linalg.norm(vec)
    vec  = (vec / max(norm, 1e-9)).tolist()
    return vec


def _dot(a: list, b: list) -> float:
    """Dot product of two already-normalised vectors = cosine similarity."""
    return sum(x * y for x, y in zip(a, b))


def _retrieve(query_vec, pids, vecs, gx, pid2idx, top_k=10):
    """
    Run both retrieval tracks for a single query vector.

    Track A — Cosine baseline: rank by cosine_similarity (= dot product,
    since vectors are L2-normalised).

    Track B — Graph-reranked:
      combined = 0.7 × cosine_similarity + 0.3 × norm_pagerank
    norm_pagerank is taken directly from graphx_results.csv
    (already normalised to [0,1] by GraphXJob.scala as
    (PR - PR_min) / (PR_max - PR_min)).
    Isolated passages (absent from GraphX output) receive
    norm_pagerank = FALLBACK_NORM_PR = 0.5 (the midpoint, documented
    explicitly so no passage is silently discarded).

    Returns (cosine_results, graph_results) where each is a list of dicts:
    [{ passage_id, passage_idx, cosine, norm_pr, combined, is_isolated }, ...]
    sorted by respective score descending.
    """
    scored = []
    for pid, vec in zip(pids, vecs):
        cos = _dot(query_vec, vec)
        idx = pid2idx.get(pid, -1)
        if idx in gx:
            norm_pr    = gx[idx]["norm_pagerank"]
            pr_raw     = gx[idx]["pagerank"]
            degree     = gx[idx]["degree"]
            isolated   = False
        else:
            norm_pr    = FALLBACK_NORM_PR
            pr_raw     = None
            degree     = 0
            isolated   = True
        combined = LAMBDA * cos + (1 - LAMBDA) * norm_pr
        scored.append({
            "passage_id":  pid,
            "passage_idx": idx,
            "cosine":      cos,
            "norm_pr":     norm_pr,
            "pr_raw":      pr_raw,
            "degree":      degree,
            "combined":    combined,
            "isolated":    isolated,
        })

    cos_ranked  = sorted(scored, key=lambda x: -x["cosine"])[:top_k]
    grp_ranked  = sorted(scored, key=lambda x: -x["combined"])[:top_k]
    return cos_ranked, grp_ranked


# ── Tab 5 rendering ───────────────────────────────────────────────────────────

with tab5:
    st.header("🔎 Live Retrieval")
    st.caption(
        "Enter any query to compare cosine-similarity retrieval against "
        "PageRank-reranked retrieval in real time. "
        "Passage embeddings and graph metrics are loaded from the project's "
        "pre-computed artifacts — no pipeline rerun required."
    )

    # Method explanation
    with st.expander("ℹ️ How the two ranking methods work"):
        st.markdown(f"""
**Track A — Cosine Baseline**
Query is embedded with *all-MiniLM-L6-v2* (ONNX, CPU).
Each passage is ranked by `cosine_similarity(query, passage)`.
Since all vectors are L2-normalised, this equals their dot product.

**Track B — Graph-Reranked** (λ = {LAMBDA})
```
combined_score = {LAMBDA} × cosine_similarity
               + {1-LAMBDA:.1f} × norm_pagerank
```
`norm_pagerank` is the passage's GraphX PageRank normalised to [0, 1] by
`(PR − PR_min) / (PR_max − PR_min)` (pre-computed by `GraphXJob.scala`).
The 117 passages isolated from the graph (no cosine-similar neighbours above
the 0.30 threshold) receive `norm_pagerank = {FALLBACK_NORM_PR}` (midpoint
fallback) so they are ranked, not dropped.

**Research note**: MS MARCO relevance labels are retrieval-relevance labels,
not hallucination ground truth. This demo does not perform hallucination detection.
        """)

    st.divider()

    # ── Query input ───────────────────────────────────────────────────────────
    query_text = st.text_input(
        "Enter your query:",
        placeholder="e.g. what causes a pimple",
        key="live_query",
    )
    run_btn = st.button("🔍 Run Retrieval", type="primary")

    # ── Guard: empty query ────────────────────────────────────────────────────
    if run_btn and not query_text.strip():
        st.warning("Please enter a non-empty query before running retrieval.")
        run_btn = False

    if run_btn and query_text.strip():

        # Load data (cached after first call)
        pids, vecs  = _load_passage_embeddings()
        passage_txt = _load_passage_text()
        gx          = _load_graph_metrics()
        pid2idx     = _load_pid_to_idx()
        tokenizer, session, err = _load_embedder()

        if err:
            st.error(
                f"Failed to load ONNX embedding model: {err}\n\n"
                "Ensure `onnxruntime` and `tokenizers` are installed "
                "(`pip install onnxruntime tokenizers huggingface_hub`)."
            )
            st.stop()

        # ── Embed query ───────────────────────────────────────────────────────
        with st.spinner("Embedding query…"):
            try:
                q_vec = _embed_query(query_text.strip(), tokenizer, session)
            except Exception as e:
                st.error(f"Embedding error: {e}")
                st.stop()

        st.success(
            f"Query embedded — 384-dim vector, "
            f"L2-norm ≈ {math.sqrt(sum(x*x for x in q_vec)):.4f} (expect 1.000)"
        )

        # ── Retrieve ──────────────────────────────────────────────────────────
        with st.spinner("Ranking 1,053 passages…"):
            cos_results, grp_results = _retrieve(
                q_vec, pids, vecs, gx, pid2idx, top_k=10
            )

        # ── Build display tables ──────────────────────────────────────────────
        def _make_display(ranked, include_graph_cols):
            rows = []
            for rank, r in enumerate(ranked, start=1):
                pid  = r["passage_id"]
                text = passage_txt.get(pid, "—")
                row  = {
                    "Rank":       rank,
                    "Passage ID": pid,
                    "Cosine":     round(r["cosine"], 4),
                    "Passage":    text[:200] + "…" if len(text) > 200 else text,
                }
                if include_graph_cols:
                    row["Norm PR"]  = round(r["norm_pr"], 4)
                    row["Combined"] = round(r["combined"], 4)
                    row["Isolated"] = "⚠️ yes" if r["isolated"] else "no"
                rows.append(row)
            return rows

        cos_display = _make_display(cos_results, False)
        grp_display = _make_display(grp_results, True)

        # Build rank-change data
        cos_pid_rank = {r["passage_id"]: i+1 for i, r in enumerate(cos_results)}
        grp_pid_rank = {r["passage_id"]: i+1 for i, r in enumerate(grp_results)}
        all_pids_in_top10 = set(cos_pid_rank) | set(grp_pid_rank)

        # ── Side-by-side tables ───────────────────────────────────────────────
        st.subheader(f"Results for: *\"{query_text.strip()}\"*")

        col_cos, col_grp = st.columns(2)
        with col_cos:
            st.markdown("**Track A — Cosine Baseline**")
            md_table(cos_display, max_col_width=90)
        with col_grp:
            st.markdown(f"**Track B — Graph-Reranked (λ={LAMBDA})**")
            md_table(grp_display, max_col_width=90)

        # ── Rank-change comparison ────────────────────────────────────────────
        st.divider()
        st.subheader("Rank Changes")

        rank_rows = []
        for pid in sorted(all_pids_in_top10):
            cos_r = cos_pid_rank.get(pid)
            grp_r = grp_pid_rank.get(pid)
            if cos_r and grp_r:
                delta = cos_r - grp_r   # positive = moved up in graph ranking
                arrow = ("⬆️" if delta > 0 else ("⬇️" if delta < 0 else "➡️"))
                status = f"{arrow} {abs(delta):+d}" if delta != 0 else "➡️ no change"
            elif cos_r:
                status = "❌ dropped from graph top-10"
                delta  = None
            else:
                status = "🆕 new in graph top-10"
                delta  = None
            rank_rows.append({
                "Passage ID":    pid,
                "Cosine rank":   cos_r or "—",
                "Graph rank":    grp_r or "—",
                "Change":        status,
                "Passage":       passage_txt.get(pid, "—")[:100] + "…",
            })
        md_table(rank_rows, max_col_width=80)

        # ── Score visualisation ───────────────────────────────────────────────
        st.divider()
        st.subheader("Top-10 Score Comparison")

        # Build score comparison table (cosine vs combined for graph top-10)
        score_rows = [
            {
                "Graph rank": f"#{i+1}",
                "Passage ID": r["passage_id"],
                "Cosine": round(r["cosine"], 4),
                "Combined": round(r["combined"], 4),
                "Δ (combined−cosine)": round(r["combined"] - r["cosine"], 4),
                "Isolated": "⚠️" if r["isolated"] else "",
            }
            for i, r in enumerate(grp_results)
        ]
        md_table(score_rows)
        st.caption(
            "Bars show cosine and combined scores for the graph-reranked top-10. "
            "When combined > cosine the passage has above-average PageRank; "
            "when combined < cosine the passage is isolated or has low centrality."
        )

        # ── Summary explanation ───────────────────────────────────────────────
        st.divider()
        n_moved  = sum(1 for r in rank_rows if isinstance(r["Change"], str)
                       and ("⬆️" in r["Change"] or "⬇️" in r["Change"]))
        n_only_cos = sum(1 for r in rank_rows if r["Graph rank"] == "—")
        n_only_grp = sum(1 for r in rank_rows if r["Cosine rank"] == "—")
        n_isolated = sum(1 for r in grp_results if r["isolated"])

        st.markdown(
            f"**Summary**: Of the {len(all_pids_in_top10)} distinct passages appearing "
            f"in either top-10 list: **{n_moved} changed rank**, "
            f"**{n_only_cos} appeared only in the cosine list**, and "
            f"**{n_only_grp} appeared only in the graph-reranked list**. "
            f"**{n_isolated}** of the graph top-10 passages are isolated (no graph "
            f"edges above threshold) and received the fallback "
            f"`norm_pagerank = {FALLBACK_NORM_PR}`. "
        )
        st.info(
            "**Research integrity**: No ground-truth relevance labels are available "
            "for ad-hoc queries entered here. Neither ranking can be declared "
            "'better' without labelled judgements for this specific query. "
            "This demo shows the mechanical effect of adding PageRank to the score."
        )
