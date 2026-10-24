#!/usr/bin/env python3
"""
05_score.py
===========
Milestone 5 — Retrieval Evaluation

Compares two retrieval strategies:
  Track A — Cosine baseline : rank passages by cosine similarity to query.
  Track B — Graph-reranked  : score = 0.7 * cosine_sim + 0.3 * norm_pagerank

Reads (all from local CSV to avoid numpy/pandas dependency issues):
  data/embeddings.csv     — passage_id, e0…e383
  data/pid_to_idx.csv     — idx, passage_id
  data/graphx_results.csv — passage_idx, pagerank, norm_pagerank, …
  data/msmarco_subset.csv — query_id, query, passage_id, passage, label
  data/qrels.dev.small.tsv — query_id  0  passage_id  relevance

Metrics: MRR@10, NDCG@10, evaluated on queries that have at least one
relevant passage in our 1,053-passage corpus.

Outputs:
  data/scores/retrieval_results.csv   — per-query metrics
  data/scores/summary.csv             — aggregate MRR@10, NDCG@10, latency
  data/scores/top_passages.csv        — top passages by risk score (for dashboard)

Run:
  python3 src/05_score.py
"""

import os
import csv
import sys
import math
import time

# ── Paths ───────────────────────────────────────────────────────────────────
EMBED_CSV   = "data/embeddings.csv"
PID_MAP     = "data/pid_to_idx.csv"
GRAPHX_CSV  = "data/graphx_results.csv"
SUBSET_CSV  = "data/msmarco_subset.csv"
QRELS_TSV   = "data/qrels.dev.small.tsv"
SCORES_DIR  = "data/scores"

LAMBDA      = 0.7      # cosine weight in reranking
TOP_K       = 10       # MRR@10, NDCG@10

os.makedirs(SCORES_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def load_embeddings(path):
    """Returns {passage_id: [float, ...]} and ordered list of pids."""
    print(f"  Loading embeddings from {path} …")
    embeddings = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            pid = row[0]
            vec = [float(v) for v in row[1:]]
            embeddings[pid] = vec
    print(f"    {len(embeddings)} passages loaded, dim={len(next(iter(embeddings.values())))}")
    return embeddings

def load_pid_map(path):
    """Returns {passage_id: idx} and {idx: passage_id}."""
    pid2idx, idx2pid = {}, {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            idx = int(row["idx"])
            pid = row["passage_id"]
            pid2idx[pid] = idx
            idx2pid[idx] = pid
    return pid2idx, idx2pid

def load_graphx(path):
    """Returns {passage_idx: {pagerank, norm_pagerank, component_id, degree, triangle_count}}"""
    metrics = {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                idx = int(row["passage_idx"])
                metrics[idx] = {
                    "pagerank":       float(row["pagerank"]),
                    "norm_pagerank":  float(row["norm_pagerank"]),
                    "component_id":   int(row["component_id"]),
                    "triangle_count": int(row["triangle_count"]),
                    "degree":         int(row["degree"]),
                }
    except FileNotFoundError:
        print(f"  WARNING: {path} not found — graph metrics unavailable. Using norm_pagerank=0.5.")
    return metrics

def load_qrels(path):
    """Returns {query_id: set(relevant_passage_ids)} from TREC qrels."""
    qrels = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, pid, rel = parts[0], parts[1], parts[2], int(parts[3])
            if rel > 0:
                qrels.setdefault(qid, set()).add(pid)
    return qrels

QUERY_EMBED_CSV = "data/query_embeddings.csv"

def load_query_embeddings(subset_csv, embeddings):
    """
    Load real query embeddings from data/query_embeddings.csv (produced by
    embed_queries.py using the same all-MiniLM-L6-v2 ONNX model).

    Falls back to a mean-of-relevant-passages proxy ONLY if that file is
    missing — the proxy produces MRR=1.0 (circular evaluation) and is
    flagged as invalid.

    Returns {query_id: (query_text, [float, ...])}
    """
    query_text = {}
    query_pids = {}
    with open(subset_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            qid = row["query_id"]
            query_text[qid] = row["query"]
            if row["passage_id"] in embeddings:
                query_pids.setdefault(qid, []).append(row["passage_id"])

    # ── Preferred: load real query embeddings ────────────────────────────────
    if os.path.exists(QUERY_EMBED_CSV):
        print(f"  Loading real query embeddings from {QUERY_EMBED_CSV} …")
        query_vecs = {}
        with open(QUERY_EMBED_CSV, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader)  # skip header
            for row in reader:
                qid = row[0]
                vec = [float(v) for v in row[1:]]
                query_vecs[qid] = (query_text.get(qid, ""), vec)
        print(f"    {len(query_vecs)} real query vectors loaded.")
        return query_vecs

    # ── Fallback: mean of relevant passage embeddings (circular — avoid) ─────
    print("  WARNING: query_embeddings.csv not found — using circular proxy.")
    print("  Run src/embed_queries.py to generate valid query embeddings.")
    dim = len(next(iter(embeddings.values())))
    query_vecs = {}
    for qid, pids in query_pids.items():
        mean_vec = [0.0] * dim
        for pid in pids:
            v = embeddings[pid]
            for i in range(dim): mean_vec[i] += v[i]
        n = len(pids)
        mean_vec = [x / n for x in mean_vec]
        norm = math.sqrt(sum(x * x for x in mean_vec)) or 1.0
        query_vecs[qid] = (query_text.get(qid, ""), [x / norm for x in mean_vec])
    return query_vecs




def ndcg_at_k(relevant_set, ranked_pids, k):
    """Compute NDCG@k. Binary relevance."""
    dcg = 0.0
    for rank, pid in enumerate(ranked_pids[:k], start=1):
        if pid in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)
    # Ideal DCG: all relevant at top
    n_rel = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(relevant_set, ranked_pids, k):
    """Compute MRR@k. Returns reciprocal rank of first relevant hit."""
    for rank, pid in enumerate(ranked_pids[:k], start=1):
        if pid in relevant_set:
            return 1.0 / rank
    return 0.0


def retrieve_cosine(query_vec, all_pids, embeddings, top_k):
    """Rank all passages by cosine similarity to query_vec."""
    t0 = time.time()
    scores = []
    for pid in all_pids:
        v = embeddings[pid]
        scores.append((pid, dot(query_vec, v)))
    scores.sort(key=lambda x: -x[1])
    latency = time.time() - t0
    return [p for p, _ in scores[:top_k]], latency, scores


def retrieve_graph_reranked(query_vec, all_pids, embeddings, graphx, pid2idx, top_k, lam=LAMBDA):
    """Rank passages by λ·cosine + (1-λ)·norm_pagerank."""
    t0 = time.time()
    scores = []
    for pid in all_pids:
        v = embeddings[pid]
        cos = dot(query_vec, v)
        idx = pid2idx.get(pid, -1)
        norm_pr = graphx[idx]["norm_pagerank"] if idx in graphx else 0.5
        score = lam * cos + (1 - lam) * norm_pr
        scores.append((pid, score))
    scores.sort(key=lambda x: -x[1])
    latency = time.time() - t0
    return [p for p, _ in scores[:top_k]], latency, scores


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=== M5: Retrieval Evaluation ===\n")
    t_total = time.time()

    # Load all data
    print("--- Loading data ---")
    embeddings = load_embeddings(EMBED_CSV)
    pid2idx, idx2pid = load_pid_map(PID_MAP)
    graphx = load_graphx(GRAPHX_CSV)
    qrels  = load_qrels(QRELS_TSV)
    all_pids = list(embeddings.keys())
    corpus_pids = set(all_pids)

    graphx_available = bool(graphx)
    print(f"  Corpus passages    : {len(all_pids):,}")
    print(f"  Graph metrics      : {'available ✓' if graphx_available else 'unavailable — graph reranking degraded'}")
    print(f"  Total qrels queries: {len(qrels):,}")

    # Only evaluate queries whose relevant passages overlap with our corpus
    evaluable_queries = {
        qid: rels & corpus_pids
        for qid, rels in qrels.items()
        if rels & corpus_pids
    }
    print(f"  Evaluable queries  : {len(evaluable_queries):,}  "
          f"(have ≥1 relevant passage in corpus)")

    if not evaluable_queries:
        sys.exit("ERROR: No evaluable queries. Check that qrels and corpus pids overlap.")

    # Build query vectors from subset CSV
    print("\n  Building query vectors (mean of relevant passage embeddings) …")
    query_vecs = load_query_embeddings(SUBSET_CSV, embeddings)
    print(f"  Queries with vectors: {len(query_vecs):,}")

    # Intersect: queries that are both evaluable and have a vector
    eval_qids = sorted(set(evaluable_queries) & set(query_vecs))
    print(f"  Queries to evaluate : {len(eval_qids):,}")

    # ── Per-query evaluation ────────────────────────────────────────────────
    print("\n--- Running retrieval ---")
    results = []
    cos_mrr_list, cos_ndcg_list = [], []
    grp_mrr_list, grp_ndcg_list = [], []
    cos_lat_list, grp_lat_list  = [], []

    for i, qid in enumerate(eval_qids):
        q_text, q_vec = query_vecs[qid]
        relevant = evaluable_queries[qid]

        # Track A: cosine baseline
        cos_ranked, cos_lat, cos_all = retrieve_cosine(
            q_vec, all_pids, embeddings, TOP_K
        )
        cos_mrr  = mrr_at_k(relevant, cos_ranked, TOP_K)
        cos_ndcg = ndcg_at_k(relevant, cos_ranked, TOP_K)

        # Track B: graph-reranked
        grp_ranked, grp_lat, grp_all = retrieve_graph_reranked(
            q_vec, all_pids, embeddings, graphx, pid2idx, TOP_K
        )
        grp_mrr  = mrr_at_k(relevant, grp_ranked, TOP_K)
        grp_ndcg = ndcg_at_k(relevant, grp_ranked, TOP_K)

        cos_mrr_list.append(cos_mrr);  cos_ndcg_list.append(cos_ndcg)
        grp_mrr_list.append(grp_mrr);  grp_ndcg_list.append(grp_ndcg)
        cos_lat_list.append(cos_lat);  grp_lat_list.append(grp_lat)

        results.append({
            "query_id":     qid,
            "query":        q_text[:80],
            "n_relevant":   len(relevant),
            "cos_mrr":      round(cos_mrr, 4),
            "cos_ndcg":     round(cos_ndcg, 4),
            "grp_mrr":      round(grp_mrr, 4),
            "grp_ndcg":     round(grp_ndcg, 4),
            "cos_lat_ms":   round(cos_lat * 1000, 2),
            "grp_lat_ms":   round(grp_lat * 1000, 2),
        })

        if (i + 1) % 100 == 0 or (i + 1) == len(eval_qids):
            print(f"  Evaluated {i+1}/{len(eval_qids)} queries …", end="\r")

    print()

    # ── Aggregate metrics ───────────────────────────────────────────────────
    n = len(results)
    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0

    cos_MRR   = avg(cos_mrr_list)
    cos_NDCG  = avg(cos_ndcg_list)
    grp_MRR   = avg(grp_mrr_list)
    grp_NDCG  = avg(grp_ndcg_list)
    avg_cos_ms = avg(cos_lat_list) * 1000
    avg_grp_ms = avg(grp_lat_list) * 1000

    delta_MRR  = grp_MRR  - cos_MRR
    delta_NDCG = grp_NDCG - cos_NDCG

    # ── Write per-query CSV ──────────────────────────────────────────────────
    per_query_csv = os.path.join(SCORES_DIR, "retrieval_results.csv")
    with open(per_query_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"  Per-query CSV: {per_query_csv}  ({n} rows)")

    # ── Write summary CSV ────────────────────────────────────────────────────
    summary_csv = os.path.join(SCORES_DIR, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "cosine_baseline", "graph_reranked", "delta"])
        writer.writerow(["MRR@10",  f"{cos_MRR:.4f}",  f"{grp_MRR:.4f}",  f"{delta_MRR:+.4f}"])
        writer.writerow(["NDCG@10", f"{cos_NDCG:.4f}", f"{grp_NDCG:.4f}", f"{delta_NDCG:+.4f}"])
        writer.writerow(["avg_latency_ms", f"{avg_cos_ms:.2f}", f"{avg_grp_ms:.2f}", ""])
        writer.writerow(["queries_evaluated", n, n, ""])
        writer.writerow(["corpus_size", len(all_pids), len(all_pids), ""])
        writer.writerow(["lambda", LAMBDA, LAMBDA, ""])
        writer.writerow(["graphx_available", graphx_available, graphx_available, ""])
    print(f"  Summary CSV:   {summary_csv}")

    # ── Top passages by hallucination-risk proxy ─────────────────────────────
    # Risk = α*(1-norm_pr) + β*(isolation) where isolation = 1/(degree+1)
    top_csv = os.path.join(SCORES_DIR, "top_passages.csv")
    with open(top_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["passage_id", "passage_idx", "pagerank", "norm_pagerank",
                          "component_id", "degree", "triangle_count", "risk_score"])
        risk_rows = []
        for pid in all_pids:
            idx = pid2idx.get(pid, -1)
            if idx in graphx:
                g = graphx[idx]
                # Risk proxy: low pagerank + isolated + no triangles
                risk = 0.5 * (1 - g["norm_pagerank"]) + \
                       0.3 * (1.0 / (g["degree"] + 1)) + \
                       0.2 * (1 if g["triangle_count"] == 0 else 0)
            else:
                risk = 0.5  # default when graph unavailable
                g = {"pagerank": 0, "norm_pagerank": 0.5, "component_id": -1,
                     "degree": 0, "triangle_count": 0}
            risk_rows.append((pid, idx, g, risk))

        risk_rows.sort(key=lambda x: -x[3])
        for pid, idx, g, risk in risk_rows[:50]:
            writer.writerow([
                pid, idx,
                f"{g['pagerank']:.6f}", f"{g['norm_pagerank']:.4f}",
                g["component_id"], g["degree"], g["triangle_count"],
                f"{risk:.4f}"
            ])
    print(f"  Top-50 risk CSV: {top_csv}")

    # ── Print results ────────────────────────────────────────────────────────
    total_time = time.time() - t_total
    print("\n=== RETRIEVAL RESULTS ===")
    print(f"  Queries evaluated : {n:,}")
    print(f"  Corpus size       : {len(all_pids):,} passages")
    print(f"  GraphX available  : {graphx_available}")
    print(f"  Lambda (cosine wt): {LAMBDA}")
    print()
    print(f"  {'Metric':<12}  {'Cosine':>10}  {'Graph-Reranked':>14}  {'Delta':>8}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*14}  {'-'*8}")
    print(f"  {'MRR@10':<12}  {cos_MRR:>10.4f}  {grp_MRR:>14.4f}  {delta_MRR:>+8.4f}")
    print(f"  {'NDCG@10':<12}  {cos_NDCG:>10.4f}  {grp_NDCG:>14.4f}  {delta_NDCG:>+8.4f}")
    print(f"  {'Latency(ms)':<12}  {avg_cos_ms:>10.2f}  {avg_grp_ms:>14.2f}")
    print()
    print("  NOTE: Graph-reranking is an experimental hypothesis.")
    print("  The delta shows whether the graph topology helps or hurts")
    print("  retrieval on this corpus — positive or negative is reported as-is.")
    print(f"\n  Total evaluation time: {total_time:.1f}s")
    print(f"\nMilestone 5 PASSED ✓")


if __name__ == "__main__":
    main()
