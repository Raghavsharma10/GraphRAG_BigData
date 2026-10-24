"""
tests/test_pipeline.py
======================
Automated tests for M3 (edges), M4 (GraphX), and M5 (scores).
All tests read from local CSV files — no PySpark or HDFS needed.
"""

import csv
import os
import pytest

EDGES_CSV   = "data/edges.csv"
GRAPHX_CSV  = "data/graphx_results.csv"
SUMMARY_CSV = "data/scores/summary.csv"
RESULTS_CSV = "data/scores/retrieval_results.csv"
EMBED_CSV   = "data/embeddings.csv"
PID_MAP_CSV = "data/pid_to_idx.csv"


# ── Fixtures ──────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ── M3: Edge tests ────────────────────────────────────────────────────────────

def test_edges_file_exists():
    assert os.path.exists(EDGES_CSV), f"{EDGES_CSV} not found"


def test_edges_nonzero():
    rows = load_csv(EDGES_CSV)
    assert len(rows) > 0, "Edge file is empty"


def test_no_self_edges():
    rows = load_csv(EDGES_CSV)
    self_edges = [r for r in rows if int(r["src_id"]) == int(r["dst_id"])]
    assert len(self_edges) == 0, f"Found {len(self_edges)} self-edges"


def test_edges_undirected_no_duplicates():
    """No pair (a, b) should appear more than once (canonical: a < b)."""
    rows = load_csv(EDGES_CSV)
    pairs = set()
    duplicates = []
    for r in rows:
        s, d = int(r["src_id"]), int(r["dst_id"])
        pair = (min(s, d), max(s, d))
        if pair in pairs:
            duplicates.append(pair)
        pairs.add(pair)
    assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate edge pairs"


def test_edge_similarity_range():
    """All similarity values must be in [0.30, 1.01]."""
    rows = load_csv(EDGES_CSV)
    bad = [r for r in rows if not (0.29 <= float(r["similarity"]) <= 1.01)]
    assert len(bad) == 0, f"{len(bad)} edges have similarity out of [0.30, 1.01]"


# ── M4: GraphX tests ──────────────────────────────────────────────────────────

def test_graphx_file_exists():
    assert os.path.exists(GRAPHX_CSV), f"{GRAPHX_CSV} not found"


def test_graphx_nonzero():
    rows = load_csv(GRAPHX_CSV)
    assert len(rows) > 0, "GraphX results file is empty"


def test_graphx_pagerank_positive():
    rows = load_csv(GRAPHX_CSV)
    bad = [r for r in rows if float(r["pagerank"]) <= 0]
    assert len(bad) == 0, f"{len(bad)} passages have non-positive PageRank"


def test_graphx_norm_pagerank_range():
    rows = load_csv(GRAPHX_CSV)
    bad = [r for r in rows if not (-0.01 <= float(r["norm_pagerank"]) <= 1.01)]
    assert len(bad) == 0, f"{len(bad)} rows have norm_pagerank outside [0, 1]"


def test_graphx_degree_nonnegative():
    rows = load_csv(GRAPHX_CSV)
    bad = [r for r in rows if int(r["degree"]) < 0]
    assert len(bad) == 0, f"{len(bad)} rows have negative degree"


# ── M2: Embedding tests ───────────────────────────────────────────────────────

def test_embedding_file_exists():
    assert os.path.exists(EMBED_CSV), f"{EMBED_CSV} not found"


def test_embedding_row_count():
    rows = load_csv(EMBED_CSV)
    assert len(rows) == 1053, f"Expected 1053 embedding rows, got {len(rows)}"


def test_embedding_dimension():
    """Each row should have 384 embedding columns (e0…e383)."""
    with open(EMBED_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        # header = [passage_id, e0, e1, ..., e383]
        dim = len(header) - 1
    assert dim == 384, f"Expected dim=384, got {dim}"


def test_embedding_normalised():
    """Sample first 10 rows: L2 norm should be close to 1.0."""
    import math
    rows = load_csv(EMBED_CSV)
    for row in rows[:10]:
        vec = [float(v) for k, v in row.items() if k != "passage_id"]
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-3, \
            f"Embedding for {row['passage_id']} not normalised: norm={norm:.6f}"


# ── M5: Score tests ───────────────────────────────────────────────────────────

def test_summary_file_exists():
    assert os.path.exists(SUMMARY_CSV), f"{SUMMARY_CSV} not found"


def test_results_file_exists():
    assert os.path.exists(RESULTS_CSV), f"{RESULTS_CSV} not found"


def test_results_query_count():
    rows = load_csv(RESULTS_CSV)
    assert len(rows) == 1000, f"Expected 1000 result rows, got {len(rows)}"


def test_mrr_in_range():
    rows = load_csv(RESULTS_CSV)
    bad = [r for r in rows if not (0.0 <= float(r["cos_mrr"]) <= 1.0 + 1e-6)]
    assert len(bad) == 0, f"{len(bad)} rows have cosine MRR outside [0, 1]"


def test_ndcg_in_range():
    rows = load_csv(RESULTS_CSV)
    bad = [r for r in rows if not (0.0 <= float(r["cos_ndcg"]) <= 1.0 + 1e-6)]
    assert len(bad) == 0, f"{len(bad)} rows have cosine NDCG outside [0, 1]"


def test_latency_positive():
    rows = load_csv(RESULTS_CSV)
    bad = [r for r in rows if float(r["cos_lat_ms"]) <= 0]
    assert len(bad) == 0, f"{len(bad)} rows have non-positive latency"
