#!/usr/bin/env python3
"""
03_edges.py
===========
Milestone 3 — k-NN Graph Edge Construction

Reads the embedding Parquet from HDFS, builds a FAISS IndexFlatIP
(inner product on L2-normalised vectors = cosine similarity) to find
the k nearest neighbours of every passage, then writes a filtered edge
list back to HDFS via PySpark.

Edge filtering rules
--------------------
- No self-edges  (src_id == dst_id)
- Undirected dedup: keep only src_id < dst_id
- Threshold: similarity >= SIM_THRESHOLD (default 0.30)

Output HDFS path
----------------
  hdfs://localhost:9000/graphrag/edges/
  Schema: src_id LONG, dst_id LONG, similarity FLOAT

The src_id / dst_id values are integer row indices (0 … N-1) that
correspond to the ordered passage list collected from HDFS.
A mapping file  data/pid_to_idx.csv  is written locally so downstream
scripts can join back to passage_id strings.

Run
---
  export PYTHONPATH=/opt/homebrew/Cellar/apache-spark/4.2.0/libexec/python:\\
/opt/homebrew/Cellar/apache-spark/4.2.0/libexec/python/lib/py4j-0.10.9.9-src.zip

  PYSPARK_PYTHON=/opt/anaconda3/bin/python3 \\
  JAVA_HOME=/opt/homebrew/Cellar/openjdk@17/17.0.20.1/libexec/openjdk.jdk/Contents/Home \\
  python3 src/03_edges.py
"""

import os
import sys
import time
import csv

try:
    import numpy as np
except ImportError:
    sys.exit("numpy not found.")

try:
    import faiss
except ImportError:
    sys.exit("faiss not found. Install faiss-cpu.")

try:
    from pyspark.sql import SparkSession, Row
    from pyspark.sql.types import (
        FloatType, LongType, StructField, StructType,
    )
except ImportError as e:
    sys.exit(f"Cannot import PySpark: {e}")

# ── Configuration ───────────────────────────────────────────────────────────
HDFS_EMBED    = "hdfs://localhost:9000/graphrag/embeddings"
HDFS_EDGES    = "hdfs://localhost:9000/graphrag/edges"
K_NEIGHBOURS  = 11          # query k=11; after removing self-edge → ≤10 edges/node
SIM_THRESHOLD = 0.30        # cosine similarity cutoff
EMBED_DIM     = 384
PID_MAP_CSV   = "data/pid_to_idx.csv"


def build_faiss_index(matrix: np.ndarray) -> faiss.IndexFlatIP:
    """Build FAISS inner-product index for L2-normalised vectors."""
    assert matrix.dtype == np.float32, "Matrix must be float32"
    index = faiss.IndexFlatIP(matrix.shape[1])
    faiss.normalize_L2(matrix)   # re-normalise to be safe
    index.add(matrix)
    return index


def main() -> None:
    t_start = time.time()

    # ── Spark session ─────────────────────────────────────────────────────────
    spark = (
        SparkSession.builder
        .appName("GraphRAG-Edges")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark {spark.version}")

    # ── Load embeddings from HDFS ────────────────────────────────────────────
    print(f"\n=== Loading embeddings from {HDFS_EMBED} ===")
    embed_df = spark.read.parquet(HDFS_EMBED).orderBy("passage_id")
    rows = embed_df.collect()
    pids = [r.passage_id for r in rows]
    vecs = np.array([r.embedding for r in rows], dtype=np.float32)

    n = len(pids)
    print(f"  Passages loaded : {n:,}")
    print(f"  Matrix shape    : {vecs.shape}")
    assert vecs.shape == (n, EMBED_DIM), f"Unexpected shape: {vecs.shape}"

    # ── Write pid → idx mapping ───────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    with open(PID_MAP_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["idx", "passage_id"])
        for idx, pid in enumerate(pids):
            writer.writerow([idx, pid])
    print(f"  Wrote pid→idx map: {PID_MAP_CSV} ({n} rows)")

    # ── Build FAISS index ────────────────────────────────────────────────────
    print(f"\n=== Building FAISS IndexFlatIP (k={K_NEIGHBOURS}) ===")
    t_idx = time.time()
    index = build_faiss_index(vecs.copy())
    print(f"  Index built in {time.time() - t_idx:.2f}s  | ntotal={index.ntotal:,}")

    # ── k-NN search ──────────────────────────────────────────────────────────
    print(f"\n=== Running k-NN search ===")
    t_search = time.time()
    # Re-normalise query vectors (same as index vectors)
    query = vecs.copy()
    faiss.normalize_L2(query)
    sims, indices = index.search(query, K_NEIGHBOURS)
    print(f"  Search done in {time.time() - t_search:.2f}s")
    print(f"  sims shape: {sims.shape}  indices shape: {indices.shape}")

    # ── Filter edges ──────────────────────────────────────────────────────────
    print(f"\n=== Filtering edges (threshold={SIM_THRESHOLD}, no self, undirected) ===")
    edge_set: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()

    for src_idx in range(n):
        for j in range(K_NEIGHBOURS):
            dst_idx = int(indices[src_idx, j])
            sim     = float(sims[src_idx, j])

            if dst_idx < 0:                # FAISS sentinel for missing neighbour
                continue
            if dst_idx == src_idx:         # self-edge
                continue
            if sim < SIM_THRESHOLD:        # below threshold
                continue

            # Undirected dedup: canonical pair (min, max)
            pair = (min(src_idx, dst_idx), max(src_idx, dst_idx))
            if pair in seen:
                continue
            seen.add(pair)
            edge_set.append((pair[0], pair[1], sim))

    n_edges = len(edge_set)
    print(f"  Edges after filtering : {n_edges:,}")
    if n_edges == 0:
        spark.stop()
        sys.exit("ERROR: Zero edges produced. Lower SIM_THRESHOLD or increase K.")

    # Verify no self-edges
    self_edges = [(s, d) for s, d, _ in edge_set if s == d]
    print(f"  Self-edges (must be 0): {len(self_edges)}")
    assert len(self_edges) == 0, "Self-edges found — bug in filtering!"

    # ── Write edges to HDFS via PySpark ─────────────────────────────────────
    print(f"\n=== Writing edge list to {HDFS_EDGES} ===")
    EDGE_SCHEMA = StructType([
        StructField("src_id",     LongType(),  nullable=False),
        StructField("dst_id",     LongType(),  nullable=False),
        StructField("similarity", FloatType(), nullable=False),
    ])
    edge_rows = [Row(src_id=s, dst_id=d, similarity=w) for s, d, w in edge_set]
    edge_df = spark.createDataFrame(edge_rows, schema=EDGE_SCHEMA)
    edge_df.write.mode("overwrite").parquet(HDFS_EDGES)
    print("  Write complete.")

    # ── Verify from HDFS ─────────────────────────────────────────────────────
    print(f"\n=== Verifying {HDFS_EDGES} ===")
    result_df = spark.read.parquet(HDFS_EDGES)
    result_df.printSchema()
    n_hdfs = result_df.count()
    print(f"  Edges in HDFS : {n_hdfs:,}")

    sample = result_df.orderBy("similarity", ascending=False).limit(5).collect()
    print("  Top-5 edges by similarity:")
    for r in sample:
        print(f"    src={r.src_id:4d}  dst={r.dst_id:4d}  sim={r.similarity:.4f}")

    # ── Local CSV backup (for dashboard and debugging) ────────────────────────
    edge_csv = "data/edges.csv"
    with open(edge_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["src_id", "dst_id", "similarity"])
        for s, d, w in edge_set:
            writer.writerow([s, d, f"{w:.6f}"])
    print(f"\n  Local backup: {edge_csv}  ({n_edges} rows, "
          f"{os.path.getsize(edge_csv)//1024} KB)")

    # ── Summary ───────────────────────────────────────────────────────────────
    t_total = time.time() - t_start
    match = (n_hdfs == n_edges)
    avg_degree = (2 * n_edges) / n if n > 0 else 0
    print("\n=== EDGE SUMMARY ===")
    print(f"  Passages        : {n:,}")
    print(f"  Edges (filtered): {n_edges:,}")
    print(f"  Avg degree      : {avg_degree:.2f}")
    print(f"  HDFS output     : {HDFS_EDGES}")
    print(f"  Local backup    : {edge_csv}")
    print(f"  Total time      : {t_total:.1f}s")
    print(f"\nMilestone 3 {'PASSED ✓' if match else 'FAILED ✗'}")

    spark.stop()


if __name__ == "__main__":
    main()
