# GraphRAG-Scale: Graph-Augmented Retrieval on MS MARCO

A big-data pipeline that builds a knowledge graph from MS MARCO passages, computes graph-topology metrics with Apache Spark GraphX, and evaluates whether PageRank-based re-ranking improves retrieval over a cosine-similarity baseline. The pipeline covers the full stack: HDFS ingestion via PySpark, ONNX-based dense embeddings, FAISS k-NN graph construction, native Scala GraphX analytics, MRR@10 / NDCG@10 evaluation, and a Streamlit dashboard — all running locally on a single MacBook M1.

---

## Architecture

```
MS MARCO qrels (7 K queries)
         │
         ▼
 00_prepare_data.py ─── HTTP streaming ─── HuggingFace parquet shards
         │
         ▼  data/msmarco_subset.csv  (1,053 passages)
         │
         ▼
 01_ingest.py  (PySpark)
         │
         ▼  HDFS: /graphrag/raw/verified/  (Parquet, 1,053 rows)
         │
         ▼
 02_embed.py  (ONNX Runtime + PySpark)
         │   all-MiniLM-L6-v2 · 384-dim · L2-normalised
         ▼  HDFS: /graphrag/embeddings/  (Parquet, 1,053 × 384)
         │
         ▼
 03_edges.py  (FAISS + PySpark)
         │   k=11, cosine similarity ≥ 0.30, undirected dedup
         ▼  HDFS: /graphrag/edges/  (2,291 edges)
         │
         ▼
 GraphXJob.scala  (native Spark GraphX, spark-shell -i)
         │   PageRank · ConnectedComponents · TriangleCount · Degree
         ▼  HDFS: /graphrag/graphx_out/  (936 vertices with metrics)
         │
         ▼
 05_score.py  (pure Python)
         │   Cosine baseline  vs.  0.7·cosine + 0.3·norm_pagerank
         ▼  data/scores/  (MRR@10, NDCG@10, top-50 risk passages)
         │
         ▼
 dashboard.py  (Streamlit) ── reads local CSVs only
```

---

## Prerequisites

| Component | Version | Path |
|-----------|---------|------|
| macOS (Apple M1, 8 GB) | — | — |
| Python (Anaconda) | 3.12.4 | `/opt/anaconda3/bin/python3` |
| Java (Spark / PySpark) | 17.0.20.1 | `/opt/homebrew/Cellar/openjdk@17/17.0.20.1/libexec/openjdk.jdk/Contents/Home` |
| Java (Hadoop / HDFS) | 21.0.12.1 | `/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home` |
| Apache Hadoop | 3.3.6 | `/Users/raghav/hadoop` |
| Apache Spark | 4.2.0 | `/opt/homebrew/Cellar/apache-spark/4.2.0/libexec` |
| GraphX JAR | `spark-graphx_2.13-4.2.0.jar` | bundled with Spark |
| ONNX Runtime | 1.22.0 | pip |
| faiss-cpu | 1.11.0 | pip |
| Streamlit | 1.32.0 | pip |

> **Note:** `sentence_transformers`, `pandas`, and standard `pyarrow` column APIs are **not** used — see [Known Issues](#known-issues-and-workarounds).

---

## Setup

1. **Start HDFS**

   ```bash
   JAVA_HOME=/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home \
     /Users/raghav/hadoop/sbin/start-dfs.sh
   ```

2. **Set the PySpark environment** (required for M1–M3; add to your shell profile or run before each session)

   ```bash
   export PYTHONPATH=/opt/homebrew/Cellar/apache-spark/4.2.0/libexec/python:/opt/homebrew/Cellar/apache-spark/4.2.0/libexec/python/lib/py4j-0.10.9.9-src.zip
   export PYSPARK_PYTHON=/opt/anaconda3/bin/python3
   export JAVA_HOME=/opt/homebrew/Cellar/openjdk@17/17.0.20.1/libexec/openjdk.jdk/Contents/Home
   ```

3. **Install Python dependencies**

   ```bash
   pip install onnxruntime faiss-cpu streamlit huggingface_hub tokenizers pytest
   ```

4. **Prepare data** (one-time; downloads MS MARCO subset via HTTP streaming)

   ```bash
   python3 src/00_prepare_data.py
   ```

   Produces `data/msmarco_subset.csv` (1,053 passage–query pairs).

---

## Run the Full Pipeline

Run steps in order. Each step's output is the next step's input.

```bash
# M1 — Ingest passages into HDFS (Parquet)
python3 src/01_ingest.py

# M2 — Compute passage embeddings via ONNX Runtime
#       Downloads ~90 MB all-MiniLM-L6-v2 ONNX model on first run
python3 src/02_embed.py

# Embed queries (required before M5)
python3 src/embed_queries.py

# M3 — Build k-NN graph with FAISS; write edges to HDFS
python3 src/03_edges.py

# M4 — Spark GraphX analytics (PageRank, ConnectedComponents, etc.)
#       Uses Java 17 and native spark-shell
JAVA_HOME=/opt/homebrew/Cellar/openjdk@17/17.0.20.1/libexec/openjdk.jdk/Contents/Home \
  /opt/homebrew/Cellar/apache-spark/4.2.0/libexec/bin/spark-shell \
  --driver-memory 4g \
  --conf spark.hadoop.fs.defaultFS=hdfs://localhost:9000 \
  --conf spark.ui.enabled=false \
  -i src/GraphXJob.scala

# M5 — Retrieval evaluation (MRR@10, NDCG@10)
python3 src/05_score.py

# M6 — Launch dashboard
streamlit run src/dashboard.py
```

---

## Output Files Reference

| File | Description |
|------|-------------|
| `data/msmarco_subset.csv` | 1,053 passage–query pairs from MS MARCO |
| `data/embeddings.csv` | 1,053 × 384 L2-normalised passage embeddings |
| `data/query_embeddings.csv` | 1,000 × 384 query embeddings |
| `data/edges.csv` | 2,291 undirected k-NN edges (cosine ≥ 0.30) |
| `data/graphx_results.csv` | 936-vertex table: PageRank, component ID, triangle count, degree |
| `data/scores/summary.csv` | MRR@10 / NDCG@10 / latency for both methods |
| `data/scores/retrieval_results.csv` | Per-query retrieval results |
| `data/scores/top_passages.csv` | Top-50 passages by composite risk score |

HDFS paths mirror the above under `hdfs://localhost:9000/graphrag/`.

---

## Dashboard

```bash
streamlit run src/dashboard.py
```

The dashboard reads only **local CSV files** — no HDFS or Spark required after the pipeline has run.

### Tabs

| # | Tab | Description |
|---|-----|-------------|
| 1 | 📋 Pipeline Summary | Milestone status, dataset stats, corpus / graph / retrieval metrics |
| 2 | 🕸️ Graph Metrics | PageRank histogram, degree distribution, component sizes |
| 3 | 📊 Retrieval Comparison | MRR@10 / NDCG@10 comparison table, per-query scatter |
| 4 | ⚠️ Risk Passages | Top-50 passages ranked by composite graph risk proxy |
| 5 | 🔎 Live Retrieval | **Interactive query comparison** — enter any free-text query and compare cosine-similarity retrieval against PageRank-reranked retrieval in real time |

### Live Retrieval tab (Tab 5)

- Enter any query in the text box and press **Run Retrieval**.
- The query is embedded on the fly using the same ONNX *all-MiniLM-L6-v2* model used in the offline pipeline (no re-download after first run).
- **Track A — Cosine Baseline**: passages ranked by `cosine_similarity(query, passage)`.
- **Track B — Graph-Reranked**: passages ranked by `0.7 × cosine + 0.3 × norm_pagerank`.
- `norm_pagerank` is pre-computed by `GraphXJob.scala` as `(PR − PR_min) / (PR_max − PR_min)` and stored in `data/graphx_results.csv`.
- The 117 isolated passages (no edge above the 0.30 threshold) receive `norm_pagerank = 0.5` (midpoint fallback).
- Results are shown side-by-side with a rank-change table and a bar chart comparing cosine and combined scores.

> **Research integrity note**: No ground-truth relevance labels exist for ad-hoc queries entered in this tab. Neither ranking can be declared "better" without human judgements. The tab demonstrates the mechanical effect of adding PageRank to the retrieval score.

---

## Known Issues and Workarounds

| Issue | Workaround |
|-------|-----------|
| `sentence_transformers` fails at import — sklearn → scipy → numpy 1.x binary incompatibility with numpy 2.2.6 | Use **ONNX Runtime** + HuggingFace `tokenizers` directly (`src/02_embed.py`, `src/embed_queries.py`) |
| `pandas` fails with numpy 2.2.6 | All scripts use Python `csv` stdlib only — no pandas anywhere in the codebase |
| `pyarrow` columns API (`.column()`, `.to_pandas()`) fails with numpy 2.2.6 | Use `pq.ParquetFile().iter_batches()` or let the Spark JVM read Parquet files directly |

---

## Test Results

All **33 / 33** tests pass.

```bash
python3 -m pytest tests/ -v
```

| Test file | Tests | Status |
|-----------|-------|--------|
| `tests/test_dataset.py` | 13 | ✅ PASS |
| `tests/test_pipeline.py` | 20 | ✅ PASS |

---

## Results Summary

Evaluation over 1,000 MS MARCO queries against 1,053 passages.

| Metric | Cosine Baseline | Graph Re-ranked | Δ |
|--------|:--------------:|:---------------:|:-:|
| MRR@10 | **0.9874** | 0.9635 | −0.0239 |
| NDCG@10 | **0.9899** | 0.9715 | −0.0184 |
| Latency (ms) | 16.61 | 16.66 | +0.05 |

Re-ranking formula: `0.7 × cosine_score + 0.3 × normalised_PageRank`

> **Honest negative result.** Graph topology (PageRank) does not improve ranking on this corpus. High baseline scores (MRR ≈ 0.99) occur because the evaluation corpus is exactly the set of passages referenced in the qrels — retrieval is nearly deterministic in this closed setting. The meaningful comparison is the **delta between methods**, not the absolute values.

---

## Project Limitations

1. **Closed corpus.** The 1,053 passages are the exact passages referenced by the 1,000 query relevance judgements. This inflates absolute MRR/NDCG scores and reduces the discriminative power of graph re-ranking.
2. **Small scale.** The dataset fits comfortably in memory on a single laptop. HDFS and Spark add infrastructure overhead without the distributional benefit they would provide at true big-data scale.
3. **Simple re-ranking formula.** The λ = 0.7 / 0.3 split between cosine and PageRank was chosen manually. A learned fusion weight or more sophisticated graph signal (e.g. personalised PageRank per query) would be more principled.
4. **117 isolated passages.** Passages with no neighbour above the 0.30 cosine threshold receive no PageRank signal and are at a structural disadvantage in graph re-ranking.
5. **Single machine.** GraphX runs in local mode with `--driver-memory 4g`; true distributed execution would require a multi-node cluster.
6. **Static graph.** The passage graph is built once offline. A production system would need incremental graph updates as the corpus grows.
