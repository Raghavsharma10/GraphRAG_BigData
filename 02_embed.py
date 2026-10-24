#!/usr/bin/env python3
"""
02_embed.py
===========
Milestone 2 — Embedding Generation

Reads unique passages from the HDFS Parquet written by 01_ingest.py,
computes 384-dim sentence embeddings using all-MiniLM-L6-v2 (ONNX),
and writes the result as Parquet back to HDFS via PySpark.

Strategy
--------
- Uses ONNX Runtime + HuggingFace tokenizers (pure Rust) for inference.
  This avoids importing sentence_transformers/sklearn/scipy which all fail
  with the numpy 2.x vs numpy 1.x binary incompatibility in this environment.
- Downloads the ONNX model (~23 MB) once via huggingface_hub if not cached.
- Computes embeddings in the driver (1,053 passages x 384 dims ≈ 1.6 MB).
- Creates a PySpark DataFrame and writes to HDFS as Parquet — genuine
  Spark integration preserved.

Output HDFS path
----------------
  hdfs://localhost:9000/graphrag/embeddings/
  Schema: passage_id STRING, embedding ARRAY<FLOAT>

Run
---
  export PYTHONPATH=/opt/homebrew/Cellar/apache-spark/4.2.0/libexec/python:\\
/opt/homebrew/Cellar/apache-spark/4.2.0/libexec/python/lib/py4j-0.10.9.9-src.zip

  PYSPARK_PYTHON=/opt/anaconda3/bin/python3 \\
  JAVA_HOME=/opt/homebrew/Cellar/openjdk@17/17.0.20.1/libexec/openjdk.jdk/Contents/Home \\
  python3 src/02_embed.py
"""

import os
import sys
import time

# ── Dependencies (no sklearn/scipy/sentence_transformers) ───────────────────
try:
    import numpy as np
except ImportError:
    sys.exit("numpy not found. Install numpy.")

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("onnxruntime not found. Install onnxruntime.")

try:
    from tokenizers import Tokenizer
except ImportError:
    sys.exit("tokenizers not found. Install tokenizers.")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("huggingface_hub not found. Install huggingface_hub.")

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import ArrayType, FloatType, StringType, StructField, StructType
    from pyspark.sql import Row
except ImportError as e:
    sys.exit(f"Cannot import PySpark: {e}")

# ── Configuration ───────────────────────────────────────────────────────────
HDFS_INGEST   = "hdfs://localhost:9000/graphrag/raw/verified"
HDFS_EMBED    = "hdfs://localhost:9000/graphrag/embeddings"
MODEL_REPO    = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM     = 384
BATCH_SIZE    = 64
MAX_SEQ_LEN   = 128


# ── ONNX Embedder (no sklearn/scipy) ────────────────────────────────────────

class MiniLMEmbedder:
    """
    Loads all-MiniLM-L6-v2 ONNX model and fast tokenizer from HuggingFace Hub.
    Computes mean-pooled, L2-normalised 384-dim embeddings on CPU.
    """

    def __init__(self):
        print(f"  Downloading/loading ONNX model from {MODEL_REPO} …")
        # Download ONNX model weights and tokenizer config
        self.onnx_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="onnx/model.onnx",
        )
        tokenizer_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="tokenizer.json",
        )
        print(f"  ONNX model : {self.onnx_path}")
        print(f"  Tokenizer  : {tokenizer_path}")

        # Load fast tokenizer (pure Rust, no sklearn)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_padding(
            pad_id=0, pad_token="[PAD]", length=MAX_SEQ_LEN
        )
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)

        # Load ONNX session (CPU provider)
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 2
        sess_opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            self.onnx_path,
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        print("  ONNX session ready.")

    def _mean_pool_and_normalise(
        self, token_embeddings: np.ndarray, attention_mask: np.ndarray
    ) -> np.ndarray:
        """Mean pool over non-padding tokens, then L2-normalise."""
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = (token_embeddings * mask).sum(axis=1)
        count  = mask.sum(axis=1).clip(min=1e-9)
        vecs   = summed / count
        norms  = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
        return (vecs / norms).astype(np.float32)

    def encode(self, texts: list) -> np.ndarray:
        """
        Encode a list of strings.
        Returns float32 array of shape (N, 384), L2-normalised.
        """
        all_vecs = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            encoded = self.tokenizer.encode_batch(batch)
            input_ids      = np.array([e.ids      for e in encoded], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

            outputs = self.session.run(
                None,
                {
                    "input_ids":      input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                },
            )
            # outputs[0] shape: (batch, seq_len, 384) — last hidden state
            token_emb = outputs[0]
            vecs = self._mean_pool_and_normalise(token_emb, attention_mask)
            all_vecs.append(vecs)
            pct = min(start + BATCH_SIZE, len(texts))
            print(f"    Encoded {pct}/{len(texts)}", end="\r", flush=True)

        print()  # newline after progress
        return np.vstack(all_vecs)


# ── Output schema ────────────────────────────────────────────────────────────
EMBED_SCHEMA = StructType([
    StructField("passage_id", StringType(),           nullable=False),
    StructField("embedding",  ArrayType(FloatType()), nullable=False),
])


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two L2-normalised vectors = cosine similarity."""
    return float(np.dot(a, b))


def main() -> None:
    t_start = time.time()

    # ── Spark session ────────────────────────────────────────────────────────
    spark = (
        SparkSession.builder
        .appName("GraphRAG-Embed")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark {spark.version}")

    # ── Load passages from HDFS (JVM reads Parquet, no pyarrow involved) ─────
    print(f"\n=== Loading passages from {HDFS_INGEST} ===")
    passages_df = (
        spark.read
        .parquet(HDFS_INGEST)
        .select("passage_id", "passage")
        .dropDuplicates(["passage_id"])
    )
    passages_df = passages_df.filter(
        passages_df["passage_id"].isNotNull() & passages_df["passage"].isNotNull()
    )



    # Collect to driver — 1,053 rows × ~100 chars ≈ trivial for 8 GB RAM
    rows = passages_df.collect()
    pids  = [r.passage_id for r in rows]
    texts = [r.passage    for r in rows]
    print(f"  Passages collected: {len(pids):,}")

    if len(pids) == 0:
        spark.stop()
        sys.exit("ERROR: No passages found in HDFS. Re-run 01_ingest.py first.")

    # ── Embed on driver (ONNX, no sklearn) ──────────────────────────────────
    print(f"\n=== Computing embeddings with all-MiniLM-L6-v2 (ONNX) ===")
    print("  (Downloading model on first run; ~23 MB)")
    embedder = MiniLMEmbedder()

    t_enc = time.time()
    vecs = embedder.encode(texts)
    t_enc_done = time.time()
    print(f"  Encoding done in {t_enc_done - t_enc:.1f}s")
    print(f"  Output shape: {vecs.shape}  (expect ({len(pids)}, {EMBED_DIM}))")

    assert vecs.shape == (len(pids), EMBED_DIM), \
        f"Unexpected embedding shape: {vecs.shape}"

    # ── Build Spark DataFrame and write to HDFS ───────────────────────────────
    print(f"\n=== Writing embeddings to {HDFS_EMBED} ===")
    embed_rows = [
        Row(passage_id=pid, embedding=vec.tolist())
        for pid, vec in zip(pids, vecs)
    ]
    embed_df = spark.createDataFrame(embed_rows, schema=EMBED_SCHEMA)
    embed_df.write.mode("overwrite").parquet(HDFS_EMBED)
    print(f"  Write complete.")

    # ── Verify from HDFS ─────────────────────────────────────────────────────
    print(f"\n=== Verifying {HDFS_EMBED} ===")
    result_df = spark.read.parquet(HDFS_EMBED)
    result_df.printSchema()

    n_embedded = result_df.count()
    print(f"  Rows in HDFS : {n_embedded:,}")

    samples = result_df.limit(4).collect()
    dims = [len(r.embedding) for r in samples]
    dim_ok = all(d == EMBED_DIM for d in dims)
    print(f"  Dims (sample): {dims}  → {'OK ✓' if dim_ok else 'WRONG ✗'}")

    # ── Sanity check: self-similarity ─────────────────────────────────────────
    print("\n=== Sanity check: cosine similarity ===")
    if len(samples) >= 2:
        v0 = np.array(samples[0].embedding, dtype=np.float32)
        v1 = np.array(samples[1].embedding, dtype=np.float32)
        self_sim = cosine_sim(v0, v0)
        cross_sim = cosine_sim(v0, v1)
        tol_ok = abs(self_sim - 1.0) < 1e-4
        print(f"  sim(p0, p0) = {self_sim:.6f}  (expect ~1.000)  "
              f"{'OK ✓' if tol_ok else 'WARN ✗'}")
        print(f"  sim(p0, p1) = {cross_sim:.6f}  (should be < 1.000)")
    else:
        print("  WARNING: fewer than 2 samples for sanity check")

    # ── Also write passage_id → embedding as local CSV (for fallback) ─────────
    local_csv = "data/embeddings.csv"
    print(f"\n=== Writing local backup: {local_csv} ===")
    with open(local_csv, "w") as fh:
        fh.write("passage_id," + ",".join(f"e{i}" for i in range(EMBED_DIM)) + "\n")
        for pid, vec in zip(pids, vecs):
            fh.write(pid + "," + ",".join(f"{v:.6f}" for v in vec) + "\n")
    size_kb = os.path.getsize(local_csv) / 1024
    print(f"  {local_csv}: {size_kb:.0f} KB, {len(pids)} rows")

    # ── Summary ───────────────────────────────────────────────────────────────
    t_total = time.time() - t_start
    match = (n_embedded == len(pids)) and dim_ok
    print("\n=== EMBEDDING SUMMARY ===")
    print(f"  Passages in     : {len(pids):,}")
    print(f"  Embeddings out  : {n_embedded:,}")
    print(f"  Dimension       : {EMBED_DIM}")
    print(f"  Model           : all-MiniLM-L6-v2 ONNX (CPU, L2-normalised)")
    print(f"  HDFS output     : {HDFS_EMBED}")
    print(f"  Local backup    : {local_csv}")
    print(f"  Total time      : {t_total:.1f}s")
    print(f"\nMilestone 2 {'PASSED ✓' if match else 'FAILED ✗'}")

    spark.stop()


if __name__ == "__main__":
    main()
