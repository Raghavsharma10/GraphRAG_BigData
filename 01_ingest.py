#!/usr/bin/env python3
"""
01_ingest.py
============
Milestone 1 — HDFS Ingestion

Reads the prepared MS MARCO CSV files from the local data/ directory,
uploads them to HDFS, reads them back from HDFS using PySpark to verify
schema and row counts, and writes a schema-validated Parquet to HDFS.

HDFS paths used
---------------
  hdfs://localhost:9000/graphrag/raw/msmarco_subset.csv
  hdfs://localhost:9000/graphrag/raw/passages_only.csv
  hdfs://localhost:9000/graphrag/raw/verified/        ← Parquet output

Run
---
  export PYTHONPATH=/opt/homebrew/Cellar/apache-spark/4.2.0/libexec/python:\
/opt/homebrew/Cellar/apache-spark/4.2.0/libexec/python/lib/py4j-0.10.9.9-src.zip

  PYSPARK_PYTHON=/opt/anaconda3/bin/python3 \
  JAVA_HOME=/opt/homebrew/Cellar/openjdk@17/17.0.20.1/libexec/openjdk.jdk/Contents/Home \
  python3 src/01_ingest.py
"""

import os
import subprocess
import sys
import time

# ── PySpark import ──────────────────────────────────────────────────────────
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType, LongType, StringType, StructField, StructType
except ImportError as e:
    sys.exit(
        f"Cannot import PySpark: {e}\n"
        "Set PYTHONPATH to include Spark's python/ and py4j zip before running."
    )

# ── Configuration ──────────────────────────────────────────────────────────
HADOOP_BIN   = "/Users/raghav/hadoop/bin/hdfs"
HDFS_RAW     = "hdfs://localhost:9000/graphrag/raw"
HDFS_PARQUET = "hdfs://localhost:9000/graphrag/raw/verified"
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
SUBSET_CSV   = os.path.join(DATA_DIR, "msmarco_subset.csv")
PASSAGES_CSV = os.path.join(DATA_DIR, "passages_only.csv")

# ── Expected schema ─────────────────────────────────────────────────────────
SUBSET_SCHEMA = StructType([
    StructField("query_id",   StringType(), nullable=False),
    StructField("query",      StringType(), nullable=False),
    StructField("passage_id", StringType(), nullable=False),
    StructField("passage",    StringType(), nullable=False),
    StructField("label",      IntegerType(), nullable=False),
])

PASSAGES_SCHEMA = StructType([
    StructField("passage_id", StringType(), nullable=False),
    StructField("passage",    StringType(), nullable=False),
])


def hdfs_put(local_path: str, hdfs_dest: str) -> None:
    """Copy a local file to HDFS, overwriting if it exists."""
    cmd_rm = [HADOOP_BIN, "dfs", "-rm", "-f", f"{hdfs_dest}/{os.path.basename(local_path)}"]
    subprocess.run(cmd_rm, capture_output=True)
    cmd_put = [HADOOP_BIN, "dfs", "-put", local_path, hdfs_dest]
    result = subprocess.run(cmd_put, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"hdfs dfs -put failed:\n{result.stderr}")
    print(f"  Uploaded {os.path.basename(local_path)} → {hdfs_dest}")


def verify_csv_from_hdfs(spark: SparkSession, hdfs_path: str,
                          schema: StructType, label: str) -> int:
    """Read a CSV from HDFS with the expected schema and return row count."""
    df = spark.read \
        .option("header", "true") \
        .option("quote", '"') \
        .option("escape", '"') \
        .schema(schema) \
        .csv(hdfs_path)

    count = df.count()
    null_qid   = df.filter(F.col("passage_id").isNull()).count()

    print(f"  [{label}] Rows: {count:,}  |  Null passage_id: {null_qid}")
    if null_qid > 0:
        print(f"  WARNING: {null_qid} rows have null passage_id in {label}")

    return count


def main() -> None:
    # ── Sanity-check local files ────────────────────────────────────────────
    for path in [SUBSET_CSV, PASSAGES_CSV]:
        if not os.path.exists(path):
            sys.exit(f"Missing input file: {path}\nRun src/00_prepare_data.py first.")

    # ── Upload to HDFS ──────────────────────────────────────────────────────
    print("=== Stage 1: Uploading CSV files to HDFS ===")
    hdfs_put(SUBSET_CSV,   HDFS_RAW)
    hdfs_put(PASSAGES_CSV, HDFS_RAW)

    # ── Start Spark session ─────────────────────────────────────────────────
    print("\n=== Stage 2: Starting PySpark session ===")
    spark = (
        SparkSession.builder
        .appName("GraphRAG-Ingest")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
        # Suppress verbose Hadoop native library warnings
        .config("spark.hadoop.fs.hdfs.impl",
                "org.apache.hadoop.hdfs.DistributedFileSystem")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"  Spark version : {spark.version}")
    print(f"  Hadoop version: {spark._jvm.org.apache.hadoop.util.VersionInfo.getVersion()}")

    # ── Read and verify from HDFS ────────────────────────────────────────────
    print("\n=== Stage 3: Reading and verifying from HDFS ===")
    t0 = time.time()

    subset_hdfs   = f"{HDFS_RAW}/msmarco_subset.csv"
    passages_hdfs = f"{HDFS_RAW}/passages_only.csv"

    n_subset   = verify_csv_from_hdfs(spark, subset_hdfs,   SUBSET_SCHEMA,   "msmarco_subset")
    n_passages = verify_csv_from_hdfs(spark, passages_hdfs, PASSAGES_SCHEMA, "passages_only")

    elapsed = time.time() - t0
    print(f"\n  Read latency: {elapsed:.1f}s")

    # ── Write Parquet to HDFS ────────────────────────────────────────────────
    print("\n=== Stage 4: Writing verified Parquet to HDFS ===")
    passages_df = (
        spark.read
        .option("header", "true")
        .option("quote", '"')
        .option("escape", '"')
        .schema(PASSAGES_SCHEMA)
        .csv(passages_hdfs)
        .dropDuplicates(["passage_id"])      # safety dedup
        .filter(F.col("passage_id").isNotNull())
        .filter(F.col("passage").isNotNull())
    )

    passages_df.write \
        .mode("overwrite") \
        .parquet(HDFS_PARQUET)

    parquet_count = spark.read.parquet(HDFS_PARQUET).count()
    print(f"  Parquet rows written to {HDFS_PARQUET}: {parquet_count:,}")

    # ── Schema print ─────────────────────────────────────────────────────────
    print("\n=== Stage 5: Schema confirmation ===")
    spark.read.parquet(HDFS_PARQUET).printSchema()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n=== INGESTION SUMMARY ===")
    print(f"  msmarco_subset.csv rows   : {n_subset:,}")
    print(f"  passages_only.csv rows     : {n_passages:,}")
    print(f"  Parquet unique passages    : {parquet_count:,}")
    print(f"  HDFS Parquet path          : {HDFS_PARQUET}")
    print(f"  Total elapsed              : {elapsed:.1f}s")
    print("\nMilestone 1 PASSED ✓" if parquet_count > 0 else "\nMilestone 1 FAILED ✗")

    spark.stop()


if __name__ == "__main__":
    main()
