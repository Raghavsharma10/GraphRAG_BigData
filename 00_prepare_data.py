#!/usr/bin/env python3
"""
00_prepare_data.py
==================
Builds the MS MARCO subset CSV from corpus parquet shards and qrels.

Usage – run once per shard, accumulates into passages_found.csv:
    python3 src/00_prepare_data.py --shard data/corpus_shard0.parquet
    python3 src/00_prepare_data.py --shard data/corpus_shard5.parquet
    ...
    python3 src/00_prepare_data.py --finalise

Outputs (all in data/):
    passages_found.csv     – passage_id, passage  (grows each shard run)
    coverage.txt           – running coverage summary
    msmarco_subset.csv     – query_id, query, passage_id, passage, label
    passages_only.csv      – passage_id, passage  (unique; for embedding)

Notes:
  - Uses pyarrow column API only (no .to_pandas()) to avoid numpy mismatch.
  - MAX_QUERIES=1000 caps the subset to keep RAM and disk manageable.
  - --finalise joins found passages with query text and writes final CSVs.
"""

import argparse
import csv
import os
import sys

import pyarrow.parquet as pq

DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
QRELS_FILE   = os.path.join(DATA_DIR, "qrels.dev.small.tsv")
QUERIES_PARQ = os.path.join(DATA_DIR, "passages_validation.parquet")
FOUND_CSV    = os.path.join(DATA_DIR, "passages_found.csv")
COVERAGE_TXT = os.path.join(DATA_DIR, "coverage.txt")
SUBSET_CSV   = os.path.join(DATA_DIR, "msmarco_subset.csv")
PASSAGES_CSV = os.path.join(DATA_DIR, "passages_only.csv")
MAX_QUERIES  = 1000


# ── Helpers ────────────────────────────────────────────────────────────────

def load_qrels(max_queries: int):
    qrels: dict[str, list[str]] = {}
    with open(QRELS_FILE) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 4 or p[3] != "1":
                continue
            qid, pid = p[0], p[2]
            if qid not in qrels:
                if len(qrels) >= max_queries:
                    continue
                qrels[qid] = []
            qrels[qid].append(pid)
    wanted = {pid for pids in qrels.values() for pid in pids}
    return qrels, wanted


def load_found_csv() -> dict[str, str]:
    """Load already-extracted passages from previous shard runs."""
    found: dict[str, str] = {}
    if not os.path.exists(FOUND_CSV):
        return found
    with open(FOUND_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            found[row["passage_id"]] = row["passage"]
    return found


def load_queries() -> dict[str, str]:
    """Load query_id → query text from the validation parquet."""
    pf = pq.ParquetFile(QUERIES_PARQ)
    queries: dict[str, str] = {}
    for batch in pf.iter_batches(columns=["query_id", "query"]):
        for qid, qtxt in zip(
            batch.column("query_id").to_pylist(),
            batch.column("query").to_pylist(),
        ):
            queries[qid] = qtxt
    return queries



def write_coverage(qrels, found, wanted):
    lines = [
        f"queries : {len(qrels)}",
        f"wanted  : {len(wanted)}",
        f"found   : {len(found)}",
        f"missing : {len(wanted) - len(found)}",
        f"coverage: {100*len(found)/len(wanted):.1f}%",
    ]
    print("\n".join(lines))
    with open(COVERAGE_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Shard extraction ───────────────────────────────────────────────────────

def extract_shard(shard_path: str):
    print(f"Loading qrels (max {MAX_QUERIES} queries)...")
    qrels, wanted = load_qrels(MAX_QUERIES)
    print(f"  Wanted: {len(wanted)} passage IDs from {len(qrels)} queries")

    print("Loading already-found passages...")
    found = load_found_csv()
    print(f"  Already found: {len(found)}")

    remaining = wanted - set(found.keys())
    if not remaining:
        print("All passages already found — nothing to do.")
        write_coverage(qrels, found, wanted)
        return

    print(f"  Still need: {len(remaining)}")
    print(f"Scanning shard: {shard_path} ...")

    pf = pq.ParquetFile(shard_path)
    print(f"  Shard rows: {pf.metadata.num_rows:,}")

    new_found: dict[str, str] = {}
    for batch in pf.iter_batches(batch_size=50000, columns=["docid", "text"]):
        docids = batch.column("docid").to_pylist()
        texts  = batch.column("text").to_pylist()
        for d, tx in zip(docids, texts):
            if d in remaining:
                new_found[d] = tx
        if len(new_found) == len(remaining):
            break   # found everything we need, stop early


    print(f"  Found in this shard: {len(new_found)}")

    # Append new passages to found CSV
    file_exists = os.path.exists(FOUND_CSV)
    with open(FOUND_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["passage_id", "passage"])
        for pid, text in new_found.items():
            writer.writerow([pid, text])

    found.update(new_found)
    write_coverage(qrels, found, wanted)
    print(f"  Total found so far: {len(found)} / {len(wanted)}")


# ── Finalise: build msmarco_subset.csv and passages_only.csv ──────────────

def finalise():
    print("Loading qrels...")
    qrels, wanted = load_qrels(MAX_QUERIES)

    print("Loading found passages...")
    found = load_found_csv()
    if not found:
        sys.exit("No passages found yet — run with --shard first.")

    print("Loading query texts...")
    queries = load_queries()

    write_coverage(qrels, found, wanted)

    # msmarco_subset.csv: one row per (query, relevant_passage)
    rows_written = 0
    with open(SUBSET_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["query_id", "query", "passage_id", "passage", "label"])
        for qid, pids in sorted(qrels.items()):
            if qid not in queries:
                continue
            qtext = queries[qid]
            for pid in pids:
                if pid not in found:
                    continue
                writer.writerow([qid, qtext, pid, found[pid], 1])
                rows_written += 1

    # passages_only.csv: unique passages for embedding stage
    with open(PASSAGES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["passage_id", "passage"])
        for pid, text in found.items():
            writer.writerow([pid, text])

    print(f"\n=== FINALISE COMPLETE ===")
    print(f"  msmarco_subset.csv : {rows_written} rows")
    print(f"  passages_only.csv  : {len(found)} unique passages")
    print(f"  coverage           : {100*len(found)/len(wanted):.1f}% of wanted passages")
    if len(found) < len(wanted):
        print(f"  NOTE: {len(wanted)-len(found)} passage IDs not recovered "
              "(spread across undownloaded shards). "
              "Report coverage accurately.")


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shard", help="Path to a corpus shard parquet to scan")
    group.add_argument("--finalise", action="store_true",
                       help="Build final CSVs from accumulated passages")
    args = parser.parse_args()

    if args.shard:
        if not os.path.exists(args.shard):
            sys.exit(f"Shard file not found: {args.shard}")
        extract_shard(args.shard)
    else:
        finalise()


if __name__ == "__main__":
    main()
