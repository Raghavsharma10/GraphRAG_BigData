#!/usr/bin/env python3
"""
extract_streaming.py
====================
Streams a HuggingFace corpus parquet shard over HTTP, reading one
row-group at a time, and appends matching passages to passages_found.csv.
Never writes the full shard to disk — uses only the row-group buffer.

Usage:
    python3 src/extract_streaming.py --url <parquet_url> [--max-rg N]

The HuggingFace parquet API URL is:
  https://huggingface.co/api/datasets/Tevatron/msmarco-passage-corpus/parquet/default/train/<N>.parquet
"""

import argparse
import csv
import os
import sys

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
QRELS_FILE   = os.path.join(DATA_DIR, "qrels.dev.small.tsv")
FOUND_CSV    = os.path.join(DATA_DIR, "passages_found.csv")
COVERAGE_TXT = os.path.join(DATA_DIR, "coverage.txt")
MAX_QUERIES  = 1000


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


def load_found() -> dict[str, str]:
    found: dict[str, str] = {}
    if not os.path.exists(FOUND_CSV):
        return found
    with open(FOUND_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            found[row["passage_id"]] = row["passage"]
    return found


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


def _flush_to_csv(new_found: dict, found_csv: str) -> None:
    """Append new_found to passages_found.csv."""
    file_exists = os.path.exists(found_csv)
    with open(found_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        if not file_exists:
            writer.writerow(["passage_id", "passage"])
        for pid, text in new_found.items():
            writer.writerow([pid, text])


def stream_extract(url: str, remaining: set, found_csv: str,
                   max_rg: int | None) -> dict[str, str]:
    """
    Open parquet over HTTP via fsspec and read row-groups one at a time.
    Flushes found passages to CSV every FLUSH_EVERY new finds (incremental save).
    Returns dict of {passage_id: text} newly found in this run.
    """
    import fsspec
    import pyarrow.parquet as pq

    FLUSH_EVERY = 50   # save to disk every 50 new passages found

    print(f"Opening stream: {url}")
    fs = fsspec.filesystem("http")

    try:
        f = fs.open(url, "rb")
    except Exception as e:
        sys.exit(f"Cannot open URL: {e}")

    pf = pq.ParquetFile(f)
    n_rg = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows
    print(f"  Row groups: {n_rg}  |  Total rows: {total_rows:,}")

    new_found: dict[str, str] = {}
    unflushed: dict[str, str] = {}
    rg_limit = min(n_rg, max_rg) if max_rg else n_rg

    try:
        for rg_idx in range(rg_limit):
            if not (remaining - set(new_found.keys())):
                print(f"  All targets found — stopping early at row-group {rg_idx}")
                break

            batch = pf.read_row_group(rg_idx, columns=["docid", "text"])
            docids = batch.column("docid").to_pylist()
            texts  = batch.column("text").to_pylist()

            for d, tx in zip(docids, texts):
                if d in remaining and d not in new_found:
                    new_found[d] = tx
                    unflushed[d] = tx

            # Incremental flush every FLUSH_EVERY new passages
            if len(unflushed) >= FLUSH_EVERY:
                _flush_to_csv(unflushed, found_csv)
                unflushed.clear()

            if rg_idx % 5 == 0 or rg_idx == rg_limit - 1:
                print(f"  RG {rg_idx+1}/{rg_limit}: found {len(new_found)} so far, "
                      f"still need {len(remaining) - len(new_found)}")
    except Exception as e:
        print(f"  Stream interrupted at RG {rg_idx}: {e}")
        print(f"  Saving {len(unflushed)} unflushed passages before exit...")
    finally:
        # Always flush remaining unflushed passages
        if unflushed:
            _flush_to_csv(unflushed, found_csv)
            print(f"  Flushed {len(unflushed)} remaining passages to disk.")
        try:
            f.close()
        except Exception:
            pass

    return new_found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="HuggingFace parquet URL to stream")
    parser.add_argument("--max-rg", type=int, default=None,
                        help="Max row-groups to read (for testing)")
    args = parser.parse_args()

    print(f"Loading qrels (max {MAX_QUERIES} queries)...")
    qrels, wanted = load_qrels(MAX_QUERIES)
    print(f"  Wanted: {len(wanted)} passage IDs")

    print("Loading already-found passages...")
    found = load_found()
    print(f"  Already found: {len(found)}")

    remaining = wanted - set(found.keys())
    if not remaining:
        print("All passages already found.")
        write_coverage(qrels, found, wanted)
        return

    print(f"  Still need: {len(remaining)}")

    new_found = stream_extract(args.url, remaining, FOUND_CSV, args.max_rg)
    print(f"\nNew passages written to disk: {len(new_found)}")
    found.update(new_found)


    write_coverage(qrels, found, wanted)


if __name__ == "__main__":
    main()
