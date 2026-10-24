#!/usr/bin/env python3
"""
tests/test_dataset.py
Tests for data preparation outputs (msmarco_subset.csv, passages_only.csv).
Run: python3 -m pytest tests/test_dataset.py -v
"""
import csv
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SUBSET_CSV   = os.path.join(DATA_DIR, "msmarco_subset.csv")
PASSAGES_CSV = os.path.join(DATA_DIR, "passages_only.csv")
FOUND_CSV    = os.path.join(DATA_DIR, "passages_found.csv")
COVERAGE_TXT = os.path.join(DATA_DIR, "coverage.txt")


def _load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── passages_found.csv ─────────────────────────────────────────────────────

def test_found_csv_exists():
    assert os.path.exists(FOUND_CSV), "passages_found.csv not created yet"


def test_found_csv_has_header():
    rows = _load_csv(FOUND_CSV)
    assert rows, "passages_found.csv is empty"
    assert "passage_id" in rows[0]
    assert "passage" in rows[0]


def test_found_no_empty_passage():
    rows = _load_csv(FOUND_CSV)
    empties = [r for r in rows if not r["passage"].strip()]
    assert len(empties) == 0, f"{len(empties)} rows with empty passage text"


def test_found_no_duplicate_passage_ids():
    rows = _load_csv(FOUND_CSV)
    pids = [r["passage_id"] for r in rows]
    assert len(pids) == len(set(pids)), "Duplicate passage_ids in passages_found.csv"


# ── msmarco_subset.csv ─────────────────────────────────────────────────────

def test_subset_csv_exists():
    assert os.path.exists(SUBSET_CSV), "msmarco_subset.csv not created yet"


def test_subset_columns():
    rows = _load_csv(SUBSET_CSV)
    assert rows, "msmarco_subset.csv is empty"
    expected = {"query_id", "query", "passage_id", "passage", "label"}
    assert expected == set(rows[0].keys()), f"Unexpected columns: {set(rows[0].keys())}"


def test_subset_labels_are_binary():
    rows = _load_csv(SUBSET_CSV)
    bad = [r for r in rows if r["label"] not in ("0", "1")]
    assert len(bad) == 0, f"{len(bad)} rows with invalid label"


def test_subset_no_empty_fields():
    rows = _load_csv(SUBSET_CSV)
    for col in ("query_id", "query", "passage_id", "passage"):
        empties = [r for r in rows if not r[col].strip()]
        assert len(empties) == 0, f"{len(empties)} empty values in column '{col}'"


def test_subset_passage_ids_match_found():
    """Every passage_id in subset must be in passages_found."""
    found_pids = {r["passage_id"] for r in _load_csv(FOUND_CSV)}
    subset_pids = {r["passage_id"] for r in _load_csv(SUBSET_CSV)}
    extra = subset_pids - found_pids
    assert len(extra) == 0, f"{len(extra)} passage_ids in subset missing from found"


# ── passages_only.csv ──────────────────────────────────────────────────────

def test_passages_only_exists():
    assert os.path.exists(PASSAGES_CSV), "passages_only.csv not created yet"


def test_passages_only_no_self_edges_prep():
    """All passage_ids must be unique (precondition for edge dedup)."""
    rows = _load_csv(PASSAGES_CSV)
    pids = [r["passage_id"] for r in rows]
    assert len(pids) == len(set(pids)), "Duplicate passage_ids in passages_only.csv"


# ── coverage.txt ───────────────────────────────────────────────────────────

def test_coverage_file_exists():
    assert os.path.exists(COVERAGE_TXT), "coverage.txt not created yet"


def test_coverage_above_zero():
    with open(COVERAGE_TXT) as f:
        content = f.read()
    for line in content.splitlines():
        if line.startswith("found"):
            n_found = int(line.split(":")[1].strip())
            assert n_found > 0, "Zero passages found"
            return
    assert False, "No 'found' line in coverage.txt"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
