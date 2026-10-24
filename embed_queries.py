#!/usr/bin/env python3
"""
embed_queries.py
================
Embed the 1,000 MS MARCO query texts using the same all-MiniLM-L6-v2
ONNX model used for passages, and save to data/query_embeddings.csv.

This produces valid query vectors for retrieval evaluation — avoiding
the circular approach of using passage embeddings as query proxies.
"""

import os
import sys
import csv
import math
import time

try:
    import numpy as np
except ImportError:
    sys.exit("numpy not found.")

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("onnxruntime not found.")

try:
    from tokenizers import Tokenizer
except ImportError:
    sys.exit("tokenizers not found.")

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("huggingface_hub not found.")

MODEL_REPO   = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM    = 384
MAX_SEQ_LEN  = 128
BATCH_SIZE   = 64
SUBSET_CSV   = "data/msmarco_subset.csv"
OUT_CSV      = "data/query_embeddings.csv"


class MiniLMEmbedder:
    def __init__(self):
        onnx_path = hf_hub_download(repo_id=MODEL_REPO, filename="onnx/model.onnx")
        tok_path  = hf_hub_download(repo_id=MODEL_REPO, filename="tokenizer.json")
        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=MAX_SEQ_LEN)
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts):
        all_vecs = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch   = texts[start: start + BATCH_SIZE]
            encoded = self.tokenizer.encode_batch(batch)
            ids  = np.array([e.ids            for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            tids = np.zeros_like(ids)
            out  = self.session.run(None, {"input_ids": ids,
                                           "attention_mask": mask,
                                           "token_type_ids": tids})[0]
            # mean pool + L2 norm
            m    = mask[:, :, np.newaxis].astype(np.float32)
            vecs = (out * m).sum(axis=1) / m.sum(axis=1).clip(min=1e-9)
            nrm  = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
            all_vecs.append((vecs / nrm).astype(np.float32))
        return np.vstack(all_vecs)


def main():
    print("=== Embedding query texts with all-MiniLM-L6-v2 ONNX ===")
    # Load unique queries
    queries = {}
    with open(SUBSET_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            queries[row["query_id"]] = row["query"]
    qids  = list(queries.keys())
    texts = [queries[q] for q in qids]
    print(f"  Queries to embed: {len(qids)}")

    embedder = MiniLMEmbedder()
    t0   = time.time()
    vecs = embedder.encode(texts)
    print(f"  Encoded {len(qids)} queries in {time.time()-t0:.1f}s, shape={vecs.shape}")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        header = ["query_id"] + [f"e{i}" for i in range(EMBED_DIM)]
        writer = csv.writer(fh)
        writer.writerow(header)
        for qid, vec in zip(qids, vecs):
            writer.writerow([qid] + [f"{v:.6f}" for v in vec])

    print(f"  Saved: {OUT_CSV}  ({os.path.getsize(OUT_CSV)//1024} KB, {len(qids)} rows)")
    print("Done.")


if __name__ == "__main__":
    main()
