"""Shared utilities for Step B text preprocessing and BERT feature extraction."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


TIME_BINS = 50
EXPECTED_HIDDEN_SIZE = 768


def normalize_text(value) -> str:
    """Apply reversible normalization only; preserve words, punctuation and negation."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text.strip())


def uniform_bin(token_index: int, token_count: int, bins: int = TIME_BINS) -> int:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    return min((token_index * bins) // token_count, bins - 1)


def make_bucket_features(token_embeddings: np.ndarray, token_count: int, bins: int = TIME_BINS):
    """Average content-token embeddings into fixed uniform token bins."""
    if token_embeddings.ndim != 2:
        raise ValueError(f"expected (tokens, hidden), got {token_embeddings.shape}")
    if token_count != token_embeddings.shape[0]:
        raise ValueError("token_count and token_embeddings length differ")
    hidden = token_embeddings.shape[1]
    features = np.zeros((bins, hidden), dtype=np.float32)
    mask = np.zeros((bins, 1), dtype=np.uint8)
    counts = np.zeros((bins,), dtype=np.int32)
    starts = np.full((bins,), -1, dtype=np.int32)
    ends = np.full((bins,), -1, dtype=np.int32)
    for i in range(token_count):
        b = uniform_bin(i, token_count, bins)
        if counts[b] == 0:
            starts[b] = i
        features[b] += token_embeddings[i].astype(np.float32, copy=False)
        counts[b] += 1
        ends[b] = i + 1
    nonempty = counts > 0
    features[nonempty] /= counts[nonempty, None]
    mask[nonempty, 0] = 1
    return features, mask, counts, starts, ends


def load_step_a_index(workspace: Path) -> pd.DataFrame:
    path = workspace / "02_features_q1" / "stepA_index" / "data" / "sample_index.csv"
    frame = pd.read_csv(path, dtype={"sample_id": str, "video_id": str, "clip_id": str})
    required = {"sample_id", "raw_text", "index_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Step A index missing columns: {sorted(missing)}")
    suspicious = frame["sample_id"].eq("#NAME?") | frame["video_id"].eq("#NAME?")
    if suspicious.any():
        raise ValueError(
            "Step A index contains Excel-corrupted '#NAME?' IDs. Rebuild it from "
            "stepA_index/code/build_sample_index.py before running Step B."
        )
    if not frame["sample_id"].is_unique:
        raise ValueError("Step A sample_id is not unique")
    return frame


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_text_index(workspace: Path) -> pd.DataFrame:
    """Create the deterministic, model-independent normalized text index."""
    frame = load_step_a_index(workspace).copy()
    frame["normalized_text"] = frame["raw_text"].map(normalize_text)
    frame["text_status"] = np.where(frame["normalized_text"].eq(""), "empty", "ready_for_tokenizer")
    frame["normalization"] = "NFC+strip+collapse_whitespace"
    frame["tokenizer_name"] = "pending_model_run"
    frame["tokenizer_version"] = "pending_model_run"
    return frame


def write_text_index(frame: pd.DataFrame, output_path: Path):
    columns = [
        "sample_id", "video_id", "clip_id", "raw_text", "normalized_text",
        "text_status", "normalization", "tokenizer_name", "tokenizer_version",
        "label", "annotation", "index_status",
    ]
    existing = [column for column in columns if column in frame.columns]
    frame[existing].to_csv(output_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)


def save_sample_npz(
    path: Path,
    features,
    mask,
    counts,
    starts,
    ends,
    sample_id: str,
    method: str,
):
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_id),
        text_features=features.astype(np.float32),
        text_mask=mask.astype(np.uint8),
        token_count_per_bin=counts.astype(np.int32),
        token_start_per_bin=starts.astype(np.int32),
        token_end_per_bin=ends.astype(np.int32),
        text_valid_length=np.asarray(int(mask.sum()), dtype=np.int32),
        text_alignment_method=np.asarray(method),
    )
