"""Rebuild Step B CSV indexes from canonical artifacts if spreadsheet software rewrote IDs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from text_feature_common import normalize_text, write_text_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    root = workspace / "02_features_q1"
    data_dir = root / "stepB_text" / "data"
    index = pd.read_csv(root / "stepA_index" / "data" / "sample_index.csv", dtype=str, keep_default_na=False)
    validation = pd.read_csv(data_dir / "text_feature_validation.csv", dtype=str, keep_default_na=False)
    validation_map = validation.set_index("sample_id")["status"].to_dict()
    records = [json.loads(line) for line in (data_dir / "text_token_map.jsonl").read_text(encoding="utf-8").splitlines()]
    record_map = {row["sample_id"]: row for row in records}
    expected = set(index.sample_id)
    if set(record_map) != expected:
        raise ValueError("token JSONL sample IDs do not match the rebuilt Step A index")
    if set(validation_map) != expected:
        raise ValueError("text validation sample IDs do not match the rebuilt Step A index")
    run = json.loads((data_dir / "text_feature_run.json").read_text(encoding="utf-8"))

    index["normalized_text"] = index["raw_text"].map(normalize_text)
    index["text_status"] = [
        "empty" if not text else ("PASS" if validation_map[sample_id] == "PASS" else validation_map[sample_id])
        for sample_id, text in zip(index.sample_id, index.normalized_text)
    ]
    index["normalization"] = "NFC+strip+collapse_whitespace"
    index["tokenizer_name"] = run["model_name"]
    index["tokenizer_version"] = run["tokenizer_class"]
    write_text_index(index, data_dir / "text_index.csv")

    rows = []
    for sample_id in index.sample_id:
        record = record_map[sample_id]
        feature_path = data_dir / "text_features" / "bert-base-uncased" / f"{sample_id}.npz"
        with np.load(feature_path, allow_pickle=False) as arrays:
            valid_length = int(arrays["text_valid_length"])
            shape = str(tuple(arrays["text_features"].shape))
        rows.append({
            "sample_id": sample_id,
            "text_status": "PASS" if validation_map[sample_id] == "PASS" else validation_map[sample_id],
            "token_count": int(record["token_count"]),
            "original_token_count": int(record["original_token_count"]),
            "truncated": bool(record["truncated"]),
            "text_valid_length": valid_length,
            "feature_shape": shape,
            "feature_path": feature_path.relative_to(workspace).as_posix(),
            "device": run["device"],
            "model_name": run["model_name"],
            "max_length": run["max_length"],
        })
    with (data_dir / "text_extraction_log.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "recovered_from": [
            "stepA_index/data/sample_index.csv",
            "text_token_map.jsonl",
            "text_feature_validation.csv",
            "text_features/bert-base-uncased/{sample_id}.npz",
            "text_feature_run.json",
        ],
        "sample_count": len(index),
        "status_counts": validation.status.value_counts().to_dict(),
        "warning": "Do not edit/save these machine-readable CSV files with spreadsheet formula inference enabled.",
    }
    (data_dir / "text_metadata_recovery.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
