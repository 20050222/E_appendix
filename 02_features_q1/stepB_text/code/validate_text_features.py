"""Validate completed Step B text features without loading the BERT model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_NPZ_KEYS = {
    "sample_id",
    "text_features",
    "text_mask",
    "token_count_per_bin",
    "token_start_per_bin",
    "token_end_per_bin",
    "text_valid_length",
    "text_alignment_method",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--model-name", default="bert-base-uncased")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    step_a_path = workspace / "02_features_q1" / "stepA_index" / "data" / "sample_index.csv"
    data_root = workspace / "02_features_q1" / "stepB_text" / "data"
    feature_dir = data_root / "text_features" / args.model_name.replace("/", "__")
    log_path = data_root / "text_extraction_log.csv"
    token_map_path = data_root / "text_token_map.jsonl"
    run_path = data_root / "text_feature_run.json"

    expected = pd.read_csv(step_a_path, dtype=str, keep_default_na=False)["sample_id"].tolist()
    expected_set = set(expected)
    errors: list[str] = []
    rows: list[dict] = []

    if not log_path.is_file():
        errors.append(f"missing extraction log: {log_path}")
        log_ids = set()
    else:
        log = pd.read_csv(log_path, dtype=str, keep_default_na=False)
        log_ids = set(log.get("sample_id", []))
        if log_ids != expected_set:
            errors.append("text_extraction_log.csv sample_id set differs from Step A")

    token_ids: list[str] = []
    if not token_map_path.is_file():
        errors.append(f"missing token map: {token_map_path}")
    else:
        with token_map_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                token_ids.append(str(record["sample_id"]))
                tokens = record.get("tokens", [])
                for token in tokens:
                    if "bin_index" not in token or not 0 <= int(token["bin_index"]) < 50:
                        errors.append(f"invalid token bin at JSONL line {line_number}")
                        break
                if len(record.get("bucket_map", [])) != 50:
                    errors.append(f"bucket_map length is not 50 at JSONL line {line_number}")
        if token_ids != expected:
            errors.append("text_token_map.jsonl sample_id order differs from Step A")

    for sample_id in expected:
        path = feature_dir / f"{sample_id}.npz"
        sample_errors: list[str] = []
        if not path.is_file():
            sample_errors.append("missing_npz")
        else:
            with np.load(path, allow_pickle=False) as data:
                missing_keys = REQUIRED_NPZ_KEYS - set(data.files)
                if missing_keys:
                    sample_errors.append(f"missing_keys={sorted(missing_keys)}")
                else:
                    features = data["text_features"]
                    mask = data["text_mask"]
                    counts = data["token_count_per_bin"]
                    starts = data["token_start_per_bin"]
                    ends = data["token_end_per_bin"]
                    if features.shape != (50, 768):
                        sample_errors.append(f"feature_shape={features.shape}")
                    if mask.shape != (50, 1):
                        sample_errors.append(f"mask_shape={mask.shape}")
                    if not np.isfinite(features).all():
                        sample_errors.append("features_have_nan_or_inf")
                    if not set(np.unique(mask)).issubset({0, 1}):
                        sample_errors.append("mask_not_binary")
                    if not np.array_equal(mask[:, 0], (counts > 0).astype(mask.dtype)):
                        sample_errors.append("mask_count_mismatch")
                    if int(data["text_valid_length"]) != int(mask.sum()):
                        sample_errors.append("valid_length_mask_mismatch")
                    if np.any((counts == 0) & ((starts != -1) | (ends != -1))):
                        sample_errors.append("empty_bin_boundary_mismatch")
                    if np.any((counts > 0) & ((ends - starts) != counts)):
                        sample_errors.append("nonempty_bin_boundary_mismatch")
                    if str(data["sample_id"]) != sample_id:
                        sample_errors.append("sample_id_mismatch")
                    if str(data["text_alignment_method"]) != "uniform_tokens":
                        sample_errors.append("alignment_method_mismatch")
        rows.append({
            "sample_id": sample_id,
            "status": "PASS" if not sample_errors else "FAIL",
            "issues": ";".join(sample_errors),
            "feature_path": str(path.relative_to(workspace)),
        })
        errors.extend(f"{sample_id}: {issue}" for issue in sample_errors)

    if not run_path.is_file():
        errors.append(f"missing run metadata: {run_path}")

    report_path = data_root / "text_feature_validation.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "expected_samples": len(expected),
        "passed_samples": sum(row["status"] == "PASS" for row in rows),
        "failed_samples": sum(row["status"] == "FAIL" for row in rows),
        "global_error_count": len(errors),
        "validation_report": str(report_path),
        "status": "PASS" if not errors else "FAIL",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        for error in errors[:20]:
            print(f"ERROR: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
