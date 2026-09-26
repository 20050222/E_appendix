"""Validate aligned-50 sample IDs, labels, masks, shapes, times and finite values."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


SHAPES = {"text": (50, 768), "audio": (50, 74), "vision": (50, 35)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    root = workspace / "02_features_q1"
    index = pd.read_csv(root / "stepA_index" / "data" / "sample_index.csv", dtype=str, keep_default_na=False)
    audit = pd.read_csv(root / "stepE_alignment" / "data" / "multimodal_sample_audit.csv", dtype=str, keep_default_na=False)
    feature_dir = root / "stepE_alignment" / "data" / "aligned_50"
    rows = []
    global_errors = []
    expected_ids = index.sample_id.tolist()
    if len(expected_ids) != 100 or len(set(expected_ids)) != 100:
        global_errors.append("Step A does not contain 100 unique sample IDs")
    if audit.sample_id.tolist() != expected_ids:
        global_errors.append("multimodal audit IDs/order differ from Step A")

    truth = index.set_index("sample_id")
    for sample_id in expected_ids:
        errors = []
        path = feature_dir / f"{sample_id}.npz"
        if not path.is_file():
            errors.append("missing_npz")
        else:
            with np.load(path, allow_pickle=False) as data:
                if str(data["sample_id"]) != sample_id:
                    errors.append("sample_id_mismatch")
                for name, shape in SHAPES.items():
                    values, mask = data[name], data[f"{name}_mask"]
                    if values.shape != shape:
                        errors.append(f"{name}_shape_{values.shape}")
                    if mask.shape != (50, 1) or not set(np.unique(mask)).issubset({0, 1}):
                        errors.append(f"{name}_mask_invalid")
                    if not np.isfinite(values).all():
                        errors.append(f"{name}_nan_or_inf")
                if data["time_start"].shape != (50,) or data["time_end"].shape != (50,):
                    errors.append("time_shape_invalid")
                elif not np.allclose(data["time_end"][:-1], data["time_start"][1:]) or np.any(data["time_end"] <= data["time_start"]):
                    errors.append("time_intervals_not_contiguous_positive")
                expected_label = truth.loc[sample_id, "annotation"]
                expected_regression = float(truth.loc[sample_id, "label"])
                if str(data["classification_label"]) != expected_label:
                    errors.append("classification_label_mismatch")
                if not np.isclose(float(data["regression_label"]), expected_regression, atol=1e-6):
                    errors.append("regression_label_mismatch")
        rows.append({"sample_id": sample_id, "status": "PASS" if not errors else "FAIL", "issues": ";".join(errors)})
    report_path = root / "stepE_alignment" / "data" / "aligned_50_validation.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    failed = sum(row["status"] != "PASS" for row in rows)
    summary = {
        "sample_count": len(rows),
        "structurally_valid_samples": len(rows) - failed,
        "structural_failures": failed,
        "complete_three_modality_samples": int((audit.status == "PASS").sum()),
        "incomplete_samples_retained_with_masks": int((audit.status != "PASS").sum()),
        "global_errors": global_errors,
        "validation_csv": str(report_path),
        "status": "PASS" if not global_errors and failed == 0 else "FAIL",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if global_errors or failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
