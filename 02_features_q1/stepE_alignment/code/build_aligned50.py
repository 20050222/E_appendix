"""Join Step B/C/D features by sample_id and create the aligned-50 interface."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


SPECS = {
    "text": ("stepB_text/data/text_features/bert-base-uncased", "text_features", "text_mask", 768),
    "audio": ("stepC_audio/data/audio_features/audio74_v1", "audio_features", "audio_mask", 74),
    "vision": ("stepD_vision/data/vision_features/face35_v1", "vision_features", "vision_mask", 35),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    root = workspace / "02_features_q1"
    index = pd.read_csv(root / "stepA_index" / "data" / "sample_index.csv", dtype=str, keep_default_na=False)
    status_frames = {
        "audio": pd.read_csv(root / "stepC_audio" / "data" / "audio_extraction_log.csv", dtype=str, keep_default_na=False).set_index("sample_id"),
        "vision": pd.read_csv(root / "stepD_vision" / "data" / "vision_extraction_log.csv", dtype=str, keep_default_na=False).set_index("sample_id"),
    }
    text_validation = pd.read_csv(root / "stepB_text" / "data" / "text_feature_validation.csv", dtype=str, keep_default_na=False).set_index("sample_id")
    out = root / "stepE_alignment" / "data"
    feature_out = out / "aligned_50"
    feature_out.mkdir(parents=True, exist_ok=True)
    rows, bin_rows = [], []

    for _, item in index.iterrows():
        sample_id = item["sample_id"]
        duration = float(item["duration_sec"])
        arrays: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}
        statuses: dict[str, str] = {}
        for modality, (relative_dir, feature_key, mask_key, dim) in SPECS.items():
            path = root / relative_dir / f"{sample_id}.npz"
            data = np.zeros((50, dim), dtype=np.float32)
            mask = np.zeros((50, 1), dtype=np.uint8)
            status = "FAIL_MISSING_FILE"
            if path.is_file():
                try:
                    with np.load(path, allow_pickle=False) as archive:
                        if str(archive["sample_id"]) != sample_id:
                            raise ValueError("sample_id_mismatch")
                        data = archive[feature_key].astype(np.float32)
                        mask = archive[mask_key].astype(np.uint8)
                        count_key = {"text": "token_count_per_bin", "audio": "audio_frame_count", "vision": "vision_frame_count"}[modality]
                        valid_key = {"text": "text_valid_length", "audio": "audio_valid_length", "vision": "vision_valid_length"}[modality]
                        counts = archive[count_key]
                        valid_length = int(archive[valid_key])
                    if data.shape != (50, dim):
                        raise ValueError(f"shape_{data.shape}_expected_(50,{dim})")
                    if mask.shape != (50, 1) or not set(np.unique(mask)).issubset({0, 1}):
                        raise ValueError(f"invalid_mask_shape_or_values_{mask.shape}")
                    if not np.isfinite(data).all():
                        raise ValueError("feature_nan_or_inf")
                    if counts.shape != (50,) or not np.array_equal(mask[:, 0], (counts > 0).astype(mask.dtype)):
                        raise ValueError("mask_source_count_mismatch")
                    if valid_length != int(mask.sum()):
                        raise ValueError("valid_length_mask_mismatch")
                    status = "PASS"
                    if modality in status_frames:
                        source_log = status_frames[modality].loc[sample_id]
                        log_status = source_log["status"]
                        if log_status != "PASS":
                            detail = source_log.get("error", "")
                            status = f"FAIL_EXTRACTION:{log_status}" + (f"({detail})" if detail else "")
                    elif text_validation.loc[sample_id, "status"] != "PASS":
                        status = f"FAIL_VALIDATION:{text_validation.loc[sample_id, 'status']}"
                except Exception as exc:
                    data = np.zeros((50, dim), dtype=np.float32)
                    mask = np.zeros((50, 1), dtype=np.uint8)
                    status = f"FAIL_{type(exc).__name__}:{exc}"
            arrays[modality] = data
            masks[modality] = mask
            statuses[modality] = status

        time_start = np.linspace(0.0, duration, 50, endpoint=False, dtype=np.float32)
        time_end = np.linspace(0.0, duration, 51, dtype=np.float32)[1:]
        status = "PASS" if all(value == "PASS" for value in statuses.values()) else "INCOMPLETE"
        np.savez_compressed(
            feature_out / f"{sample_id}.npz",
            sample_id=np.asarray(sample_id),
            text=arrays["text"], audio=arrays["audio"], vision=arrays["vision"],
            text_mask=masks["text"], audio_mask=masks["audio"], vision_mask=masks["vision"],
            time_start=time_start, time_end=time_end,
            classification_label=np.asarray(item["annotation"]),
            regression_label=np.asarray(float(item["label"]), dtype=np.float32),
            text_alignment_method=np.asarray("uniform_tokens_non_timestamp"),
            multimodal_status=np.asarray(status),
        )
        valid = {name: int(mask.sum()) for name, mask in masks.items()}
        rows.append({"sample_id": sample_id, "status": status, **{f"{name}_status": value for name, value in statuses.items()},
                     "text_valid_bins": valid["text"], "audio_valid_bins": valid["audio"], "vision_valid_bins": valid["vision"],
                     "duration_sec": duration, "feature_path": str((feature_out / f"{sample_id}.npz").relative_to(workspace))})
        for b in range(50):
            for modality in SPECS:
                bin_rows.append({"sample_id": sample_id, "modality": modality, "bin_index": b,
                                 "bin_start_sec": float(time_start[b]), "bin_end_sec": float(time_end[b]),
                                 "valid": int(masks[modality][b, 0]),
                                 "alignment_interpretation": "approximate token sequence bin; not timestamp" if modality == "text" else "video-time bin",
                                 "status": statuses[modality]})

    pd.DataFrame(rows).to_csv(out / "multimodal_sample_audit.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    pd.DataFrame(bin_rows).to_csv(out / "multimodal_alignment_log.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    summary = {
        "sample_count": len(index),
        "complete_count": sum(row["status"] == "PASS" for row in rows),
        "incomplete_count": sum(row["status"] != "PASS" for row in rows),
        "modality_pass_counts": {name: sum(row[f"{name}_status"] == "PASS" for row in rows) for name in SPECS},
        "valid_bin_totals": {name: int(sum(row[f"{name}_valid_bins"] for row in rows)) for name in SPECS},
        "alignment": "50 equal-duration bins over each video's duration for audio/vision; text uses uniform token bins and is not word-timestamp aligned",
        "feature_dims": {"text": 768, "audio": 74, "vision": 35},
        "output_dir": str(feature_out),
    }
    (out / "multimodal_alignment_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    schema = {
        "schema": "aligned_50_multimodal_v1", "sample_id_rule": "video_id + '_' + clip_id",
        "time_bins": 50, "interval": "per-video equal-duration bins [kD/50,(k+1)D/50)",
        "modalities": {"text": {"shape": [50, 768], "source": "Step B BERT", "alignment": "uniform token index; not actual timestamp"},
                       "audio": {"shape": [50, 74], "source": "Step C audio74_v1", "alignment": "frame centers aggregated into video-time bins"},
                       "vision": {"shape": [50, 35], "source": "Step D face35_v1", "alignment": "sampled frames aggregated into video-time bins"}},
        "masks": "1 means a valid modality observation in the bin; 0 means unavailable or extraction failed, consult sample audit",
        "labels": {"classification_label": "annotation", "regression_label": "label"},
    }
    (out / "aligned_50_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
