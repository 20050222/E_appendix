#!/usr/bin/env python3
"""Run Question 3 predictions and perturbation explanations with Q2 assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Some supplied PKL files were serialized by NumPy 2.x. The Q2 CPU
# environment uses NumPy 1.26, where these private module names moved. The
# aliases only affect unpickling in this process and never rewrite raw files.
try:  # pragma: no cover - depends on the local NumPy major version
    import numpy.core.numeric as _numpy_numeric
    sys.modules.setdefault("numpy._core.numeric", _numpy_numeric)
    import numpy.core.multiarray as _numpy_multiarray
    sys.modules.setdefault("numpy._core.multiarray", _numpy_multiarray)
except Exception:
    pass

from q3_explanation_core import compute_modal_contributions, make_window_specs, mask_fraction


LABELS = ("Negative", "Neutral", "Positive")
MODALITIES = ("text", "audio", "vision")
MODALITY_INDEX = {name: index for index, name in enumerate(MODALITIES)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_attachment4(workspace: Path) -> Path:
    base = workspace / "00_raw_readonly" / "E题数据"
    candidates = []
    for path in base.rglob("对齐版本"):
        if path.is_dir() and len(list(path.glob("*.pkl"))) == 20 and len(list((path / "videos").glob("*.mp4"))) == 20:
            candidates.append(path)
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one attachment-4 aligned directory, found {len(candidates)}")
    return candidates[0]


def import_q2(q2_root: Path):
    src = q2_root / "src"
    sys.path.insert(0, str(src))
    data = importlib.import_module("data")
    train = importlib.import_module("train")
    return data, train


def load_attachment4_aligned(data_module, folder: Path, q2_root: Path) -> dict[str, np.ndarray]:
    """Adapt attachment-4 single-sample PKLs to the Q2 batched interface."""
    rows = []
    stats = np.load(q2_root / "artifacts" / "normalization.npz")
    for path in sorted(folder.glob("*.pkl")):
        with path.open("rb") as handle:
            sample = pickle.load(handle)
        batched = {}
        for key in ("text_bert", "audio", "vision"):
            value = np.asarray(sample[key])
            batched[key] = value[None, ...]
        adapted = data_module.adapt(batched)
        for modality, index in (("audio", 1), ("vision", 2)):
            adapted[modality] = np.clip(
                (adapted[modality] - stats[f"{modality}_mean"]) / stats[f"{modality}_std"],
                -5,
                5,
            ) * adapted["obs"][:, :, index, None]
        adapted["sample_id"] = np.asarray([str(sample.get("id", path.stem))])
        rows.append(adapted)
    if not rows:
        raise ValueError(f"no attachment-4 PKL files found in {folder}")
    return {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}


def predict_one(train_module, model, d: dict[str, np.ndarray], device, kept: np.ndarray | None = None) -> dict[str, Any]:
    raw, logits, weights = train_module.predict(model, d, device, kept=kept, batch_size=max(1, len(d["ids"])))
    probabilities = np.asarray(__import__("scipy").special.softmax(logits, axis=1), dtype=np.float64)
    projected = train_module.project_intensity(raw, logits)
    return {"raw": raw, "logits": logits, "probabilities": probabilities, "intensity": projected, "weights": weights}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--q2-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--window-ratio", type=float, default=0.20)
    parser.add_argument("--stride-ratio", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    q2_root = (args.q2_root or workspace / "04_robust_model" / "question2").resolve()
    q3_root = workspace / "05_explainable_model"
    out = q3_root / "data"
    q2_src = q2_root / "src"
    checkpoint = q2_root / "artifacts" / "final_model.pt"
    normalization = q2_root / "artifacts" / "normalization.npz"
    if not checkpoint.is_file() or not normalization.is_file():
        raise FileNotFoundError("Q2 checkpoint or normalization.npz is missing")
    data_module, train_module = import_q2(q2_root)
    device = train_module.device_setup(args.device, args.threads)
    model, state = train_module.load_checkpoint(checkpoint, device)
    folder = find_attachment4(workspace)
    d = load_attachment4_aligned(data_module, folder, q2_root)
    n = len(d["sample_id"])
    if n != 20:
        raise ValueError(f"attachment-4 aligned input must contain 20 samples, got {n}")
    full = predict_one(train_module, model, d, device)
    target_classes = np.argmax(full["probabilities"], axis=1)

    ablated: dict[str, dict[str, Any]] = {}
    for modality in MODALITIES:
        kept = d["obs"].copy()
        kept[:, :, MODALITY_INDEX[modality]] = False
        ablated[modality] = predict_one(train_module, model, d, device, kept=kept)

    contribution_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    bins = d["content"].shape[1]
    window_size = max(1, min(bins, round(bins * args.window_ratio)))
    stride = max(1, min(window_size, round(bins * args.stride_ratio)))
    windows = make_window_specs(bins, window_size, stride)
    for index, sample_id in enumerate(d["sample_id"]):
        sample_id = str(sample_id)
        contribution = compute_modal_contributions(
            full["probabilities"][index],
            {m: ablated[m]["probabilities"][index] for m in MODALITIES},
            float(full["intensity"][index]),
            {m: float(ablated[m]["intensity"][index]) for m in MODALITIES},
            predicted_class=int(target_classes[index]),
        )
        prediction_row: dict[str, Any] = {
            "sample_id": sample_id,
            "source_version": "attachment4_aligned_50",
            "predicted_class_index": contribution.predicted_class,
            "predicted_class": LABELS[contribution.predicted_class],
            "predicted_intensity": float(full["intensity"][index]),
            "prob_negative": float(full["probabilities"][index, 0]),
            "prob_neutral": float(full["probabilities"][index, 1]),
            "prob_positive": float(full["probabilities"][index, 2]),
            "model_variant": state["config"].get("variant", "robust_norec"),
            "model_seed": state["config"].get("seed", 42),
            "best_epoch": state.get("best_epoch", ""),
        }
        for modality in MODALITIES:
            prediction_row[f"fusion_weight_{modality}"] = float(full["weights"][index, MODALITY_INDEX[modality]])
        prediction_rows.append(prediction_row)
        contribution_row: dict[str, Any] = {
            "sample_id": sample_id,
            "predicted_class": LABELS[contribution.predicted_class],
            "classification_main_modality": contribution.classification_main_modality,
            "regression_main_modality": contribution.regression_main_modality,
            "classification_status": contribution.classification_status,
            "regression_status": contribution.regression_status,
        }
        for modality in MODALITIES:
            contribution_row[f"classification_signed_{modality}"] = contribution.classification_signed[modality]
            contribution_row[f"classification_abs_{modality}"] = contribution.classification_abs[modality]
            contribution_row[f"classification_normalized_{modality}"] = contribution.classification_normalized[modality]
            contribution_row[f"regression_signed_{modality}"] = contribution.regression_signed[modality]
            contribution_row[f"regression_abs_{modality}"] = contribution.regression_abs[modality]
            contribution_row[f"regression_normalized_{modality}"] = contribution.regression_normalized[modality]
        contribution_rows.append(contribution_row)

        for modality in MODALITIES:
            m = MODALITY_INDEX[modality]
            mask = d["obs"][index, :, m] & d["content"][index]
            for window in windows:
                kept = d["obs"][index:index + 1].copy()
                kept[:, window["start"]:window["end"], m] = False
                local_d = {key: value[index:index + 1] for key, value in d.items() if isinstance(value, np.ndarray) and value.shape[0] == n}
                local = predict_one(train_module, model, local_d, device, kept=kept)
                evidence_rows.append({
                    "sample_id": sample_id,
                    "modality": modality,
                    "start_bin": window["start"],
                    "end_bin_exclusive": window["end"],
                    "normalized_start": window["start"] / bins,
                    "normalized_end": window["end"] / bins,
                    "position_unit": "normalized_video_time" if modality in ("audio", "vision") else "token_bin",
                    "target_probability_after_occlusion": float(local["probabilities"][0, contribution.predicted_class]),
                    "classification_delta_signed": float(full["probabilities"][index, contribution.predicted_class] - local["probabilities"][0, contribution.predicted_class]),
                    "classification_delta_abs": float(abs(full["probabilities"][index, contribution.predicted_class] - local["probabilities"][0, contribution.predicted_class])),
                    "intensity_after_occlusion": float(local["intensity"][0]),
                    "regression_delta_signed": float(full["intensity"][index] - local["intensity"][0]),
                    "regression_delta_abs": float(abs(full["intensity"][index] - local["intensity"][0])),
                    "valid_fraction": mask_fraction(mask, window["start"], window["end"]),
                    "evidence_status": "token_index_only" if modality == "text" else "normalized_time_only",
                })

    evidence_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in evidence_rows:
        evidence_by_key.setdefault((row["sample_id"], row["modality"]), []).append(row)
    for rows in evidence_by_key.values():
        for metric, suffix in (("classification_delta_abs", "classification_rank"), ("regression_delta_abs", "regression_rank")):
            for rank, row in enumerate(sorted(rows, key=lambda value: (-value[metric], value["start_bin"])), start=1):
                row[suffix] = rank
        for row in rows:
            row["selected_classification"] = row["classification_rank"] <= args.top_k
            row["selected_regression"] = row["regression_rank"] <= args.top_k

    write_csv(out / "q3_prediction.csv", prediction_rows)
    write_csv(out / "q3_modal_contribution.csv", contribution_rows)
    write_csv(out / "q3_temporal_evidence.csv", evidence_rows)
    metadata = {
        "checkpoint": str(checkpoint.relative_to(workspace)),
        "checkpoint_sha256": sha256(checkpoint),
        "normalization": str(normalization.relative_to(workspace)),
        "normalization_sha256": sha256(normalization),
        "q2_source": str(q2_src.relative_to(workspace)),
        "attachment4_aligned_dir": str(folder.relative_to(workspace)),
        "sample_count": n,
        "window_ratio": args.window_ratio,
        "window_size": window_size,
        "stride_ratio": args.stride_ratio,
        "stride": stride,
        "top_k": args.top_k,
        "device": str(device),
        "status": "attachment4_inference_completed; explanation_protocol_not_validated_on_attachment2" ,
    }
    protocol_path = out / "q3_explanation_protocol.json"
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if (
            protocol.get("status") == "validated_on_attachment2_full_valid"
            and float(protocol.get("selected_window_ratio", -1)) == float(args.window_ratio)
            and int(protocol.get("selected_window_size", -1)) == window_size
        ):
            metadata["status"] = "attachment4_inference_completed_with_full_validated_explanation_protocol"
            metadata["protocol"] = str(protocol_path.relative_to(workspace))
            metadata["protocol_sha256"] = sha256(protocol_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / "q3_run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
