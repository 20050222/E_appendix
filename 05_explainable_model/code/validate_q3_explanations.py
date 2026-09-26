#!/usr/bin/env python3
"""Select and validate the Q3 temporal occlusion protocol on Q2 validation data."""

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


MODALITIES = ("text", "audio", "vision")
MODALITY_INDEX = {name: index for index, name in enumerate(MODALITIES)}


def import_q2(q2_root: Path):
    sys.path.insert(0, str(q2_root / "src"))
    return importlib.import_module("data"), importlib.import_module("train")


def find_aligned_pkl(workspace: Path) -> Path:
    candidates = list((workspace / "00_raw_readonly" / "E题数据").rglob("aligned_50.pkl"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one attachment-2 aligned_50.pkl, found {len(candidates)}")
    return candidates[0]


def load_raw_split(data_module, workspace: Path, q2_root: Path, split: str) -> dict[str, np.ndarray]:
    source = find_aligned_pkl(workspace)
    with source.open("rb") as handle:
        original = pickle.load(handle)
    adapted = data_module.adapt(original[split])
    stats = np.load(q2_root / "artifacts" / "normalization.npz")
    for modality, index in (("audio", 1), ("vision", 2)):
        adapted[modality] = np.clip(
            (adapted[modality] - stats[f"{modality}_mean"]) / stats[f"{modality}_std"],
            -5,
            5,
        ) * adapted["obs"][:, :, index, None]
    adapted["sample_id"] = np.asarray(original[split]["id"])
    return adapted


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=1, keepdims=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def window_specs(length: int, size: int, stride: int) -> list[tuple[int, int]]:
    starts = list(range(0, length - size + 1, stride))
    last = length - size
    if starts[-1] != last:
        starts.append(last)
    return [(start, start + size) for start in starts]


def batch_window_predict(train_module, model, d: dict[str, np.ndarray], device, index: int, modality: str, windows: list[tuple[int, int]]):
    count = len(windows)
    repeated = {key: np.repeat(value[index:index + 1], count, axis=0) for key, value in d.items() if isinstance(value, np.ndarray) and value.shape[0] == len(d["ids"])}
    kept = np.repeat(d["obs"][index:index + 1], count, axis=0)
    m = MODALITY_INDEX[modality]
    for j, (start, end) in enumerate(windows):
        kept[j, start:end, m] = False
    raw, logits, _ = train_module.predict(model, repeated, device, kept=kept, batch_size=min(128, count))
    return softmax(logits), train_module.project_intensity(raw, logits), logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--q2-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 means all validation samples")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.10, 0.20, 0.30])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    q2_root = (args.q2_root or workspace / "04_robust_model" / "question2").resolve()
    out = workspace / "05_explainable_model" / "data"
    data_module, train_module = import_q2(q2_root)
    device = train_module.device_setup(args.device, args.threads)
    checkpoint = q2_root / "artifacts" / "final_model.pt"
    model, state = train_module.load_checkpoint(checkpoint, device)
    d = load_raw_split(data_module, workspace, q2_root, "valid")
    n = len(d["ids"])
    limit = n if args.limit <= 0 else min(n, args.limit)
    rows: list[dict[str, Any]] = []
    for index in range(limit):
        one = {key: value[index:index + 1] for key, value in d.items() if isinstance(value, np.ndarray) and value.shape[0] == n}
        raw, logits, _ = train_module.predict(model, one, device, batch_size=1)
        probabilities = softmax(logits)
        target = int(np.argmax(probabilities[0]))
        full_probability = float(probabilities[0, target])
        full_intensity = float(train_module.project_intensity(raw, logits)[0])
        sample_id = str(d["sample_id"][index])
        for ratio in args.ratios:
            size = max(1, min(50, round(50 * ratio)))
            stride = max(1, min(size, round(50 * ratio / 2)))
            windows = window_specs(50, size, stride)
            for modality in MODALITIES:
                probabilities_occ, intensities_occ, logits_occ = batch_window_predict(train_module, model, d, device, index, modality, windows)
                cls_scores = np.abs(full_probability - probabilities_occ[:, target])
                reg_scores = np.abs(full_intensity - intensities_occ)
                top_indices_cls = np.argsort(-cls_scores, kind="stable")[:min(args.top_k, len(windows))]
                top_indices_reg = np.argsort(-reg_scores, kind="stable")[:min(args.top_k, len(windows))]
                seed_material = f"{sample_id}|{modality}|{ratio:.6f}".encode()
                seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
                rng = np.random.default_rng(seed)
                random_indices = rng.choice(len(windows), size=min(args.top_k, len(windows)), replace=False)
                rows.append({
                    "sample_id": sample_id,
                    "modality": modality,
                    "window_ratio": ratio,
                    "window_size": size,
                    "window_count": len(windows),
                    "topk_classification_effect": float(cls_scores[top_indices_cls].mean()),
                    "random_classification_effect": float(cls_scores[random_indices].mean()),
                    "classification_faithfulness_gap": float(cls_scores[top_indices_cls].mean() - cls_scores[random_indices].mean()),
                    "topk_regression_effect": float(reg_scores[top_indices_reg].mean()),
                    "random_regression_effect": float(reg_scores[random_indices].mean()),
                    "regression_faithfulness_gap": float(reg_scores[top_indices_reg].mean() - reg_scores[random_indices].mean()),
                    "topk_classification_flip_rate": float(np.mean(np.argmax(logits_occ[top_indices_cls], axis=1) != target)),
                    "random_classification_flip_rate": float(np.mean(np.argmax(logits_occ[random_indices], axis=1) != target)),
                    "device": str(device),
                })
    out.mkdir(parents=True, exist_ok=True)
    path = out / "q3_explanation_validation.csv"
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    summary: list[dict[str, Any]] = []
    for ratio in args.ratios:
        selected = [row for row in rows if float(row["window_ratio"]) == ratio]
        cls_gap = float(np.mean([row["classification_faithfulness_gap"] for row in selected]))
        reg_gap = float(np.mean([row["regression_faithfulness_gap"] for row in selected]))
        summary.append({"window_ratio": ratio, "classification_gap": cls_gap, "regression_gap": reg_gap, "combined_gap": cls_gap + reg_gap, "records": len(selected)})
    selected = max(summary, key=lambda row: (row["combined_gap"], -row["window_ratio"]))
    metadata = {
        "checkpoint": str(checkpoint.relative_to(workspace)),
        "checkpoint_sha256": sha256(checkpoint),
        "split": "附件2 valid",
        "sample_count_evaluated": limit,
        "top_k": args.top_k,
        "candidate_protocols": summary,
        "selected_window_ratio": selected["window_ratio"],
        "selected_window_size": max(1, round(50 * selected["window_ratio"])),
        "selection_rule": "maximize mean classification faithfulness gap + mean regression faithfulness gap against equal-count random windows",
        "status": "validated_on_attachment2_subset" if limit < n else "validated_on_attachment2_full_valid",
        "model_variant": state["config"].get("variant", "robust_norec"),
    }
    (out / "q3_explanation_protocol.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
