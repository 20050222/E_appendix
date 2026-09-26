#!/usr/bin/env python3
"""Audit the attachment-4 inputs without changing raw files.

The script only reads the aligned/unaligned PKL files and the bundled videos,
then writes a manifest and schema summary under 05_explainable_model/data.
It intentionally does not train or run a prediction model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_ALIGNED = {
    "audio": [50, 74],
    "vision": [50, 35],
    "text": [50, 768],
    "text_bert": [3, 50],
}
EXPECTED_UNALIGNED = {
    "audio": [500, 74],
    "vision": [500, 35],
    "text": [50, 768],
    "text_bert": [3, 50],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def numeric_stats(value: Any) -> tuple[int | str, int | str, int | str]:
    if not isinstance(value, np.ndarray) or not np.issubdtype(value.dtype, np.number):
        return "", "", ""
    return int(np.isnan(value).sum()), int(np.isinf(value).sum()), int(np.all(value == 0, axis=1).sum()) if value.ndim == 2 else ""


def find_root(workspace: Path) -> Path:
    base = workspace / "00_raw_readonly" / "E题数据"
    candidates = [
        p for p in base.rglob("*")
        if p.is_dir()
        and p.name == "对齐版本"
        and (p.parent / "未对齐版本").is_dir()
        and len(list(p.glob("*.pkl"))) == 20
        and len(list((p / "videos").glob("*.mp4"))) == 20
    ]
    if not candidates:
        raise FileNotFoundError(f"cannot find attachment-4 aligned directory below {base}")
    return candidates[0].parent


def audit(workspace: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = find_root(workspace)
    rows: list[dict[str, Any]] = []
    schema: dict[str, Any] = {"root": str(root), "versions": {}}
    for version in ("对齐版本", "未对齐版本"):
        version_dir = root / version
        pkl_files = sorted(version_dir.glob("*.pkl"))
        video_dir = version_dir / "videos"
        expected = EXPECTED_ALIGNED if version == "对齐版本" else EXPECTED_UNALIGNED
        schema["versions"][version] = {
            "pkl_count": len(pkl_files),
            "video_count": len(list(video_dir.glob("*.mp4"))),
            "expected_shapes": expected,
            "fields_seen": {},
        }
        for path in pkl_files:
            with path.open("rb") as f:
                obj = pickle.load(f)
            if not isinstance(obj, dict):
                raise TypeError(f"{path} is not a dictionary")
            sample_id = str(obj.get("id", path.stem)).strip()
            row: dict[str, Any] = {
                "version": version,
                "file": path.name,
                "sample_id": sample_id,
                "pkl_relative_path": str(path.relative_to(workspace)),
                "pkl_size_bytes": path.stat().st_size,
                "pkl_sha256": sha256(path),
                "video_relative_path": str((video_dir / f"{path.stem}.mp4").relative_to(workspace)),
                "video_exists": (video_dir / f"{path.stem}.mp4").is_file(),
                "raw_text_chars": len(str(obj.get("raw_text", ""))),
                "audio_shape": list(np.asarray(obj["audio"]).shape) if "audio" in obj else "",
                "vision_shape": list(np.asarray(obj["vision"]).shape) if "vision" in obj else "",
                "text_shape": list(np.asarray(obj["text"]).shape) if "text" in obj else "",
                "text_bert_shape": list(np.asarray(obj["text_bert"]).shape) if "text_bert" in obj else "",
                "audio_lengths": obj.get("audio_lengths", ""),
                "vision_lengths": obj.get("vision_lengths", ""),
            }
            for field, value in obj.items():
                arr = np.asarray(value) if isinstance(value, np.ndarray) else None
                schema["versions"][version]["fields_seen"].setdefault(field, {
                    "python_type": type(value).__name__,
                    "shape": list(arr.shape) if arr is not None else [],
                    "dtype": str(arr.dtype) if arr is not None else "",
                })
                if field in ("audio", "vision", "text", "text_bert"):
                    nan_count, inf_count, zero_rows = numeric_stats(arr)
                    row[f"{field}_nan"] = nan_count
                    row[f"{field}_inf"] = inf_count
                    if field in ("audio", "vision"):
                        row[f"{field}_zero_rows"] = zero_rows
            text_bert = np.asarray(obj.get("text_bert", []))
            row["text_bert_attention_sum"] = int(text_bert[1].sum()) if text_bert.ndim == 2 and text_bert.shape[0] >= 2 else ""
            row["shape_status"] = "pass" if all(row.get(f"{k}_shape") == v for k, v in expected.items()) else "check"
            row["id_status"] = "pass" if sample_id == path.stem else "check"
            rows.append(row)
    return rows, schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    out = workspace / "05_explainable_model" / "data"
    out.mkdir(parents=True, exist_ok=True)
    rows, schema = audit(workspace)
    columns = sorted({key for row in rows for key in row})
    with (out / "attachment4_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (out / "attachment4_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {version: sum(row["version"] == version for row in rows) for version in ("对齐版本", "未对齐版本")}
    report = [
        "# 附件4输入审计报告",
        "",
        "本报告由 `code/audit_attachment4.py` 只读生成；未训练模型、未修改原始 PKL 或视频。",
        "",
        f"- 对齐版本 PKL：{counts['对齐版本']} 个；未对齐版本 PKL：{counts['未对齐版本']} 个。",
        "- 当前审计确认：样本 ID 与文件名一致性、字段/shape、NaN/Inf、全零行、文本 attention mask 和视频文件存在性。",
        "- 对齐版本没有显式 `audio_lengths`/`vision_lengths` 字段；其有效位置需要沿用第二问的 mask 恢复规则，不能仅凭全零值断言缺失。",
        "- 附件4无真实标签，因此本阶段不计算 Accuracy、F1、MAE 或 Pearson。",
        "- 最终推理仍依赖第二问的 `final_model.pt`、`normalization.npz`、模型配置和推理代码；这些文件当前未在 `04_robust_model` 目录发现。",
    ]
    (out / "attachment4_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(out / "attachment4_manifest.csv"), "schema": str(out / "attachment4_schema.json"), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
