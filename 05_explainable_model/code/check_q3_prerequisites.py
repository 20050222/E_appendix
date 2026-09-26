#!/usr/bin/env python3
"""Check the artifacts required before running Q3 explanations.

This command is intentionally a gate: it does not train, infer, or modify raw
data. It records exactly which Q2 artifacts are available in the workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--q2-root", type=Path, default=None)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    q2 = (args.q2_root or workspace / "04_robust_model" / "question2").resolve()
    q3 = workspace / "05_explainable_model"

    required = {
        "checkpoint": [q2 / "artifacts" / "final_model.pt", q2 / "final_model.pt"],
        "normalization": [q2 / "artifacts" / "normalization.npz", q2 / "normalization.npz"],
        "inference_source": [q2 / "src" / "infer.py", q2 / "infer.py"],
        "model_source": [q2 / "src" / "model.py", q2 / "model.py"],
        "model_config": [q2 / "artifacts" / "final_model_config.json", q2 / "config.yaml", q2 / "artifacts" / "config.yaml"],
    }
    found: dict[str, dict[str, object]] = {}
    for label, candidates in required.items():
        existing = next((path for path in candidates if path.is_file()), None)
        if label == "model_config" and existing is None and (q2 / "artifacts" / "final_model.pt").is_file():
            # Q2 stores the network config and BERT config inside final_model.pt.
            existing = q2 / "artifacts" / "final_model.pt"
        found[label] = {
            "status": "present" if existing else "missing",
            "path": str(existing.relative_to(workspace)) if existing else "",
            "sha256": sha256(existing) if existing else "",
            "size_bytes": existing.stat().st_size if existing else "",
            "source": "embedded_in_checkpoint" if label == "model_config" and existing and not any(path.is_file() for path in candidates) else "file",
        }

    attachment_manifest = q3 / "data" / "attachment4_manifest.csv"
    schema = q3 / "data" / "attachment4_schema.json"
    support = {
        "attachment4_manifest": {"status": "present" if attachment_manifest.is_file() else "missing", "path": str(attachment_manifest.relative_to(workspace))},
        "attachment4_schema": {"status": "present" if schema.is_file() else "missing", "path": str(schema.relative_to(workspace))},
    }
    ready = all(item["status"] == "present" for item in found.values()) and all(item["status"] == "present" for item in support.values())
    result = {
        "ready_for_q3_inference": ready,
        "required_q2_artifacts": found,
        "q3_inputs": support,
        "q2_root": str(q2),
        "note": "A missing checkpoint or preprocessing artifact blocks prediction and explanation generation.",
    }
    out = q3 / "data"
    out.mkdir(parents=True, exist_ok=True)
    (out / "q3_prerequisite_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 问题3启动前置检查", "", f"- 是否可以进入问题3推理：`{'是' if ready else '否'}`", ""]
    for label, item in found.items():
        lines.append(f"- {label}: `{item['status']}`" + (f"（{item['path']}）" if item["path"] else ""))
    for label, item in support.items():
        lines.append(f"- {label}: `{item['status']}`（{item['path']}）")
    if not ready:
        lines.extend(["", "当前不能运行附件4预测；请先补齐缺失的第二问模型资产。"])
    (out / "q3_prerequisite_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
