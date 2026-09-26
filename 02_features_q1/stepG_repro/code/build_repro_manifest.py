"""Build the unified extraction log and reproducibility manifest for Q1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_value(frame: pd.DataFrame, sample_id: str, column: str, default: str = "") -> str:
    if sample_id not in frame.index or column not in frame.columns:
        return default
    return str(frame.loc[sample_id, column])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    root = workspace / "02_features_q1"
    stepg = root / "stepG_repro"
    data_dir = stepg / "data"
    docs_dir = stepg / "docs"
    data_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    index = pd.read_csv(root / "stepA_index/data/sample_index.csv", dtype=str, keep_default_na=False)
    text = pd.read_csv(root / "stepB_text/data/text_extraction_log.csv", dtype=str, keep_default_na=False).set_index("sample_id")
    audio = pd.read_csv(root / "stepC_audio/data/audio_extraction_log.csv", dtype=str, keep_default_na=False).set_index("sample_id")
    vision = pd.read_csv(root / "stepD_vision/data/vision_extraction_log.csv", dtype=str, keep_default_na=False).set_index("sample_id")
    audit = pd.read_csv(root / "stepE_alignment/data/multimodal_sample_audit.csv", dtype=str, keep_default_na=False).set_index("sample_id")
    validation = pd.read_csv(root / "stepE_alignment/data/aligned_50_validation.csv", dtype=str, keep_default_na=False).set_index("sample_id")
    text_run = json.loads((root / "stepB_text/data/text_feature_run.json").read_text(encoding="utf-8"))
    audio_run = json.loads((root / "stepC_audio/data/audio_feature_run.json").read_text(encoding="utf-8"))
    vision_run = json.loads((root / "stepD_vision/data/vision_feature_run.json").read_text(encoding="utf-8"))
    config_path = data_dir / "processing_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config_hash = sha256(config_path)

    records: list[dict[str, object]] = []
    common = {
        "start_time": "",
        "end_time": "",
        "time_recording_status": "not_recorded_in_source_log",
        "config_sha256": config_hash,
        "error_type": "",
        "error_message": "",
    }
    for sample_id in index.sample_id.astype(str):
        records.append({**common, "sample_id": sample_id, "stage": "A", "modality": "index", "status": row_value(index.set_index("sample_id"), sample_id, "index_status"), "output_shape": "1 row", "valid_length": "", "mask_ratio": "", "duration_sec": row_value(index.set_index("sample_id"), sample_id, "duration_sec"), "tool_version": row_value(index.set_index("sample_id"), sample_id, "metadata_source"), "source_log": "stepA_index/data/sample_index.csv"})
        records.append({**common, "sample_id": sample_id, "stage": "B", "modality": "text", "status": row_value(text, sample_id, "text_status"), "output_shape": row_value(text, sample_id, "feature_shape"), "valid_length": row_value(text, sample_id, "text_valid_length"), "mask_ratio": "", "duration_sec": row_value(index.set_index("sample_id"), sample_id, "duration_sec"), "tool_version": f"{text_run.get('model_name')} / transformers {text_run.get('transformers_version')}", "source_log": "stepB_text/data/text_extraction_log.csv"})
        records.append({**common, "sample_id": sample_id, "stage": "C", "modality": "audio", "status": row_value(audio, sample_id, "status"), "output_shape": row_value(audio, sample_id, "feature_shape"), "valid_length": row_value(audio, sample_id, "valid_bins"), "mask_ratio": row_value(audio, sample_id, "mask_ratio"), "duration_sec": row_value(audio, sample_id, "video_duration_sec"), "tool_version": f"librosa {audio_run.get('librosa')} / FFmpeg {audio_run.get('ffmpeg_version', '')}", "source_log": "stepC_audio/data/audio_extraction_log.csv"})
        records.append({**common, "sample_id": sample_id, "stage": "D", "modality": "vision", "status": row_value(vision, sample_id, "status"), "output_shape": row_value(vision, sample_id, "feature_shape"), "valid_length": row_value(vision, sample_id, "valid_bins"), "mask_ratio": row_value(vision, sample_id, "mask_ratio"), "duration_sec": row_value(vision, sample_id, "video_duration_sec"), "tool_version": f"OpenCV {vision_run.get('opencv')} / MediaPipe {vision_run.get('mediapipe')}", "source_log": "stepD_vision/data/vision_extraction_log.csv"})
        records.append({**common, "sample_id": sample_id, "stage": "E", "modality": "aligned_50", "status": row_value(validation, sample_id, "status"), "output_shape": "(50,768)|(50,74)|(50,35)", "valid_length": f"text={row_value(audit, sample_id, 'text_valid_bins')};audio={row_value(audit, sample_id, 'audio_valid_bins')};vision={row_value(audit, sample_id, 'vision_valid_bins')}", "mask_ratio": "", "duration_sec": row_value(audit, sample_id, "duration_sec"), "tool_version": "build_aligned50.py + validate_aligned50.py", "source_log": "stepE_alignment/data/aligned_50_validation.csv"})
    unified = pd.DataFrame(records)
    unified.to_csv(data_dir / "unified_extraction_log.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    code_files = sorted((root).rglob("*.py"))
    tracked_files = [
        workspace / "01_manifest/csv_output/file_manifest.csv",
        root / "stepA_index/data/sample_index.csv",
        root / "stepB_text/data/text_feature_run.json",
        root / "stepC_audio/data/audio_feature_run.json",
        root / "stepD_vision/data/vision_feature_run.json",
        root / "stepE_alignment/data/aligned_50_schema.json",
        root / "stepF_validation/data/final_sample_validation.csv",
    ]
    tracked_hashes = {str(path.relative_to(workspace)): sha256(path) for path in tracked_files if path.is_file()}
    code_hashes = {str(path.relative_to(workspace)): sha256(path) for path in code_files if "__pycache__" not in path.parts}
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "sample_count": int(len(index)),
        "random_seed": 42,
        "config": {"path": str(config_path.relative_to(workspace)), "sha256": config_hash},
        "environment_files": [
            "01_manifest/env/environment_cpu.yaml",
            "01_manifest/env/environment_cuda4090.yaml",
        ],
        "run_metadata": {
            "text": text_run,
            "audio": audio_run,
            "vision": vision_run,
        },
        "tracked_input_and_result_sha256": tracked_hashes,
        "code_sha256": code_hashes,
        "rerun_order": [
            "Step A sample index",
            "Step B text features",
            "Step C audio features",
            "Step D vision features",
            "Step E aligned-50 merge",
            "Step F validation and traceability",
            "Step G manifest generation",
        ],
        "integrity_scope": "hashes identify the recorded inputs, schemas, logs and scripts; raw videos remain in 00_raw_readonly",
    }
    (data_dir / "reproducibility_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# 阶段 G：配置、日志和复现协议",
        "",
        "本目录固化问题 1 的处理配置、统一 extraction log、运行元数据和 SHA-256 清单。原始数据保持在 `00_raw_readonly/`，本阶段只读取并登记，不覆盖原始文件。",
        f"音频阶段记录的总耗时为 {float(audio_run.get('elapsed_sec', 0.0)):.2f} 秒，视觉阶段为 {float(vision_run.get('elapsed_sec', 0.0)):.2f} 秒；文本、索引与对齐阶段没有同口径总耗时。",
        "",
        "历史 Step A-E 逐样本日志没有保存准确的开始/结束时间，因此统一日志中的 `start_time`、`end_time` 留空，`time_recording_status=not_recorded_in_source_log`；不得将其解读为实测运行时间。`generated_at_utc` 仅表示本复现清单生成时间。",
        "统一日志目前没有纳入 manifest 的 SHA-256 清单，100 个单样本 aligned NPZ 也未逐文件登记哈希；`tracked_input_and_result_sha256` 仅覆盖其中列出的关键文件。",
        "",
        "## 文件",
        "",
        "- `data/processing_config.yaml`：三模态特征、时间桶、mask 和环境参数；",
        "- `data/unified_extraction_log.csv`：A-E 每条样本的统一处理状态；",
        "- `data/reproducibility_manifest.json`：配置、输入/结果文件和代码哈希；",
        "- `../stepF_validation/data/representative_traceability.csv`：典型样本逐桶回溯。",
        "- `image/publication/fig8_reproducibility_record_audit/fig8_reproducibility_record_audit.png`：阶段状态和复现证据覆盖图；",
        "",
        "## 复现顺序",
        "",
        "1. 创建 `01_manifest/env/environment_cpu.yaml` 或 `environment_cuda4090.yaml`；",
        "2. 检查 `01_manifest/csv_output/file_manifest.csv` 和 `00_raw_readonly/`；",
        "3. 依次运行 Step A、B、C、D、E；",
        "4. 运行 Step F 生成最终验收和典型回溯；",
        "5. 运行本脚本生成统一日志和哈希清单；",
        "6. 对比 `reproducibility_manifest.json` 中的哈希，确认复现输入和代码版本一致。",
        "",
    ]
    (docs_dir / "reproducibility_protocol.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "PASS", "sample_count": len(index), "unified_log_rows": len(unified), "config_sha256": config_hash, "tracked_files": len(tracked_hashes), "code_files": len(code_hashes)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
