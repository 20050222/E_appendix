"""Build the 100-sample index for E-question Step A.

This script is read-only with respect to 00_raw_readonly. It reads the
attachment-1 label sheet and video metadata, then writes:

  02_features_q1/stepA_index/data/sample_index.csv
  02_features_q1/stepA_index/docs/stepA_index_report.md

FFmpeg is used in stream-copy mode to inspect the video stream. The script
does not decode and store all video frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
import pandas as pd


REQUIRED_COLUMNS = {"video_id", "clip_id", "text", "label", "annotation"}
DURATION_RE = re.compile(r"Duration:\s+(\d+):(\d+):(\d+(?:\.\d+)?)")
FPS_RE = re.compile(r",\s*(\d+(?:\.\d+)?)\s+fps(?:[,\s]|$)")
FRAME_RE = re.compile(r"frame=\s*(\d+)")


def parse_args() -> argparse.Namespace:
    # .../E_workspace/02_features_q1/stepA_index/code/script.py -> E_workspace
    default_workspace = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=default_workspace,
        help="E_workspace directory; inferred from this script by default.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="Per-video FFmpeg metadata timeout in seconds.",
    )
    return parser.parse_args()


def stable_cell(value) -> str:
    """Convert an Excel cell to a stable ID component without changing text IDs."""
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def make_sample_id(video_id, clip_id) -> str:
    return f"{stable_cell(video_id)}_{stable_cell(clip_id)}"


def parse_duration(value: str) -> float | None:
    match = DURATION_RE.search(value)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_fps(value: str) -> float | None:
    match = FPS_RE.search(value)
    return float(match.group(1)) if match else None


def parse_frame_count(value: str) -> int | None:
    matches = FRAME_RE.findall(value)
    return int(matches[-1]) if matches else None


def inspect_video(path: Path, ffmpeg_exe: str, timeout_sec: int) -> dict:
    """Read stream metadata with FFmpeg copy mode; do not decode frames."""
    command = [
        ffmpeg_exe,
        "-hide_banner",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-c",
        "copy",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "duration_sec": "",
            "fps": "",
            "frame_count": "",
            "metadata_source": "ffmpeg_stream_copy",
            "frame_count_source": "",
            "metadata_status": "FAIL",
            "metadata_error": f"timeout_after_{timeout_sec}s",
        }
    output = f"{completed.stderr}\n{completed.stdout}"
    duration = parse_duration(output)
    fps = parse_fps(output)
    frame_count = parse_frame_count(output)
    errors = []
    if completed.returncode != 0:
        errors.append(f"ffmpeg_returncode_{completed.returncode}")
    if duration is None:
        errors.append("duration_unavailable")
    if fps is None:
        errors.append("fps_unavailable")
    if frame_count is None:
        errors.append("frame_count_unavailable")
    return {
        "duration_sec": f"{duration:.6f}" if duration is not None else "",
        "fps": f"{fps:.6f}" if fps is not None else "",
        "frame_count": frame_count if frame_count is not None else "",
        "metadata_source": "ffmpeg_stream_copy",
        "frame_count_source": "ffmpeg_stream_copy_summary" if frame_count is not None else "",
        "metadata_status": "PASS" if not errors else "PARTIAL",
        "metadata_error": ";".join(errors),
    }


def load_manifest(workspace: Path) -> dict[str, dict]:
    candidates = [
        workspace / "01_manifest" / "file_manifest.csv",
        workspace / "01_manifest" / "csv_output" / "file_manifest.csv",
    ]
    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None:
        return {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            row["relative_path"]: row
            for row in rows
            if row.get("extension", "").lower() == ".mp4"
            and row.get("attachment") == "附件1"
        }


def write_report(path: Path, *, label_count: int, video_count: int, duplicate_ids: list[str],
                 missing: list[str], ambiguous: dict[str, list[str]], rows: list[dict],
                 label_columns: list[str], ffmpeg_exe: str) -> None:
    status_counts = Counter(row["index_status"] for row in rows)
    metadata_counts = Counter(row["metadata_status"] for row in rows)
    failed = [row for row in rows if row["index_status"] != "PASS"]
    lines = [
        "# Step A 样本索引验收报告",
        "",
        f"> 生成时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        "> 原始数据：`00_raw_readonly/E题数据/`（只读）",
        "> 本步骤只读取标签和视频元信息，不解码保存全部视频帧，不提取文本/音频/视觉特征。",
        "",
        "## 结论",
        "",
        f"- 标签行数：**{label_count}**，预期 100。",
        f"- 附件 1 MP4 文件数：**{video_count}**，预期 100。",
        f"- `sample_id` 重复数：**{len(duplicate_ids)}**。",
        f"- 标签找不到视频：**{len(missing)}**。",
        f"- 一个标签对应多个视频候选：**{len(ambiguous)}**。",
        f"- 索引状态统计：`{dict(status_counts)}`。",
        f"- 视频元信息状态统计：`{dict(metadata_counts)}`。",
        "",
        "## 验收表",
        "",
        "| 检查项 | 结果 | 说明 |",
        "|---|---|---|",
        f"| 标签行数 | {'PASS' if label_count == 100 else 'FAIL'} | {label_count} |",
        f"| 视频数量 | {'PASS' if video_count == 100 else 'FAIL'} | {video_count} |",
        f"| `sample_id` 唯一 | {'PASS' if not duplicate_ids else 'FAIL'} | 重复 {len(duplicate_ids)} |",
        f"| 标签-视频一一对应 | {'PASS' if not missing and not ambiguous else 'FAIL'} | 缺失 {len(missing)}，歧义 {len(ambiguous)} |",
        f"| 元信息可读取 | {'PASS' if all(row['metadata_status'] == 'PASS' for row in rows) else 'PARTIAL'} | {dict(metadata_counts)} |",
        f"| 可回溯字段 | {'PASS' if all(row['sample_id'] and row['video_path'] for row in rows) else 'FAIL'} | sample_id + 原始路径均保留 |",
        "",
        "## 输入字段",
        "",
        f"`label-100.xlsx` 工作表 `label` 字段：`{', '.join(label_columns)}`",
        "",
        "## 元信息读取方式",
        "",
        f"- FFmpeg：`{ffmpeg_exe}`",
        "- 命令采用 `-map 0:v:0 -c copy -f null -`，读取视频流摘要，不将全部帧载入内存。",
        "- `frame_count` 来源标记为 `ffmpeg_stream_copy_summary`；后续若需要逐帧视觉抽取，应以实际解码帧时间戳为准。",
        "",
        "## 失败或待核验样本",
        "",
    ]
    if failed:
        lines += ["| sample_id | status | failure_reason | video_path |", "|---|---|---|---|"]
        lines += [
            f"| `{row['sample_id']}` | {row['index_status']} | {row['failure_reason']} | `{row['video_path']}` |"
            for row in failed
        ]
    else:
        lines.append("无。")
    lines += [
        "",
        "## 输出",
        "",
        "- `02_features_q1/stepA_index/data/sample_index.csv`：100 条标签样本的索引和视频元信息。",
        "- 本报告不代表文本、音频或视觉特征已经生成；这些属于后续 Step B-D。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    raw = workspace / "00_raw_readonly" / "E题数据"
    step_root = workspace / "02_features_q1" / "stepA_index"
    data_output = step_root / "data"
    report_path = step_root / "docs" / "stepA_index_report.md"
    data_output.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw.is_dir():
        raise FileNotFoundError(f"raw data directory not found: {raw}")

    label_path = next(raw.rglob("label-100.xlsx"), None)
    if label_path is None:
        raise FileNotFoundError("label-100.xlsx not found under raw data")
    labels = pd.read_excel(label_path, sheet_name="label")
    missing_columns = REQUIRED_COLUMNS - set(labels.columns)
    if missing_columns:
        raise ValueError(f"label sheet missing columns: {sorted(missing_columns)}")

    video_root = label_path.parent
    video_paths = sorted(video_root.rglob("*.mp4"), key=lambda p: p.as_posix())
    candidates: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in video_paths:
        candidates[(path.parent.name, stable_cell(path.stem))].append(path)

    manifest = load_manifest(workspace)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    rows = []
    duplicate_ids = []
    seen_ids = set()
    missing = []
    ambiguous = {}

    for row_number, row in labels.iterrows():
        video_id = stable_cell(row["video_id"])
        clip_id = stable_cell(row["clip_id"])
        sample_id = make_sample_id(video_id, clip_id)
        failure_reasons = []
        if sample_id in seen_ids:
            duplicate_ids.append(sample_id)
            failure_reasons.append("duplicate_sample_id")
        seen_ids.add(sample_id)
        choices = candidates.get((video_id, clip_id), [])
        if not choices:
            missing.append(sample_id)
            failure_reasons.append("video_not_found")
        elif len(choices) > 1:
            ambiguous[sample_id] = [str(path) for path in choices]
            failure_reasons.append("multiple_video_candidates")
        video_path = choices[0] if choices else None
        metadata = {
            "duration_sec": "",
            "fps": "",
            "frame_count": "",
            "metadata_source": "",
            "frame_count_source": "",
            "metadata_status": "FAIL" if video_path is None else "PENDING",
            "metadata_error": "video_not_found" if video_path is None else "",
        }
        video_relative_path = ""
        video_checksum = ""
        video_file_size = ""
        if video_path is not None:
            metadata = inspect_video(video_path, ffmpeg_exe, args.timeout_sec)
            if metadata["metadata_status"] != "PASS":
                failure_reasons.append(f"video_metadata_{metadata['metadata_status'].lower()}")
            video_relative_path = video_path.relative_to(raw).as_posix()
            manifest_row = manifest.get(video_relative_path, {})
            video_checksum = manifest_row.get("checksum", "")
            video_file_size = manifest_row.get("file_size", str(video_path.stat().st_size))

        index_status = "PASS" if not failure_reasons and metadata["metadata_status"] == "PASS" else "FAIL"
        rows.append({
            "sample_id": sample_id,
            "video_id": video_id,
            "clip_id": clip_id,
            "video_path": str(video_path) if video_path else "",
            "video_relative_path": video_relative_path,
            "video_exists": "1" if video_path else "0",
            "video_file_size": video_file_size,
            "video_sha256": video_checksum,
            "duration_sec": metadata["duration_sec"],
            "fps": metadata["fps"],
            "frame_count": metadata["frame_count"],
            "frame_count_source": metadata["frame_count_source"],
            "raw_text": str(row["text"]) if not pd.isna(row["text"]) else "",
            "label": row["label"],
            "annotation": row["annotation"],
            "index_status": index_status,
            "failure_reason": ";".join(failure_reasons + ([metadata["metadata_error"]] if metadata["metadata_error"] else [])),
            "metadata_status": metadata["metadata_status"],
            "metadata_source": metadata["metadata_source"],
            "source_label_file": str(label_path),
            "source_label_row": row_number + 2,
        })

    fieldnames = [
        "sample_id", "video_id", "clip_id", "video_path", "video_relative_path",
        "video_exists", "video_file_size", "video_sha256", "duration_sec", "fps",
        "frame_count", "frame_count_source", "raw_text", "label", "annotation",
        "index_status", "failure_reason", "metadata_status", "metadata_source",
        "source_label_file", "source_label_row",
    ]
    sample_index_path = data_output / "sample_index.csv"
    with sample_index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        # Quote every field so Excel does not reinterpret video IDs such as
        # `-HwX2H8Z4hY` as formulas and rewrite them as `#NAME?`.
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    write_report(
        report_path,
        label_count=len(labels),
        video_count=len(video_paths),
        duplicate_ids=duplicate_ids,
        missing=missing,
        ambiguous=ambiguous,
        rows=rows,
        label_columns=list(labels.columns),
        ffmpeg_exe=ffmpeg_exe,
    )
    print(json.dumps({
        "sample_index": str(sample_index_path),
        "report": str(report_path),
        "label_count": len(labels),
        "video_count": len(video_paths),
        "pass_count": sum(row["index_status"] == "PASS" for row in rows),
        "fail_count": sum(row["index_status"] != "PASS" for row in rows),
        "duplicate_sample_ids": duplicate_ids,
        "missing_videos": missing,
        "ambiguous_matches": sorted(ambiguous),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
