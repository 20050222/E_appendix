"""Audit XLSX labels against attachment-2 aligned/unaligned PKL labels.

The script is read-only with respect to 00_raw_readonly. It writes only:
  - label_summary.csv
  - label_consistency_report.md

It avoids materializing the large feature arrays in aligned_50.pkl and
unaligned_50.pkl. Only IDs and the small label arrays are retained.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


VALID_MODES = {"train", "valid", "test"}
LABEL_KEYS = ("classification_labels", "regression_labels")


class ArrayProxy:
    """Discard large ndarray payloads while retaining small label arrays."""

    def __init__(self, *_args):
        self.state = None

    def __setstate__(self, state):
        shape, dtype, fortran, data = state[1], state[2], state[3], state[4]
        keep_data = data if isinstance(data, (bytes, bytearray)) and len(data) <= 2_000_000 else None
        self.state = (tuple(shape), dtype, fortran, keep_data)


class LightweightUnpickler(pickle.Unpickler):
    """Replace NumPy ndarray reconstruction with ArrayProxy."""

    def find_class(self, module, name):
        if module == "numpy.core.multiarray" and name == "_reconstruct":
            return ArrayProxy
        return super().find_class(module, name)


def parse_args():
    default_workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=default_workspace,
        help="E_workspace directory; defaults to the directory inferred from this script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory; defaults to WORKSPACE/01_manifest/csv_output.",
    )
    return parser.parse_args()


def clip_text(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def make_sample_id(video_id, clip_id) -> str:
    return f"{str(video_id).strip()}_{clip_text(clip_id)}"


def normalize_pkl_id(value) -> str:
    value = str(value).strip()
    return value.replace("$_$", "_", 1)


def read_feature_labels(path: Path) -> dict:
    """Read split IDs and labels without materializing feature arrays."""
    with path.open("rb") as handle:
        data = LightweightUnpickler(handle).load()

    result = {}
    for split, split_data in data.items():
        result[split] = {"id": list(split_data.get("id", []))}
        for key in LABEL_KEYS:
            proxy = split_data.get(key)
            if proxy is None or proxy.state is None or proxy.state[3] is None:
                result[split][key] = None
                continue
            shape, dtype, _fortran, raw = proxy.state
            result[split][key] = np.frombuffer(raw, dtype=dtype).reshape(shape).tolist()
    return result


def result_status(ok: bool) -> str:
    return "PASS" if bool(ok) else "FAIL"


def add(rows, source, scope, metric, value, details=""):
    rows.append(
        {
            "source": source,
            "scope": scope,
            "metric": metric,
            "value": value,
            "details": details,
        }
    )


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    raw = workspace / "00_raw_readonly" / "E题数据"
    output = (args.output_dir or workspace / "01_manifest" / "csv_output").resolve()
    output.mkdir(parents=True, exist_ok=True)

    label100_path = next(raw.rglob("label-100.xlsx"))
    label_path = raw / "附件2-数据集特征文件" / "label.xlsx"
    df100 = pd.read_excel(label100_path, sheet_name="label")
    labels = pd.read_excel(label_path, sheet_name="label")
    for frame in (df100, labels):
        frame["sample_id_expected"] = [
            make_sample_id(video, clip)
            for video, clip in zip(frame["video_id"], frame["clip_id"])
        ]

    rows = []
    for source, frame in (("label-100.xlsx", df100), ("label.xlsx", labels)):
        add(rows, source, "label", "row_count", len(frame))
        add(
            rows,
            source,
            "label",
            "sample_id_formula",
            result_status(frame["sample_id_expected"].notna().all()),
            'sample_id=video_id+"_"+clip_id',
        )
        add(rows, source, "label", "duplicate_sample_id_count", int(frame["sample_id_expected"].duplicated().sum()))
        for label, count in frame["annotation"].value_counts(dropna=False).items():
            add(rows, source, "annotation", str(label), int(count))

    for mode, count in labels["mode"].value_counts(dropna=False).items():
        add(rows, "label.xlsx", "mode", str(mode), int(count))

    label_by_id = {
        item.sample_id_expected: item for item in labels.itertuples(index=False)
    }
    split_sets = {}
    classification_pairs = Counter()
    regression_total = regression_mismatch = 0

    for version in ("aligned", "unaligned"):
        feature_path = raw / "附件2-数据集特征文件" / f"{version}_50.pkl"
        parsed = read_feature_labels(feature_path)
        for split, split_data in parsed.items():
            ids = [normalize_pkl_id(item) for item in split_data["id"]]
            split_sets[(version, split)] = set(ids)
            matched = [label_by_id.get(item) for item in ids]
            missing = [item for item, row in zip(ids, matched) if row is None]
            mode_matches = not missing and all(getattr(row, "mode", None) == split for row in matched)
            source = f"{version}_50.pkl"
            add(rows, source, split, "id_count", len(ids))
            add(rows, source, split, "label_lookup_missing", len(missing), repr(missing[:3]))
            add(rows, source, split, "mode_matches_split", result_status(mode_matches), split)

            class_values = split_data["classification_labels"]
            regression_values = split_data["regression_labels"]
            if class_values is None or regression_values is None:
                add(rows, source, split, "label_array_alignment", "FAIL", "label array unavailable")
                continue

            regression_errors = 0
            for sample_id, class_value, regression_value, row in zip(
                ids, class_values, regression_values, matched
            ):
                if row is None:
                    continue
                classification_pairs[(int(round(float(class_value))), str(row.annotation))] += 1
                regression_total += 1
                if not math.isclose(float(regression_value), float(row.label), abs_tol=1e-5, rel_tol=0):
                    regression_errors += 1
            regression_mismatch += regression_errors
            add(rows, source, split, "regression_label_compared", len([x for x in matched if x is not None]), "abs_tol=1e-5")
            add(rows, source, split, "regression_mismatch_count", regression_errors, "abs_tol=1e-5")

    by_class = defaultdict(Counter)
    for (class_value, annotation), count in classification_pairs.items():
        by_class[class_value][annotation] += count
    mapping = {
        class_value: counts.most_common(1)[0][0]
        for class_value, counts in by_class.items()
        if counts
    }
    mapping_ok = (
        all(len(counts) == 1 for counts in by_class.values())
        and set(mapping) == {0, 1, 2}
        and set(mapping.values()) == {"Positive", "Neutral", "Negative"}
    )
    for class_value in sorted(by_class):
        add(
            rows,
            "label.xlsx vs attachment2",
            "classification",
            f"class_{class_value}_annotation_mapping",
            mapping.get(class_value, ""),
            json.dumps(dict(by_class[class_value]), ensure_ascii=False),
        )
    add(
        rows,
        "label.xlsx vs attachment2",
        "classification",
        "classification_labels_annotation_consistency",
        result_status(mapping_ok),
        "one-to-one mapping inferred from both aligned and unaligned PKL",
    )

    for version in ("aligned", "unaligned"):
        parsed = read_feature_labels(raw / "附件2-数据集特征文件" / f"{version}_50.pkl")
        total = mismatch = 0
        for split_data in parsed.values():
            for sample_id, class_value in zip(
                [normalize_pkl_id(item) for item in split_data["id"]],
                split_data["classification_labels"],
            ):
                row = label_by_id.get(sample_id)
                if row is None:
                    continue
                total += 1
                mismatch += mapping.get(int(round(float(class_value)))) != str(row.annotation)
        add(rows, f"{version}_50.pkl", "classification", "classification_annotation_mismatch_count", mismatch, f"compared={total}")

    mode_values = sorted(str(value) for value in labels["mode"].dropna().unique())
    mode_ok = set(mode_values).issubset(VALID_MODES) and labels["mode"].notna().all()
    add(rows, "label.xlsx", "mode", "allowed_values_only", result_status(mode_ok), ",".join(mode_values))
    add(rows, "label.xlsx", "mode", "null_mode_count", int(labels["mode"].isna().sum()))

    for version in ("aligned", "unaligned"):
        for split in ("train", "valid", "test"):
            expected = set(labels.loc[labels["mode"] == split, "sample_id_expected"])
            actual = split_sets[(version, split)]
            add(rows, f"{version}_50.pkl", split, "xlsx_pkl_id_set_missing_in_pkl", len(expected - actual))
            add(rows, f"{version}_50.pkl", split, "pkl_id_set_extra_vs_xlsx", len(actual - expected))

    with (output / "label_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "scope", "metric", "value", "details"])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# E题标签一致性审计报告",
        "",
        "> 本报告由 `label_consistency_audit.py` 生成。只读检查 XLSX 与附件2 PKL，未修改原始数据，未训练模型。",
        "> `classification_labels`/`regression_labels` 使用轻量代理提取，未物化大尺寸模态特征数组。",
        "",
        "## 结论摘要",
        "",
        f'- `sample_id = video_id + "_" + clip_id`：**{result_status(all(frame["sample_id_expected"].notna().all() and frame["sample_id_expected"].duplicated().sum() == 0 for frame in (df100, labels)))}**。',
        f'- `classification_labels` 与 `annotation`：**{result_status(mapping_ok)}**；映射为 `{json.dumps(mapping, ensure_ascii=False)}`。',
        f'- `regression_labels` 与 `label`：比较 {regression_total} 条，不一致 {regression_mismatch} 条（绝对误差 `1e-5`）；**{result_status(regression_mismatch == 0)}**。',
        f'- `mode` 白名单：实际取值 `{", ".join(mode_values)}`；**{result_status(mode_ok)}**。',
        "",
        "## 分类标签映射",
        "",
        "| classification_labels | annotation | 样本数 |",
        "|---:|---|---:|",
    ]
    for class_value in sorted(by_class):
        for annotation, count in by_class[class_value].items():
            lines.append(f"| {class_value} | {annotation} | {count} |")
    lines += [
        "",
        "## mode 分布",
        "",
        f'- `{json.dumps({str(k): int(v) for k, v in labels["mode"].value_counts(dropna=False).items()}, ensure_ascii=False)}`',
        "",
        "## split ID 集合检查",
        "",
        "| PKL 版本 | split | XLSX 中但 PKL 缺失 | PKL 中但 XLSX 多出 |",
        "|---|---|---:|---:|",
    ]
    for version in ("aligned", "unaligned"):
        for split in ("train", "valid", "test"):
            expected = set(labels.loc[labels["mode"] == split, "sample_id_expected"])
            actual = split_sets[(version, split)]
            lines.append(f"| `{version}` | `{split}` | {len(expected - actual)} | {len(actual - expected)} |")
    lines += [
        "",
        "## 说明",
        "",
        "- PKL 中的 `video_id$_$clip_id` 已规范化为 `video_id_clip_id` 后比较。",
        "- 回归标签比较使用绝对误差 `1e-5`，仅容忍浮点存储误差，不修改标签。",
        "- 输出文件位于 `01_manifest/csv_output/`；原始文件保持只读。",
        "",
    ]
    (output / "label_consistency_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "summary": str(output / "label_summary.csv"),
        "report": str(output / "label_consistency_report.md"),
        "classification": result_status(mapping_ok),
        "regression_mismatch": regression_mismatch,
        "mode": result_status(mode_ok),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
