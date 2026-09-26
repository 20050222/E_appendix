# E题标签一致性审计报告

> 审计范围：只读检查 `label-100.xlsx`、`label.xlsx` 与附件2 aligned/unaligned PKL；未修改原始数据，未训练模型。
> 工作区：`E_workspace/`（实际目录名）；输出目录：`01_manifest/csv_output/`。

## 结论摘要

- `sample_id = video_id + "_" + clip_id`：**通过**。
- `classification_labels` 与 `annotation`：**通过**；推断映射为 `{"2": "Positive", "0": "Negative", "1": "Neutral"}`。
- `regression_labels` 与 `label`：比较总数 9700，容差 `1e-5` 不一致 0 条；**通过**。
- `mode` 白名单：实际取值 `test, train, valid`；**通过**。

## 1. sample_id 检查

| 文件 | 行数 | 公式生成非空数 | 重复 sample_id | 结果 |
|---|---:|---:|---:|---|
| `label-100.xlsx` | 100 | 100 | 0 | PASS |
| `label.xlsx` | 4850 | 4850 | 0 | PASS |

附件2 PKL 的 ID 使用 `video_id$_$clip_id` 表示；审计时将该分隔符规范化为下划线后，与 XLSX 的 `sample_id` 比较。aligned 和 unaligned 两套 split 的 ID 集合均已比较。

## 2. classification_labels 与 annotation

| classification_labels | annotation | 样本数 |
|---:|---|---:|
| 0 | Negative | 2760 |
| 1 | Neutral | 2200 |
| 2 | Positive | 4740 |

推断的一对一映射：`{"2": "Positive", "0": "Negative", "1": "Neutral"}`。

| PKL 版本 | 比较数 | 不一致数 | 结果 |
|---|---:|---:|---|
| `aligned` | 4850 | 0 | PASS |
| `unaligned` | 4850 | 0 | PASS |

## 3. regression_labels 与 label

| PKL 版本 | 比较数 | 不一致数 | 容差 | 结果 |
|---|---:|---:|---:|---|
| `aligned` | 4850 | 0 | 1e-5 | PASS |
| `unaligned` | 4850 | 0 | 1e-5 | PASS |

## 4. mode 检查

- 允许集合：`train`, `valid`, `test`。
- 实际非空取值：`test, train, valid`。
- 各 mode 数量：`{"train": 3395, "valid": 728, "test": 727}`。
- 白名单检查：**PASS**。

## 5. XLSX 与 PKL split 对齐

| PKL 版本 | split | XLSX 中但 PKL 缺失 | PKL 中但 XLSX 多出 |
|---|---|---:|---:|
| `aligned` | `train` | 0 | 0 |
| `aligned` | `valid` | 0 | 0 |
| `aligned` | `test` | 0 | 0 |
| `unaligned` | `train` | 0 | 0 |
| `unaligned` | `valid` | 0 | 0 |
| `unaligned` | `test` | 0 | 0 |

## 审计说明

- `classification_labels`/`regression_labels` 从附件2 PKL 读取；大数组使用轻量代理提取标签，未物化模态特征数组。
- 回归比较使用绝对误差 `1e-5`，用于容忍 PKL 中的浮点表示差异。
- 本报告只记录一致性，不对标签值进行修正，也不生成清洗数据。
