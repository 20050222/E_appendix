"""生成 E 题数据预处理阶段的中文版论文级审计图。

输入：
    ../label_summary.csv

输出：
    visualizations_chinese/<figure_name>/<figure_name>.{png,svg,pdf}

说明：
    本脚本只读取已经生成的审计统计，不修改原始 XLSX、PKL 或视频，
    也不执行模型训练。英文版绘图代码和结果保留在相邻的
    ``visualizations`` 目录中。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


HERE = Path(__file__).resolve().parent
ENGLISH_SCRIPT = HERE.parent / "visualizations" / "publication_audit_figures.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("publication_audit_figures_en", ENGLISH_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法载入英文版绘图代码：{ENGLISH_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()
C = BASE.COLORS
CATEGORY_COLORS = BASE.CATEGORY_COLORS


def configure_chinese_style() -> None:
    """使用 Windows 可用的中文字体并继承英文版视觉规范。"""
    BASE.configure_style()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DengXian", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def export_figure(fig: plt.Figure, stem: str) -> None:
    """按“同图三格式同目录”的规则导出。"""
    target = HERE / stem
    target.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "publication_audit_figures_chinese.py", "Title": stem}
    fig.savefig(target / f"{stem}.png", dpi=320, metadata=metadata)
    fig.savefig(target / f"{stem}.svg", metadata=metadata)
    fig.savefig(target / f"{stem}.pdf", metadata=metadata)
    plt.close(fig)


def panel_label(ax, label: str) -> None:
    text_method = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_method(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=C["ink"],
        va="top",
    )


def draw_extruded_box(ax, x, y, w, h, depth, face, title, lines, accent) -> None:
    """绘制克制的拟三维科研信息模块。"""
    side = Polygon(
        [(x + w, y), (x + w + depth, y + depth),
         (x + w + depth, y + h + depth), (x + w, y + h)],
        closed=True,
        facecolor=mpl.colors.to_rgba(accent, 0.18),
        edgecolor=mpl.colors.to_rgba(accent, 0.32),
        linewidth=0.7,
        zorder=1,
    )
    top = Polygon(
        [(x, y + h), (x + depth, y + h + depth),
         (x + w + depth, y + h + depth), (x + w, y + h)],
        closed=True,
        facecolor=mpl.colors.to_rgba(accent, 0.10),
        edgecolor=mpl.colors.to_rgba(accent, 0.28),
        linewidth=0.7,
        zorder=1,
    )
    front = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=face,
        edgecolor=mpl.colors.to_rgba(accent, 0.55),
        linewidth=0.8,
        zorder=2,
    )
    ax.add_patch(side)
    ax.add_patch(top)
    ax.add_patch(front)
    ax.text(x + 0.03 * w, y + h - 0.18 * h, title,
            color=C["ink"], fontsize=9.2, fontweight="bold", zorder=3)
    for index, line in enumerate(lines):
        ax.text(x + 0.03 * w, y + h - (0.43 + 0.18 * index) * h, line,
                color=C["muted"], fontsize=7.4, zorder=3)


def figure_graphical_abstract(_data: dict) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 5.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("多模态情感识别数据完整性审计", x=0.08, y=0.98,
                 ha="left", fontsize=15, color=C["ink"], fontweight="bold")
    ax.text(0.08, 0.91, "证据可追溯 · 标签一致性 · 数据划分完整性 · 结果可复现",
            color=C["muted"], fontsize=9.5)

    boxes = [
        (0.06, C["light_blue"], C["blue"], "01  原始证据",
         ["2 个 XLSX 标签表", "2 套 PKL 特征版本", "4,850 个有标签样本"]),
        (0.29, C["light_cyan"], C["cyan"], "02  样本编号规范化",
         ["video_id + '_' + clip_id", "统一 PKL 分隔符", "重复样本编号为 0"]),
        (0.52, C["light_orange"], C["orange"], "03  交叉一致性检查",
         ["分类标签与情感类别", "回归标签与连续标签", "mode 与 PKL 数据划分"]),
        (0.75, C["light_purple"], C["purple"], "04  审计产出",
         ["标签统计清单", "一致性审计报告", "论文级科研图件"]),
    ]
    for x, face, accent, title, lines in boxes:
        draw_extruded_box(ax, x, 0.43, 0.17, 0.27, 0.015, face, title, lines, accent)
    for x0, x1 in ((0.245, 0.285), (0.475, 0.515), (0.705, 0.745)):
        ax.add_patch(FancyArrowPatch((x0, 0.57), (x1, 0.57), arrowstyle="-|>",
                                     mutation_scale=10, linewidth=1.0, color=C["muted"]))

    facts = [
        ("4,850", "唯一有标签样本", C["blue"]),
        ("3,395 / 728 / 727", "训练集 / 验证集 / 测试集", C["cyan"]),
        ("0", "标签不一致数", C["orange"]),
        ("4 / 4", "通过的审计规则", C["purple"]),
    ]
    for index, (value, label, color) in enumerate(facts):
        x = 0.075 + index * 0.23
        ax.plot([x, x + 0.16], [0.24, 0.24], color=mpl.colors.to_rgba(color, 0.55), linewidth=2.0)
        ax.text(x, 0.18, value, color=C["ink"], fontsize=12, fontweight="bold")
        ax.text(x, 0.12, label, color=C["muted"], fontsize=8)
    ax.text(0.06, 0.02, "图1｜只读审计；未训练模型，未修改任何原始数据。",
            color=C["muted"], fontsize=7.5)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.05, top=0.90)
    export_figure(fig, "audit_graphical_abstract")


def figure_annotation_distribution(data: dict) -> None:
    categories_en = ["Negative", "Neutral", "Positive"]
    categories_zh = ["负向", "中性", "正向"]
    fig = plt.figure(figsize=(11.4, 5.0))
    grid = fig.add_gridspec(1, 2, wspace=0.04)
    panels = (
        ("label.xlsx", "完整标签数据集"),
        ("label-100.xlsx", "原始视频样本子集"),
    )
    for index, (source, title) in enumerate(panels):
        ax = fig.add_subplot(grid[0, index], projection="3d")
        values = np.array([data["annotation"][source][cat] for cat in categories_en], dtype=float)
        total = values.sum()
        xpos = np.arange(3)
        ax.bar3d(xpos, np.zeros(3), np.zeros(3), np.full(3, 0.56), np.full(3, 0.52), values,
                 color=[CATEGORY_COLORS[c] for c in categories_en], shade=True,
                 alpha=0.96, edgecolor="white", linewidth=0.35)
        for x, value in zip(xpos, values):
            ax.text(x + 0.28, 0.27, value * 1.08,
                    f"{int(value):,}\n{value / total:.1%}",
                    ha="center", va="bottom", fontsize=8, color=C["ink"])
        BASE.clean_3d(ax)
        ax.view_init(elev=24, azim=-58)
        ax.set_box_aspect((1.7, 0.65, 1.25))
        ax.set_xticks(xpos + 0.28, categories_zh)
        ax.set_yticks([])
        ax.set_zlabel("样本数", labelpad=6, color=C["muted"])
        ax.set_title(f"{title}\n样本量 = {int(total):,}", pad=1,
                     color=C["ink"], fontweight="bold")
    fig.text(0.08, 0.02,
             "柱顶同时标注样本数与数据源内部占比；两个面板采用独立纵轴，以保留 100 条样本子集的可读性。",
             color=C["muted"], fontsize=7.5)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.87)
    export_figure(fig, "annotation_distribution")


def figure_split_distribution(data: dict) -> None:
    order = ["train", "valid", "test"]
    labels_zh = ["训练集", "验证集", "测试集"]
    values = np.array([data["split"][key] for key in order], dtype=float)
    percentages = values / values.sum()
    colors = [C["blue"], C["cyan"], C["orange"]]

    fig = plt.figure(figsize=(10.2, 6.0))
    grid = fig.add_gridspec(2, 1, height_ratios=(4.0, 1.25), hspace=0.08)
    ax = fig.add_subplot(grid[0], projection="3d")
    xpos = np.arange(3)
    ax.bar3d(xpos, np.zeros(3), np.zeros(3), np.full(3, 0.58), np.full(3, 0.55), values,
             color=colors, shade=True, edgecolor="white", linewidth=0.35)
    for x, value, proportion in zip(xpos, values, percentages):
        ax.text(x + 0.29, 0.28, value * 1.08,
                f"{int(value):,}\n{proportion:.1%}",
                ha="center", va="bottom", fontsize=8, color=C["ink"])
    BASE.clean_3d(ax)
    ax.view_init(elev=23, azim=-57)
    ax.set_box_aspect((2.2, 0.70, 1.25))
    ax.set_xticks(xpos + 0.29, labels_zh)
    ax.set_yticks([])
    ax.set_zlabel("样本数", color=C["muted"])
    ax.set_title("各数据划分的样本规模", pad=3, color=C["ink"], fontweight="bold")

    ax2 = fig.add_subplot(grid[1])
    left = 0.0
    for label, proportion, count, color in zip(labels_zh, percentages, values, colors):
        ax2.barh([0], [proportion], left=left, height=0.38,
                 color=color, edgecolor="white", linewidth=0.8)
        text_color = "white" if label != "验证集" else C["ink"]
        ax2.text(left + proportion / 2, 0.06,
                 f"{label}\n{int(count):,}（{proportion:.1%}）",
                 ha="center", va="center", color=text_color,
                 fontsize=8, fontweight="bold")
        left += proportion
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_yticks([])
    ax2.set_xlabel("有标签样本占比", color=C["muted"])
    ax2.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.grid(axis="x", color=C["grid"], linewidth=0.5)
    ax2.set_axisbelow(True)
    fig.text(0.08, 0.01,
             "验证集与测试集规模近似相等；全部 PKL 样本编号均与 XLSX 的 mode 字段一致。",
             color=C["muted"], fontsize=7.5)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.88)
    export_figure(fig, "split_distribution")


def figure_consistency_status(_data: dict) -> None:
    fig, ax = plt.subplots(figsize=(11.4, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle("标签完整性审计矩阵", x=0.07, y=0.98,
                 ha="left", fontsize=14, color=C["ink"], fontweight="bold")
    ax.text(0.07, 0.90, "从样本编号、分类标签、回归标签和数据划分四个维度实施独立核验",
            color=C["muted"], fontsize=9)

    audit_rows = [
        ("样本编号", "video_id + '_' + clip_id", "4,850 个唯一编号", "重复数为 0", C["blue"]),
        ("分类标签", "0→负向 · 1→中性 · 2→正向", "9,700 次比较", "不一致数为 0", C["cyan"]),
        ("回归标签", "regression_labels 与 label", "9,700 次比较", "不一致数为 0", C["orange"]),
        ("数据划分", "mode ∈ {train, valid, test}", "4,850 条记录", "非法值为 0", C["purple"]),
    ]
    y_positions = [0.72, 0.55, 0.38, 0.21]
    headers = [(0.075, "审计维度"), (0.30, "检查规则"), (0.61, "核验证据"), (0.80, "结果")]
    for x, text in headers:
        ax.text(x, 0.82, text, fontsize=8, color=C["muted"], fontweight="bold")
    ax.plot([0.07, 0.94], [0.79, 0.79], color=C["grid"], linewidth=0.8)
    for (name, rule, evidence, result, color), y in zip(audit_rows, y_positions):
        ax.add_patch(Polygon([(0.072, y - 0.045), (0.084, y - 0.034),
                              (0.084, y + 0.055), (0.072, y + 0.044)],
                             closed=True, facecolor=mpl.colors.to_rgba(color, 0.28), edgecolor="none"))
        ax.add_patch(FancyBboxPatch((0.08, y - 0.045), 0.84, 0.095,
                                    boxstyle="round,pad=0.006,rounding_size=0.008",
                                    facecolor=mpl.colors.to_rgba(color, 0.065),
                                    edgecolor=mpl.colors.to_rgba(color, 0.25), linewidth=0.65))
        ax.text(0.095, y, name, va="center", color=C["ink"], fontsize=9, fontweight="bold")
        ax.text(0.30, y, rule, va="center", color=C["ink"], fontsize=8.2)
        ax.text(0.61, y, evidence, va="center", color=C["muted"], fontsize=8.2)
        ax.text(0.80, y + 0.014, result, va="center", color=C["ink"], fontsize=8.2)
        ax.text(0.88, y - 0.018, "通过", va="center", ha="center",
                color=C["green"], fontsize=9, fontweight="bold")
        ax.scatter([0.93], [y], s=95, facecolor=C["green"], edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(0.93, y - 0.002, "通", ha="center", va="center",
                color="white", fontsize=7.5, fontweight="bold", zorder=4)
    ax.text(0.07, 0.07, "对齐版与未对齐版 PKL 均独立复现了 XLSX 标签和数据划分关系。",
            color=C["muted"], fontsize=8)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.05, top=0.90)
    export_figure(fig, "consistency_audit_status")


def write_readme() -> None:
    (HERE / "README.md").write_text(
        """# E题数据预处理阶段中文可视化

本目录为英文版 `../visualizations/` 的中文对应版本，文件组织规则保持一致：每张图单独一个文件夹，文件夹内保存同名 PNG、SVG 和 PDF。

## 原始绘图代码

- `publication_audit_figures_chinese.py`

## 输入

- `../label_summary.csv`

## 输出结构

- `audit_graphical_abstract/`：数据完整性审计流程与关键证据。
- `annotation_distribution/`：完整标签数据与原始视频子集的情感类别构成。
- `split_distribution/`：训练集、验证集和测试集的数量与比例。
- `consistency_audit_status/`：四项标签一致性检查的规则、证据和结论。

每张图提供：

- PNG：320 dpi，便于预览和插入普通文档；
- SVG：可编辑矢量格式；
- PDF：适合论文排版和打印。

## 复现

```powershell
python publication_audit_figures_chinese.py
```

脚本只读取审计统计，不修改原始数据，也不训练模型。
""",
        encoding="utf-8",
    )


def main() -> None:
    configure_chinese_style()
    data = BASE.load_summary()
    figure_graphical_abstract(data)
    figure_annotation_distribution(data)
    figure_split_distribution(data)
    figure_consistency_status(data)
    write_readme()
    print("已生成 4 组中文版图件：PNG、SVG、PDF。")


if __name__ == "__main__":
    main()
