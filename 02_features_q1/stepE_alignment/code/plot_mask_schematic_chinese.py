"""绘制三模态有效观测与整维置零的示意图。

该图用于解释 mask 的语义，不代表某一条真实样本，也不读取实验 CSV。
如需恢复原图，只需恢复本脚本中的 STYLE、STATES 和输出参数。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


STYLE = {
    "ink": "#243447",
    "muted": "#6B7C8F",
    "grid": "#D9E2EA",
    "track": "#F3F6F8",
    "valid": "#7EA8C0",
    "zero": "#E7AD91",
    "zero_text": "#7E4636",
    "text": "#3D6F90",
    "audio": "#D27C45",
    "vision": "#3B8D86",
}

# 16 个位置仅用于解释 mask 的含义，不是实际样本的统计结果。
STATES = {
    "文本 T": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    "语音 A": [1] * 16,
    "视觉 V": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
}

ROW_COLORS = {
    "文本 T": STYLE["text"],
    "语音 A": STYLE["audio"],
    "视觉 V": STYLE["vision"],
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "text.color": STYLE["ink"],
            "axes.labelcolor": STYLE["ink"],
            "xtick.color": STYLE["muted"],
            "ytick.color": STYLE["ink"],
        }
    )


def plot_mask_schematic(output_dir: Path) -> None:
    configure_style()
    n_positions = len(next(iter(STATES.values())))
    x = range(n_positions)
    fig, ax = plt.subplots(figsize=(10.2, 3.8), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 先画统一的浅色轨道，再覆盖有效观测和整维置零状态。
    for row_idx, (label, states) in enumerate(STATES.items()):
        y = 2 - row_idx
        for pos, state in zip(x, states):
            face = STYLE["valid"] if state else STYLE["zero"]
            rect = Rectangle(
                (pos + 0.10, y - 0.25),
                0.80,
                0.50,
                facecolor=face,
                edgecolor="white",
                linewidth=1.4,
                joinstyle="round",
            )
            ax.add_patch(rect)
            if not state:
                ax.text(
                    pos + 0.50,
                    y,
                    "0",
                    ha="center",
                    va="center",
                    color=STYLE["zero_text"],
                    fontsize=10,
                    fontweight="bold",
                )

        valid_count = sum(states)
        ax.text(
            n_positions + 0.55,
            y,
            f"有效 {valid_count}/{n_positions}",
            ha="left",
            va="center",
            color=ROW_COLORS[label],
            fontsize=9.5,
            fontweight="bold",
        )

    # 行标签和分隔线让三行状态可快速比较。
    for row_idx, label in enumerate(STATES):
        y = 2 - row_idx
        ax.text(
            -0.45,
            y,
            label,
            ha="right",
            va="center",
            color=ROW_COLORS[label],
            fontsize=11,
            fontweight="bold",
        )
        if row_idx < len(STATES) - 1:
            ax.axhline(y - 0.50, color=STYLE["grid"], linewidth=0.8, zorder=0)

    ax.set_xlim(-0.75, n_positions + 2.25)
    ax.set_ylim(-0.9, 2.95)
    ax.set_xticks([0, 3, 7, 11, 15], ["1", "4", "8", "12", "16"])
    ax.set_xlabel("统一序列位置（示意）", labelpad=9, fontsize=10)
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        0.0,
        2.72,
        "三模态观测 mask 的位置语义",
        transform=ax.transData,
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=STYLE["ink"],
    )
    ax.text(
        n_positions + 2.18,
        2.72,
        "示意图",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=STYLE["muted"],
    )

    legend = ax.legend(
        handles=[
            Patch(facecolor=STYLE["valid"], edgecolor="none", label="有效观测（mask=1）"),
            Patch(facecolor=STYLE["zero"], edgecolor="none", label="整维置零（mask=0）"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.50, 1.00),
        ncol=2,
        frameon=False,
        fontsize=9,
        handlelength=1.3,
        columnspacing=1.8,
    )
    for text in legend.get_texts():
        text.set_color(STYLE["ink"])

    ax.text(
        0.0,
        -0.68,
        "mask=0 仅表示该位置不可观测；对应特征填零，不表示情感值为 0。",
        ha="left",
        va="top",
        fontsize=8.5,
        color=STYLE["muted"],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"fig2_mask_schematic.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制三模态 mask 位置语义示意图")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "image/vpublication_chinese/fig2_mask_schematic",
    )
    args = parser.parse_args()
    plot_mask_schematic(args.output_dir.resolve())
    print(f"示意图输出目录: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
