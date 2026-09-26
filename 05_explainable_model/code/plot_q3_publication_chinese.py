"""第三问中文出版版图表。

读取第三问已经冻结的 CSV 结果，输出与英文出版版完全对应的
PNG、SVG、PDF 三种格式。中文脚本与英文脚本并列保存，互不覆盖。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from plot_q3_publication import (
    CLASS_COLORS,
    COLORS,
    MODALITIES,
    MODALITY_KEYS,
    despine,
    natural_ids,
    save_triplet,
)


CHINESE_MODALITIES = ["文本", "语音", "视觉"]
CHINESE_CLASS = {"Negative": "负面", "Neutral": "中性", "Positive": "正面"}


def configure_chinese_style() -> None:
    """设置适合论文排版的中文字体和克制的出版级样式。"""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans SC", "SimHei", "Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def protocol_selection_zh(validation: pd.DataFrame, out_root: Path) -> None:
    agg = (
        validation.groupby("window_ratio", as_index=False)
        .agg(
            classification_gap=("classification_faithfulness_gap", "mean"),
            regression_gap=("regression_faithfulness_gap", "mean"),
        )
        .sort_values("window_ratio")
    )
    ratios = agg["window_ratio"].to_numpy() * 100
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.65), sharex=True)
    for ax, col, title in zip(
        axes,
        ["classification_gap", "regression_gap"],
        ["分类任务", "回归任务"],
    ):
        vals = agg[col].to_numpy()
        ax.plot(ratios, vals, marker="o", markersize=4.5, linewidth=1.6, color="#3D596B")
        ax.fill_between(ratios, vals, 0, color="#3D596B", alpha=0.07)
        ax.axvline(10, color="#8F3B3B", linestyle=(0, (3, 2)), linewidth=1.0)
        ax.text(10.8, ax.get_ylim()[1] * 0.93, "最终选择", color="#8F3B3B", fontsize=7)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("候选窗口比例（%）")
        ax.set_ylabel("忠实度差距")
        ax.set_xticks(ratios)
        despine(ax)
    fig.suptitle("解释窗口协议选择", x=0.03, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.03, -0.02, "差距越大，表示选定证据窗口与随机窗口的区分度越强。", fontsize=7.5, color="#555555")
    fig.tight_layout(w_pad=2.0)
    save_triplet(fig, out_root, "q3_fig1_protocol_selection")


def modal_contribution_zh(modal: pd.DataFrame, out_root: Path) -> None:
    rows = []
    for modality, key in zip(CHINESE_MODALITIES, MODALITY_KEYS):
        for task, prefix in [("分类任务", "classification"), ("回归任务", "regression")]:
            vals = modal[f"{prefix}_normalized_{key}"].astype(float)
            rows.append({"modality": modality, "task": task, "mean": vals.mean(), "sd": vals.std(ddof=1)})
    summary = pd.DataFrame(rows)
    x = np.arange(len(CHINESE_MODALITIES))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.9, 3.25))
    for j, task in enumerate(["分类任务", "回归任务"]):
        cur = summary[summary.task == task]
        vals = cur["mean"].to_numpy()
        errs = cur["sd"].to_numpy()
        bars = ax.bar(
            x + (j - 0.5) * width,
            vals,
            width,
            yerr=errs,
            capsize=2.5,
            color=[COLORS[en] for en in MODALITIES],
            alpha=0.78 if j == 0 else 0.45,
            edgecolor="#2B2B2B",
            linewidth=0.45,
            label=task,
        )
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, CHINESE_MODALITIES)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("平均归一化扰动贡献")
    ax.set_title("测试样本的模态贡献", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(0.0, -0.19, "柱高为均值，误差线为样本标准差；贡献由模态遮挡扰动计算，不等同于内部融合权重。", transform=ax.transAxes, fontsize=7.2, color="#555555")
    despine(ax)
    fig.tight_layout()
    save_triplet(fig, out_root, "q3_fig2_modal_contribution")


def contribution_heatmap_zh(modal: pd.DataFrame, out_root: Path) -> None:
    ids = natural_ids(modal["sample_id"])
    ordered = modal.set_index("sample_id").loc[ids]
    cmap = LinearSegmentedColormap.from_list("contribution_zh", ["#F4F5F2", "#B7C9BE", "#2F5D62"])
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 5.0), sharey=True, constrained_layout=True)
    for ax, prefix, title in zip(
        axes,
        ["classification", "regression"],
        ["分类贡献", "回归贡献"],
    ):
        matrix = ordered[[f"{prefix}_normalized_{key}" for key in MODALITY_KEYS]].to_numpy()
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(3), CHINESE_MODALITIES)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("模态")
        ax.set_yticks(range(len(ids)), ids if ax is axes[0] else [])
        if ax is axes[0]:
            ax.set_ylabel("附件4样本")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.3, color="white" if value > 0.58 else "#263238")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label("归一化贡献", rotation=90)
    cbar.outline.set_linewidth(0.4)
    fig.suptitle("样本级可解释性贡献结构", x=0.03, ha="left", fontsize=11, fontweight="bold")
    save_triplet(fig, out_root, "q3_fig3_sample_contribution_heatmap")


def temporal_evidence_zh(temporal: pd.DataFrame, modal: pd.DataFrame, out_root: Path) -> None:
    vision_rows = modal[
        (modal["classification_main_modality"] == "vision")
        | (modal["regression_main_modality"] == "vision")
    ]
    if len(vision_rows):
        candidate = vision_rows.assign(
            score=vision_rows["classification_normalized_vision"] + vision_rows["regression_normalized_vision"]
        ).sort_values("score", ascending=False).iloc[0]["sample_id"]
    else:
        candidate = modal.assign(
            score=modal[[f"classification_normalized_{k}" for k in MODALITY_KEYS] + [f"regression_normalized_{k}" for k in MODALITY_KEYS]].sum(axis=1)
        ).sort_values("score", ascending=False).iloc[0]["sample_id"]
    sample = temporal[temporal.sample_id.astype(str) == str(candidate)].copy()
    sample["modality_label"] = sample["modality"].map({"text": "文本", "audio": "语音", "vision": "视觉"})
    y_positions = {"文本": 2, "语音": 1, "视觉": 0}
    cmap = mpl.colormaps["RdBu_r"]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.25), sharex=True, constrained_layout=True)
    for ax, delta_col, selected_col, title in [
        (axes[0], "classification_delta_signed", "selected_classification", "分类证据"),
        (axes[1], "regression_delta_signed", "selected_regression", "回归证据"),
    ]:
        for modality in CHINESE_MODALITIES:
            y = y_positions[modality]
            cur = sample[(sample.modality_label == modality) & sample[selected_col].astype(bool)].copy()
            if cur.empty:
                continue
            max_abs = max(float(sample[delta_col].abs().max()), 1e-9)
            norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
            for _, row in cur.iterrows():
                width = float(row.normalized_end - row.normalized_start)
                rect = mpl.patches.Rectangle(
                    (row.normalized_start, y - 0.28), width, 0.56,
                    facecolor=cmap(norm(float(row[delta_col]))), edgecolor="#34495E", linewidth=0.45,
                )
                ax.add_patch(rect)
        ax.set_ylim(-0.65, 2.65)
        ax.set_yticks([2, 1, 0], CHINESE_MODALITIES)
        ax.set_xlim(0, 1)
        ax.set_ylabel("模态")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#E1E1E1", linewidth=0.5)
        despine(ax, grid=False)
    axes[1].set_xlabel("归一化位置（文本为 token/bin；语音和视觉为时间位置）")
    sm = mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("有符号证据差值（相对尺度）")
    fig.suptitle(f"Top-k 局部证据窗口：样本 {candidate}", x=0.03, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.03, -0.02, "仅显示冻结解释协议选出的窗口；由于附件4 PKL缺少时长元数据，位置使用归一化坐标。", fontsize=7.2, color="#555555")
    save_triplet(fig, out_root, "q3_fig4_temporal_evidence")


def prediction_profile_zh(prediction: pd.DataFrame, out_root: Path) -> None:
    ids = natural_ids(prediction["sample_id"])
    df = prediction.set_index("sample_id").loc[ids]
    x = np.arange(len(df))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.7), gridspec_kw={"height_ratios": [1.35, 1]}, sharex=True)
    bottom = np.zeros(len(df))
    for cls, col in [("Negative", "prob_negative"), ("Neutral", "prob_neutral"), ("Positive", "prob_positive")]:
        vals = df[col].to_numpy()
        axes[0].bar(x, vals, bottom=bottom, width=0.76, color=CLASS_COLORS[cls], label=CHINESE_CLASS[cls], edgecolor="white", linewidth=0.35)
        bottom += vals
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("类别概率")
    axes[0].set_title("附件4测试样本预测结果", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    despine(axes[0])
    intensities = df["predicted_intensity"].to_numpy()
    colors = [CLASS_COLORS[str(c)] for c in df["predicted_class"]]
    axes[1].bar(x, intensities, color=colors, width=0.76, edgecolor="#2B2B2B", linewidth=0.35)
    axes[1].axhline(0, color="#444444", linewidth=0.65)
    axes[1].set_ylabel("预测情感强度")
    axes[1].set_xlabel("附件4样本")
    axes[1].set_xticks(x, ids)
    axes[1].tick_params(axis="x", rotation=45)
    despine(axes[1], grid=False)
    handles = [Patch(facecolor=CLASS_COLORS[c], edgecolor="none", label=CHINESE_CLASS[c]) for c in CLASS_COLORS]
    axes[1].legend(handles=handles, frameon=False, ncol=3, loc="upper right")
    fig.tight_layout(h_pad=1.25)
    save_triplet(fig, out_root, "q3_fig5_prediction_profile")


def polarity_distribution_3d_zh(prediction: pd.DataFrame, out_root: Path) -> None:
    """绘制附件4预测极性与强度的三维分箱分布。"""
    labels = ["Negative", "Neutral", "Positive"]
    chinese_labels = [CHINESE_CLASS[label] for label in labels]
    bins = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
    centers = (bins[:-1] + bins[1:]) / 2
    dx, dy = 0.48, 0.34

    fig = plt.figure(figsize=(8.2, 5.7))
    # 固定绘制层级：3D 自动深度排序会把远处的小柱顶标签遮住。
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    counts = prediction["predicted_class"].value_counts().reindex(labels, fill_value=0)
    for class_index, label in enumerate(labels):
        values = prediction.loc[prediction["predicted_class"] == label, "predicted_intensity"].to_numpy(dtype=float)
        hist, _ = np.histogram(values, bins=bins)
        x = np.full_like(centers, class_index, dtype=float) - dx / 2
        y = centers - dy / 2
        z = np.zeros_like(centers)
        ax.bar3d(
            x, y, z, dx, dy, hist,
            color=CLASS_COLORS[label], alpha=0.88, shade=True,
            edgecolor="white", linewidth=0.35, zsort="min", zorder=1,
        )
        for y_center, height in zip(centers, hist):
            if height:
                ax.text(
                    class_index, y_center, height + 0.14, str(int(height)),
                    ha="center", va="bottom", fontsize=7.2, zorder=100,
                    bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.86},
                )

    total = int(len(prediction))
    summary = "；".join(
        f"{name} {int(counts[label])} ({counts[label] / total:.0%})"
        for label, name in zip(labels, chinese_labels)
    )
    ax.text2D(0.03, 0.95, f"样本总数 n = {total}\n{summary}", transform=ax.transAxes, fontsize=8.5, va="top")
    ax.set(
        xticks=np.arange(3), xticklabels=chinese_labels,
        yticks=np.arange(-2.0, 2.1, 0.5),
        xlabel="预测情感极性", ylabel="预测情感强度区间", zlabel="样本数",
        title="附件4预测情感极性与强度分布",
        zlim=(0, max(6, int(counts.max()) + 1)),
    )
    ax.set_box_aspect((1.55, 1.2, 0.95))
    ax.view_init(elev=24, azim=-58)
    ax.tick_params(axis="both", labelsize=8, pad=1)
    ax.tick_params(axis="z", labelsize=8, pad=2)
    ax.xaxis.pane.set_facecolor((0.94, 0.95, 0.96, 0.55))
    ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 0.45))
    ax.zaxis.pane.set_facecolor((0.96, 0.96, 0.96, 0.35))
    ax.grid(True, linewidth=0.45, alpha=0.35)
    fig.text(0.03, 0.015, "柱体按预测情感强度分箱；附件4无真实标签，图中仅描述模型输出分布。", fontsize=7.5, color="#555555")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.91)
    save_triplet(fig, out_root, "q3_fig6_polarity_distribution_3d")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "image" / "publication_chinese")
    args = parser.parse_args()
    configure_chinese_style()
    validation = pd.read_csv(args.data / "q3_explanation_validation.csv")
    modal = pd.read_csv(args.data / "q3_modal_contribution.csv")
    temporal = pd.read_csv(args.data / "q3_temporal_evidence.csv")
    prediction = pd.read_csv(args.data / "q3_prediction.csv")
    for frame in [modal, temporal, prediction]:
        frame["sample_id"] = frame["sample_id"].astype(str)
    for frame in [validation, modal, temporal, prediction]:
        numeric = frame.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("输入 CSV 存在 NaN 或无穷数值")
    protocol_selection_zh(validation, args.out)
    modal_contribution_zh(modal, args.out)
    contribution_heatmap_zh(modal, args.out)
    temporal_evidence_zh(temporal, modal, args.out)
    prediction_profile_zh(prediction, args.out)
    polarity_distribution_3d_zh(prediction, args.out)
    print(f"已生成中文第三问出版版图表：{args.out}")


if __name__ == "__main__":
    main()
