"""生成三模态质量控制中文出版风格图。

本脚本复用英文绘图脚本的数据读取、样本排序、统计和文件导出逻辑，
只替换图中文字与输出目录。英文源代码仍保留在同一 code 文件夹中。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import numpy as np

import plot_multimodal_publication as en


CN = {
    "text": "文本",
    "audio": "语音",
    "vision": "视觉",
    "negative": "负面",
    "neutral": "中性",
    "positive": "正面",
}


def configure_chinese_style() -> None:
    en.configure_style()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans SC", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )


def panel(ax: plt.Axes, label: str) -> None:
    en.panel_label(ax, label)


def clean(ax: plt.Axes, grid: bool = False) -> None:
    en.clean_axis(ax, grid=grid)


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    en.save_figure(fig, output_dir, stem)


def plot_overview(table, matrices, out: Path) -> None:
    values = {m: matrices[m].sum(axis=1) for m in en.MODALITIES}
    n_samples = len(table)
    n_bins = matrices["text"].shape[1]
    coverage_counts = {m: int(matrices[m].sum()) for m in en.MODALITIES}
    any_valid = {m: int((values[m] > 0).sum()) for m in en.MODALITIES}
    all_valid = {m: int((values[m] == n_bins).sum()) for m in en.MODALITIES}

    fig = plt.figure(figsize=(9.4, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.28, 1.0])
    dist_axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])]
    rng = np.random.default_rng(42)
    for ax, modality in zip(dist_axes, en.MODALITIES):
        vals = values[modality]
        violin = ax.violinplot(vals, positions=[1], widths=0.72, showextrema=False)
        violin["bodies"][0].set_facecolor(en.COLORS[modality])
        violin["bodies"][0].set_edgecolor("none")
        violin["bodies"][0].set_alpha(0.22)
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(1 + jitter, vals, s=11, color=en.COLORS[modality], alpha=0.48,
                   linewidth=0, rasterized=True, zorder=2)
        ax.boxplot(vals, positions=[1], widths=0.24, patch_artist=True, showfliers=False,
                   medianprops={"color": en.COLORS["dark"], "linewidth": 1.2},
                   whiskerprops={"color": en.COLORS["dark"], "linewidth": 0.8},
                   capprops={"color": en.COLORS["dark"], "linewidth": 0.8},
                   boxprops={"facecolor": "white", "edgecolor": en.COLORS["dark"], "linewidth": 0.8},
                   zorder=3)
        if modality == "audio":
            ax.set_ylim(46.5, 50.5)
            ax.set_yticks([47, 48, 49, 50])
            title = f"{CN[modality]}（局部放大）\n均值 {np.mean(vals):.1f}"
        else:
            ax.set_ylim(-1, 53)
            ax.set_yticks([0, 10, 20, 30, 40, 50])
            title = f"{CN[modality]}\n均值 {np.mean(vals):.1f}"
        ax.set_xlim(0.48, 1.52)
        ax.set_xticks([])
        ax.set_title(title, color=en.COLORS[modality], fontweight="bold", pad=7)
        clean(ax, True)
    dist_axes[0].set_ylabel("每个样本的有效桶数")
    dist_axes[1].text(0.5, 0.025, "纵轴：46.5–50.5 桶",
                      transform=dist_axes[1].transAxes, ha="center", va="bottom",
                      fontsize=7, color=en.COLORS["muted"])

    y = np.arange(1, len(en.MODALITIES) + 1)
    ax = fig.add_subplot(gs[1, 0:3])
    for yi, modality in zip(y, en.MODALITIES):
        count = coverage_counts[modality]
        pct = 100 * count / (n_samples * n_bins)
        ax.barh(yi, 100, color="#E8EDF1", height=0.58, edgecolor="none", zorder=0)
        ax.barh(yi, pct, color=en.COLORS[modality], height=0.58, edgecolor="none", zorder=1)
        ax.text(103, yi, f"{pct:.1f}%  ·  {count:,}/{n_samples * n_bins:,}",
                va="center", ha="left", color=en.COLORS["dark"], fontsize=7.5)
    ax.set_yticks(y, [CN[m] for m in en.MODALITIES])
    ax.set_ylim(4, 0)
    ax.set_xlim(0, 157)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("桶级比例（%）")
    ax.set_title("桶级观测构成", loc="left")
    ax.legend(handles=[
        Patch(facecolor=en.COLORS["text"], label="有效观测"),
        Patch(facecolor="#E8EDF1", label="无观测"),
    ], loc="upper right", ncol=2, frameon=False, fontsize=7)
    clean(ax, True)

    ax = fig.add_subplot(gs[1, 3:6])
    state_colors = {"empty": "#D8E0E5", "partial": "#E5A14A", "complete": "#368F88"}
    states = [
        ("empty", "无有效桶"),
        ("partial", "部分有效"),
        ("complete", "50 桶全有效"),
    ]
    for yi, modality in zip(y, en.MODALITIES):
        counts = {
            "empty": n_samples - any_valid[modality],
            "partial": any_valid[modality] - all_valid[modality],
            "complete": all_valid[modality],
        }
        left = 0.0
        for state, _label in states:
            width = 100 * counts[state] / n_samples
            if width > 0:
                ax.barh(yi, width, left=left, height=0.58, color=state_colors[state], edgecolor="white",
                        linewidth=0.7, zorder=1)
                text_color = en.COLORS["dark"] if state == "empty" else "white"
                ax.text(left + width / 2, yi, str(counts[state]), ha="center", va="center",
                        color=text_color, fontsize=7.5, fontweight="bold", zorder=2)
            left += width
    ax.set_yticks(y, [CN[m] for m in en.MODALITIES])
    ax.set_ylim(4, 0)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("样本构成（%）")
    ax.set_title("样本级完整性", loc="left")
    ax.legend(handles=[Patch(facecolor=state_colors[s], edgecolor="none", label=label)
                       for s, label in states],
              loc="upper center", ncol=3, frameon=False, fontsize=7)
    clean(ax, True)

    fig.suptitle("aligned-50 三模态质量概览", fontsize=13, fontweight="bold", color=en.COLORS["dark"])
    save(fig, out, "fig1_multimodal_qc_overview")


def plot_masks(table, matrices, out: Path) -> None:
    """将三模态观测掩码绘制为三个对齐的 3D 可用性地形面。"""
    text_order = np.argsort(table["token_count"].to_numpy())
    av_order = np.lexsort((table["audio_valid_bins"].to_numpy(), table["vision_valid_bins"].to_numpy()))[::-1]
    ordered = {"text": text_order, "audio": av_order, "vision": av_order}
    fig = plt.figure(figsize=(11.2, 5.8), constrained_layout=True)
    axes = [fig.add_subplot(1, 3, idx + 1, projection="3d") for idx in range(3)]
    x_grid, y_grid = np.meshgrid(np.arange(1, 51), np.arange(1, 101))
    for ax, modality in zip(axes, en.MODALITIES):
        mask = matrices[modality][ordered[modality]].astype(float)
        z_grid = 0.04 + 0.96 * mask
        valid_rgba = matplotlib.colors.to_rgba(en.COLORS[modality], 0.90)
        missing_rgba = matplotlib.colors.to_rgba(en.COLORS[modality], 0.22)
        facecolors = np.where(mask[:-1, :-1, None] > 0.5, valid_rgba, missing_rgba)
        ax.plot_surface(x_grid, y_grid, np.zeros_like(z_grid), color=en.MASK_VALID_COLOR,
                        alpha=0.22, linewidth=0, shade=False)
        ax.plot_surface(x_grid, y_grid, z_grid, facecolors=facecolors,
                        linewidth=0, antialiased=False, shade=True)
        profile = mask.mean(axis=0) * 100
        ax.plot(np.arange(1, 51), np.full(50, -7.0), 1.06 + 0.22 * profile / 100.0,
                color=en.COLORS[modality], linewidth=2.0)
        ax.set_title(f"{CN[modality]}  ·  {mask.mean() * 100:.1f}% 覆盖率",
                     color=en.COLORS[modality], fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("序列位置", labelpad=5)
        ax.set_ylabel("排序后样本", labelpad=5)
        ax.set_zlabel("" if modality != "vision" else "可用状态（0=无效，1=有效）", labelpad=5)
        ax.set_xlim(1, 50); ax.set_ylim(-9, 100); ax.set_zlim(0, 1.35)
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.set_yticks([1, 25, 50, 75, 100])
        ax.set_zticks([0, 1]); ax.set_zticklabels(["无效", "有效"])
        ax.view_init(elev=28, azim=-58)
        ax.set_box_aspect((1.5, 2.0, 0.62))
        ax.grid(True, linewidth=0.35, alpha=0.35)
    axes[-1].text2D(0.02, -0.11, "前方曲线表示按序列位置汇总的可用率。",
                    transform=axes[-1].transAxes, fontsize=7, color=en.COLORS["muted"])
    fig.legend(
        handles=[Patch(facecolor=en.COLORS["text"], alpha=0.9, label="有效观测面"),
                 Patch(facecolor=en.COLORS["text"], alpha=0.22, label="不可用/空桶"),
                 Line2D([0], [0], color=en.COLORS["dark"], linewidth=2, label="可用率曲线")],
        loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle("三模态序列位置可用性与观测掩码（3D）",
                 fontsize=12, fontweight="bold", color=en.COLORS["dark"])
    save(fig, out, "fig2_mask_structure_revised")


def plot_profiles(matrices, out: Path) -> None:
    # 三类模态共用归一化序列位置轴；文本轴语义仍为均匀词元位置。
    x = np.linspace(0.0, 1.0, 50)
    text_profile = matrices["text"].mean(axis=0) * 100
    audio_profile = matrices["audio"].mean(axis=0) * 100
    vision_profile = matrices["vision"].mean(axis=0) * 100
    both = ((matrices["audio"] == 1) & (matrices["vision"] == 1)).mean(axis=0) * 100
    audio_only = ((matrices["audio"] == 1) & (matrices["vision"] == 0)).mean(axis=0) * 100
    vision_only = ((matrices["audio"] == 0) & (matrices["vision"] == 1)).mean(axis=0) * 100
    neither = ((matrices["audio"] == 0) & (matrices["vision"] == 0)).mean(axis=0) * 100
    state_colors = ["#5C7F6F", en.COLORS["audio"], en.COLORS["vision"], "#D7DEE4"]
    state_labels = ["两者均有效", "仅语音有效", "仅视觉有效", "两者均无效"]
    fig = plt.figure(figsize=(9.3, 6.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    profile_by_modality = (("text", text_profile, 2.8), ("audio", audio_profile, 1.8),
                           ("vision", vision_profile, 0.8))
    for modality, values, y_value in profile_by_modality:
        ax.plot(x, np.full_like(x, y_value), values, color=en.COLORS[modality], linewidth=2.4,
                marker="o", markersize=2.8, markevery=5, label=CN[modality])
        ax.plot([0, 1], [y_value, y_value], [values.mean(), values.mean()],
                color=en.COLORS["muted"], linewidth=0.7, linestyle=(0, (3, 2)), alpha=0.85)
        ax.text(1.015, y_value, float(values[-1]), f"{values[-1]:.0f}%",
                color=en.COLORS[modality], fontsize=8, ha="left", va="center")
    x_centers = (np.arange(50) + 0.5) / 50.0
    left = np.zeros_like(x_centers)
    for values, color in zip((both, audio_only, vision_only, neither), state_colors):
        ax.bar3d(x_centers, np.full(50, 0.06), left, np.full(50, 0.016),
                 np.full(50, 0.34), values, color=color, alpha=0.92, shade=True,
                 linewidth=0.05, edgecolor="white")
        left += values
    ax.set_xlim(0, 1.08); ax.set_ylim(-0.05, 3.2); ax.set_zlim(0, 110)
    ax.set_xticks(np.linspace(0, 1, 5), ["0.00", "0.25", "0.50", "0.75", "1.00"])
    ax.set_yticks([0.23, 0.8, 1.8, 2.8], ["联合状态", "视觉", "语音", "文本"])
    ax.set_zticks([0, 25, 50, 75, 100])
    ax.set_xlabel("归一化序列位置", labelpad=7)
    ax.set_ylabel("模态", labelpad=8)
    ax.set_zlabel("可用率 / 状态构成（%）", labelpad=7)
    ax.view_init(elev=25, azim=-61)
    ax.set_box_aspect((1.65, 1.55, 1.0))
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(handles=[Line2D([0], [0], color=en.COLORS[m], linewidth=2.4, label=CN[m])
                       for m in en.MODALITIES] +
              [Patch(facecolor=c, alpha=0.92, label=l) for c, l in zip(state_colors, state_labels)],
              loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=2, frameon=False, fontsize=7)
    ax.text2D(0.02, -0.08, "文本轴为均匀词元位置；语音和视觉轴为归一化视频时间桶。",
              transform=ax.transAxes, fontsize=7.2, color=en.COLORS["muted"])
    ax.set_title("序列位置多模态可用率剖面（3D）", fontsize=12, fontweight="bold",
                 color=en.COLORS["dark"], pad=12)
    save(fig, out, "fig3_alignment_profiles")


def plot_label_stratified(table, out: Path) -> None:
    categories = ["Negative", "Neutral", "Positive"]
    cn_categories = ["负面", "中性", "正面"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True, constrained_layout=True)
    rng = np.random.default_rng(42)
    for ax, modality in zip(axes, en.MODALITIES):
        groups = [table.loc[table["annotation"] == category, f"{modality}_valid_bins"].dropna().to_numpy() for category in categories]
        en.draw_raincloud(ax, groups, en.COLORS[modality], rng, summary_label="中位数")
        ax.set_xticks(np.arange(3), cn_categories); ax.set_ylim(-1, 55); ax.set_xlabel("情感标注")
        ax.set_xlim(-0.48, 2.58); ax.set_ylim(0, 50)
        ax.set_title(f"{CN[modality]}有效桶数", fontsize=9, pad=18, color=en.COLORS[modality], fontweight="bold")
        clean(ax, True)
        ax.grid(axis="y", color="#D9E1E7", linewidth=0.45, alpha=0.55)
        ax.spines["left"].set_linewidth(0.65); ax.spines["bottom"].set_linewidth(0.65)
    axes[0].set_ylabel("每个样本的有效桶数（0–50）")
    axes[1].tick_params(labelleft=False); axes[2].tick_params(labelleft=False)
    fig.suptitle("按情感标签分层的特征可用率", fontsize=11, fontweight="bold", color=en.COLORS["dark"])
    save(fig, out, "fig4_label_stratified_availability")


def plot_diagnostics(table, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.25), constrained_layout=True)
    ax = axes[0]
    ax.scatter(table["token_count"], table["text_valid_bins"], s=18, color=en.COLORS["text"], alpha=0.68,
               edgecolor="white", linewidth=0.35)
    line_x = np.linspace(0, max(90, table["token_count"].max()), 200)
    ax.plot(line_x, np.minimum(line_x, 50), color=en.COLORS["dark"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xlabel("每个样本的内容词元数"); ax.set_ylabel("有效文本桶数"); ax.set_title("词元数与文本桶数的确定性关系")
    ax.text(0.03, 0.96, "有效桶数 = min（词元数，50）", transform=ax.transAxes, va="top",
            color=en.COLORS["dark"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5})
    clean(ax, True); panel(ax, "a")
    ax = axes[1]
    ax.scatter(table["audio_duration_diff_ms"], table["audio_valid_bins"], s=18, color=en.COLORS["audio"], alpha=0.68,
               edgecolor="white", linewidth=0.35)
    ax.axvline(0, color=en.COLORS["muted"], linewidth=0.7, linestyle=(0, (3, 2)))
    ax.set_xlabel("解码音频 − 视频时长\n（毫秒）"); ax.set_ylabel("有效语音桶数"); ax.set_title("音频尾部时长差异与有效桶数")
    ax.set_ylim(46.8, 50.2); ax.set_yticks([47, 48, 49, 50])
    audio_counts = table["audio_valid_bins"].round().astype(int).value_counts().sort_index()
    for level, count in audio_counts.items():
        ax.annotate(f"n={count}", xy=(1.0, level), xycoords=("axes fraction", "data"), xytext=(-4, 2),
                    textcoords="offset points", ha="right", va="bottom", fontsize=6.5, color=en.COLORS["dark"],
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0})
    clean(ax, True); panel(ax, "b")
    ax = axes[2]
    sizes = 12 + 28 * np.sqrt(table["multiple_face_samples"] / max(1, table["multiple_face_samples"].max()))
    status_color = np.where(table["vision_extraction_status"].eq("PASS"), en.COLORS["vision"], en.COLORS["incomplete"])
    rng = np.random.default_rng(42)
    face_x = np.clip(table["face_detection_rate"].to_numpy() * 100 + rng.uniform(-0.8, 0.8, len(table)), 0, 102)
    ax.scatter(face_x, table["vision_valid_bins"], s=sizes, c=status_color, alpha=0.68,
               edgecolor="white", linewidth=0.35)
    ax.set_xlabel("检测到人脸的抽样帧比例（%）"); ax.set_ylabel("有效视觉桶数"); ax.set_title("人脸检测与视觉覆盖率")
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=en.COLORS["vision"], markeredgecolor="white", markersize=5.5, label="通过"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=en.COLORS["incomplete"], markeredgecolor="white", markersize=5.5, label="未检测到人脸"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=en.COLORS["vision"], markeredgecolor="white", markersize=8.0, label="点大小：多脸抽样帧数"),
    ], loc="upper left", frameon=False, fontsize=6.2, handletextpad=0.35, borderpad=0.2)
    clean(ax, True); panel(ax, "c")
    values = [
        en.spearman(table["token_count"], table["text_valid_bins"]),
        en.spearman(table["audio_duration_diff_ms"], table["audio_valid_bins"]),
        en.spearman(table["face_detection_rate"], table["vision_valid_bins"]),
    ]
    axes[1].text(0.03, 0.08, f"斯皮尔曼 ρ = {values[1]:.2f}", transform=axes[1].transAxes, va="bottom",
                 color=en.COLORS["dark"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5})
    axes[2].text(0.97, 0.96, f"斯皮尔曼 ρ = {values[2]:.2f}", transform=axes[2].transAxes, va="top", ha="right",
                 color=en.COLORS["dark"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5})
    save(fig, out, "fig5_extraction_diagnostics")


def plot_traceability(data, table, out: Path) -> None:
    def passed(frame, column):
        return int(frame[column].astype(str).eq("PASS").sum())
    stages = ["视频与标签\n已建索引", "文本特征\n已验证", "语音特征\n已提取", "视觉有效性\n检查", "对齐文件\n已验证"]
    counts = np.array([passed(data["index"], "index_status"), passed(data["text_validation"], "status"),
                       passed(data["audio"], "status"), int((table["vision_valid_bins"] > 0).sum()), passed(data["aligned_validation"], "status")])
    fig, ax = plt.subplots(figsize=(8.0, 3.2), constrained_layout=True)
    x = np.arange(len(stages))
    total = np.full(len(stages), 100)
    visual_valid = int(counts[3])
    # 总保留数与视觉可观测数是两个指标；第二条线从视觉检查阶段开始。
    ax.plot(x, total, color=en.COLORS["dark"], linewidth=2.0, marker="o", markersize=6,
            label="样本保留数")
    visual_series = np.array([np.nan, np.nan, np.nan, visual_valid, visual_valid], dtype=float)
    ax.plot(x, visual_series, color=en.COLORS["vision"], linewidth=2.0, marker="o", markersize=6,
            label="视觉有效样本数")
    stage_colors = [en.COLORS["dark"], en.COLORS["text"], en.COLORS["audio"], en.COLORS["vision"], "#5C7F6F"]
    ax.scatter(x, total, s=80, color=stage_colors, zorder=3, edgecolor="white", linewidth=0.7)
    for xi in x:
        ax.text(xi, 101.0, "100", ha="center", va="bottom", fontweight="bold", color=en.COLORS["dark"], fontsize=8)
    ax.text(3, visual_valid - 1.0, f"{visual_valid}", ha="center", va="top", fontweight="bold", color=en.COLORS["vision"], fontsize=8)
    ax.text(4, visual_valid - 1.0, f"{visual_valid}", ha="center", va="top", fontweight="bold", color=en.COLORS["vision"], fontsize=8)
    ax.vlines(3, visual_valid, 100, color=en.COLORS["incomplete"], linewidth=1.2, linestyle=(0, (2, 2)), zorder=2)
    ax.annotate(f"{100 - visual_valid} 个样本未检测到有效人脸，\n但保留在 aligned-50 中（vision_mask = 0）",
                xy=(3, (visual_valid + 100) / 2), xytext=(2.15, 87.0),
                arrowprops={"arrowstyle": "-", "color": en.COLORS["incomplete"], "linewidth": 0.8},
                color=en.COLORS["incomplete"], ha="center")
    ax.set_xticks(x, stages); ax.set_ylim(86, 103); ax.set_ylabel("样本数（n = 100）")
    ax.set_title("各阶段检查结果与缺失视觉样本保留情况", loc="left", fontsize=11, fontweight="bold", color=en.COLORS["dark"])
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    clean(ax, True); save(fig, out, "fig6_pipeline_traceability")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    root = workspace / "02_features_q1"
    out = root / "stepE_alignment/image/vpublication_chinese"
    configure_chinese_style()
    data = en.read_data(root)
    matrices = en.mask_matrices(data)
    table = en.merged_sample_table(data)
    plot_overview(table, matrices, out)
    plot_masks(table, matrices, out)
    plot_profiles(matrices, out)
    plot_label_stratified(table, out)
    plot_diagnostics(table, out)
    plot_traceability(data, table, out)
    print(f"中文图输出目录: {out}")
    print(f"样本数: {len(table)}；三模态完整样本: {(table['multimodal_status'] == 'PASS').sum()}")


if __name__ == "__main__":
    main()
