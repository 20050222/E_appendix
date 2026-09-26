"""Create publication-style QC figures for the aligned three-modality features.

The figures deliberately keep text token-position bins separate from the
audio/vision video-time bins.  They summarize measured extraction logs only;
no model is trained and no missing value is imputed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import pandas as pd


COLORS = {
    "text": "#365F8D",
    "audio": "#D07A35",
    "vision": "#3F8F8A",
    "dark": "#243447",
    "muted": "#708090",
    "light": "#E9EEF2",
    "incomplete": "#C65353",
}
MODALITIES = ("text", "audio", "vision")
DISPLAY = {"text": "Text", "audio": "Audio", "vision": "Vision"}
MASK_ALPHA = 0.52
MASK_VALID_COLOR = "#FBFCFB"
MASK_FIGURE_WIDTH_IN = 10.2
MASK_SUMMARY_HEIGHT_IN = 1.05
MASK_ROW_HEIGHT_IN = 1.18
MASK_LEGEND_HEIGHT_IN = 0.28


def configure_style() -> None:
    """Compact two-dimensional style suitable for journal-width figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        color=COLORS["dark"],
    )


def clean_axis(ax: plt.Axes, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", color="#D9E1E7", linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    # Keep every figure and its three export formats in one named folder.
    # The plotting source remains in stepE_alignment/code/ and is not copied
    # into the image archive, preventing multiple diverging code versions.
    figure_dir = output_dir / stem
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{stem}.png", dpi=600)
    fig.savefig(figure_dir / f"{stem}.pdf")
    fig.savefig(figure_dir / f"{stem}.svg")
    plt.close(fig)


def read_data(root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "index": root / "stepA_index/data/sample_index.csv",
        "text": root / "stepB_text/data/text_extraction_log.csv",
        "text_validation": root / "stepB_text/data/text_feature_validation.csv",
        "audio": root / "stepC_audio/data/audio_extraction_log.csv",
        "audio_alignment": root / "stepC_audio/data/audio_alignment_log.csv",
        "vision": root / "stepD_vision/data/vision_extraction_log.csv",
        "vision_alignment": root / "stepD_vision/data/vision_alignment_log.csv",
        "vision_frames": root / "stepD_vision/data/vision_frame_log.csv",
        "audit": root / "stepE_alignment/data/multimodal_sample_audit.csv",
        "alignment": root / "stepE_alignment/data/multimodal_alignment_log.csv",
        "coverage": root / "stepE_alignment/data/multimodal_coverage_summary.csv",
        "aligned_validation": root / "stepE_alignment/data/aligned_50_validation.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required CSV files:\n" + "\n".join(missing))
    return {name: pd.read_csv(path, keep_default_na=False) for name, path in paths.items()}


def mask_matrices(data: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    order = data["index"]["sample_id"].astype(str).tolist()
    alignment = data["alignment"].copy()
    alignment["sample_id"] = alignment["sample_id"].astype(str)
    alignment["bin_index"] = pd.to_numeric(alignment["bin_index"], errors="raise").astype(int)
    alignment["valid"] = pd.to_numeric(alignment["valid"], errors="raise").astype(int)
    matrices: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        subset = alignment.loc[alignment["modality"].str.lower() == modality]
        table = subset.pivot(index="sample_id", columns="bin_index", values="valid")
        table = table.reindex(index=order, columns=range(50), fill_value=0).fillna(0)
        matrices[modality] = table.to_numpy(dtype=np.uint8)
    return matrices


def merged_sample_table(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    index = data["index"][["sample_id", "annotation", "label", "duration_sec"]].copy()
    audit = data["audit"][
        ["sample_id", "status", "text_valid_bins", "audio_valid_bins", "vision_valid_bins"]
    ].rename(columns={"status": "multimodal_status"})
    text = data["text"][["sample_id", "token_count", "original_token_count", "truncated"]]
    audio = data["audio"][
        [
            "sample_id",
            "decoded_audio_duration_sec",
            "audio_video_duration_diff_sec",
            "frame_count",
            "status",
        ]
    ].rename(columns={"status": "audio_extraction_status", "frame_count": "audio_frame_count"})
    vision = data["vision"][
        [
            "sample_id",
            "sampled_frames",
            "face_detected_frames",
            "multiple_face_samples",
            "timestamp_source",
            "status",
        ]
    ].rename(columns={"status": "vision_extraction_status"})
    merged = index.merge(audit, on="sample_id", validate="one_to_one")
    merged = merged.merge(text, on="sample_id", validate="one_to_one")
    merged = merged.merge(audio, on="sample_id", validate="one_to_one")
    merged = merged.merge(vision, on="sample_id", validate="one_to_one")
    numeric = [
        "label",
        "duration_sec",
        "text_valid_bins",
        "audio_valid_bins",
        "vision_valid_bins",
        "token_count",
        "original_token_count",
        "decoded_audio_duration_sec",
        "audio_video_duration_diff_sec",
        "audio_frame_count",
        "sampled_frames",
        "face_detected_frames",
        "multiple_face_samples",
    ]
    for column in numeric:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["face_detection_rate"] = np.divide(
        merged["face_detected_frames"],
        merged["sampled_frames"],
        out=np.zeros(len(merged), dtype=float),
        where=merged["sampled_frames"].to_numpy() > 0,
    )
    merged["audio_duration_diff_ms"] = 1000.0 * merged["audio_video_duration_diff_sec"]
    return merged


def draw_raincloud(
    ax: plt.Axes,
    groups: list[np.ndarray],
    color: str,
    rng: np.random.Generator,
    summary_label: str = "Median",
) -> None:
    """Draw one modality's three label groups as half-violins, boxes and raw points."""
    positions = np.arange(len(groups), dtype=float)
    y_grid = np.linspace(0.0, 50.0, 301)
    for position, values in zip(positions, groups):
        values = np.asarray(values, dtype=float)
        spread = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        bandwidth = 1.06 * spread * max(len(values), 1) ** (-0.2)
        bandwidth = float(np.clip(bandwidth if bandwidth > 0 else 0.8, 0.7, 4.0))
        density = np.exp(-0.5 * ((y_grid[:, None] - values[None, :]) / bandwidth) ** 2).sum(axis=1)
        density /= density.max() if density.max() else 1.0

        # Half violin: the left side encodes the density while the raw points
        # and compact box occupy the right side of the same label position.
        ax.fill_betweenx(
            y_grid,
            position - 0.32 * density,
            position,
            color=color,
            alpha=0.25,
            linewidth=0,
            zorder=1,
        )
        ax.boxplot(
            [values],
            positions=[position + 0.08],
            widths=0.18,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": color, "edgecolor": COLORS["dark"], "linewidth": 0.8, "alpha": 0.62},
            medianprops={"color": COLORS["dark"], "linewidth": 1.2},
            whiskerprops={"color": COLORS["dark"], "linewidth": 0.8},
            capprops={"color": COLORS["dark"], "linewidth": 0.8},
            zorder=3,
        )
        jitter = rng.uniform(-0.095, 0.095, len(values))
        ax.scatter(
            position + 0.25 + jitter,
            values,
            s=16,
            color=color,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.25,
            rasterized=True,
            zorder=4,
        )
        ax.text(
            position,
            1.012,
            f"n={len(values)}\n{summary_label}={np.median(values):.1f}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=COLORS["muted"],
            linespacing=1.1,
            clip_on=False,
        )


def plot_overview(
    table: pd.DataFrame, matrices: dict[str, np.ndarray], output_dir: Path
) -> None:
    values = [matrices[m].sum(axis=1) for m in MODALITIES]
    coverage = np.array([matrices[m].mean() * 100 for m in MODALITIES])
    any_valid = np.array([(v > 0).sum() for v in values])
    all_valid = np.array([(v == 50).sum() for v in values])
    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0])

    ax = fig.add_subplot(gs[0, :])
    rng = np.random.default_rng(42)
    for idx, (modality, vals) in enumerate(zip(MODALITIES, values), start=1):
        violin = ax.violinplot(vals, positions=[idx], widths=0.72, showextrema=False)
        body = violin["bodies"][0]
        body.set_facecolor(COLORS[modality])
        body.set_edgecolor("none")
        body.set_alpha(0.25)
        jitter = rng.uniform(-0.14, 0.14, size=len(vals))
        ax.scatter(
            np.full(len(vals), idx) + jitter,
            vals,
            s=8,
            color=COLORS[modality],
            alpha=0.42,
            linewidth=0,
            rasterized=True,
        )
        bp = ax.boxplot(
            vals,
            positions=[idx],
            widths=0.25,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": COLORS["dark"], "linewidth": 1.2},
            whiskerprops={"color": COLORS["dark"], "linewidth": 0.8},
            capprops={"color": COLORS["dark"], "linewidth": 0.8},
            boxprops={"facecolor": "white", "edgecolor": COLORS["dark"], "linewidth": 0.8},
        )
        _ = bp
        ax.text(idx, 51.1, f"mean {np.mean(vals):.1f}", ha="center", va="bottom", color=COLORS[modality])
    ax.set_xticks([1, 2, 3], [DISPLAY[m] for m in MODALITIES])
    ax.set_ylabel("Valid bins per sample (0–50)")
    ax.set_ylim(-1, 55)
    ax.set_title("Distribution of valid sequence positions")
    clean_axis(ax, grid=True)
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[1, 0])
    y = np.arange(3)
    ax.barh(y, coverage, color=[COLORS[m] for m in MODALITIES], height=0.58)
    for yi, value in zip(y, coverage):
        ax.text(min(value + 1.4, 101.5), yi, f"{value:.1f}%", va="center", color=COLORS["dark"])
    ax.set_yticks(y, [DISPLAY[m] for m in MODALITIES])
    ax.invert_yaxis()
    ax.set_xlim(0, 106)
    ax.set_xlabel("Valid observations among 5,000 bins (%)")
    ax.set_title("Overall mask coverage")
    clean_axis(ax, grid=False)
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    for yi, modality, n_any, n_all in zip(y, MODALITIES, any_valid, all_valid):
        ax.plot([n_all, n_any], [yi, yi], color="#C8D2DA", linewidth=2.0, zorder=1)
        ax.scatter(n_any, yi, s=34, color=COLORS[modality], marker="o", zorder=2)
        ax.scatter(n_all, yi, s=34, facecolor="white", edgecolor=COLORS[modality], marker="o", zorder=2)
        ax.text(n_any + 1.5, yi - 0.12, str(n_any), va="center", color=COLORS["dark"])
        ax.text(n_all - 1.5, yi + 0.18, str(n_all), ha="right", va="center", color=COLORS["dark"])
    ax.set_yticks(y, [DISPLAY[m] for m in MODALITIES])
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("Samples (filled: ≥1 valid bin; open: all 50)")
    ax.set_title("Coverage thresholds")
    clean_axis(ax, grid=False)
    panel_label(ax, "c")
    fig.suptitle("Aligned-50 multimodal quality overview", fontsize=12, fontweight="bold", color=COLORS["dark"])
    save_figure(fig, output_dir, "fig1_multimodal_qc_overview")


def plot_mask_structure(
    table: pd.DataFrame, matrices: dict[str, np.ndarray], output_dir: Path
) -> None:
    """Render the availability masks as three aligned 3D terrain panels."""
    text_order = np.argsort(table["token_count"].to_numpy())
    av_order = np.lexsort(
        (
            table["audio_valid_bins"].to_numpy(),
            table["vision_valid_bins"].to_numpy(),
        )
    )[::-1]
    ordered = {"text": text_order, "audio": av_order, "vision": av_order}
    fig = plt.figure(figsize=(11.2, 5.8), constrained_layout=True)
    axes = [fig.add_subplot(1, 3, idx + 1, projection="3d") for idx in range(3)]
    x_grid, y_grid = np.meshgrid(np.arange(1, 51), np.arange(1, 101))
    for ax, modality in zip(axes, MODALITIES):
        mask = matrices[modality][ordered[modality]].astype(float)
        z_grid = 0.04 + 0.96 * mask
        valid_rgba = mcolors.to_rgba(COLORS[modality], 0.90)
        missing_rgba = mcolors.to_rgba(COLORS[modality], 0.22)
        facecolors = np.where(mask[:-1, :-1, None] > 0.5, valid_rgba, missing_rgba)
        ax.plot_surface(x_grid, y_grid, np.zeros_like(z_grid), color=MASK_VALID_COLOR,
                        alpha=0.22, linewidth=0, shade=False)
        ax.plot_surface(x_grid, y_grid, z_grid, facecolors=facecolors,
                        linewidth=0, antialiased=False, shade=True)
        profile = mask.mean(axis=0) * 100
        ax.plot(np.arange(1, 51), np.full(50, -7.0), 1.06 + 0.22 * profile / 100.0,
                color=COLORS[modality], linewidth=2.0, label="availability profile")
        ax.set_title(f"{DISPLAY[modality]}  ·  {mask.mean() * 100:.1f}% coverage",
                     color=COLORS[modality], fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Sequence position", labelpad=5)
        ax.set_ylabel("Ranked sample", labelpad=5)
        ax.set_zlabel("" if modality != "vision" else "Availability state (0/1)", labelpad=5)
        ax.set_xlim(1, 50); ax.set_ylim(-9, 100); ax.set_zlim(0, 1.35)
        ax.set_xticks([1, 10, 20, 30, 40, 50])
        ax.set_yticks([1, 25, 50, 75, 100])
        ax.set_zticks([0, 1]); ax.set_zticklabels(["empty", "valid"])
        ax.view_init(elev=28, azim=-58)
        ax.set_box_aspect((1.5, 2.0, 0.62))
        ax.grid(True, linewidth=0.35, alpha=0.35)
    axes[-1].text2D(0.02, -0.11, "Profile line is placed in front of the mask surface.",
                    transform=axes[-1].transAxes, fontsize=7, color=COLORS["muted"])
    fig.legend(
        handles=[Patch(facecolor=COLORS["text"], alpha=0.9, label="Valid observation surface"),
                 Patch(facecolor=COLORS["text"], alpha=0.22, label="Unavailable / empty bin"),
                 Line2D([0], [0], color=COLORS["dark"], linewidth=2, label="Availability profile")],
        loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle("Three-modality sequence availability and observation masks (3D)",
                 fontsize=12, fontweight="bold", color=COLORS["dark"])
    save_figure(fig, output_dir, "fig2_mask_structure_revised")


def plot_alignment_profiles(matrices: dict[str, np.ndarray], output_dir: Path) -> None:
    # All profiles share one normalized sequence-position axis. Text uses the
    # same 50-bin position index, while its semantics remain uniform token
    # position rather than video time.
    x = np.linspace(0.0, 1.0, 50)
    text_profile = matrices["text"].mean(axis=0) * 100
    audio_profile = matrices["audio"].mean(axis=0) * 100
    vision_profile = matrices["vision"].mean(axis=0) * 100
    both = ((matrices["audio"] == 1) & (matrices["vision"] == 1)).mean(axis=0) * 100
    audio_only = ((matrices["audio"] == 1) & (matrices["vision"] == 0)).mean(axis=0) * 100
    vision_only = ((matrices["audio"] == 0) & (matrices["vision"] == 1)).mean(axis=0) * 100
    neither = ((matrices["audio"] == 0) & (matrices["vision"] == 0)).mean(axis=0) * 100

    state_colors = ["#5C7F6F", COLORS["audio"], COLORS["vision"], "#D7DEE4"]
    state_labels = ["Both valid", "Audio only", "Vision only", "Neither"]
    fig = plt.figure(figsize=(9.3, 6.0), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    profile_by_modality = (("text", text_profile, 2.8), ("audio", audio_profile, 1.8),
                           ("vision", vision_profile, 0.8))
    for modality, values, y_value in profile_by_modality:
        ax.plot(x, np.full_like(x, y_value), values, color=COLORS[modality], linewidth=2.4,
                marker="o", markersize=2.8, markevery=5, label=DISPLAY[modality])
        ax.plot([0, 1], [y_value, y_value], [values.mean(), values.mean()],
                color=COLORS["muted"], linewidth=0.7, linestyle=(0, (3, 2)), alpha=0.85)
        ax.text(1.015, y_value, float(values[-1]), f"{values[-1]:.0f}%",
                color=COLORS[modality], fontsize=8, ha="left", va="center")
    joint_values = (both, audio_only, vision_only, neither)
    x_centers = (np.arange(50) + 0.5) / 50.0
    left = np.zeros_like(x_centers)
    for values, color in zip(joint_values, state_colors):
        ax.bar3d(x_centers, np.full(50, 0.06), left, np.full(50, 0.016),
                 np.full(50, 0.34), values, color=color, alpha=0.92, shade=True,
                 linewidth=0.05, edgecolor="white")
        left += values
    ax.set_xlim(0, 1.08); ax.set_ylim(-0.05, 3.2); ax.set_zlim(0, 110)
    ax.set_xticks(np.linspace(0, 1, 5), ["0.00", "0.25", "0.50", "0.75", "1.00"])
    ax.set_yticks([0.23, 0.8, 1.8, 2.8], ["Joint state", "Vision", "Audio", "Text"])
    ax.set_zticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Normalized sequence position", labelpad=7)
    ax.set_ylabel("Modality", labelpad=8)
    ax.set_zlabel("Availability / composition (%)", labelpad=7)
    ax.view_init(elev=25, azim=-61)
    ax.set_box_aspect((1.65, 1.55, 1.0))
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(handles=[Line2D([0], [0], color=COLORS[m], linewidth=2.4, label=DISPLAY[m])
                       for m in MODALITIES] +
              [Patch(facecolor=c, alpha=0.92, label=l) for c, l in zip(state_colors, state_labels)],
              loc="upper left", bbox_to_anchor=(0.02, 0.98), ncol=2, frameon=False, fontsize=7)
    ax.text2D(0.02, -0.08, "Text uses uniform token position; audio and vision use normalized video-time bins.",
              transform=ax.transAxes, fontsize=7.2, color=COLORS["muted"])
    ax.set_title("Multimodal sequence-position availability profile (3D)", fontsize=12,
                 fontweight="bold", color=COLORS["dark"], pad=12)
    save_figure(fig, output_dir, "fig3_alignment_profiles")


def plot_label_stratified(table: pd.DataFrame, output_dir: Path) -> None:
    categories = ["Negative", "Neutral", "Positive"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), sharey=True, constrained_layout=True)
    rng = np.random.default_rng(42)
    for ax, modality in zip(axes, MODALITIES):
        column = f"{modality}_valid_bins"
        groups = [table.loc[table["annotation"] == category, column].dropna().to_numpy() for category in categories]
        draw_raincloud(ax, groups, COLORS[modality], rng)
        ax.set_xticks(np.arange(3), ["Neg.", "Neu.", "Pos."])
        ax.set_xlim(-0.48, 2.58)
        ax.set_ylim(0, 50)
        ax.set_xlabel("Sentiment annotation")
        ax.set_title(f"{DISPLAY[modality]} valid bins", fontsize=9, pad=18, color=COLORS[modality], fontweight="bold")
        clean_axis(ax, grid=True)
        ax.grid(axis="y", color="#D9E1E7", linewidth=0.45, alpha=0.55)
        ax.spines["left"].set_linewidth(0.65)
        ax.spines["bottom"].set_linewidth(0.65)
    axes[0].set_ylabel("Valid bins per sample (0–50)")
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    fig.suptitle("Feature availability stratified by sentiment label", fontsize=11, fontweight="bold", color=COLORS["dark"])
    save_figure(fig, output_dir, "fig4_label_stratified_availability")


def spearman(x: pd.Series, y: pd.Series) -> float:
    frame = pd.concat([x, y], axis=1).dropna()
    ranks = frame.rank(method="average")
    return float(ranks.iloc[:, 0].corr(ranks.iloc[:, 1], method="pearson"))


def plot_extraction_diagnostics(table: pd.DataFrame, output_dir: Path) -> dict[str, float]:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.25), constrained_layout=True)

    ax = axes[0]
    ax.scatter(
        table["token_count"],
        table["text_valid_bins"],
        s=18,
        color=COLORS["text"],
        alpha=0.68,
        edgecolor="white",
        linewidth=0.35,
    )
    line_x = np.linspace(0, max(90, table["token_count"].max()), 200)
    ax.plot(line_x, np.minimum(line_x, 50), color=COLORS["dark"], linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_xlabel("Content tokens per sample")
    ax.set_ylabel("Valid text bins")
    ax.set_title("Token count and text-bin occupancy")
    ax.text(
        0.03,
        0.96,
        "Valid bins = min(tokens, 50)",
        transform=ax.transAxes,
        va="top",
        color=COLORS["dark"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    clean_axis(ax, grid=True)
    panel_label(ax, "a")

    ax = axes[1]
    audio_x = table["audio_duration_diff_ms"]
    ax.scatter(
        audio_x,
        table["audio_valid_bins"],
        s=18,
        color=COLORS["audio"],
        alpha=0.68,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.axvline(0, color=COLORS["muted"], linewidth=0.7, linestyle=(0, (3, 2)))
    ax.set_xlabel("Decoded audio − video duration\n(ms)")
    ax.set_ylabel("Valid audio bins")
    ax.set_title("Audio tail shortfall and valid bins")
    ax.set_ylim(46.8, 50.2)
    ax.set_yticks([47, 48, 49, 50])
    audio_counts = table["audio_valid_bins"].round().astype(int).value_counts().sort_index()
    for level, count in audio_counts.items():
        ax.annotate(
            f"n={count}",
            xy=(1.0, level),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=6.5,
            color=COLORS["dark"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
        )
    clean_axis(ax, grid=True)
    panel_label(ax, "b")

    ax = axes[2]
    sizes = 12 + 28 * np.sqrt(table["multiple_face_samples"] / max(1, table["multiple_face_samples"].max()))
    status_color = np.where(table["vision_extraction_status"].eq("PASS"), COLORS["vision"], COLORS["incomplete"])
    rng = np.random.default_rng(42)
    face_x = np.clip(table["face_detection_rate"].to_numpy() * 100 + rng.uniform(-0.8, 0.8, len(table)), 0, 102)
    ax.scatter(
        face_x,
        table["vision_valid_bins"],
        s=sizes,
        c=status_color,
        alpha=0.68,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xlabel("Sampled frames with a detected face (%)")
    ax.set_ylabel("Valid vision bins")
    ax.set_title("Face detection and visual coverage")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["vision"], markeredgecolor="white", markersize=5.5, label="PASS"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["incomplete"], markeredgecolor="white", markersize=5.5, label="No face detected"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["vision"], markeredgecolor="white", markersize=8.0, label="Size: multiple-face frames"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=6.2,
        handletextpad=0.35,
        borderpad=0.2,
    )
    clean_axis(ax, grid=True)
    panel_label(ax, "c")

    correlations = {
        "token_count_vs_text_valid_bins_spearman": spearman(table["token_count"], table["text_valid_bins"]),
        "audio_duration_diff_ms_vs_audio_valid_bins_spearman": spearman(
            table["audio_duration_diff_ms"], table["audio_valid_bins"]
        ),
        "face_detection_rate_vs_vision_valid_bins_spearman": spearman(
            table["face_detection_rate"], table["vision_valid_bins"]
        ),
    }
    axes[1].text(
        0.03,
        0.08,
        f"Spearman ρ = {correlations['audio_duration_diff_ms_vs_audio_valid_bins_spearman']:.2f}",
        transform=axes[1].transAxes,
        va="bottom",
        color=COLORS["dark"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    axes[2].text(
        0.97,
        0.96,
        f"Spearman ρ = {correlations['face_detection_rate_vs_vision_valid_bins_spearman']:.2f}",
        transform=axes[2].transAxes,
        va="top",
        ha="right",
        color=COLORS["dark"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.5},
    )
    save_figure(fig, output_dir, "fig5_extraction_diagnostics")
    return correlations


def plot_pipeline_traceability(data: dict[str, pd.DataFrame], table: pd.DataFrame, output_dir: Path) -> None:
    def passed(frame: pd.DataFrame, column: str, value: str = "PASS") -> int:
        return int(frame[column].astype(str).eq(value).sum())

    stages = [
        "Indexed\nvideo + label",
        "Text feature\nvalidated",
        "Audio feature\nextracted",
        "Vision validity\ncheck",
        "Aligned file\nvalidated",
    ]
    counts = np.array(
        [
            passed(data["index"], "index_status"),
            passed(data["text_validation"], "status"),
            passed(data["audio"], "status"),
            int((table["vision_valid_bins"] > 0).sum()),
            passed(data["aligned_validation"], "status"),
        ]
    )
    fig, ax = plt.subplots(figsize=(8.0, 3.2), constrained_layout=True)
    x = np.arange(len(stages))
    total = np.full(len(stages), 100)
    visual_valid = int(counts[3])
    # Total retention and visual observability are different quantities. The
    # second line begins when the visual check is performed and stays at 90.
    ax.plot(x, total, color=COLORS["dark"], linewidth=2.0, marker="o", markersize=6,
            label="Samples retained")
    visual_series = np.array([np.nan, np.nan, np.nan, visual_valid, visual_valid], dtype=float)
    ax.plot(x, visual_series, color=COLORS["vision"], linewidth=2.0, marker="o", markersize=6,
            label="Samples with valid vision")
    stage_colors = [COLORS["dark"], COLORS["text"], COLORS["audio"], COLORS["vision"], "#5C7F6F"]
    ax.scatter(x, total, s=80, color=stage_colors, zorder=3, edgecolor="white", linewidth=0.7)
    for xi in x:
        ax.text(xi, 101.0, "100", ha="center", va="bottom", fontweight="bold", color=COLORS["dark"], fontsize=8)
    ax.text(3, visual_valid - 1.0, f"{visual_valid}", ha="center", va="top", fontweight="bold", color=COLORS["vision"], fontsize=8)
    ax.text(4, visual_valid - 1.0, f"{visual_valid}", ha="center", va="top", fontweight="bold", color=COLORS["vision"], fontsize=8)
    ax.vlines(3, visual_valid, 100, color=COLORS["incomplete"], linewidth=1.2, linestyle=(0, (2, 2)), zorder=2)
    ax.annotate(
        f"{100 - visual_valid} samples have no valid face\nbut remain in aligned-50 (vision_mask = 0)",
        xy=(3, (visual_valid + 100) / 2), xytext=(2.15, 87.0),
        arrowprops={"arrowstyle": "-", "color": COLORS["incomplete"], "linewidth": 0.8},
        color=COLORS["incomplete"],
        ha="center",
    )
    ax.set_xticks(x, stages)
    ax.set_ylim(86, 103)
    ax.set_ylabel("Samples (n = 100)")
    ax.set_title("Pipeline checks and retained missing-vision samples", loc="left", fontsize=11, fontweight="bold", color=COLORS["dark"])
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    clean_axis(ax, grid=True)
    save_figure(fig, output_dir, "fig6_pipeline_traceability")


def write_statistics(
    root: Path,
    data: dict[str, pd.DataFrame],
    table: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    correlations: dict[str, float],
) -> Path:
    records: list[dict[str, object]] = []
    for modality in MODALITIES:
        valid = matrices[modality].sum(axis=1)
        records.extend(
            [
                {"section": "modality", "metric": f"{modality}_coverage_pct", "value": matrices[modality].mean() * 100},
                {"section": "modality", "metric": f"{modality}_valid_bins_median", "value": np.median(valid)},
                {"section": "modality", "metric": f"{modality}_valid_bins_mean", "value": np.mean(valid)},
                {"section": "modality", "metric": f"{modality}_samples_any_valid", "value": (valid > 0).sum()},
                {"section": "modality", "metric": f"{modality}_samples_all_50", "value": (valid == 50).sum()},
            ]
        )
    records.extend(
        [
            {
                "section": "pipeline",
                "metric": "complete_three_modality_samples",
                "value": table["multimodal_status"].eq("PASS").sum(),
            },
            {
                "section": "pipeline",
                "metric": "aligned_structural_pass",
                "value": data["aligned_validation"]["status"].eq("PASS").sum(),
            },
        ]
    )
    records.extend(
        {"section": "diagnostic", "metric": key, "value": value}
        for key, value in correlations.items()
    )
    output = root / "stepE_alignment/data/publication_figure_statistics.csv"
    pd.DataFrame(records).to_csv(output, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    root = workspace / "02_features_q1"
    output_dir = root / "stepE_alignment/image/publication"
    configure_style()
    data = read_data(root)
    matrices = mask_matrices(data)
    table = merged_sample_table(data)

    plot_overview(table, matrices, output_dir)
    plot_mask_structure(table, matrices, output_dir)
    plot_alignment_profiles(matrices, output_dir)
    plot_label_stratified(table, output_dir)
    correlations = plot_extraction_diagnostics(table, output_dir)
    plot_pipeline_traceability(data, table, output_dir)
    statistics_path = write_statistics(root, data, table, matrices, correlations)

    print(f"publication_figures={output_dir}")
    print(f"statistics={statistics_path}")
    print(f"samples={len(table)} complete={(table['multimodal_status'] == 'PASS').sum()}")
    for key, value in correlations.items():
        print(f"{key}={value:.4f}")


if __name__ == "__main__":
    main()
