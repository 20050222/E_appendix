"""Publication-style figures for Question 3.

The script consumes only the frozen Question 3 CSV outputs.  Each figure is
written to its own directory as PNG, SVG, and PDF so that raster previews and
editable/vector artwork are both retained.
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


MODALITIES = ["Text", "Audio", "Vision"]
MODALITY_KEYS = ["text", "audio", "vision"]
COLORS = {
    "Text": "#2F5D62",
    "Audio": "#C17C5B",
    "Vision": "#5A7D62",
}
CLASS_COLORS = {
    "Negative": "#6B7A8F",
    "Neutral": "#C39B4A",
    "Positive": "#4C806B",
}


def configure_style() -> None:
    """Set restrained, vector-first defaults suitable for manuscript figures."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
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


def natural_ids(values: pd.Series) -> list[str]:
    return sorted(values.astype(str).unique(), key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else s))


def despine(ax: mpl.axes.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#4A4A4A")
    ax.spines["bottom"].set_color("#4A4A4A")
    if grid:
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.7)
        ax.set_axisbelow(True)


def save_triplet(fig: mpl.figure.Figure, out_root: Path, stem: str) -> None:
    folder = out_root / stem
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.png", dpi=600)
    fig.savefig(folder / f"{stem}.svg")
    fig.savefig(folder / f"{stem}.pdf")
    plt.close(fig)


def protocol_selection(validation: pd.DataFrame, out_root: Path) -> None:
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
        ["Classification", "Regression"],
    ):
        vals = agg[col].to_numpy()
        ax.plot(ratios, vals, marker="o", markersize=4.5, linewidth=1.6, color="#3D596B")
        ax.fill_between(ratios, vals, 0, color="#3D596B", alpha=0.07)
        ax.axvline(10, color="#8F3B3B", linestyle=(0, (3, 2)), linewidth=1.0)
        ax.text(10.8, ax.get_ylim()[1] * 0.93, "selected", color="#8F3B3B", fontsize=7)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Candidate window ratio (%)")
        ax.set_ylabel("Faithfulness gap")
        ax.set_xticks(ratios)
        despine(ax)
    fig.suptitle("Explanation-window protocol selection", x=0.03, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.03, -0.02, "Larger gaps indicate stronger separation between selected and random evidence windows.", fontsize=7.5, color="#555555")
    fig.tight_layout(w_pad=2.0)
    save_triplet(fig, out_root, "q3_fig1_protocol_selection")


def modal_contribution(modal: pd.DataFrame, out_root: Path) -> None:
    rows = []
    for modality, key in zip(MODALITIES, MODALITY_KEYS):
        for task, prefix in [("Classification", "classification"), ("Regression", "regression")]:
            vals = modal[f"{prefix}_normalized_{key}"].astype(float)
            rows.append({"modality": modality, "task": task, "mean": vals.mean(), "sd": vals.std(ddof=1)})
    summary = pd.DataFrame(rows)
    x = np.arange(len(MODALITIES))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.9, 3.25))
    for j, task in enumerate(["Classification", "Regression"]):
        cur = summary[summary.task == task]
        vals = cur["mean"].to_numpy()
        errs = cur["sd"].to_numpy()
        bars = ax.bar(
            x + (j - 0.5) * width,
            vals,
            width,
            yerr=errs,
            capsize=2.5,
            color=[COLORS[m] for m in MODALITIES],
            alpha=0.78 if j == 0 else 0.45,
            edgecolor="#2B2B2B",
            linewidth=0.45,
            label=task,
        )
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.025, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, MODALITIES)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Mean normalized perturbation contribution")
    ax.set_title("Formal modality contribution across test samples", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.text(0.0, -0.19, "Bars show mean ± sample SD; contributions are based on modality occlusion, not internal fusion weights.", transform=ax.transAxes, fontsize=7.2, color="#555555")
    despine(ax)
    fig.tight_layout()
    save_triplet(fig, out_root, "q3_fig2_modal_contribution")


def contribution_heatmap(modal: pd.DataFrame, out_root: Path) -> None:
    ids = natural_ids(modal["sample_id"])
    ordered = modal.set_index("sample_id").loc[ids]
    cmap = LinearSegmentedColormap.from_list("contribution", ["#F4F5F2", "#B7C9BE", "#2F5D62"])
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 5.0), sharey=True, constrained_layout=True)
    for ax, prefix, title in zip(
        axes,
        ["classification", "regression"],
        ["Classification contribution", "Regression contribution"],
    ):
        matrix = ordered[[f"{prefix}_normalized_{key}" for key in MODALITY_KEYS]].to_numpy()
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(3), ["Text", "Audio", "Vision"])
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Modality")
        ax.set_yticks(range(len(ids)), ids if ax is axes[0] else [])
        if ax is axes[0]:
            ax.set_ylabel("Attachment 4 sample")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.3, color="white" if value > 0.58 else "#263238")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    cbar.set_label("Normalized contribution", rotation=90)
    cbar.outline.set_linewidth(0.4)
    fig.suptitle("Sample-level explanation profiles", x=0.03, ha="left", fontsize=11, fontweight="bold")
    save_triplet(fig, out_root, "q3_fig3_sample_contribution_heatmap")


def temporal_evidence(temporal: pd.DataFrame, modal: pd.DataFrame, out_root: Path) -> None:
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
    sample["modality_label"] = sample["modality"].map({"text": "Text", "audio": "Audio", "vision": "Vision"})
    y_positions = {"Text": 2, "Audio": 1, "Vision": 0}
    cmap = mpl.colormaps["RdBu_r"]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.25), sharex=True, constrained_layout=True)
    for ax, delta_col, selected_col, title in [
        (axes[0], "classification_delta_signed", "selected_classification", "Classification evidence"),
        (axes[1], "regression_delta_signed", "selected_regression", "Regression evidence"),
    ]:
        for modality in MODALITIES:
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
        ax.set_yticks([2, 1, 0], MODALITIES)
        ax.set_xlim(0, 1)
        ax.set_ylabel("Modality")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color="#E1E1E1", linewidth=0.5)
        despine(ax, grid=False)
    axes[1].set_xlabel("Normalized position (token/bin for text; time for audio and vision)")
    sm = mpl.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Signed evidence delta (relative scale)")
    fig.suptitle(f"Top-k local evidence windows: sample {candidate}", x=0.03, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.03, -0.02, "Only windows selected by the frozen explanation protocol are shown; positions are normalized because attachment 4 PKL lacks duration metadata.", fontsize=7.2, color="#555555")
    save_triplet(fig, out_root, "q3_fig4_temporal_evidence")


def prediction_profile(prediction: pd.DataFrame, out_root: Path) -> None:
    ids = natural_ids(prediction["sample_id"])
    df = prediction.set_index("sample_id").loc[ids]
    x = np.arange(len(df))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.7), gridspec_kw={"height_ratios": [1.35, 1]}, sharex=True)
    bottom = np.zeros(len(df))
    for cls, col in [("Negative", "prob_negative"), ("Neutral", "prob_neutral"), ("Positive", "prob_positive")]:
        vals = df[col].to_numpy()
        axes[0].bar(x, vals, bottom=bottom, width=0.76, color=CLASS_COLORS[cls], label=cls, edgecolor="white", linewidth=0.35)
        bottom += vals
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Class probability")
    axes[0].set_title("Prediction profile on attachment 4", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=3, loc="upper right")
    despine(axes[0])
    intensities = df["predicted_intensity"].to_numpy()
    colors = [CLASS_COLORS[str(c)] for c in df["predicted_class"]]
    axes[1].bar(x, intensities, color=colors, width=0.76, edgecolor="#2B2B2B", linewidth=0.35)
    axes[1].axhline(0, color="#444444", linewidth=0.65)
    axes[1].set_ylabel("Predicted intensity")
    axes[1].set_xlabel("Attachment 4 sample")
    axes[1].set_xticks(x, ids)
    axes[1].tick_params(axis="x", rotation=45)
    despine(axes[1], grid=False)
    handles = [Patch(facecolor=CLASS_COLORS[c], edgecolor="none", label=c) for c in CLASS_COLORS]
    axes[1].legend(handles=handles, frameon=False, ncol=3, loc="upper right")
    fig.tight_layout(h_pad=1.25)
    save_triplet(fig, out_root, "q3_fig5_prediction_profile")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "image" / "publication")
    args = parser.parse_args()
    configure_style()
    validation = pd.read_csv(args.data / "q3_explanation_validation.csv")
    modal = pd.read_csv(args.data / "q3_modal_contribution.csv")
    temporal = pd.read_csv(args.data / "q3_temporal_evidence.csv")
    prediction = pd.read_csv(args.data / "q3_prediction.csv")
    # Keep sample identifiers stable across CSVs (some files are inferred as int,
    # others as string because the validation set also contains composite IDs).
    for frame in [modal, temporal, prediction]:
        frame["sample_id"] = frame["sample_id"].astype(str)
    for frame in [validation, modal, temporal, prediction]:
        numeric = frame.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError("Input CSV contains NaN or infinite numeric values")
    protocol_selection(validation, args.out)
    modal_contribution(modal, args.out)
    contribution_heatmap(modal, args.out)
    temporal_evidence(temporal, modal, args.out)
    prediction_profile(prediction, args.out)
    print(f"Generated five Q3 publication figures under: {args.out}")


if __name__ == "__main__":
    main()
