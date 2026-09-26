"""Chinese figures with the same data, layout and styles as make_report.py."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

NAMES = {
    "fusion_clean": "完整融合", "fusion_aug": "融合+缺失增强",
    "robust_norec": "掩码+动态融合", "robust_rec": "掩码+融合+重建",
    "robust_nogate": "掩码+重建", "text_only": "仅文本",
}
SCENARIO = {"T": "文本", "A": "语音", "V": "视觉", "TAV": "三模态", "TA": "文本+语音", "TV": "文本+视觉", "AV": "语音+视觉"}
POSITION = {"front": "前段", "middle": "中段", "back": "后段"}
PATTERN = {"block": "连续区间", "two_blocks": "双区间", "points": "离散位置"}
FONT_TTC = Path("C:/Windows/Fonts/msyh.ttc")
FONT_CACHE = Path(tempfile.gettempdir()) / "cpmcm_q2_msyh_equivalent.ttf"


def configure_style() -> None:
    if not FONT_TTC.is_file():
        raise FileNotFoundError(f"未找到中文字体文件：{FONT_TTC}")
    from fontTools.ttLib import TTFont
    with TTFont(str(FONT_TTC), fontNumber=0) as font:
        font.flavor = None
        names = {1: "CPMCM Microsoft YaHei", 4: "CPMCM Microsoft YaHei", 6: "CPMCM-MicrosoftYaHei", 16: "CPMCM Microsoft YaHei"}
        for record in font["name"].names:
            if record.nameID in names:
                record.string = names[record.nameID].encode(record.getEncoding())
        font.save(FONT_CACHE)
    font_manager.fontManager.addfont(str(FONT_CACHE))
    font_name = font_manager.FontProperties(fname=str(FONT_CACHE)).get_name()
    # Same rcParams as make_report.py, with only the font family changed.
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [font_name, "DejaVu Sans"],
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .2, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def save(fig: plt.Figure, root: Path, stem: str) -> None:
    out = root / stem
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def draw_rates(s: pd.DataFrame, root: Path, output: Path) -> None:
    # 修改记录（可恢复）：
    # Q2-06-A 绝对 MAE 图保留共享 y 轴，便于四种缺失场景横向比较。
    # Q2-06-B 配对退化图改为“主图 + A/V 局部放大”，避免小幅变化贴在零线上。
    # Q2-06-C 退化 y 轴明确写 ΔMAE = MAE_missing - MAE_clean；阴影为遮挡种子 SD。
    # Q2-06-D 图例移至整图上方，说明固定代表模型与 3 个遮挡种子。
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), layout="constrained", sharey=True)
    for ax, mod in zip(axs.flat, ["T", "A", "V", "TAV"]):
        for model in s.variant.unique():
            z = s[(s.variant == model) & (s.group == "rate") & (s.scenario == mod)].groupby("rate").mae.agg(["mean", "std"])
            clean = s[(s.variant == model) & (s.group == "clean")].mae.iloc[0]
            xx = np.r_[0, z.index.to_numpy()]; yy = np.r_[clean, z["mean"].to_numpy()]; err = np.r_[0, z["std"].fillna(0).to_numpy()]
            ax.plot(xx, yy, "o-", label=NAMES[model]); ax.fill_between(xx, yy - err, yy + err, alpha=.15)
        ax.set(title=f"缺失{SCENARIO[mod]}", xlabel="内容位置的目标缺失比例", ylabel="MAE（越低越好）", xticks=[0, .1, .3, .5, .7])
    axs[0, 0].legend(fontsize=8); fig.suptitle("验证集鲁棒性：3 个遮挡实现的均值 ± 标准差")
    save(fig, output, "01_missing_rates")

    # Paired degradation: retain a common-scale overview and add local A/V zoom panels.
    fig = plt.figure(figsize=(12, 7.8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0])
    axs = np.asarray([
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ], dtype=object)
    for ax, mod in zip(axs, ["T", "A", "V", "TAV"]):
        for model in s.variant.unique():
            z = s[(s.variant == model) & (s.group == "rate") & (s.scenario == mod)].groupby("rate").mae.agg(["mean", "std"])
            clean = s[(s.variant == model) & (s.group == "clean")].mae.iloc[0]
            xx = np.r_[0, z.index.to_numpy()]; yy = np.r_[0, z["mean"].to_numpy() - clean]; err = np.r_[0, z["std"].fillna(0).to_numpy()]
            ax.plot(xx, yy, "o-", label=NAMES[model]); ax.fill_between(xx, yy - err, yy + err, alpha=.15)
        ax.axhline(0, color="gray", lw=.8)
        ax.set(title=f"缺失{SCENARIO[mod]}", xlabel="", ylabel="", xticks=[0, .1, .3, .5, .7])
        ax.grid(True, alpha=.2)
    # Common-scale panels show the overall size of degradation.
    axs[0].set_ylim(-.012, .105)
    axs[1].set_ylim(-.012, .105)
    # Local panels make the low-amplitude audio/vision effects readable.
    axs[2].set_ylim(-.008, .018)
    axs[3].set_ylim(-.010, .040)
    axs[2].set_title("缺失语音（局部放大）")
    axs[3].set_title("缺失视觉（局部放大）")
    axs[0].set_title("缺失文本（总览）")
    axs[1].set_title("缺失三模态（总览）")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("验证集配对退化：ΔMAE = 缺失输入 MAE − 完整输入 MAE", y=0.995, fontsize=13)
    fig.text(0.018, 0.50, "ΔMAE", rotation=90, va="center", ha="center", fontsize=10)
    fig.text(0.50, 0.085, "内容位置的目标缺失比例", ha="center", fontsize=10)
    fig.text(0.5, 0.018, "阴影表示 3 个遮挡种子的标准差；结果来自固定代表模型，完整输入点统一为 ΔMAE = 0。", ha="center", fontsize=9, color="0.35")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.16, top=0.86, hspace=0.48, wspace=0.16)
    save(fig, output, "06_paired_degradation")


def draw_missing_types(s: pd.DataFrame, selected: dict, output: Path) -> None:
    name = selected["selected_variant"]
    z = s[(s.variant == name) & (s.group == "rate")].pivot_table(index="scenario", columns="rate", values="macro_f1").reindex(["T", "A", "V", "TA", "TV", "AV", "TAV"])
    z.index = [SCENARIO[x] for x in z.index]
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained"); im = ax.imshow(z.to_numpy(), cmap="YlGnBu", aspect="auto"); ax.grid(False)
    ax.set(xticks=range(4), xticklabels=["10%", "30%", "50%", "70%"], yticks=range(7), yticklabels=z.index, xlabel="目标缺失比例", title="验证集不同缺失模态组合的 Macro-F1")
    threshold = (z.min().min() + z.max().max()) / 2
    for i in range(7):
        for j in range(4): ax.text(j, i, f"{z.iloc[i, j]:.3f}", ha="center", va="center", color="white" if z.iloc[i, j] > threshold else "black")
    fig.colorbar(im, ax=ax, label="Macro-F1（越高越好）"); save(fig, output, "02_missing_types")


def draw_position_gap(s: pd.DataFrame, selected: dict, output: Path) -> None:
    name = selected["selected_variant"]; fig, axs = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    pos = s[(s.variant == name) & (s.group == "position")].pivot_table(index="scenario", columns="position", values="mae").reindex(["T", "A", "V"])[["front", "middle", "back"]]
    pos.index = [SCENARIO[x] for x in pos.index]; pos.columns = [POSITION[x] for x in pos.columns]; pos.plot.bar(ax=axs[0], rot=0); axs[0].set(title="30% 缺失的位置影响", ylabel="MAE", xlabel="缺失模态")
    pat = s[(s.variant == name) & (s.group == "pattern")].pivot_table(index="scenario", columns="pattern", values="mae").reindex(["T", "A", "V"])[["block", "two_blocks", "points"]]
    pat.index = [SCENARIO[x] for x in pat.index]; pat.columns = [PATTERN[x] for x in pat.columns]; pat.plot.bar(ax=axs[1], rot=0); axs[1].set(title="30% 缺失的区间结构影响", ylabel="MAE", xlabel="缺失模态")
    for ax in axs: ax.set_ylim(0, max(pos.max().max(), pat.max().max()) * 1.22); ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=False)
    save(fig, output, "03_position_and_gap")


def draw_errors(root: Path, output: Path) -> None:
    v = np.load(root / "reports" / "validation_predictions.npz"); predcls = v["logits"].argmax(1); cm = confusion_matrix(v["cls"], predcls, labels=[0, 1, 2])
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2), layout="constrained", gridspec_kw={"width_ratios": [0.92, 1.08]})
    axs[0].imshow(cm, cmap="Blues"); axs[0].grid(False)
    row_totals = cm.sum(axis=1, keepdims=True)
    row_rates = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)
    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            axs[0].text(j, i, f"{cm[i, j]}\n({row_rates[i, j]:.1%})", ha="center", va="center", color=color, linespacing=1.25)
    axs[0].set(xticks=[0, 1, 2], yticks=[0, 1, 2], xticklabels=["负向", "中性", "正向"], yticklabels=["负向", "中性", "正向"], xlabel="预测类别", ylabel="真实类别", title="验证集分类混淆矩阵")

    errors = v["pred"] - v["y"]
    mae = float(np.mean(np.abs(errors))); rmse = float(np.sqrt(np.mean(errors ** 2)))
    pearson = float(np.corrcoef(v["y"], v["pred"])[0, 1])
    axs[1].scatter(v["y"], v["pred"], s=9, alpha=.2, color="#2f80b7", edgecolors="none", rasterized=True)
    axs[1].plot([-3, 3], [-3, 3], "--", color="black", lw=1, label="理想一致线 $y=x$")
    axs[1].text(.04, .96, f"MAE = {mae:.3f}\nRMSE = {rmse:.3f}\nPearson $\\rho$ = {pearson:.3f}", transform=axs[1].transAxes, ha="left", va="top", fontsize=8.5,
                bbox={"boxstyle": "round,pad=.3", "facecolor": "white", "edgecolor": "0.75", "alpha": .9})
    axs[1].set(xlim=(-3.1, 3.1), ylim=(-3.1, 3.1), xlabel="真实情感强度", ylabel="预测情感强度", title="验证集情感强度（投影后输出）")
    axs[1].legend(loc="lower right", fontsize=8, frameon=False)
    save(fig, output, "04_validation_errors")


def draw_ablation(rep: pd.DataFrame, output: Path) -> None:
    keys = ["variant", "seed", "split", "condition"]; measures = ["accuracy", "macro_f1", "weighted_f1", "mae", "pearson", "raw_mae"]
    c = rep.groupby(keys, as_index=False)[measures].mean(); fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    g = c[(c.split == "valid") & (c.condition == "TAV30")].groupby("variant")[["mae", "macro_f1"]].agg(["mean", "std"])
    order = [x for x in ["text_only", "fusion_clean", "fusion_aug", "robust_norec", "robust_rec", "robust_nogate"] if x in g.index]
    for ax, metric in zip(axs, ["mae", "macro_f1"]):
        vals = g.loc[order, metric]; ax.barh(range(len(order)), vals["mean"], xerr=vals["std"], capsize=3, color="#387ea0"); ax.set(yticks=range(len(order)), yticklabels=[NAMES[x] for x in order], xlabel="MAE" if metric == "mae" else "Macro-F1", title=f"30% 三模态缺失：{'MAE' if metric == 'mae' else 'Macro-F1'}"); ax.invert_yaxis()
    save(fig, output, "05_ablation")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); args = parser.parse_args(); root = args.root.resolve(); output = root / "reports" / "figures" / "publication_chinese"
    configure_style(); rep = pd.read_csv(root / "reports" / "core_metrics.csv"); s = pd.read_csv(root / "reports" / "missing_scenarios.csv"); selected = json.loads((root / "reports" / "selection.json").read_text(encoding="utf-8"))
    draw_rates(s, root, output); draw_missing_types(s, selected, output); draw_position_gap(s, selected, output); draw_errors(root, output); draw_ablation(rep, output)
    print(f"已生成与英文版一一对应的中文图表：{output}")


if __name__ == "__main__": main()
