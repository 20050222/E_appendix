$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$figureRoot = Join-Path $workspace 'reports\figures\publication_chinese'
$backup = Join-Path $figureRoot 'archive_previous\06_paired_degradation_before_20260926'
$script = Join-Path $workspace 'src\plot_publication_chinese.py'
$target = Join-Path $figureRoot '06_paired_degradation'

if (-not (Test-Path $backup)) { throw "Missing figure backup: $backup" }

# Restore the original draw_rates() implementation from the version before Q2-06-A..D.
$source = Get-Content -Raw $script
$originalFunction = @'
def draw_rates(s: pd.DataFrame, root: Path, output: Path) -> None:
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

    fig, axs = plt.subplots(2, 2, figsize=(11, 7), layout="constrained", sharey=True)
    for ax, mod in zip(axs.flat, ["T", "A", "V", "TAV"]):
        for model in s.variant.unique():
            z = s[(s.variant == model) & (s.group == "rate") & (s.scenario == mod)].groupby("rate").mae.agg(["mean", "std"])
            clean = s[(s.variant == model) & (s.group == "clean")].mae.iloc[0]
            xx = np.r_[0, z.index.to_numpy()]; yy = np.r_[0, z["mean"].to_numpy() - clean]; err = np.r_[0, z["std"].fillna(0).to_numpy()]
            ax.plot(xx, yy, "o-", label=NAMES[model]); ax.fill_between(xx, yy - err, yy + err, alpha=.15)
        ax.axhline(0, color="gray", lw=.8); ax.set(title=f"缺失{SCENARIO[mod]}", xlabel="内容位置的目标缺失比例", ylabel="相对完整输入的 MAE 增量", xticks=[0, .1, .3, .5, .7])
    axs[0, 0].legend(fontsize=8); fig.suptitle("验证集退化：固定训练种子的 3 个配对遮挡实现")
    save(fig, output, "06_paired_degradation")


'@
$pattern = '(?s)def draw_rates\(.*?\n\ndef draw_missing_types'
$replacement = $originalFunction + 'def draw_missing_types'
$restored = [regex]::Replace($source, $pattern, $replacement)
if ($restored -eq $source) { throw 'Could not locate draw_rates() for restoration.' }
Set-Content -Path $script -Value $restored -Encoding UTF8
New-Item -ItemType Directory -Force $target | Out-Null
foreach ($ext in 'png','pdf','svg') {
    Copy-Item (Join-Path $backup "06_paired_degradation.$ext") (Join-Path $target "06_paired_degradation.$ext") -Force
}
Write-Host "Restored plot_publication_chinese.py and 06_paired_degradation exports."
