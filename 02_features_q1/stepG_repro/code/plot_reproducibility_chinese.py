"""Generate the Chinese Step G figure beside its image outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    feature_root = workspace / "02_features_q1"
    step_f_code = feature_root / "stepF_validation/code"
    sys.path.insert(0, str(step_f_code))

    import plot_fg_publication_chinese as figures

    data = figures.read_inputs(workspace)
    figures.configure_style()
    output_root = feature_root / "stepG_repro/image/publication_chinese"
    figures.draw_reproducibility(workspace, data, output_root=output_root)
    print(f"中文复现图输出目录: {output_root}")


if __name__ == "__main__":
    main()
