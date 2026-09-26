"""Prepare Step B normalized text inputs without requiring a GPU or Transformer."""

from __future__ import annotations

import argparse
from pathlib import Path

from text_feature_common import prepare_text_index, write_text_index, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    out = workspace / "02_features_q1" / "stepB_text" / "data"
    out.mkdir(parents=True, exist_ok=True)
    frame = prepare_text_index(workspace)
    write_text_index(frame, out / "text_index.csv")
    records = [
        {
            "sample_id": row.sample_id,
            "raw_text": row.raw_text,
            "normalized_text": row.normalized_text,
            "tokenizer_name": "pending_model_run",
            "tokenizer_version": "pending_model_run",
            "tokenization_status": "pending_transformers",
        }
        for row in frame.itertuples(index=False)
    ]
    write_jsonl(out / "text_token_map.jsonl", records)
    print({"rows": len(frame), "empty_text": int((frame.text_status == "empty").sum()), "output": str(out)})


if __name__ == "__main__":
    main()
