"""Extract BERT text features and organize them into 50 uniform token bins.

Run this script from either the CPU wrapper or the RTX4090 wrapper. The model
must be a 768-dimensional BERT checkpoint such as bert-base-uncased.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

from text_feature_common import (
    EXPECTED_HIDDEN_SIZE,
    make_bucket_features,
    prepare_text_index,
    save_sample_npz,
    write_jsonl,
    write_text_index,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda:0"])
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Step B requires torch and transformers. Install the selected environment first; "
            f"original error: {exc}"
        ) from exc

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is False")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    workspace = args.workspace.resolve()
    out = workspace / "02_features_q1" / "stepB_text" / "data"
    feature_dir = out / "text_features" / args.model_name.replace("/", "__")
    feature_dir.mkdir(parents=True, exist_ok=True)
    frame = prepare_text_index(workspace)
    model_source = args.model_path or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=args.local_files_only, use_fast=True)
    model = AutoModel.from_pretrained(model_source, local_files_only=args.local_files_only)
    model.to(args.device)
    model.eval()
    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size != EXPECTED_HIDDEN_SIZE:
        raise ValueError(f"expected 768-dimensional BERT, got hidden_size={hidden_size}")

    rows = []
    token_records = []
    with torch.inference_mode():
        for start in range(0, len(frame), args.batch_size):
            batch = frame.iloc[start:start + args.batch_size]
            texts = batch["normalized_text"].tolist()
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping").tolist()
            model_inputs = {key: value.to(args.device) for key, value in encoded.items()}
            outputs = model(**model_inputs).last_hidden_state.detach().cpu().numpy()
            attention = encoded["attention_mask"].cpu().numpy()
            input_ids = encoded["input_ids"].cpu().numpy()

            for j, (_, item) in enumerate(batch.iterrows()):
                content_indices = []
                token_map = []
                for pos, (token_id, attn, offset) in enumerate(zip(input_ids[j], attention[j], offsets[j])):
                    if not attn or tuple(offset) == (0, 0):
                        continue
                    token = tokenizer.convert_ids_to_tokens(int(token_id))
                    if token in tokenizer.all_special_tokens:
                        continue
                    content_indices.append(pos)
                    token_map.append({
                        "token_index": len(token_map),
                        "model_position": pos,
                        "token_id": int(token_id),
                        "token": token,
                        "offset": [int(offset[0]), int(offset[1])],
                    })
                token_count = len(content_indices)
                sample_id = str(item["sample_id"])
                full_encoding = tokenizer(
                    item["normalized_text"],
                    add_special_tokens=False,
                    truncation=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                original_token_count = len(full_encoding["input_ids"])
                truncated = original_token_count > token_count
                if token_count:
                    token_embeddings = outputs[j, content_indices, :]
                    features, mask, counts, starts, ends = make_bucket_features(token_embeddings, token_count)
                    status = "PASS"
                else:
                    features = np.zeros((50, EXPECTED_HIDDEN_SIZE), dtype=np.float32)
                    mask = np.zeros((50, 1), dtype=np.uint8)
                    counts = np.zeros((50,), dtype=np.int32)
                    starts = np.full((50,), -1, dtype=np.int32)
                    ends = np.full((50,), -1, dtype=np.int32)
                    status = "EMPTY_TEXT"
                for token in token_map:
                    token["bin_index"] = int(
                        min((token["token_index"] * 50) // token_count, 49)
                    ) if token_count else -1
                    token["token_start_char"] = token["offset"][0]
                    token["token_end_char"] = token["offset"][1]
                bucket_map = [
                    {
                        "bin_index": index,
                        "token_start": int(starts[index]),
                        "token_end": int(ends[index]),
                        "token_count": int(counts[index]),
                        "text_mask": int(mask[index, 0]),
                    }
                    for index in range(50)
                ]
                save_sample_npz(
                    feature_dir / f"{sample_id}.npz",
                    features,
                    mask,
                    counts,
                    starts,
                    ends,
                    sample_id,
                    "uniform_tokens",
                )
                token_records.append({
                    "sample_id": sample_id,
                    "raw_text": item["raw_text"],
                    "normalized_text": item["normalized_text"],
                    "token_count": token_count,
                    "original_token_count": original_token_count,
                    "max_length": args.max_length,
                    "truncated": truncated,
                    "tokenizer_name": args.model_name,
                    "tokenizer_version": getattr(tokenizer, "__class__", type(tokenizer)).__name__,
                    "text_alignment_method": "uniform_tokens",
                    "tokens": token_map,
                    "bucket_map": bucket_map,
                })
                rows.append({
                    "sample_id": sample_id,
                    "text_status": status,
                    "token_count": token_count,
                    "original_token_count": original_token_count,
                    "truncated": truncated,
                    "text_valid_length": int(mask.sum()),
                    "feature_shape": "(50,768)",
                    "feature_path": str((feature_dir / f"{sample_id}.npz").relative_to(workspace)),
                    "device": args.device,
                    "model_name": args.model_name,
                    "max_length": args.max_length,
                })

    write_text_index(frame.assign(tokenizer_name=args.model_name, tokenizer_version=tokenizer.__class__.__name__), out / "text_index.csv")
    write_jsonl(out / "text_token_map.jsonl", token_records)
    with (out / "text_extraction_log.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "model_name": args.model_name,
        "model_source": model_source,
        "device": args.device,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "hidden_size": hidden_size,
        "max_length": args.max_length,
        "time_bins": 50,
        "random_seed": args.seed,
        "alignment_method": "uniform_tokens",
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "raw_index": "stepA_index/data/sample_index.csv",
    }
    (out / "text_feature_run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "pass": sum(x["text_status"] == "PASS" for x in rows), "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
