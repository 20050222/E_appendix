"""Extract 74-D audio frame features from Step A videos and aggregate to 50 bins."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import pandas as pd


SR = 16000
FRAME = 400
HOP = 160
N_MELS = 27
BINS = 50


def feature_names() -> list[str]:
    return ([f"mfcc_{i:02d}" for i in range(1, 14)]
            + [f"mfcc_delta_{i:02d}" for i in range(1, 14)]
            + [f"mfcc_delta2_{i:02d}" for i in range(1, 14)]
            + ["rms", "zero_crossing_rate", "f0_yin_hz", "voicing_autocorr_peak"]
            + ["spectral_centroid_hz", "spectral_bandwidth_hz", "spectral_rolloff_hz", "spectral_flux"]
            + [f"logmel_{i:02d}" for i in range(1, N_MELS + 1)])


def decode_audio(video_path: Path, ffmpeg_exe: str) -> np.ndarray:
    command = [ffmpeg_exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(video_path),
               "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-1200:] or f"ffmpeg_exit_{result.returncode}")
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if audio.size < FRAME:
        raise RuntimeError(f"audio_too_short_samples_{audio.size}")
    if not np.isfinite(audio).all():
        raise RuntimeError("decoded_audio_contains_nan_or_inf")
    return audio


def frame_features(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stft = librosa.stft(y, n_fft=FRAME, hop_length=HOP, win_length=FRAME, center=False)
    power = np.abs(stft) ** 2
    mel = librosa.feature.melspectrogram(S=power, sr=SR, n_fft=FRAME, n_mels=N_MELS, fmin=50, fmax=SR / 2)
    logmel = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=13)
    delta = librosa.feature.delta(mfcc, width=9, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, width=9, mode="nearest")
    rms = librosa.feature.rms(S=np.abs(stft), frame_length=FRAME)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME, hop_length=HOP, center=False)[0]
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=SR, frame_length=FRAME, hop_length=HOP, center=False)
    centroid = librosa.feature.spectral_centroid(S=np.abs(stft), sr=SR)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=np.abs(stft), sr=SR)[0]
    rolloff = librosa.feature.spectral_rolloff(S=np.abs(stft), sr=SR, roll_percent=0.85)[0]
    magnitude = np.abs(stft)
    flux = np.zeros(magnitude.shape[1], dtype=np.float32)
    if magnitude.shape[1] > 1:
        diff = np.maximum(magnitude[:, 1:] - magnitude[:, :-1], 0)
        flux[1:] = np.sqrt(np.sum(diff ** 2, axis=0))
    autocorr_peak = np.zeros_like(f0, dtype=np.float32)
    low_lag, high_lag = max(1, SR // 400), min(FRAME - 1, SR // 60)
    for idx in range(len(f0)):
        start = idx * HOP
        frame = y[start:start + FRAME]
        if len(frame) < FRAME:
            continue
        frame = frame - frame.mean()
        energy = float(np.dot(frame, frame))
        if energy <= 1e-10:
            continue
        corr = np.correlate(frame, frame, mode="full")[FRAME - 1:] / energy
        autocorr_peak[idx] = float(np.max(corr[low_lag:high_lag + 1]))
    matrix = np.vstack([mfcc, delta, delta2, rms, zcr, f0, autocorr_peak,
                        centroid, bandwidth, rolloff, flux, logmel]).T.astype(np.float32)
    if matrix.shape[1] != 74:
        raise AssertionError(f"audio feature dimension is {matrix.shape[1]}, expected 74")
    frame_times = (np.arange(matrix.shape[0]) * HOP + FRAME / 2) / SR
    return matrix, frame_times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit for smoke tests")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    step_root = workspace / "02_features_q1" / "stepC_audio"
    out_dir = step_root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(workspace / "02_features_q1" / "stepA_index" / "data" / "sample_index.csv", dtype=str, keep_default_na=False)
    if args.limit > 0:
        index = index.head(args.limit)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    feature_dir = out_dir / "audio_features" / "audio74_v1"
    feature_dir.mkdir(parents=True, exist_ok=True)
    log_rows, align_rows = [], []
    started = time.time()
    names = feature_names()

    for _, row in index.iterrows():
        sample_id = row["sample_id"]
        duration = float(row["duration_sec"])
        path = Path(row["video_path"])
        status, error, valid_length, decoded_sec = "PASS", "", 0, 0.0
        features = np.zeros((BINS, 74), dtype=np.float32)
        mask = np.zeros((BINS, 1), dtype=np.uint8)
        frame_counts = np.zeros(BINS, dtype=np.int32)
        try:
            audio = decode_audio(path, ffmpeg_exe)
            decoded_sec = len(audio) / SR
            frames, times = frame_features(audio)
            bucket = np.minimum((times * BINS / duration).astype(int), BINS - 1)
            for b in range(BINS):
                selected = bucket == b
                frame_counts[b] = int(selected.sum())
                if selected.any():
                    features[b] = frames[selected].mean(axis=0)
                    mask[b, 0] = 1
            valid_length = int(mask.sum())
            if valid_length == 0:
                status, error = "FAIL", "no_audio_frames_in_video_bins"
        except Exception as exc:
            status, error = "FAIL", f"{type(exc).__name__}: {exc}"

        np.savez_compressed(feature_dir / f"{sample_id}.npz", sample_id=np.asarray(sample_id),
                            audio_features=features, audio_mask=mask, audio_frame_count=frame_counts,
                            audio_valid_length=np.asarray(valid_length, dtype=np.int32),
                            audio_feature_names=np.asarray(names), audio_sample_rate=np.asarray(SR, dtype=np.int32),
                            audio_frame_ms=np.asarray(25.0), audio_hop_ms=np.asarray(10.0),
                            audio_time_start=np.linspace(0, duration, BINS, endpoint=False, dtype=np.float32),
                            audio_time_end=np.linspace(0, duration, BINS + 1, dtype=np.float32)[1:])
        for b in range(BINS):
            t0, t1 = b * duration / BINS, (b + 1) * duration / BINS
            align_rows.append({"sample_id": sample_id, "modality": "audio", "bin_index": b,
                               "bin_start_sec": t0, "bin_end_sec": t1, "source_count": int(frame_counts[b]),
                               "valid": int(mask[b, 0]), "aggregation": "mean", "alignment_method": "equal_duration_50_bins",
                               "quality_status": status, "source_time_min": "", "source_time_max": ""})
        log_rows.append({"sample_id": sample_id, "status": status, "error": error, "video_duration_sec": duration,
                         "decoded_audio_duration_sec": decoded_sec, "audio_video_duration_diff_sec": decoded_sec - duration if decoded_sec else "",
                         "feature_shape": "(50,74)", "valid_bins": valid_length, "mask_ratio": valid_length / BINS,
                         "frame_count": int(frame_counts.sum()), "sample_rate": SR, "frame_ms": 25, "hop_ms": 10,
                         "feature_schema": "audio74_v1", "ffmpeg_executable": ffmpeg_exe})
        print(f"[{len(log_rows)}/{len(index)}] {sample_id}: {status}" + (f" ({error[:120]})" if error else ""), flush=True)

    pd.DataFrame(log_rows).to_csv(out_dir / "audio_extraction_log.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    pd.DataFrame(align_rows).to_csv(out_dir / "audio_alignment_log.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    schema = {"schema": "audio74_v1", "sample_rate_hz": SR, "channels": 1, "frame_ms": 25, "hop_ms": 10,
              "window": "librosa default hann", "features": names, "dimension": 74, "time_bins": 50,
              "aggregation": "mean by frame-center in equal-duration video bins", "frame_interval": "left_closed_right_open",
              "mask": "1 when at least one decoded audio frame center falls in bin"}
    (out_dir / "audio_feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {"python": sys.version, "librosa": librosa.__version__, "numpy": np.__version__, "ffmpeg": ffmpeg_exe,
            "ffmpeg_version": subprocess.run([ffmpeg_exe, "-version"], capture_output=True, text=True).stdout.splitlines()[0],
            "elapsed_sec": time.time() - started, "sample_count": len(index), "feature_schema": "audio74_v1"}
    (out_dir / "audio_feature_run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sample_count": len(index), "pass": sum(r["status"] == "PASS" for r in log_rows),
                      "fail": sum(r["status"] != "PASS" for r in log_rows), "output": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
