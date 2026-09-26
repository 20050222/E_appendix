"""Extract 35-D MediaPipe face geometry/quality features and aggregate to 50 bins."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


BINS = 50
SAMPLE_FPS = 10.0
LANDMARK_IDS = [1, 152, 33, 133, 362, 263, 61, 291, 13, 14, 70, 105, 334, 300, 10]
LANDMARK_NAMES = ["nose_tip", "chin", "left_eye_outer", "left_eye_inner", "right_eye_inner", "right_eye_outer",
                  "mouth_left", "mouth_right", "upper_lip", "lower_lip", "left_brow_outer", "left_brow_inner",
                  "right_brow_inner", "right_brow_outer", "forehead_center"]


def feature_names() -> list[str]:
    points = [f"{name}_{axis}_face_norm" for name in LANDMARK_NAMES for axis in ("x", "y")]
    return points + ["face_center_x_frame_norm", "face_center_y_frame_norm", "face_width_frame_norm",
                     "face_height_frame_norm", "face_gray_mean_0_1"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit for smoke tests")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    step_root = workspace / "02_features_q1" / "stepD_vision"
    out_dir = step_root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = out_dir / "vision_features" / "face35_v1"
    feature_dir.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(workspace / "02_features_q1" / "stepA_index" / "data" / "sample_index.csv", dtype=str, keep_default_na=False)
    if args.limit > 0:
        index = index.head(args.limit)
    log_rows, frame_rows, align_rows = [], [], []
    started = time.time()
    names = feature_names()

    for sample_num, (_, row) in enumerate(index.iterrows(), start=1):
        sample_id = row["sample_id"]
        duration = float(row["duration_sec"])
        path = str(row["video_path"])
        features = np.zeros((BINS, 35), dtype=np.float32)
        mask = np.zeros((BINS, 1), dtype=np.uint8)
        counts = np.zeros(BINS, dtype=np.int32)
        status, error, timestamp_source = "PASS", "", "opencv_pos_msec"
        per_bin: list[list[np.ndarray]] = [[] for _ in range(BINS)]
        sampled_count = 0
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            status, error = "FAIL", "opencv_video_open_failed"
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                fps = float(row["fps"] or 0)
                timestamp_source = "frame_index_fps_fallback"
            frame_index = 0
            next_sample_sec = 0.0
            face_count_total = 0
            multiple_face_count = 0
            try:
                with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=4,
                        refine_landmarks=False, min_detection_confidence=0.5) as face_mesh:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        if not np.isfinite(pts) or pts <= 0:
                            pts = frame_index / fps if fps > 0 else float("nan")
                            timestamp_source = "frame_index_fps_fallback"
                        frame_index += 1
                        if not np.isfinite(pts) or pts + 1e-6 < next_sample_sec:
                            continue
                        next_sample_sec += 1.0 / args.sample_fps
                        if pts >= duration:
                            continue
                        sampled_count += 1
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = face_mesh.process(rgb)
                        detected = result.multi_face_landmarks or []
                        face_count_total += len(detected)
                        if len(detected) > 1:
                            multiple_face_count += 1
                        chosen = None
                        chosen_area = -1.0
                        chosen_bbox = None
                        h, w = frame.shape[:2]
                        for face_i, face in enumerate(detected):
                            xy = np.array([(p.x * w, p.y * h) for p in face.landmark], dtype=np.float32)
                            x0, y0 = np.clip(xy.min(axis=0), [0, 0], [w - 1, h - 1])
                            x1, y1 = np.clip(xy.max(axis=0), [0, 0], [w - 1, h - 1])
                            area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
                            if area > chosen_area:
                                chosen, chosen_area = (face_i, face), area
                                chosen_bbox = (x0, y0, x1, y1, xy)
                        b = min(int(pts * BINS / duration), BINS - 1)
                        if chosen is None:
                            frame_rows.append({"sample_id": sample_id, "sample_time_sec": pts, "frame_index": frame_index - 1,
                                               "face_count": 0, "chosen_face_index": "", "face_bbox_xyxy": "",
                                               "valid": 0, "failure_reason": "no_face_detected", "timestamp_source": timestamp_source})
                            continue
                        face_i, face = chosen
                        x0, y0, x1, y1, xy = chosen_bbox
                        width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
                        selected = xy[LANDMARK_IDS]
                        relative = (selected - np.array([x0, y0])) / np.array([width, height])
                        gray_mean = float(cv2.cvtColor(frame[int(y0):max(int(y1), int(y0)+1), int(x0):max(int(x1), int(x0)+1)], cv2.COLOR_BGR2GRAY).mean() / 255.0)
                        vector = np.concatenate([relative.reshape(-1), [(x0+x1)/2/w, (y0+y1)/2/h, width/w, height/h, gray_mean]]).astype(np.float32)
                        if vector.shape != (35,) or not np.isfinite(vector).all():
                            frame_rows.append({"sample_id": sample_id, "sample_time_sec": pts, "frame_index": frame_index - 1,
                                               "face_count": len(detected), "chosen_face_index": face_i, "face_bbox_xyxy": "",
                                               "valid": 0, "failure_reason": "invalid_feature_vector", "timestamp_source": timestamp_source})
                            continue
                        per_bin[b].append(vector)
                        frame_rows.append({"sample_id": sample_id, "sample_time_sec": pts, "frame_index": frame_index - 1,
                                           "face_count": len(detected), "chosen_face_index": face_i,
                                           "face_bbox_xyxy": json.dumps([round(float(v), 2) for v in [x0, y0, x1, y1]]),
                                           "valid": 1, "failure_reason": "", "timestamp_source": timestamp_source})
            except Exception as exc:
                status, error = "FAIL", f"{type(exc).__name__}: {exc}"
            finally:
                cap.release()
        for b in range(BINS):
            counts[b] = len(per_bin[b])
            if counts[b]:
                features[b] = np.mean(per_bin[b], axis=0)
                mask[b, 0] = 1
            t0, t1 = b * duration / BINS, (b + 1) * duration / BINS
            align_rows.append({"sample_id": sample_id, "modality": "vision", "bin_index": b,
                               "bin_start_sec": t0, "bin_end_sec": t1, "source_count": int(counts[b]),
                               "valid": int(mask[b, 0]), "aggregation": "mean_valid_faces", "alignment_method": "equal_duration_50_bins",
                               "quality_status": status, "source_time_min": "", "source_time_max": ""})
        valid_length = int(mask.sum())
        if status == "PASS" and valid_length == 0:
            status, error = "FAIL", "no_valid_face_frames"
        np.savez_compressed(feature_dir / f"{sample_id}.npz", sample_id=np.asarray(sample_id), vision_features=features,
                            vision_mask=mask, vision_frame_count=counts, vision_valid_length=np.asarray(valid_length, dtype=np.int32),
                            vision_feature_names=np.asarray(names), vision_time_start=np.linspace(0, duration, BINS, endpoint=False, dtype=np.float32),
                            vision_time_end=np.linspace(0, duration, BINS + 1, dtype=np.float32)[1:],
                            sample_fps=np.asarray(args.sample_fps, dtype=np.float32),)
        log_rows.append({"sample_id": sample_id, "status": status, "error": error, "video_duration_sec": duration,
                         "feature_shape": "(50,35)", "valid_bins": valid_length, "mask_ratio": valid_length / BINS,
                         "sampled_fps": args.sample_fps, "sampled_frames": sampled_count,
                         "face_detected_frames": sum(len(x) for x in per_bin), "multiple_face_samples": sum(1 for r in frame_rows if r["sample_id"] == sample_id and int(r["face_count"] or 0) > 1),
                         "timestamp_source": timestamp_source, "feature_schema": "face35_v1"})
        print(f"[{sample_num}/{len(index)}] {sample_id}: {status}, valid_bins={valid_length}" + (f" ({error[:100]})" if error else ""), flush=True)

    pd.DataFrame(log_rows).to_csv(out_dir / "vision_extraction_log.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    pd.DataFrame(frame_rows).to_csv(out_dir / "vision_frame_log.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    pd.DataFrame(align_rows).to_csv(out_dir / "vision_alignment_log.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    schema = {"schema": "face35_v1", "sampler_fps": args.sample_fps, "detector": "MediaPipe Face Mesh",
              "face_selection": "largest landmark bounding box per sampled frame", "max_faces": 4,
              "features": names, "dimension": 35, "time_bins": 50,
              "aggregation": "mean of valid sampled face frames per equal-duration video bin",
              "mask": "1 when at least one valid face frame falls in bin",
              "landmark_ids": LANDMARK_IDS, "landmark_names": LANDMARK_NAMES}
    (out_dir / "vision_feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {"python": sys.version, "opencv": cv2.__version__, "mediapipe": mp.__version__, "numpy": np.__version__,
            "elapsed_sec": time.time() - started, "sample_count": len(index), "feature_schema": "face35_v1"}
    (out_dir / "vision_feature_run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sample_count": len(index), "pass": sum(r["status"] == "PASS" for r in log_rows),
                      "fail": sum(r["status"] != "PASS" for r in log_rows), "output": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
