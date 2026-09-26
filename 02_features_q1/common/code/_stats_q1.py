from pathlib import Path
import pandas as pd

r = Path(r'E_workspace/02_features_q1')
files = [
    'stepA_index/data/sample_index.csv',
    'stepB_text/data/text_extraction_log.csv',
    'stepC_audio/data/audio_extraction_log.csv',
    'stepD_vision/data/vision_extraction_log.csv',
    'stepE_alignment/data/multimodal_sample_audit.csv',
    'stepF_validation/data/final_sample_validation.csv',
    'stepG_repro/data/unified_extraction_log.csv',
]
for f in files:
    d = pd.read_csv(r / f)
    print('\n', f, len(d), list(d.columns))
    if 'token_count' in d:
        print('tokens', d.token_count.min(), d.token_count.median(), d.token_count.mean(), d.token_count.max(), d.truncated.value_counts().to_dict())
    if 'audio_video_duration_diff_sec' in d:
        print('audio_diff', d.audio_video_duration_diff_sec.min(), d.audio_video_duration_diff_sec.median(), d.audio_video_duration_diff_sec.mean(), d.audio_video_duration_diff_sec.max(), 'valid', d.valid_bins.min(), d.valid_bins.median(), d.valid_bins.max())
    if 'face_detected_frames' in d:
        print('vision', d.sampled_frames.min(), d.sampled_frames.median(), d.sampled_frames.max(), d.face_detected_frames.min(), d.face_detected_frames.median(), d.face_detected_frames.max(), d.valid_bins.min(), d.valid_bins.median(), d.valid_bins.max(), d.status.value_counts().to_dict())
    if 'multimodal_status' in d:
        print('multi', d.multimodal_status.value_counts().to_dict(), d.annotation.value_counts().to_dict())
    if 'stage' in d:
        print('stages', d.groupby(['stage','status']).size().to_dict())
