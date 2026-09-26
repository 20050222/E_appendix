"""Versioned provenance and validated completion markers for Q2 repair iteration."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import torch
from data import ROOT

VERSION='q2-v2'
OUTPUT_RULE='argmax polarity; Neutral->0; Negative->min(raw,-1e-6); Positive->max(raw,1e-6)'

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''): h.update(block)
    return h.hexdigest()

@lru_cache(maxsize=1)
def training_fingerprint():
    paths=['src/data.py','src/model.py','src/train.py','src/protocol.py','src/tfr_generator.py',
           'artifacts/train.npz','artifacts/valid.npz','artifacts/normalization.npz',
           'pretrained/bert-tiny/model.safetensors','pretrained/bert-tiny/config.json']
    return {p:sha256(ROOT/p) for p in paths}

def completed_run_is_valid(out, config=None):
    out=Path(out)
    try:
        done=json.loads((out/'finished.json').read_text())
        saved=json.loads((out/'config.json').read_text())
        if saved.get('protocol')!=VERSION or (config is not None and saved!=config): return False
        if done.get('checkpoint_sha256')!=sha256(out/'best.pt'): return False
        state=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        if state.get('config')!=saved or not state.get('model'): return False
        if abs(done['best_score']-state['score'])>1e-10: return False
        return True
    except (OSError,ValueError,KeyError,RuntimeError,EOFError):
        return False
