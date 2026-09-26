"""Download pinned general-purpose BERT weights; never download sentiment data."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPO='google/bert_uncased_L-2_H-128_A-2'
REV='30b0a37ccaaa32f332884b96992754e246e48c5f'

def main():
    folder=ROOT/'pretrained/bert-tiny';folder.mkdir(parents=True,exist_ok=True)
    files=['config.json','vocab.txt','model.safetensors']
    for name in files:
        p=folder/name
        if not p.exists():
            subprocess.run(['curl','-fL','--retry','3','--retry-all-errors','--connect-timeout','20','--max-time','300',
                f'https://huggingface.co/{REPO}/resolve/{REV}/{name}?download=true','-o',str(p)],check=True)
    hashes={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in files}
    old=folder/'provenance.json'
    if old.exists():
        expected=json.loads(old.read_text())['files']
        if hashes!=expected:raise ValueError('Pretrained SHA256 mismatch')
    old.write_text(json.dumps(dict(repo=REPO,revision=REV,files=hashes),indent=2))

if __name__=='__main__':main()
