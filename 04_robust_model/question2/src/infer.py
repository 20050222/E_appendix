"""Offline inference with final checkpoint; no pretrained download required."""
import argparse
from pathlib import Path
import numpy as np
from data import ROOT,load_special
from train import device_setup,load_checkpoint,predict,project_intensity
from evaluate import write_csv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-dir');ap.add_argument('--checkpoint',default=str(ROOT/'artifacts/final_model.pt'))
    ap.add_argument('--output',default=str(ROOT/'reports/附件3_第二问预测结果.csv'));ap.add_argument('--device',default='auto');args=ap.parse_args()
    device=device_setup(args.device);model,_=load_checkpoint(args.checkpoint,device)
    d=load_special(args.input_dir);p,l,_=predict(model,d,device);p=project_intensity(p,l)
    assert np.isfinite(p).all()
    labels=['Negative','Neutral','Positive']
    rows=[dict(sample_id=sid,pred_polarity=labels[l[i].argmax()],pred_intensity=round(float(p[i]),6)) for i,sid in enumerate(d['sample_id'])]
    write_csv(Path(args.output),rows);print(f'Wrote {len(rows)} predictions to {args.output}')

if __name__=='__main__':main()
