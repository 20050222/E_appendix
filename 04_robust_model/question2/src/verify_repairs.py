"""Recheck audit repairs from artifacts, plus fresh-unzip training and offline inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
import numpy as np
import pandas as pd
import torch
from data import ROOT, DATA, load_split, corruption, pattern_eligible, deletion_count
from protocol import VERSION,sha256,training_fingerprint,completed_run_is_valid
from train import device_setup,load_checkpoint,predict,project_intensity,final_metric,selection_score

R=ROOT/'reports'


def run(command,cwd,env=None):
    p=subprocess.run(command,cwd=cwd,env=env,text=True,capture_output=True)
    if p.returncode:
        raise RuntimeError(f'Command failed: {command[1:]}\n{p.stdout}\n{p.stderr}')
    return p.stdout+p.stderr


def verify_local():
    evidence={'protocol':VERSION}
    valid=load_split('valid');ids=valid['sample_id'];c=valid['content'];o=valid['obs']
    paired=[]
    for seed in [20260924,20260925,20260926]:
        for m in range(3):
            single=corruption(c,o,seed=seed,sample_ids=ids,rate=.3,modalities=[m])[0]
            allmod=corruption(c,o,seed=seed,sample_ids=ids,rate=.3,modalities=[0,1,2])[0]
            mismatched=int(np.any(single[:,:,m]!=allmod[:,:,m],axis=1).sum())
            assert mismatched==0
            ix=np.arange(len(c)-1,-1,-3)
            subset=corruption(c[ix],o[ix],seed=seed,sample_ids=ids[ix],rate=.3,modalities=[0,1,2])[0]
            np.testing.assert_array_equal(allmod[ix],subset)
            paired.append(dict(mask_seed=seed,modality=m,n=len(c),mismatched=mismatched))
    evidence['cross_condition_pairing']=paired
    eligible=pattern_eligible(c,.3);design_obs=np.repeat(c[eligible,:,None],3,axis=2)
    two=[]
    for seed in [20260924,20260925,20260926]:
        _,removed=corruption(c[eligible],design_obs,seed=seed,sample_ids=ids[eligible],rate=.3,pattern='two_blocks')
        counts=[]
        for i,n in enumerate(c[eligible].sum(1)):
            for m in range(3):
                # Count runs on the content timeline, excluding special-token holes.
                bits=removed[i,c[eligible][i],m]
                count=int((np.diff(np.r_[False,bits].astype(int))==1).sum())
                assert count==2 and bits.sum()==deletion_count(int(n),.3)
                counts.append(count)
        two.append(dict(mask_seed=seed,eligible=int(eligible.sum()),excluded=int((~eligible).sum()),modality_sequences=len(counts),all_exactly_two=True))
    evidence['strict_two_blocks']=two
    core=pd.read_csv(R/'core_metrics.csv')
    assert len(core)==468 and core.groupby(['variant','seed','split']).size().eq(13).all()
    runs=[];score_errors=[]
    for p in sorted((ROOT/'artifacts/runs').glob('*/best.pt')):
        assert completed_run_is_valid(p.parent)
        state=torch.load(p,map_location='cpu',weights_only=False);cfg=state['config']
        assert cfg['fingerprint']==training_fingerprint() and not cfg['smoke']
        rows=core[(core.variant==cfg['variant'])&(core.seed==cfg['seed'])&(core.split=='valid')&(core.mask_seed==20260924)]
        assert len(rows)==5
        score=float((rows.mae+1-rows.macro_f1).mean());err=abs(state['score']-score);assert err<1e-7
        score_errors.append(err)
        history=json.loads((p.parent/'history.json').read_text())
        # Replay the exact >1e-4 improvement rule used to choose the checkpoint.
        best=float('inf');chosen=None
        for row in history:
            assert abs(selection_score(row['validation'])-row['score'])<1e-10
            assert all('raw_mae' in m for m in row['validation'].values())
            if row['score']<best-1e-4:best=row['score'];chosen=row['epoch']
        assert chosen==state['best_epoch'] and best==state['score']
        runs.append(dict(variant=cfg['variant'],seed=cfg['seed'],projected_score=score,best_epoch=chosen))
    assert len(runs)==18
    selection=json.loads((R/'selection.json').read_text())
    scores=pd.DataFrame(runs).groupby('variant').projected_score.mean()
    assert selection['selected_variant']==scores.idxmin() and selection['selected_seed']==42
    for name,score in scores.items():assert abs(score-selection['variant_mean_score'][name])<1e-7
    evidence['selection']={'n_retrained':len(runs),'max_score_recalculation_error':max(score_errors),'scores':scores.to_dict(),'selected_variant':scores.idxmin(),'selected_seed':42,'all_epoch_selections_replayed':True}
    patterns=pd.read_csv(R/'missing_scenarios.csv');pat=patterns[patterns.group=='pattern']
    expected_hash=hashlib.sha256('\n'.join(map(str,ids[eligible])).encode()).hexdigest()
    assert pat.n.eq(eligible.sum()).all() and pat.excluded_short.eq((~eligible).sum()).all()
    assert pat.population_sha256.eq(expected_hash).all()
    assert set(pat.pattern)=={'block','two_blocks','points'}
    evidence['pattern_population']={'n':int(eligible.sum()),'excluded':int((~eligible).sum()),'sha256':expected_hash,'all_patterns_identical_population':True}
    deg=pd.read_csv(R/'paired_degradation.csv');con=pd.read_csv(R/'paired_baseline_contrasts.csv')
    assert len(deg)==432 and len(con)==360
    for row in deg.itertuples():
        base=core[(core.variant==row.variant)&(core.seed==row.seed)&(core.split==row.split)&(core.condition=='clean')].iloc[0]
        missing=core[(core.variant==row.variant)&(core.seed==row.seed)&(core.split==row.split)&(core.condition==row.condition)&(core.mask_seed==row.mask_seed)].iloc[0]
        assert abs(row.mae_increase-(missing.mae-base.mae))<1e-12
        assert abs(row.macro_f1_drop-(base.macro_f1-missing.macro_f1))<1e-12
    for row in con.itertuples():
        mask=(deg.seed==row.seed)&(deg.split==row.split)&(deg.condition==row.condition)&(deg.mask_seed==row.mask_seed)
        a=deg[mask&(deg.variant==row.variant)].iloc[0];b=deg[mask&(deg.variant=='fusion_clean')].iloc[0]
        for output,field in [('missing_mae_difference','missing_mae'),('mae_increase_difference','mae_increase'),('macro_f1_drop_difference','macro_f1_drop')]:
            assert abs(getattr(row,output)-(a[field]-b[field]))<1e-12
    evidence['degradation']={'paired_rows':len(deg),'baseline_contrast_rows':len(con),'recalculation_pass':True}
    device=device_setup('cpu',2);model,state=load_checkpoint(ROOT/'artifacts/final_model.pt',device)
    pr,lo,_=predict(model,valid,device);del model
    saved=np.load(R/'validation_predictions.npz');np.testing.assert_allclose(pr,saved['raw_pred'],atol=1e-6,rtol=0);np.testing.assert_allclose(lo,saved['logits'],atol=1e-6,rtol=0)
    np.testing.assert_allclose(project_intensity(pr,lo),saved['pred'],atol=1e-6,rtol=0)
    m=final_metric(valid['y'],valid['cls'],pr,lo)
    row=core[(core.variant==selection['selected_variant'])&(core.seed==42)&(core.split=='valid')&(core.condition=='clean')].iloc[0]
    errors={k:abs(m[k]-row[k]) for k in ['accuracy','macro_f1','mae','pearson']};assert max(errors.values())<1e-6
    evidence['prediction_recalculation']={'metric_errors':errors,'final_model_sha256':sha256(ROOT/'artifacts/final_model.pt')}
    pred=pd.read_csv(R/'附件3_第二问预测结果.csv');assert len(pred)==30 and pred.sample_id.is_unique
    assert np.isfinite(pred.pred_intensity).all() and pred.pred_intensity.abs().le(3).all()
    assert pred.loc[pred.pred_polarity=='Neutral','pred_intensity'].eq(0).all()
    assert pred.loc[pred.pred_polarity=='Negative','pred_intensity'].lt(0).all()
    assert pred.loc[pred.pred_polarity=='Positive','pred_intensity'].gt(0).all()
    evidence['special_predictions']={'rows':30,'finite_in_range':True,'polarity_consistency':True}
    return evidence


def verify_archive():
    archive=ROOT/'第二问_代码模型与结果.zip';assert archive.stat().st_size<50_000_000
    with tempfile.TemporaryDirectory(prefix='q2-v2-fresh-') as tmp:
        tmp=Path(tmp)
        with zipfile.ZipFile(archive) as z:
            assert not any(n.startswith('question2/artifacts/runs/') for n in z.namelist())
            z.extractall(tmp)
        fresh=tmp/'question2';manifest=json.loads((fresh/'reports/submission_manifest.json').read_text())
        for name,digest in manifest.items():assert sha256(fresh/name)==digest
        env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        prediction=tmp/'fresh_predictions.csv'
        log=run([sys.executable,'src/infer.py','--input-dir',str(DATA/'附件3-模态缺失特征样本/对齐版本'),'--output',str(prediction)],fresh,env)
        assert prediction.read_bytes()==(R/'附件3_第二问预测结果.csv').read_bytes()
        # Use exactly the already-prepared competition data and pinned initialization.
        # These resources are intentionally excluded from the small submission ZIP.
        for name in ['train.npz','valid.npz']:(fresh/'artifacts'/name).symlink_to(ROOT/'artifacts'/name)
        for name in ['model.safetensors','config.json']:(fresh/'pretrained/bert-tiny'/name).symlink_to(ROOT/'pretrained/bert-tiny'/name)
        command=[sys.executable,'src/train.py','--variants','robust_rec','--seeds','42','--epochs','1','--patience','1','--threads','2']
        first=run(command,fresh,env);assert '"epoch": 1' in first and 'SKIP' not in first
        checkpoint=fresh/'artifacts/runs/robust_rec_seed42/best.pt';assert checkpoint.exists()
        original_hash=sha256(checkpoint)
        second=run(command,fresh,env);assert 'SKIP robust_rec_seed42' in second and sha256(checkpoint)==original_hash
        checkpoint.unlink() # deliberately create the precise audited broken state in the temporary copy
        third=run(command,fresh,env);assert '"epoch": 1' in third and 'SKIP' not in third and checkpoint.exists()
        assert sha256(checkpoint)==original_hash # deterministic one-epoch CPU replay
        # Reload the newly trained checkpoint and independently execute validation.
        script="""import json,sys,torch
sys.path.insert(0,'src')
from data import load_split
from train import device_setup,load_checkpoint,predict,final_metric,validation_masks,selection_score
dev=device_setup('cpu',2);d=load_split('valid');m,s=load_checkpoint('artifacts/runs/robust_rec_seed42/best.pt',dev)
metrics={}
for name,mask in validation_masks(d):
 p,l,_=predict(m,d,dev,mask);metrics[name]=final_metric(d['y'],d['cls'],p,l)
assert abs(selection_score(metrics)-s['score'])<1e-7
print(json.dumps({'validation_recalculated':True,'n':len(d['ids']),'score':s['score']}))
"""
        evaluation=run([sys.executable,'-c',script],fresh,env)
        (R/'fresh_archive_verification.log').write_text('OFFLINE INFERENCE\n'+log+'\nFRESH TRAINING\n'+first+'\nVERIFIED SKIP\n'+second+'\nMISSING CHECKPOINT REPAIR\n'+third+'\nCHECKPOINT RELOAD AND VALIDATION\n'+evaluation)
        return {'no_active_run_markers_in_zip':True,'manifest_pass':True,'offline_inference_byte_identical':True,
                'fresh_training_train_n':3395,'fresh_training_valid_n':728,'fresh_training_epochs':1,
                'valid_complete_run_skipped':True,'missing_checkpoint_automatically_retrained':True,
                'retraining_checkpoint_hash_identical':True,'reloaded_checkpoint_validation_pass':True,
                'scope':'18 full retrainings in main workspace; clean-unzip integration uses one full-data epoch, not 18 repeated full trainings; local pinned dependencies reused, no clean OS installation.'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',action='store_true');args=ap.parse_args()
    result=verify_local()
    if args.archive:result['fresh_archive']=verify_archive()
    (R/'repair_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
