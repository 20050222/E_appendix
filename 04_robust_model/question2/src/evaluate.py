"""Freeze validation selection, evaluate the reused holdout, and predict Annex 3."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from data import ROOT,load_split,load_special,corruption,pattern_eligible
from train import device_setup,load_checkpoint,predict,metric,validation_masks,project_intensity,final_metric,selection_score
from protocol import VERSION,OUTPUT_RULE,completed_run_is_valid,training_fingerprint

REPORT=ROOT/'reports'


def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def flat_metrics(m):
    return {k:m[k] for k in ['accuracy','macro_f1','weighted_f1','mae','pearson','polarity_regression_conflict','raw_mae','raw_pearson','raw_polarity_regression_conflict','n']}


def scenarios():
    yield dict(name='clean',mods=[],rate=0,position='random',pattern='block',group='clean')
    for name,mods in [('T',[0]),('A',[1]),('V',[2]),('TA',[0,1]),('TV',[0,2]),('AV',[1,2]),('TAV',[0,1,2])]:
        for rate in [.1,.3,.5,.7]:
            yield dict(name=name,mods=mods,rate=rate,position='random',pattern='block',group='rate')
    for name,mods in [('T',[0]),('A',[1]),('V',[2])]:
        for pos in ['front','middle','back']:
            yield dict(name=name,mods=mods,rate=.3,position=pos,pattern='block',group='position')
        for pat in ['block','two_blocks','points']:
            yield dict(name=name,mods=mods,rate=.3,position='random',pattern=pat,group='pattern')


def mask_for(d,s,seed):
    if not s['mods']:return d['obs']
    return corruption(d['content'],d['obs'],seed=seed,sample_ids=d['sample_id'],rate=s['rate'],modalities=s['mods'],position=s['position'],pattern=s['pattern'])[0]


def write_degradation(summary):
    """Pair each corrupted evaluation with its own trained model's clean evaluation."""
    clean={(r['variant'],r['seed'],r['split']):r for r in summary if r['condition']=='clean'}
    rows=[]
    for r in summary:
        if r['condition']=='clean':continue
        base=clean[r['variant'],r['seed'],r['split']]
        rows.append({**{k:r[k] for k in ['variant','seed','split','condition','mask_seed']},
            'clean_mae':base['mae'],'missing_mae':r['mae'],'mae_increase':r['mae']-base['mae'],
            'relative_mae_increase':(r['mae']-base['mae'])/base['mae'],
            'clean_macro_f1':base['macro_f1'],'missing_macro_f1':r['macro_f1'],
            'macro_f1_drop':base['macro_f1']-r['macro_f1']})
    write_csv(REPORT/'paired_degradation.csv',rows)
    controls={(r['seed'],r['split'],r['condition'],r['mask_seed']):r for r in rows if r['variant']=='fusion_clean'}
    contrasts=[]
    for r in rows:
        if r['variant']=='fusion_clean':continue
        b=controls[r['seed'],r['split'],r['condition'],r['mask_seed']]
        contrasts.append({**{k:r[k] for k in ['variant','seed','split','condition','mask_seed']},
            'missing_mae_difference':r['missing_mae']-b['missing_mae'],
            'mae_increase_difference':r['mae_increase']-b['mae_increase'],
            'macro_f1_drop_difference':r['macro_f1_drop']-b['macro_f1_drop']})
    write_csv(REPORT/'paired_baseline_contrasts.csv',contrasts)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='auto');ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--mask-seeds',type=int,nargs='+',default=[20260924,20260925,20260926]);args=ap.parse_args()
    device=device_setup(args.device,args.threads)
    runs=[]
    for p in sorted((ROOT/'artifacts/runs').glob('*/best.pt')):
        if not completed_run_is_valid(p.parent): raise ValueError(f'Invalid/incomplete run: {p.parent.name}')
        state=torch.load(p,map_location='cpu',weights_only=False)
        if state['config'].get('smoke') or state['config']['fingerprint']!=training_fingerprint():
            raise ValueError(f'Stale source/data fingerprint: {p.parent.name}')
        if abs(selection_score(state['validation'])-state['score'])>1e-10: raise ValueError('Selection score mismatch')
        runs.append(dict(path=p,variant=state['config']['variant'],seed=state['config']['seed'],score=state['score'],best_epoch=state['best_epoch'],validation=state['validation']))
        del state
    if not runs:raise ValueError('No completed training runs')
    variants=sorted({r['variant'] for r in runs})
    expected=['text_only','fusion_clean','fusion_aug','robust_norec','robust_rec','robust_nogate']
    if set(variants)!=set(expected) or len(runs)!=18: raise ValueError('Expected exactly the 18 preset runs')
    for variant in expected:
        seeds={r['seed'] for r in runs if r['variant']==variant}
        if seeds!={42,43,44}:
            raise ValueError(f'Finish all three preset seeds before final evaluation: {variant}: {seeds}')
    scores={v:float(np.mean([r['score'] for r in runs if r['variant']==v])) for v in variants}
    best_variant=min(scores,key=scores.get)
    # Fixed representative training seed: do not cherry-pick the best seed.
    selected=min([r for r in runs if r['variant']==best_variant],key=lambda r:r['seed'])
    frozen={'protocol':VERSION,'iteration_note':'V1 test was already observed; V2 is an audit repair iteration using a reused holdout, not a new blind test.','selection_rule':'lowest mean projected validation MAE+1-MacroF1 across 5 preset conditions, then lowest numeric seed for representative model',
        'variant_mean_score':scores,'variant_seed_counts':{v:sum(r['variant']==v for r in runs) for v in variants},
        'selected_variant':best_variant,'selected_seed':selected['seed'],'checkpoint':str(selected['path'].relative_to(ROOT)),
        'sha256':hashlib.sha256(selected['path'].read_bytes()).hexdigest(),
        'test_used_for_selection':False,'special_used_for_selection':False,'mask_seeds':args.mask_seeds,
        'output_rule':OUTPUT_RULE+'; checkpoint and variant selection use these same projected metrics.'}
    (REPORT/'selection.json').write_text(json.dumps(frozen,indent=2))
    print('FROZEN',json.dumps(frozen),flush=True)
    valid=load_split('valid');test=load_split('test')
    summary=[]
    for run in runs:
        model,_=load_checkpoint(run['path'],device)
        for split,d in [('valid',valid),('test',test)]:
            for mask_seed in args.mask_seeds:
                for condition,mask in validation_masks(d,mask_seed):
                    if condition=='clean' and mask_seed!=args.mask_seeds[0]: continue
                    pr,lo,_=predict(model,d,device,mask)
                    m=final_metric(d['y'],d['cls'],pr,lo)
                    summary.append(dict(variant=run['variant'],seed=run['seed'],split=split,condition=condition,mask_seed=mask_seed,best_epoch=run['best_epoch'],**flat_metrics(m)))
        print('CORE',run['variant'],run['seed'],flush=True)
        del model
    write_csv(REPORT/'core_metrics.csv',summary)
    write_degradation(summary)
    selected_model,state=load_checkpoint(selected['path'],device)
    # A fixed clean-fusion control, same representative seed.
    baseline=next((r for r in runs if r['variant']=='fusion_clean' and r['seed']==selected['seed']),None)
    detailed=[]
    comparison=[(best_variant,selected_model)]
    if baseline and best_variant!='fusion_clean':comparison.append(('fusion_clean',load_checkpoint(baseline['path'],device)[0]))
    for name,model in comparison:
        for s in scenarios():
            eligible=pattern_eligible(valid['content'],s['rate']) if s['group']=='pattern' else np.ones(len(valid['ids']),bool)
            d={k:v[eligible] for k,v in valid.items()}
            population_hash=hashlib.sha256('\n'.join(map(str,d['sample_id'])).encode()).hexdigest()
            seeds=args.mask_seeds if s['position']=='random' and s['rate']>0 else args.mask_seeds[:1]
            for seed in seeds:
                kept=mask_for(d,s,seed);pr,lo,_=predict(model,d,device,kept)
                m=final_metric(d['y'],d['cls'],pr,lo)
                removed=d['obs'] & ~kept
                available=d['content'].sum()
                rates={f'realized_{mod}':float(removed[:,:,i].sum()/available) for i,mod in enumerate(['T','A','V'])}
                detailed.append(dict(variant=name,scenario=s['name'],group=s['group'],rate=s['rate'],position=s['position'],pattern=s['pattern'],mask_seed=seed,population_sha256=population_hash,excluded_short=int((~eligible).sum()),**rates,**flat_metrics(m)))
            print('SCENARIO',name,s['name'],s['rate'],s['position'],s['pattern'],flush=True)
        write_csv(REPORT/'missing_scenarios.csv',detailed)
    pr,lo,w=predict(selected_model,valid,device)
    raw_pr=pr.copy();pr=project_intensity(pr,lo)
    np.savez_compressed(REPORT/'validation_predictions.npz',y=valid['y'],cls=valid['cls'],pred=pr,raw_pred=raw_pr,logits=lo,weights=w,sample_id=valid['sample_id'],content_length=valid['content'].sum(1),native_vision_unavailable=1-valid['obs'][:,:,2].sum(1)/valid['content'].sum(1).clip(1))
    texts=json.loads((ROOT/'artifacts/valid_text.json').read_text())
    errors=[]
    for i in np.argsort(np.abs(valid['y']-pr))[::-1][:20]:
        errors.append(dict(sample_id=valid['sample_id'][i],true_intensity=float(valid['y'][i]),pred_intensity=float(pr[i]),true_class=int(valid['cls'][i]),pred_class=int(lo[i].argmax()),absolute_error=float(abs(valid['y'][i]-pr[i])),text=texts[valid['sample_id'][i]]))
    write_csv(REPORT/'validation_error_cases.csv',errors)
    special=load_special();sp,sl,sw=predict(selected_model,special,device);raw_sp=sp.copy();sp=project_intensity(sp,sl)
    labels=['Negative','Neutral','Positive'];rows=[]
    prob=torch.softmax(torch.from_numpy(sl),-1).numpy()
    for i,sid in enumerate(special['sample_id']):
        rows.append(dict(sample_id=sid,pred_polarity=labels[sl[i].argmax()],pred_intensity=round(float(sp[i]),6)))
    write_csv(REPORT/'附件3_第二问预测结果.csv',rows)
    diagnostics=[]
    for i,sid in enumerate(special['sample_id']):
        diagnostics.append(dict(sample_id=sid,content_length=int(special['content'][i].sum()),raw_intensity=float(raw_sp[i]),
            **{f'available_{m}':int(special['obs'][i,:,j].sum()) for j,m in enumerate(['text','audio','vision'])},
            **{f'prob_{label}':float(prob[i,j]) for j,label in enumerate(labels)},
            **{f'fusion_weight_{m}':float(sw[i,j]) for j,m in enumerate(['text','audio','vision'])}))
    write_csv(REPORT/'special_diagnostics.csv',diagnostics)
    assert len(rows)==len(set(special['sample_id']))==30
    assert np.isfinite(sp).all() and (np.abs(sp)<=3).all()
    state['bert_config']['_name_or_path']='google/bert_uncased_L-2_H-128_A-2'
    state['output_rule']=frozen['output_rule']
    torch.save(state,ROOT/'artifacts/final_model.pt')
    print('DONE',len(rows),'special predictions',flush=True)


if __name__=='__main__':main()
