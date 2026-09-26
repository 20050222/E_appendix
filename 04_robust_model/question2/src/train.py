"""Reproducible CPU/MPS/CUDA training; valid-only checkpoint selection."""
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, confusion_matrix
from data import ROOT, load_split, corruption
from model import RobustModel
from protocol import VERSION, OUTPUT_RULE, training_fingerprint, completed_run_is_valid, sha256

VARIANTS={
 'text_only':dict(text_only=True,gating=False,mask_aware=True,augment=False,rec_weight=0.),
 'fusion_clean':dict(gating=False,mask_aware=False,augment=False,rec_weight=0.),
 'fusion_aug':dict(gating=False,mask_aware=False,augment=True,rec_weight=0.),
 'robust_norec':dict(gating=True,mask_aware=True,augment=True,rec_weight=0.),
 'robust_rec':dict(gating=True,mask_aware=True,augment=True,rec_weight=.05),
 'robust_nogate':dict(gating=False,mask_aware=True,augment=True,rec_weight=.05),
}


def device_setup(request='auto',threads=4):
    torch.set_num_threads(threads)
    if request!='auto': return torch.device(request)
    if torch.cuda.is_available(): return torch.device('cuda')
    # CPU is the reproducible default; MPS may be benchmarked explicitly.
    return torch.device('cpu')


def metric(y,cls,pred,logits):
    pc=np.argmax(logits,axis=1)
    corr=float(np.corrcoef(y,pred)[0,1]) if np.std(pred)>1e-8 and np.std(y)>1e-8 else None
    return {'accuracy':float(accuracy_score(cls,pc)),
            'macro_f1':float(f1_score(cls,pc,labels=[0,1,2],average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(cls,pc,labels=[0,1,2],average='weighted',zero_division=0)),
            'mae':float(mean_absolute_error(y,pred)),'pearson':corr,
            'class_f1':f1_score(cls,pc,labels=[0,1,2],average=None,zero_division=0).tolist(),
            'confusion_matrix':confusion_matrix(cls,pc,labels=[0,1,2]).tolist(),
            'polarity_regression_conflict':float(np.mean(((pc==0)&(pred>=0))|((pc==2)&(pred<=0)))),
            'n':len(y)}


def project_intensity(pred,logits):
    """Fixed label-compatible projection, no fitted parameters or test access."""
    c=np.argmax(logits,axis=1)
    return np.where(c==1,0.,np.where(c==0,np.minimum(pred,-1e-6),np.maximum(pred,1e-6))).astype(np.float32)


def final_metric(y,cls,pred,logits):
    raw=metric(y,cls,pred,logits)
    result=metric(y,cls,project_intensity(pred,logits),logits)
    result.update(raw_mae=raw['mae'],raw_pearson=raw['pearson'],raw_polarity_regression_conflict=raw['polarity_regression_conflict'])
    return result


def batch(d,ix,device,kept=None):
    keys=['ids','segments','valid','content','audio','vision']
    b={k:torch.from_numpy(d[k][ix]).to(device) for k in keys}
    original=d['obs'][ix]; now=original if kept is None else kept[ix]
    b['obs']=torch.from_numpy(now).to(device)
    b['original_obs']=torch.from_numpy(original).to(device)
    b['removed']=torch.from_numpy(original & ~now).to(device)
    for k in ['y','cls']:
        if k in d:b[k]=torch.from_numpy(d[k][ix]).to(device)
    return b


@torch.no_grad()
def predict(model,d,device,kept=None,batch_size=128):
    model.eval();pred=[];logits=[];weights=[]
    for start in range(0,len(d['ids']),batch_size):
        ix=np.arange(start,min(start+batch_size,len(d['ids'])))
        r=model(batch(d,ix,device,kept))
        pred.append(r['reg'].cpu().numpy());logits.append(r['logits'].cpu().numpy());weights.append(r['weights'].cpu().numpy())
    return np.concatenate(pred),np.concatenate(logits),np.concatenate(weights)


def selection_score(metrics):
    return float(np.mean([x['mae']+1-x['macro_f1'] for x in metrics.values()]))


def validation_masks(d, seed=20260924):
    masks=[('clean',d['obs'])]
    for name,mods in [('T',[0]),('A',[1]),('V',[2]),('TAV',[0,1,2])]:
        kept,_=corruption(d['content'],d['obs'],seed=seed,sample_ids=d['sample_id'],rate=.3,modalities=mods)
        masks.append((name+'30',kept))
    return masks


def load_checkpoint(path,device):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=RobustModel(state['config'],bert_config=state['bert_config'])
    model.load_state_dict(state['model']);model.to(device).eval()
    return model,state


def train_one(args,variant,seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=device_setup(args.device,args.threads)
    train=load_split('train');valid=load_split('valid')
    if args.smoke:
        train={k:v[:64] for k,v in train.items()};valid={k:v[:32] for k,v in valid.items()}
    cfg={**VARIANTS[variant],'hidden':32,'variant':variant,'seed':seed,
         'lr':args.lr,'bert_lr':args.bert_lr,'batch_size':args.batch_size,
         'epochs':args.epochs,'patience':args.patience,'classification_weight':.6,
         'protocol':VERSION,'output_rule':OUTPUT_RULE,'fingerprint':training_fingerprint(),
         'smoke':args.smoke,'selection':'projected mean_over(clean,T30,A30,V30,TAV30)[MAE + 1 - MacroF1]',
         'train_noise':'25% clean, otherwise uniform 1-3 modalities, rates .1/.3/.5/.7; 75% one block, 25% separated two blocks (budget<2: one block)' }
    out=ROOT/('artifacts/smoke_runs' if args.smoke else 'artifacts/runs')/f'{variant}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if completed_run_is_valid(out,cfg) and not args.overwrite:
        print('SKIP',out.name,flush=True);return
    (out/'finished.json').unlink(missing_ok=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    model=RobustModel(cfg,pretrained=ROOT/'pretrained/bert-tiny').to(device)
    other=[p for n,p in model.named_parameters() if not n.startswith('bert.')]
    optim=torch.optim.AdamW([{'params':model.bert.parameters(),'lr':args.bert_lr}, {'params':other,'lr':args.lr}],weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim,T_max=args.epochs,eta_min=1e-6)
    weights=np.bincount(train['cls'],minlength=3).astype(float)
    weights=np.sqrt(weights.sum()/(3*weights));weights=weights/weights.mean()
    class_weight=torch.tensor(weights,dtype=torch.float32,device=device)
    rng=np.random.default_rng(seed);noise_rng=np.random.default_rng(seed+100000);vmasks=validation_masks(valid)
    best=float('inf');stale=0;history=[];begin=time.time()
    print(json.dumps({'start':out.name,'device':str(device),'parameters':sum(p.numel() for p in model.parameters())}),flush=True)
    for epoch in range(1,args.epochs+1):
        model.train();losses=[];ep_start=time.time()
        kept,_=corruption(train['content'],train['obs'],noise_rng,training=True,sample_ids=train['sample_id']) if cfg['augment'] else (train['obs'],None)
        order=rng.permutation(len(train['ids']))
        for start in range(0,len(order),args.batch_size):
            ix=order[start:start+args.batch_size];b=batch(train,ix,device,kept)
            optim.zero_grad(set_to_none=True)
            result=model(b,clean_targets=cfg['rec_weight']>0)
            loss=F.smooth_l1_loss(result['reg'],b['y'])+.6*F.cross_entropy(result['logits'],b['cls'],weight=class_weight)
            if cfg['rec_weight']>0:loss=loss+cfg['rec_weight']*result['rec_loss']
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        metrics={}
        for name,mask in vmasks:
            pr,lo,_=predict(model,valid,device,mask,args.eval_batch)
            metrics[name]=final_metric(valid['y'],valid['cls'],pr,lo)
        score=selection_score(metrics)
        improved=score<best-1e-4
        row={'epoch':epoch,'loss':float(np.mean(losses)),'score':score,'seconds':time.time()-ep_start,'validation':metrics}
        history.append(row)
        (out/'history.json').write_text(json.dumps(history,indent=2))
        if improved:
            best=score;stale=0
            torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':cfg,
                        'bert_config':model.bert.config.to_dict(),'best_epoch':epoch,'validation':metrics,'score':score},out/'best.tmp')
            (out/'best.tmp').replace(out/'best.pt')
        else:stale+=1
        print(json.dumps({'run':out.name,'epoch':epoch,'loss':round(row['loss'],4),'score':round(score,4),'clean_mae':round(metrics['clean']['mae'],4),'clean_f1':round(metrics['clean']['macro_f1'],4),'TAV30_mae':round(metrics['TAV30']['mae'],4),'seconds':round(row['seconds'],1),'best':improved}),flush=True)
        if stale>=args.patience:break
    (out/'finished.json').write_text(json.dumps({'protocol':VERSION,'checkpoint_sha256':sha256(out/'best.pt'),'best_score':best,'total_seconds':time.time()-begin,'epochs_completed':epoch},indent=2))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variants',nargs='+',default=['fusion_clean','fusion_aug','robust_norec','robust_rec'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[42])
    ap.add_argument('--epochs',type=int,default=16);ap.add_argument('--patience',type=int,default=5)
    ap.add_argument('--batch-size',type=int,default=64);ap.add_argument('--eval-batch',type=int,default=128)
    ap.add_argument('--lr',type=float,default=7e-4);ap.add_argument('--bert-lr',type=float,default=1e-4)
    ap.add_argument('--device',default='auto');ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--overwrite',action='store_true');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    if args.smoke: args.epochs=1;args.patience=1
    for seed in args.seeds:
        for variant in args.variants:train_one(args,variant,seed)

if __name__=='__main__':main()
