"""Competition-only aligned input adapter; never reconstruct text from raw_text."""
from pathlib import Path
import hashlib
import json
import pickle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / 'E题数据'
MODS = ('text', 'audio', 'vision')


def adapt(s):
    tb = np.asarray(s['text_bert'])
    if not np.all(np.isfinite(tb)) or not np.all(tb == tb.astype(np.int64)):
        raise ValueError('text_bert must contain finite integer token IDs/masks')
    ids = tb[:, 0].astype(np.int64)
    if ids.min() < 0 or ids.max() >= 30522:
        raise ValueError('Not the expected BERT WordPiece vocabulary')
    a = np.nan_to_num(np.asarray(s['audio'], dtype=np.float32))
    v = np.nan_to_num(np.asarray(s['vision'], dtype=np.float32))
    oa, ov = np.any(a != 0, -1), np.any(v != 0, -1)
    # Recover only the occupied timeline, not missing values. No labels are used.
    occupied = (tb[:, 1] > 0) | (ids != 0) | oa | ov
    pos = np.arange(ids.shape[1])[None, :]
    length = np.maximum(occupied * (pos + 1), 0).max(-1)
    # A completely empty input gets one non-content sentinel position so
    # self-attention is never asked to attend to an entirely padded sequence.
    length = np.maximum(length,1)
    valid = pos < length[:, None]
    # CLS / SEP are metadata, not acoustic/visual content positions.
    content = valid & (pos != 0) & (ids != 102)
    ot = content & (ids != 0) & (tb[:, 1] > 0)
    obs = np.stack([ot, oa & content, ov & content], axis=-1)
    result = dict(ids=ids, segments=tb[:, 2].astype(np.int64), valid=valid,
                  content=content, obs=obs, audio=a, vision=v)
    if 'regression_labels' in s:
        y = np.asarray(s['regression_labels'], dtype=np.float32)
        c = (np.sign(y) + 1).astype(np.int64)
        if not np.array_equal(c, np.asarray(s['classification_labels']).astype(int)):
            raise ValueError('Classification/regression labels disagree')
        result.update(y=y, cls=c)
    return result


def prepare():
    out = ROOT / 'artifacts'; out.mkdir(exist_ok=True)
    source = DATA / '附件2-数据集特征文件/aligned_50.pkl'
    with source.open('rb') as f:
        original = pickle.load(f)
    splits = {k: adapt(v) for k, v in original.items()}
    stats = {}
    for m, j in [('audio', 1), ('vision', 2)]:
        x = splits['train'][m][splits['train']['obs'][:, :, j]]
        stats[m + '_mean'] = x.mean(0, dtype=np.float64).astype(np.float32)
        stats[m + '_std'] = np.maximum(x.std(0, dtype=np.float64), 1e-3).astype(np.float32)
    np.savez(out / 'normalization.npz', **stats)
    audit = {'source': str(source.relative_to(ROOT.parent)), 'splits': {},
             'text_input': 'text_bert only; text/raw_text are not model inputs',
             'native_zero_rule': 'unavailable (not proof of artificial deletion)',
             'native_zero_ambiguity': 'Original naturally all-zero A/V positions cannot be distinguished from deleted positions.',
             'time_rule': 'prefix through last occupied token/A/V; exclude CLS and SEP from content',
             'normalization': 'train observed content only; z-score, clip [-5,5], unavailable reset to zero'}
    for k, d in splits.items():
        for m,j in [('audio',1),('vision',2)]:
            d[m] = np.clip((d[m]-stats[m+'_mean']) / stats[m+'_std'], -5, 5)
            d[m] *= d['obs'][:, :, j, None]
        d['sample_id'] = np.asarray(original[k]['id'])
        np.savez_compressed(out / (k + '.npz'), **d)
        audit['splits'][k] = {'n': len(d['ids']), 'class_counts': np.bincount(d['cls'], minlength=3).tolist(),
            'content_length_quantiles': np.quantile(d['content'].sum(1), [0,.25,.5,.75,1]).tolist(),
            'native_unavailable_rate': {m: float(((~d['obs'][:,:,j]) & d['content']).sum()/d['content'].sum()) for j,m in enumerate(MODS)},
            'nonfinite_original': {m: int((~np.isfinite(original[k][m])).sum()) for m in ['audio','vision']}}
    audit['id_overlap'] = {}
    audit['video_overlap'] = {}
    h=hashlib.sha256()
    with source.open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''):h.update(block)
    audit['source_sha256']=h.hexdigest()
    for x,y in [('train','valid'),('train','test'),('valid','test')]:
        audit['id_overlap'][x+'_'+y] = len(set(original[x]['id']) & set(original[y]['id']))
        audit['video_overlap'][x+'_'+y] = len({s.split('$_$')[0] for s in original[x]['id']} & {s.split('$_$')[0] for s in original[y]['id']})
    # Keep readable validation text only for post-training error analysis.
    (out/'valid_text.json').write_text(json.dumps(dict(zip(original['valid']['id'],original['valid']['raw_text'].tolist())),ensure_ascii=False),encoding='utf-8')
    (ROOT/'reports/data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


def load_split(split):
    with np.load(ROOT/'artifacts'/f'{split}.npz',allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def load_special(folder=None):
    folder = Path(folder) if folder else DATA/'附件3-模态缺失特征样本/对齐版本'
    rows, names = [], []
    stats = np.load(ROOT/'artifacts/normalization.npz')
    for p in sorted(folder.glob('*.pkl')):
        with p.open('rb') as f: s=pickle.load(f)
        s=s.get('test',s); d=adapt(s)
        for m,j in [('audio',1),('vision',2)]:
            d[m]=np.clip((d[m]-stats[m+'_mean'])/stats[m+'_std'],-5,5)*d['obs'][:,:,j,None]
        rows.append(d)
        names.extend([p.stem if len(d['ids'])==1 else f'{p.stem}:{i}' for i in range(len(d['ids']))])
    if not rows: raise ValueError('No special test files found')
    result={k:np.concatenate([d[k] for d in rows]) for k in rows[0] if k not in ('y','cls')}
    result['sample_id']=np.asarray(names)
    return result


def deletion_count(n, rate):
    return min(n-1, max(1, int(round(rate*n)))) if n>=2 and rate>0 else 0


def pattern_eligible(content, rate):
    """All pattern comparisons use this same population (two nonempty blocks)."""
    return np.array([deletion_count(int(n), rate)>=2 for n in content.sum(1)])


def keyed_rng(seed, sample_id, domain):
    payload=json.dumps([int(seed),str(sample_id),str(domain)],ensure_ascii=False).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:16],'little'))


def corruption(content, obs, rng=None, rate=None, modalities=None, position='random',
               pattern='block', training=False, *, seed=None, sample_ids=None):
    """Pair masks by seed/sample ID/modality, independent of other requested modalities.

    Evaluation two_blocks rejects ineligible samples; callers must subset every
    compared pattern identically. Training explicitly falls back to one block for
    a deletion budget below two. Gap separation refers to the content timeline;
    naturally unavailable features remain unavailable and are not reconstruction targets.
    """
    if seed is None:
        if rng is None: raise ValueError('Supply seed or rng')
        seed=int(rng.integers(0,2**63-1))
    if sample_ids is None: sample_ids=np.arange(len(content))
    if len(sample_ids)!=len(content): raise ValueError('sample_ids length mismatch')
    if pattern not in ('block','two_blocks','points'): raise ValueError(pattern)
    if position not in ('random','front','middle','back'): raise ValueError(position)
    if not training and rate is not None and not 0<=rate<=1: raise ValueError('rate outside [0,1]')
    if not training and pattern=='two_blocks' and not pattern_eligible(content,float(rate or 0)).all():
        raise ValueError('two_blocks requires >=2 deletions; subset all patterns using pattern_eligible')
    kept=obs.copy()
    for i,sid in enumerate(sample_ids):
        ix=np.flatnonzero(content[i]); n=len(ix)
        if n<2: continue
        meta=keyed_rng(seed,sid,'training-meta')
        if training:
            if meta.random()<.25: continue
            chosen=meta.choice(3,size=int(meta.integers(1,4)),replace=False)
            rr=float(meta.choice([.1,.3,.5,.7]))
            pat='block' if meta.random()<.75 else 'two_blocks'
        else:
            chosen=modalities if modalities is not None else [0,1,2]
            rr=float(rate or 0); pat=pattern
        count=deletion_count(n,rr)
        if not count: continue
        for m in chosen:
            local=keyed_rng(seed,sid,f'modality-{int(m)}')
            if pat=='points': chosen_ix=local.choice(ix,size=count,replace=False)
            elif pat=='two_blocks' and count>=2:
                left=count//2; right=count-left
                # Uniform weak composition of spare positions into before/gap/after.
                slack=n-count-1
                bars=np.sort(local.choice(slack+2,size=2,replace=False))
                start=int(bars[0]); gap=int(bars[1]-bars[0])  # always >=1
                chosen_ix=np.r_[ix[start:start+left],ix[start+left+gap:start+left+gap+right]]
            else:
                start={'front':0,'middle':(n-count)//2,'back':n-count}.get(position)
                if start is None: start=int(local.integers(0,n-count+1))
                chosen_ix=ix[start:start+count]
            kept[i,chosen_ix,m]=False
    return kept,obs & ~kept


if __name__=='__main__': prepare()
