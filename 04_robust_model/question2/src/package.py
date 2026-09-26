"""Small anonymous submission bundle; exclude environments, caches and raw data."""
from pathlib import Path
import hashlib
import json
import zipfile
from data import ROOT

def main():
    files=[]
    for folder,patterns in [('src',['*.py','*LICENSE']),('tests',['*.py']),('reports',['*.json','*.csv','*.md','*.npz','*.log','figures/*.png','training_runs/*/*.json'])]:
        for pattern in patterns:files.extend((ROOT/folder).glob(pattern))
    files += [ROOT/'README.md',ROOT/'requirements.txt',ROOT/'requirements-lock.txt',
              ROOT/'artifacts/final_model.pt',ROOT/'artifacts/normalization.npz',
              ROOT/'pretrained/bert-tiny/provenance.json',ROOT/'pretrained/bert-tiny/LICENSE-APACHE-2.0.txt']
    # Historical records are not runnable completion markers. Only final weights ship.
    import shutil
    for run in (ROOT/'artifacts/runs').glob('*'):
        if not run.is_dir():continue
        target=ROOT/'reports/training_runs'/run.name;target.mkdir(parents=True,exist_ok=True)
        for name in ['config.json','history.json','finished.json']:
            shutil.copy2(run/name,target/name)
            files.append(target/name)
    files=sorted(set(p for p in files if p.is_file() and p.name!='submission_manifest.json'))
    path_markers=(bytes([47,85,115,101,114,115,47]),bytes([47,104,111,109,101,47]))
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'reports/submission_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    output=ROOT/'第二问_代码模型与结果.zip'
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:
            if p.suffix in ['.py','.md','.json','.csv','.txt','.log']:
                raw=p.read_bytes()
                if any(marker in raw for marker in path_markers):
                    raise ValueError(f'Absolute personal path in submission: {p.name}')
            z.write(p,Path('question2')/p.relative_to(ROOT))
        z.write(ROOT/'reports/submission_manifest.json','question2/reports/submission_manifest.json')
    if output.stat().st_size>50_000_000:raise ValueError('Submission exceeds 50MB')
    with zipfile.ZipFile(ROOT/'artifacts/final_model.pt') as z:
        for name in z.namelist():
            if name.endswith('.pkl'):
                b=z.read(name)
                if any(marker in b for marker in path_markers):raise ValueError('Personal path in model metadata')
    print(json.dumps({'file':output.name,'bytes':output.stat().st_size,'files':len(files)+1},ensure_ascii=False))

if __name__=='__main__':main()
