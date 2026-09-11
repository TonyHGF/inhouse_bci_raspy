"""Archive the completed experiment outputs on a CPU allocation; no training/cache regeneration."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

BASE=Path('/public/home/hugf2022/inhouse_bci_raspy')
PROJECT=Path(__file__).resolve().parents[1]
DEST=BASE/'exports/all-experiments-20260911'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Submit slurm/export_results.slurm')
    DEST.mkdir(exist_ok=False)
    specs=[
        ('overnight-inhouse.tar.gz',[(BASE/'overnight-v1/inhouse','overnight-v1/inhouse'),(BASE/'overnight-v1/config.json','overnight-v1/config.json'),(BASE/'overnight-v1/reviews','overnight-v1/reviews')],set()),
        ('overnight-bci2a.tar.gz',[(BASE/'overnight-v1/bci2a','overnight-v1/bci2a'),(BASE/'overnight-v1/config.json','overnight-v1/config.json')],set()),
        ('selection-bias-64pct.tar.gz',[(BASE/'selection-bias-v1','selection-bias-v1')],set()),
        ('legacy-local-outputs.tar.gz',[(PROJECT/'outputs','legacy-local/outputs')],{'legacy-local/outputs/bids','legacy-local/outputs/prepared','legacy-local/outputs/matplotlib','legacy-local/outputs/mne_home'}),
        ('server-validation.tar.gz',[(BASE/'server-v1/results','server-v1/results')],set()),
        ('earlier-validation.tar.gz',[(BASE/'overnight-validation-v1','overnight-validation-v1'),(BASE/'selection-bias-validation-v1','selection-bias-validation-v1'),(BASE/'inhouse-comparison-v1','inhouse-comparison-v1'),(BASE/'inhouse-comparison-v2','inhouse-comparison-v2')],{'inhouse-comparison-v1/old_pipeline','inhouse-comparison-v2/prepared'}),
        ('cluster-logs.tar.gz',[(PROJECT/'logs','cluster-logs')],set()),
    ]
    manifest={}
    latest=BASE/'exports/inhouse-comparison-v3-results-20260911.tar.gz'
    shutil.copyfile(latest,DEST/latest.name)
    for name,roots,excluded in specs:
        out=DEST/name
        def filtering(member):
            if any(member.name==x or member.name.startswith(x+'/') for x in excluded): return None
            return member
        print('PACK',name,flush=True)
        with tarfile.open(out.with_suffix('.partial'),'w:gz',compresslevel=1,dereference=False) as t:
            for path,arc in roots:
                if not path.exists(): raise FileNotFoundError(path)
                t.add(path,arcname=arc,filter=filtering)
        out.with_suffix('.partial').replace(out)
        manifest[name]={'sources':[str(p) for p,_ in roots],'excluded':sorted(excluded)}
    manifest[latest.name]={'sources':[str(latest)],'excluded':['prepared','smoke-validation']}
    lines=[]
    for name,entry in manifest.items():
        p=DEST/name
        with tarfile.open(p,'r:gz') as t:
            count=0;size=0;links=[]
            for member in t:
                if member.isfile(): count+=1;size+=member.size
                elif member.issym(): links.append({'path':member.name,'target':member.linkname})
        entry.update(bytes=p.stat().st_size,sha256=sha(p),files=count,uncompressed_bytes=size,symlinks=links)
        lines.append(entry['sha256']+'  '+name)
        print('VERIFIED',name,entry['bytes'],count,'files',len(links),'symlinks',flush=True)
    (DEST/'SHA256SUMS').write_text('\n'.join(lines)+'\n')
    (DEST/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (DEST/'README.txt').write_text('All experiment result exports, 2026-09-11.\nEach tar.gz preserves models, predictions, plots, configs and histories available in its source directories.\nlegacy-local includes the historical ablation outputs. server-validation and earlier-validation are diagnostic runs, not formal experiments.\nRaw data and the main preprocessing caches are excluded; see MANIFEST.json for exact exclusions and any stored symlink targets.\nThese are result archives, not a portable training environment. Paths inside metadata retain server values.\nVerify: shasum -a 256 -c SHA256SUMS\nExtract: tar -xzf ARCHIVE.tar.gz\n')
    print('COMPLETE',DEST,flush=True)

if __name__=='__main__': main()
