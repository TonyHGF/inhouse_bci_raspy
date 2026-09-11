"""Aligned raw-trial fivefold comparison of preprocessing and final 80% training."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import subprocess
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bci_raspy_experiments.common import PROJECT, digest, file_hash, read_json, save_json
from overnight import immutable
from selection_bias import array_hash, seeded_loader

ARMS = ('old_strict', 'new_strict', 'old_test_selected')


def config(path):
    c = read_json(path)
    for key in ('legacy_config', 'dataset_config', 'raw_cache'):
        c[key] = str((PROJECT / c[key]).resolve())
    if not Path(c['output_root']).is_relative_to('/public/home/hugf2022'):
        raise ValueError('Output must be under /public/home/hugf2022')
    if not set(c['folds']) <= set(range(5)) or len(set(c['folds'])) != len(c['folds']) or not c['folds']:
        raise ValueError('Invalid folds')
    for k in ('max_epochs', 'parallel_per_gpu', 'threads_per_process'):
        if c[k] < 1: raise ValueError(k)
    return c


def source_hash():
    paths = [Path(__file__), PROJECT/'scripts/selection_bias.py', PROJECT/'scripts/overnight.py']
    for package in ('inhouse_bci_raspy', 'bci_raspy_experiments'):
        paths.extend(sorted((PROJECT/'src'/package).rglob('*.py')))
    return digest({str(p.relative_to(PROJECT)):file_hash(p) for p in paths})


def aligned_splits(labels, seed, fraction):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, train_test_split
    result = []
    for fold, (outer, test) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels)):
        train, val = train_test_split(outer, test_size=fraction, stratify=np.asarray(labels)[outer], random_state=seed+fold)
        result.append(dict(fold=fold, training=train.tolist(), validation=val.tolist(), outer_training=outer.tolist(), test=test.tolist()))
    return result


def old_trials(c, catalog, legacy):
    import mne
    import numpy as np
    from inhouse_bci_raspy.preprocessing.pipeline import preprocess
    root = Path(c['output_root'])
    dataset = read_json(c['dataset_config'])
    dataset['bids_root'] = str(root/'old_pipeline'/'bids')
    marker = root/'old_pipeline'/'complete.json'
    if not marker.exists():
        preprocess(legacy, dataset, root/'old_pipeline')
        immutable(marker, dict(source_hash=source_hash(), mne=importlib.metadata.version('mne')))
    elif read_json(marker)['source_hash'] != source_hash():
        raise ValueError('Preprocessing source changed')
    data, sources = {}, {}
    rows = {(r['session'], r['trial']): (i,r) for i,r in enumerate(catalog['trials'])}
    for rec in legacy['recordings']:
        name = f"sub-{dataset['subject']}_ses-{rec['session']}_task-{dataset['task']}_proc-clean_epo.fif"
        path = root/'old_pipeline'/'bids'/'derivatives'/'mne-bids-pipeline'/f"sub-{dataset['subject']}"/f"ses-{rec['session']}"/'eeg'/name
        ep = mne.read_epochs(path, preload=True, verbose='error')[list(dataset['event_id'])]
        inverse = {v:k for k,v in ep.event_id.items()}
        labels = [list(dataset['event_id']).index(inverse[int(v)]) for v in ep.events[:,2]]
        ep = mne.preprocessing.compute_current_source_density(ep, copy=True)
        expected = catalog['trials'][0]['channels']
        if ep.ch_names != expected or ep.info['sfreq'] != 250 or ep.tmin != -2:
            raise ValueError('Old pipeline channel/time layout changed')
        for x, trial, label, event in zip(ep.get_data(copy=True), ep.selection, labels, ep.events):
            i, row = rows[(rec['session'], int(trial))]
            if row['label'] != label or abs(event[0]/250-row['onset']) > .005:
                raise ValueError('Raw identity/onset/label mismatch')
            data[i] = x
        sources[str(path)] = file_hash(path)
    save_json(root/'old_sources.json', sources)
    return data


def normalize_old(data, fit_ids):
    import numpy as np
    fitting = [data[i] for i in fit_ids if i in data]
    scale = np.maximum(np.sqrt(sum(np.square(x).sum(1) for x in fitting)/sum(x.shape[1] for x in fitting)), np.finfo(float).eps)
    return {i:x/scale[:,None] for i,x in data.items()}, dict(rms=scale.tolist(), retained_fit_ids=[i for i in fit_ids if i in data])


def prepare(c):
    import numpy as np
    from inhouse_bci_raspy.data.windows import make_windows
    from bci_raspy_experiments.preprocessing import Pipeline
    from bci_raspy_experiments.cache import writer_lock
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=True)
    with writer_lock(root/'prepare.lock', timeout=1):
        immutable(root/'config.json', c)
        immutable(root/'source.json', dict(sha256=source_hash()))
        catalog = read_json(Path(c['raw_cache'])/'inhouse'/'catalog.json')
        rows = catalog['trials']; labels = np.asarray([r['label'] for r in rows])
        legacy = read_json(c['legacy_config']); legacy['training']['max_epochs'] = c['max_epochs']
        immutable(root/'legacy_config.json', legacy)
        splits = aligned_splits(labels,c['seed'],c['inner_validation_fraction'])
        immutable(root/'manifest.json', dict(trials=rows,splits=splits,arms=ARMS,
                  raw_sha256=file_hash(Path(c['raw_cache'])/'inhouse'/'raw.h5'),
                  preprocessing_packages={p:importlib.metadata.version(p) for p in ('mne','mne-bids','mne-bids-pipeline','numpy','scipy')}))
        old = old_trials(c,catalog,legacy)
        for split in splits:
            fold=split['fold']
            if fold not in c['folds']: continue
            for method in ('old','new'):
                for stage, fit_key, eval_key in [('selection','training','validation'), ('final','outer_training','test')]:
                    out=root/'prepared'/f'fold_{fold}'/method/stage
                    if (out/'complete.json').exists(): continue
                    out.mkdir(parents=True,exist_ok=True)
                    fit_ids=split[fit_key]; eval_ids=split[eval_key]
                    if method=='old':
                        transformed,meta=normalize_old(old,fit_ids)
                    else:
                        pipe=Pipeline(catalog,c['raw_cache'],'full').fit([rows[i]['id'] for i in fit_ids])
                        pipe.save(out/'pipeline.pkl')
                        transformed={}
                        for i in fit_ids+eval_ids:
                            x,reason=pipe.unscaled(rows[i]['id'])
                            if reason is None: transformed[i]=x/pipe.scale[:,None]
                        meta=pipe.metadata()
                    retained={name:[i for i in ids if i in transformed] for name,ids in [('train',fit_ids),('eval',eval_ids)]}
                    rng=np.random.default_rng(c['seed']+fold)
                    for name,copies in [('train',5),('eval',1)]:
                        ids=retained[name]
                        if set(labels[ids])!={0,1,2,3}: raise ValueError('Missing class')
                        # Global indices preserved in every window block; omitted trials are never read.
                        shape=next(iter(transformed.values())).shape
                        x=np.zeros((len(rows),*shape),dtype=np.float64)
                        for i in ids: x[i]=transformed[i]
                        block=make_windows(x,labels,ids,250.,-2.,legacy['window'],copies,rng)
                        np.savez(out/f'{name}.npz',**block)
                        del x,block
                    save_json(out/'complete.json',dict(preprocessing=meta,requested={'train':fit_ids,'eval':eval_ids},retained=retained,
                              checksums={n:file_hash(out/f'{n}.npz') for n in ('train','eval')}))
                    print(f'Prepared fold={fold} {method}/{stage}: '+str({k:len(v) for k,v in retained.items()}),flush=True)
        save_json(root/'prepared.json',dict(state='complete',folds=c['folds']))


def load_block(root,fold,method,stage,name):
    import numpy as np
    out=root/'prepared'/f'fold_{fold}'/method/stage
    meta=read_json(out/'complete.json')
    if file_hash(out/f'{name}.npz')!=meta['checksums'][name]: raise ValueError('Prepared block changed')
    with np.load(out/f'{name}.npz') as f: return {k:f[k] for k in f.files}


def fit_model(block, evaluation, settings, seed, target, fixed_schedule=None, device_name='cuda:0'):
    """Same window optimizer for all arms. Fixed refit cannot accept an evaluation block."""
    import numpy as np
    import torch
    from inhouse_bci_raspy.runtime import setup_training
    from inhouse_bci_raspy.models.eegnet import EEGNet
    from inhouse_bci_raspy.training.losses import load_loss_criterion
    from inhouse_bci_raspy.training.engine import test
    if (evaluation is None) != (fixed_schedule is not None):
        raise ValueError('Refit requires a fixed schedule and no evaluation data')
    target.mkdir(parents=True,exist_ok=True)
    device=setup_training(seed,device_name)
    model=EEGNet(settings['model'],output_dim=4,n_electrodes=16).to(device)
    initial=digest({k:array_hash(v.detach().cpu().numpy()) for k,v in model.state_dict().items()})
    cfg=settings['training']; training=seeded_loader(block,cfg,'train',seed+1000)
    counts=np.bincount(block['y'],minlength=4)
    if np.any(counts==0): raise ValueError('Missing training class')
    criterion=load_loss_criterion(cfg['loss_func'],weight=torch.tensor(counts.mean()/counts,dtype=torch.float32))
    opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max')
    selection=None if evaluation is None else seeded_loader(evaluation,cfg,'val',seed+2000)
    history=[]; best_loss=float('inf'); best_epoch=0; stale=0; best=None
    limit=cfg['max_epochs'] if fixed_schedule is None else len(fixed_schedule)
    for epoch in range(limit):
        if fixed_schedule is not None:
            for g in opt.param_groups: g['lr']=fixed_schedule[epoch]
        record=dict(epoch=epoch+1,lr=opt.param_groups[0]['lr'])
        model.train(); losses=[]
        for x,y in training:
            opt.zero_grad(); loss=criterion(model(x),y.to(device))
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        record['train_loss']=float(np.mean(losses))
        if selection is not None:
            val_loss,correct,_,_=test(model,selection,criterion)
            value=float(val_loss); accuracy=100*sum(correct)/len(correct)
            record.update(selection_loss=value/len(selection),selection_accuracy=accuracy)
            scheduler.step(accuracy)
            if value<best_loss:
                best_loss=value; best_epoch=epoch; stale=0
                best=deepcopy(model.state_dict())
            else: stale+=1
        else:
            best_epoch=epoch
        history.append(record)
        save_json(target/'history.json',history)
        print(f'{target} epoch={epoch+1} train_loss={record["train_loss"]:.6g}',flush=True)
        if selection is not None and epoch>5 and stale==10: break
    if best is not None: model.load_state_dict(best)
    torch.save(dict(model=model.state_dict(),settings=settings),target/'model.pt')
    result=dict(initial_state_hash=initial,first_training_loss=history[0]['train_loss'],
                training_X_sha256=array_hash(block['X']),training_y_sha256=array_hash(block['y']),
                train_trials=sorted(np.unique(block['trial_index']).tolist()),epochs=len(history),best_epoch=best_epoch+1,
                schedule=[r['lr'] for r in history[:best_epoch+1]],refit=fixed_schedule is not None,
                selection_hit_cap=selection is not None and len(history)==limit)
    save_json(target/'training.json',result)
    return model,result


def run_fold(c,fold):
    import numpy as np
    import torch
    from inhouse_bci_raspy.evaluation import predict_probabilities
    from inhouse_bci_raspy.reporting import save_predictions
    root=Path(c['output_root']); settings=read_json(root/'legacy_config.json')
    for arm in ARMS:
        out=root/f'fold_{fold}'/arm; out.mkdir(parents=True,exist_ok=True)
        if (out/'status.json').exists() and read_json(out/'status.json')['state']=='complete': continue
        save_json(out/'status.json',dict(state='running')); start=time.monotonic()
        method='new' if arm=='new_strict' else 'old'
        try:
            schedule=None
            if arm.endswith('strict'):
                tr=load_block(root,fold,method,'selection','train'); val=load_block(root,fold,method,'selection','eval')
                model,info=fit_model(tr,val,settings,c['seed']+fold,out/'selection')
                schedule=info['schedule']; del tr,val,model
            tr=load_block(root,fold,method,'final','train')
            # Strict refit has no test array in scope. The test-selected arm deliberately supplies it.
            selection=load_block(root,fold,method,'final','eval') if arm=='old_test_selected' else None
            model,info=fit_model(tr,selection,settings,c['seed']+fold,out/'final',fixed_schedule=schedule)
            del tr,selection
            test_block=load_block(root,fold,method,'final','eval')
            p=predict_probabilities(model,test_block,torch.device('cuda:0'),settings['training']['val_batch_size'])
            save_predictions(out/'test_predictions.csv',test_block,p)
            np.savez_compressed(out/'test_predictions.npz',probabilities=p,y=test_block['y'],trial_index=test_block['trial_index'])
            save_json(out/'status.json',dict(state='complete',seconds=time.monotonic()-start))
            del model,test_block,p
        except BaseException as exc:
            save_json(out/'status.json',dict(state='failed',error=str(exc))); raise
    a,b=[read_json(root/f'fold_{fold}'/arm/'final'/'training.json') for arm in ('old_strict','old_test_selected')]
    for key in ('initial_state_hash','training_X_sha256','training_y_sha256','train_trials'):
        if a[key]!=b[key]: raise ValueError('Unmatched final training: '+key)
    if not np.isclose(a['first_training_loss'],b['first_training_loss'],rtol=0,atol=1e-7): raise ValueError('First epoch mismatch')
    immutable(root/f'fold_{fold}'/'pair_verified.json',dict(identical_old_final_training=True,identical_initialization=True,equal_first_epoch=True))


def report(c):
    import csv
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
    root=Path(c['output_root']); rows=[]; trials={a:{} for a in ARMS}; missing=[]
    for fold in c['folds']:
        if not (root/f'fold_{fold}'/'pair_verified.json').exists(): missing.append(fold); continue
        for arm in ARMS:
            path=root/f'fold_{fold}'/arm
            with np.load(path/'test_predictions.npz') as f: p,y,ids=f['probabilities'],f['y'],f['trial_index']
            for i in np.unique(ids):
                mask=ids==i; prob=p[mask].mean(0)
                trials[arm][int(i)]=dict(fold=fold,actual=int(y[mask][0]),predicted=int(prob.argmax()),
                                       hard_vote=int(np.bincount(p[mask].argmax(1),minlength=4).argmax()),probabilities=prob.tolist())
            rows.append(dict(fold=fold,arm=arm,retained_test=len(np.unique(ids)),window_accuracy=accuracy_score(y,p.argmax(1)),
                             trial_accuracy=accuracy_score([trials[arm][int(i)]['actual'] for i in np.unique(ids)],[trials[arm][int(i)]['predicted'] for i in np.unique(ids)]),
                             **read_json(path/'final'/'training.json')))
    out=root/'reports'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); out.mkdir(parents=True)
    common=sorted(set.intersection(*(set(v) for v in trials.values())))
    def metrics(values,ids):
        actual=[values[i]['actual'] for i in ids]; pred=[values[i]['predicted'] for i in ids]
        return dict(n=len(ids),accuracy=accuracy_score(actual,pred),macro_f1=f1_score(actual,pred,labels=[0,1,2,3],average='macro',zero_division=0),
                    kappa=cohen_kappa_score(actual,pred,labels=[0,1,2,3]),confusion_matrix=confusion_matrix(actual,pred,labels=[0,1,2,3]).tolist())
    summary=dict(missing_folds=missing,provisional=bool(missing) or len(c['folds'])!=5,common_test_ids=common,rows=rows)
    if common:
        summary['common_metrics']={a:metrics(v,common) for a,v in trials.items()}
        summary['retained_metrics']={a:metrics(v,sorted(v)) for a,v in trials.items()}
        summary['preprocessing_new_minus_old']=summary['common_metrics']['new_strict']['accuracy']-summary['common_metrics']['old_strict']['accuracy']
        summary['test_selection_minus_strict']=summary['common_metrics']['old_test_selected']['accuracy']-summary['common_metrics']['old_strict']['accuracy']
        for a,v in trials.items(): save_json(out/f'{a}_trials.json',v)
        with (out/'folds.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n'); w.writeheader(); w.writerows(rows)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4)); ax.bar(ARMS,[summary['common_metrics'][a]['accuracy']*100 for a in ARMS]); ax.set_ylabel('Common outer-test trial accuracy (%)'); fig.tight_layout()
        for ext in ('png','pdf'): fig.savefig(out/f'comparison.{ext}',dpi=180)
        plt.close(fig)
    save_json(out/'summary.json',summary)
    lines=['# Aligned Inhouse fivefold comparison','',f'Complete folds: {len(c["folds"])-len(missing)}/{len(c["folds"])}; provisional={summary["provisional"]}',
           '', 'All final models train on the retained outer 80%. Strict models use inner selection then fresh refit. Old preprocessing is session-wide before splitting; new preprocessing fits training data only.',
           '', 'This selection-protocol comparison includes selection/refit versus direct test-selected fitting; it is not a selection-source-only paired estimate.', '', '| Arm | Common trial accuracy | N |','|---|---:|---:|']
    for a,m in summary.get('common_metrics',{}).items(): lines.append(f'| {a} | {m["accuracy"]:.2%} | {m["n"]} |')
    (out/'report.md').write_text('\n'.join(lines)+'\n'); print('REPORT',out,flush=True)
    if missing or not common: raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['check','prepare','smoke','run','fold','report']); p.add_argument('--config',type=Path,default=Path('config/inhouse_comparison.json')); p.add_argument('--fold',type=int)
    args=p.parse_args(); c=config(args.config)
    if args.stage=='check':
        for k in ('legacy_config','dataset_config','raw_cache'):
            if not Path(c[k]).exists(): raise FileNotFoundError(c[k])
        print(json.dumps(c,indent=2)); return
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Use run_inhouse_comparison.sh server')
    if args.stage=='prepare': prepare(c); return
    root=Path(c['output_root'])
    if read_json(root/'config.json')!=c or read_json(root/'source.json')['sha256']!=source_hash(): raise ValueError('Frozen config/source changed')
    if args.stage=='smoke':
        smoke=deepcopy(c); smoke['output_root']=str(root/'smoke-validation'); smoke['folds']=[c['folds'][0]]; smoke['max_epochs']=2
        dest=Path(smoke['output_root']); dest.mkdir(exist_ok=True)
        immutable(dest/'config.json',smoke); immutable(dest/'source.json',read_json(root/'source.json'))
        settings=read_json(root/'legacy_config.json'); settings['training']['max_epochs']=2
        immutable(dest/'legacy_config.json',settings)
        if not (dest/'prepared').exists(): (dest/'prepared').symlink_to(root/'prepared',target_is_directory=True)
        run_fold(smoke,smoke['folds'][0]); report(smoke)
    elif args.stage=='report': report(c)
    elif args.stage=='run': manage(args,c)
    else:
        if args.fold not in c['folds']: raise ValueError('Fold not configured')
        def interrupt(sig,frame): raise KeyboardInterrupt('Allocation ended')
        signal.signal(signal.SIGTERM,interrupt); signal.signal(signal.SIGINT,interrupt)
        run_fold(c,args.fold)


def manage(args,c):
    if int(os.environ.get('SLURM_CPUS_PER_TASK','0')) < c['parallel_per_gpu']*c['threads_per_process']:
        raise ValueError('Insufficient CPU allocation')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one Slurm-visible GPU required')
    root=Path(c['output_root']); lock=root/'queue.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f: f.write(f'{os.uname().nodename} pid={os.getpid()} job={os.environ["SLURM_JOB_ID"]}')
    pending=iter(c['folds']); active=[]; failed=[]; stopping=False; exhausted=False
    def stop(sig,frame):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    try:
        while active or not exhausted:
            if stopping: break
            while len(active)<c['parallel_per_gpu'] and not exhausted:
                fold=next(pending,None)
                if fold is None: exhausted=True; break
                log=(root/f'fold_{fold}-{stamp}.log').open('x')
                env=dict(os.environ)
                for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
                    env[key]=str(c['threads_per_process'])
                command=[sys.executable,'-B',str(Path(__file__).resolve()),'fold','--config',str(args.config.resolve()),'--fold',str(fold)]
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
                active.append((fold,process,log)); print(f'Start paired fold={fold} pid={process.pid}',flush=True)
            for fold,process,log in list(active):
                code=process.poll()
                if code is not None:
                    log.close(); active.remove((fold,process,log))
                    if code: failed.append(fold)
                    print(f'End paired fold={fold} exit={code}',flush=True)
            if active: time.sleep(1)
    finally:
        for _,process,_ in active:
            if process.poll() is None: process.send_signal(signal.SIGINT)
        for _,process,log in active:
            try: process.wait(timeout=30)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            log.close()
        lock.unlink(missing_ok=True)
    if failed or stopping: raise SystemExit(1)



if __name__ == "__main__":
    main()
