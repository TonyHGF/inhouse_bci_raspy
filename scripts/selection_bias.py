"""Paired test-selection diagnostic on the frozen original MNE-BIDS clean data."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PureWindowsPath
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bci_raspy_experiments.common import PROJECT, digest, file_hash, read_json, save_json
from overnight import immutable

ARMS = ('validation_selected', 'test_selected')


def config(path):
    c = read_json(path)
    if c['design'] != 'matched_training':
        raise ValueError('This diagnostic requires identical training trials in both arms')
    for k in ('legacy_cache', 'legacy_folds', 'legacy_config', 'legacy_bids_root'):
        c[k] = str((PROJECT / c[k]).resolve())
    if not Path(c['output_root']).is_absolute() or not Path(c['output_root']).is_relative_to('/public/home/hugf2022'):
        raise ValueError('Use an absolute output_root under /public/home/hugf2022')
    if not 0 < c['inner_validation_fraction'] < 1:
        raise ValueError('Invalid internal validation fraction')
    if not c['folds'] or len(set(c['folds'])) != len(c['folds']) or not set(c['folds']) <= set(range(5)):
        raise ValueError('folds must be unique indices in 0..4')
    for k in ('parallel_per_gpu', 'threads_per_process', 'max_epochs'):
        if not isinstance(c[k], int) or c[k] < 1:
            raise ValueError(f'{k} must be positive')
    return c


def split_trials(y, folds, fold, c):
    import numpy as np
    from sklearn.model_selection import train_test_split
    outer_test = np.asarray(folds[fold])
    pool = np.concatenate([v for i,v in enumerate(folds) if i != fold])
    training, validation = train_test_split(pool, test_size=c['inner_validation_fraction'],
                                          stratify=y[pool], random_state=c['seed'] + fold)
    parts = (training, validation, outer_test)
    if any(set(a) & set(b) for i,a in enumerate(parts) for b in parts[i+1:]):
        raise ValueError('Overlapping trial identities')
    if any(set(y[a]) != {0,1,2,3} for a in parts):
        raise ValueError('Each partition must contain all four classes')
    return training, validation, outer_test


def prepare(c):
    import h5py
    import numpy as np
    import shutil
    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    immutable(root/'config.json', c)
    legacy = read_json(c['legacy_config'])
    legacy['training']['max_epochs'] = c['max_epochs']
    immutable(root/'legacy_config.json', legacy)
    metadata = read_json(c['legacy_folds'])
    source_hashes = {}
    for source in metadata['sources']:
        name = PureWindowsPath(source['path']).name
        matches = list(Path(c['legacy_bids_root']).glob(f'sub-*/ses-*/eeg/{name}'))
        if len(matches) != 1 or file_hash(matches[0]) != source['sha256']:
            raise ValueError(f'Legacy MNE-BIDS clean FIF missing/changed: {name}')
        source_hashes[str(matches[0])] = source['sha256']
    raw_hash = file_hash(c['legacy_cache'])
    destination = root/'clean_trials.h5'
    if destination.exists():
        if file_hash(destination) != raw_hash:
            raise ValueError('Frozen clean cache changed')
    else:
        temporary = root/'clean_trials.h5.tmp'
        shutil.copyfile(c['legacy_cache'], temporary)
        if file_hash(temporary) != raw_hash:
            raise ValueError('Copied cache checksum mismatch')
        temporary.replace(destination)
    with h5py.File(destination, 'r') as f:
        if list(f['X'].shape) != metadata['shape'] or len(metadata['channel_names']) != 16:
            raise ValueError('Legacy clean cache layout differs from metadata')
        y = f['y'][:]
        folds = [f[f'fold_{i}'][:] for i in range(5)]
        if sorted(np.concatenate(folds).tolist()) != list(range(len(y))):
            raise ValueError('Legacy folds do not cover trials exactly once')
        for i in range(5):
            if folds[i].tolist() != metadata['folds'][i]['validation_trial_indices']:
                raise ValueError('HDF5 and legacy fold metadata disagree')
        splits = []
        for fold in c['folds']:
            training, validation, test = split_trials(y, folds, fold, c)
            splits.append(dict(fold=fold, training=training.tolist(), validation=validation.tolist(), test=test.tolist()))
    immutable(root/'manifest.json', dict(cache_sha256=raw_hash, sources=source_hashes, metadata=metadata,
              splits=splits, arms=list(ARMS), note='Fixed historical session-level MNE-BIDS cleaning plus CSD; only selection-set identity changes between arms. No refit.'))
    print(f'Prepared {len(splits)} paired folds; source cache SHA256={raw_hash}', flush=True)


def array_hash(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def make_blocks(c, split, legacy):
    import h5py
    import numpy as np
    from inhouse_bci_raspy.data.windows import make_windows
    with h5py.File(Path(c['output_root'])/'clean_trials.h5', 'r') as f:
        x,y = f['X'][:],f['y'][:]
        sf,tmin = float(f.attrs['sfreq']),float(f.attrs['tmin'])
    tr = np.asarray(split['training'])
    scale = np.maximum(np.sqrt(np.mean(x[tr].transpose(0,2,1).reshape(-1,x.shape[1])**2,axis=0)), np.finfo(float).eps)
    x /= scale[None,:,None]
    rng = np.random.default_rng(c['seed']+split['fold'])
    blocks = {}
    for name,copies in [('training',legacy['window']['num_noise']+1),('validation',1),('test',1)]:
        blocks[name] = make_windows(x,y,split[name],sf,tmin,legacy['window'],copies,rng)
    return blocks,scale


def seeded_loader(block, settings, prefix, seed):
    import torch
    from inhouse_bci_raspy.data.loaders import make_loader
    loader = make_loader(block, settings, prefix)
    # Dedicated streams prevent different selection-set lengths consuming the training RNG.
    loader.generator = torch.Generator().manual_seed(seed)
    if hasattr(loader.sampler, 'generator'):
        loader.sampler.generator = torch.Generator().manual_seed(seed+1)
    return loader


def training_source_hash():
    files = [Path(__file__), PROJECT/'scripts/overnight.py']
    files += sorted((PROJECT/'src/inhouse_bci_raspy').rglob('*.py'))
    return digest({str(p.relative_to(PROJECT)): file_hash(p) for p in files})


def run_fold(c, fold):
    import numpy as np
    import torch
    from inhouse_bci_raspy.models.eegnet import EEGNet
    from inhouse_bci_raspy.training.engine import train
    from inhouse_bci_raspy.runtime import setup_training
    from inhouse_bci_raspy.evaluation import predict_probabilities
    from inhouse_bci_raspy.reporting import save_predictions
    from inhouse_bci_raspy.training.checkpoints import save_checkpoint
    root = Path(c['output_root'])
    manifest, legacy = read_json(root/'manifest.json'), read_json(root/'legacy_config.json')
    split = next(s for s in manifest['splits'] if s['fold'] == fold)
    if file_hash(root/'clean_trials.h5') != manifest['cache_sha256']:
        raise ValueError('Frozen cache checksum mismatch')
    directory = root/f'fold_{fold}'
    directory.mkdir(exist_ok=True)
    blocks, scale = make_blocks(c, split, legacy)
    paired = dict(split=split, rms=scale.tolist(), training_X_sha256=array_hash(blocks['training']['X']),
                  training_y_sha256=array_hash(blocks['training']['y']),
                  training_window_count=len(blocks['training']['y']), seed=c['seed']+fold,
                  source_hash=training_source_hash(), cache_sha256=manifest['cache_sha256'],
                  config=legacy, packages={p:importlib.metadata.version(p) for p in ('torch','numpy','scipy','mne','h5py')})
    immutable(directory/'paired_inputs.json', paired)
    for arm in ARMS:
        target = directory/arm
        target.mkdir(exist_ok=True)
        if (target/'status.json').exists() and read_json(target/'status.json')['state']=='complete':
            print(f'Fold {fold} {arm}: already complete', flush=True)
            continue
        start = time.monotonic()
        save_json(target/'status.json', dict(state='running'))
        try:
            device = setup_training(c['seed']+fold, 'cuda:0')
            torch.set_num_threads(c['threads_per_process'])
            model = EEGNet(legacy['model'], output_dim=4, n_electrodes=16).to(device)
            initial = digest({k:array_hash(v.detach().cpu().numpy()) for k,v in model.state_dict().items()})
            settings = legacy['training']
            training = seeded_loader(blocks['training'], settings, 'train', c['seed']+fold+1000)
            selection = seeded_loader(blocks['validation' if arm=='validation_selected' else 'test'],
                                      settings, 'val', c['seed']+fold+2000)
            class Writer:
                def add_scalar(self, tag, value, step):
                    if tag == 'Validation Loss':
                        print(f'Fold {fold} {arm} epoch={step+1} selection_loss={float(value):.6g}', flush=True)
            best, stats, best_index, train_losses, val_losses, _, _ = train(
                model, training, selection, settings, Writer(), device, verbosity=0)
            probabilities = predict_probabilities(best, blocks['test'], device, settings['val_batch_size'])
            save_predictions(target/'test_predictions.csv', blocks['test'], probabilities)
            np.savez_compressed(target/'test_predictions.npz', probabilities=probabilities,
                                y=blocks['test']['y'], trial_index=blocks['test']['trial_index'])
            save_checkpoint(target/'best.pt', best, legacy, manifest['metadata'],
                            dict(normalization_scales=scale.tolist()),fold,best_index)
            save_json(target/'history.json', dict(stats=stats, train_loss=train_losses, selection_loss=val_losses))
            save_json(target/'training.json', dict(initial_state_hash=initial, best_epoch=best_index+1,
                      epochs=len(train_losses), first_training_loss=train_losses[0], seconds=time.monotonic()-start,
                      test_used_for_selection=arm=='test_selected', selection_source='outer_test' if arm=='test_selected' else 'inner_validation'))
            if array_hash(blocks['training']['X']) != paired['training_X_sha256']:
                raise RuntimeError('Training arrays mutated')
            save_json(target/'status.json', dict(state='complete'))
        except BaseException as exc:
            save_json(target/'status.json', dict(state='failed', error=str(exc)))
            raise
    a,b = [read_json(directory/arm/'training.json') for arm in ARMS]
    if a['initial_state_hash'] != b['initial_state_hash'] or not np.isclose(a['first_training_loss'],b['first_training_loss'],rtol=0,atol=1e-7):
        raise RuntimeError('Paired initial states/first training epoch differ')
    immutable(directory/'pair_verified.json', dict(identical_initialization=True, identical_training_arrays=True,
              equal_first_epoch_training_loss=True, first_training_loss=a['first_training_loss']))


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


def report(c):
    import numpy as np
    import csv
    from sklearn.metrics import accuracy_score
    root=Path(c['output_root']); rows=[]; pooled={a:[] for a in ARMS}; missing=[]
    for fold in c['folds']:
        if not (root/f'fold_{fold}'/'pair_verified.json').exists():
            missing.append(fold); continue
        row=dict(fold=fold)
        for arm in ARMS:
            d=root/f'fold_{fold}'/arm
            with np.load(d/'test_predictions.npz') as f:
                p,y,ids=f['probabilities'],f['y'],f['trial_index']
            actual=[]; predicted=[]
            for i in np.unique(ids):
                mask=ids==i; actual.append(int(y[mask][0])); predicted.append(int(p[mask].mean(0).argmax()))
            training=read_json(d/'training.json')
            row[arm+'_trial_accuracy']=accuracy_score(actual,predicted)
            row[arm+'_window_accuracy']=accuracy_score(y,p.argmax(-1))
            row[arm+'_best_epoch']=training['best_epoch']
            row[arm+'_epochs']=training['epochs']
            pooled[arm].extend(zip(actual,predicted))
        row['test_selected_minus_validation_selected']=row['test_selected_trial_accuracy']-row['validation_selected_trial_accuracy']
        rows.append(row)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'); out=root/'reports'/stamp; out.mkdir(parents=True,exist_ok=False)
    result=dict(completed_pairs=len(rows),expected_pairs=len(c['folds']),missing_folds=missing,
                provisional=bool(missing) or set(c['folds'])!=set(range(5)),rows=rows,
                preprocessing='Frozen historical session-level MNE-BIDS cleaning + CSD; shared training-only RMS.',
                interpretation='Selection-source diagnostic with matched training sets, not an unbiased estimate of population optimism; not an exact reproduction of the old 80%-training baseline.')
    if rows:
        result['pooled_trial_accuracy']={arm:accuracy_score([x[0] for x in v],[x[1] for x in v]) for arm,v in pooled.items()}
        result['paired_difference']=result['pooled_trial_accuracy']['test_selected']-result['pooled_trial_accuracy']['validation_selected']
        with (out/'paired_folds.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots()
        for row in rows:
            ax.plot([0,1],[100*row[a+'_trial_accuracy'] for a in ARMS],marker='o',label=f'Fold {row["fold"]}')
        ax.set_xticks([0,1],['Internal validation selected','Test selected']); ax.set_ylabel('Outer-test trial accuracy (%)'); ax.legend(); fig.tight_layout()
        fig.savefig(out/'paired_accuracy.png',dpi=180); fig.savefig(out/'paired_accuracy.pdf'); plt.close(fig)
    save_json(out/'summary.json',result)
    lines=['# Selection-source paired diagnostic','',result['preprocessing'],'',result['interpretation'],'',f'Completed pairs: {len(rows)}/{len(c["folds"])}', '', '| Fold | Internal-validation selected | Test selected | Difference (pp) |','|---|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["fold"]} | {r["validation_selected_trial_accuracy"]:.2%} | {r["test_selected_trial_accuracy"]:.2%} | {r["test_selected_minus_validation_selected"]*100:+.2f} |')
    if rows:
        lines.extend(['',f'Pooled: {result["pooled_trial_accuracy"]}; difference={result["paired_difference"]*100:+.2f} pp.'])
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2)); print('REPORT',out)
    if not rows: raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check','prepare','run','fold','report'])
    parser.add_argument('--config',type=Path,default=Path('config/selection_bias.json'))
    parser.add_argument('--fold',type=int)
    args=parser.parse_args(); c=config(args.config)
    if args.stage=='check':
        for key in ('legacy_cache','legacy_folds','legacy_config','legacy_bids_root'):
            if not Path(c[key]).exists(): raise FileNotFoundError(c[key])
        print(json.dumps(c,indent=2)); return
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Use run_selection_bias.sh server for computation')
    if args.stage=='prepare': prepare(c); return
    if read_json(Path(c['output_root'])/'config.json') != c: raise ValueError('Frozen configuration changed')
    if args.stage=='run': manage(args,c)
    elif args.stage=='report': report(c)
    else:
        if args.fold not in c['folds']: parser.error('Fold not in configured plan')
        def interrupt(sig,frame): raise KeyboardInterrupt('Allocation ended')
        signal.signal(signal.SIGTERM,interrupt); signal.signal(signal.SIGINT,interrupt)
        run_fold(c,args.fold)


if __name__=='__main__':
    main()
