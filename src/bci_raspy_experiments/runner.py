"""Stage orchestration: strict test data is materialized only after final fit."""
from pathlib import Path
import os
import pickle
import platform
import re
import time
import importlib.metadata
import torch
from .common import digest, file_hash, read_json, save_json
from .engine import fit, load_torch, save_torch
from .evaluation import predict, summarize, aggregation, save_predictions, load_predictions
from .models import build_model
from .preprocessing import Pipeline
from .protocols import inner_split, validate
from .windows import Bags


def code_hash():
    package = Path(__file__).resolve().parent
    return digest({p.relative_to(package).as_posix(): file_hash(p)
                   for p in sorted(package.rglob('*.py')) if '__pycache__' not in p.parts})


def versions():
    return dict(python=platform.python_version(), platform=platform.platform(),
                packages={p: importlib.metadata.version(p) for p in ('torch', 'numpy', 'scipy', 'mne', 'h5py', 'scikit-learn')})


def phase_data(catalog, cfg, profile, directory, train_ids, other_ids=None):
    from .cache import writer_lock, materialize_cached
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = digest([catalog['hash'], train_ids, cfg['preprocessing'], code_hash()])
    shared = Path(profile['cache']) / 'transforms' / cfg['dataset'] / fingerprint
    shared.mkdir(parents=True, exist_ok=True)
    pipeline_path = shared / 'pipeline.pkl'
    with writer_lock(shared / 'pipeline.lock'):
        if pipeline_path.exists():
            with pipeline_path.open('rb') as f:
                pipe = pickle.load(f)
            if pipe.fit_ids != train_ids or pipe.name != cfg['preprocessing']:
                raise ValueError('Preprocessing cache identity mismatch')
            pipe.raw_path = Path(profile['cache']) / cfg['dataset'] / 'raw.h5'
        else:
            pipe = Pipeline(catalog, profile['cache'], cfg['preprocessing']).fit(train_ids)
            pipe.save(pipeline_path)
    # Rebind metadata for evaluation labels; fitted numerical parameters stay frozen.
    pipe.rows = {r['id']: r for r in catalog['trials']}
    train_path, kept = materialize_cached(pipe, train_ids, shared)
    training = Bags(train_path, kept, catalog['trials'], cfg, training=True)
    other, other_path = None, None
    if other_ids is not None:
        other_path, kept_other = materialize_cached(pipe, other_ids, shared)
        other = Bags(other_path, kept_other, catalog['trials'], cfg)
    save_json(directory / 'pipeline.json', pipe.metadata())
    save_json(directory / 'data.json', dict(pipeline_path=str(pipeline_path), shared=str(shared),
              train_path=str(train_path), selection_path=str(other_path) if other_path else None))
    return pipe, training, other


_VERIFIED_RAW = {}


def verify_raw(catalog, profile, dataset):
    if 'raw_hash' not in catalog:
        return  # synthetic test fixture
    path = Path(profile['cache']) / dataset / 'raw.h5'
    stat = path.stat()
    stamp = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if stamp not in _VERIFIED_RAW:
        _VERIFIED_RAW[stamp] = file_hash(path)
    if _VERIFIED_RAW[stamp] != catalog['raw_hash']:
        raise ValueError('Raw trial cache checksum mismatch')


def run_task(task, catalog, profile, resume=False):
    cfg = task['config']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', task['id']):
        raise ValueError('Invalid run ID')
    validate(catalog['trials'], cfg['split']['train'], cfg['split']['test'], cfg['split']['kind'] == 'subject')
    verify_raw(catalog, profile, cfg['dataset'])
    if cfg['catalog_hash'] != catalog['hash']:
        raise ValueError('Manifest/cache mismatch')
    destination = Path(profile['output']) / ('smoke' if cfg.get('smoke') else 'runs') / task['id']
    destination.mkdir(parents=True, exist_ok=True)
    frozen_path = destination / 'task.json'
    provenance = dict(task=task, source_code_hash=code_hash(), environment=versions(),
                      runtime={k: profile[k] for k in ('name', 'device', 'workers', 'threads')})
    if frozen_path.exists():
        previous = read_json(frozen_path)
        if previous != provenance:
            raise ValueError('Task/config/code/environment changed; use a new output root')
        if not resume:
            raise FileExistsError(f'Existing task: {destination}; use --resume')
        status = read_json(destination / 'status.json') if (destination / 'status.json').exists() else {}
        if status.get('state') == 'complete':
            return read_json(destination / 'metrics.json')
    else:
        save_json(frozen_path, provenance)
    # Exclusive task ownership also prevents two shards from writing the same run.
    lock = destination / '.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f'Task is locked: {lock}; verify the recorded process before removing a stale lock')
    with os.fdopen(fd, 'w') as f:
        f.write(f'{platform.node()} pid={os.getpid()}')
    started = time.monotonic()
    if profile['device'].startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(torch.device(profile['device']))
    save_json(destination / 'status.json', dict(state='running', started=time.time()))
    try:
        split = cfg['split']
        train_ids, test_ids = list(split['train']), list(split['test'])
        if cfg['selection'] == 'strict':
            inner_train, inner_val = inner_split(catalog['trials'], split)
            save_json(destination / 'split.json', dict(**split, inner_train=inner_train, inner_validation=inner_val))
            _, train, val = phase_data(catalog, cfg, profile, destination / 'inner', inner_train, inner_val)
            _, selected = fit(train, val, cfg, profile, destination / 'inner', resume=resume)
            pipe, train, _ = phase_data(catalog, cfg, profile, destination / 'final', train_ids)
            model, fitted = fit(train, None, cfg, profile, destination / 'final', resume=resume, fixed_schedule=selected['schedule'])
        else:
            save_json(destination / 'split.json', dict(**split, selection_warning='TEST_USED_FOR_MODEL_SELECTION'))
            pipe, train, val = phase_data(catalog, cfg, profile, destination / 'final', train_ids, test_ids)
            model, fitted = fit(train, val, cfg, profile, destination / 'final', resume=resume)
            selected = fitted
        save_json(destination / 'training.json', dict(selection=cfg['selection'], selected_schedule=selected['schedule'],
                                                    fitted=fitted, total_training_seconds=time.monotonic()-started))
        sample = train[0]['x']
        save_torch(destination / 'model.pt', dict(model=model.state_dict(), spec=cfg['model'],
                   channels=sample.shape[1], samples=sample.shape[2], code_hash=provenance['source_code_hash']))
        # The strict branch first accesses held-out signals here.
        from .cache import materialize_cached
        shared = Path(read_json(destination / 'final/data.json')['shared'])
        test_path, retained_test = materialize_cached(pipe, test_ids, shared)
        save_json(destination / 'test_data.json', dict(path=str(test_path), retained_ids=retained_test))
        test = Bags(test_path, retained_test, catalog['trials'], cfg)
        clean_train = Bags(train.path, train.ids, catalog['trials'], cfg)
        results = {}
        for name, data in [('train', clean_train), ('test', test)]:
            prediction = predict(model, data, cfg, profile, profile['device'])
            save_predictions(destination / (name + '_predictions'), prediction)
            results[name] = summarize(prediction, catalog['trials'])
            if name == 'test':
                save_json(destination / 'aggregation.json', aggregation(prediction, 10 if cfg.get('smoke') else 1000))
        results.update(selection=cfg['selection'], test_used_for_selection=cfg['selection'] != 'strict',
                       seconds=time.monotonic()-started, parameters=fitted['shape']['parameters'],
                       peak_cuda_bytes=torch.cuda.max_memory_allocated(torch.device(profile['device'])) if profile['device'].startswith('cuda') else 0,
                       retained_train=len(train.ids), requested_train=len(train_ids), retained_test=len(retained_test), requested_test=len(test_ids),
                       smoke_only=bool(cfg.get('smoke')))
        save_json(destination / 'metrics.json', results)
        save_json(destination / 'status.json', dict(state='complete', seconds=time.monotonic()-started))
        return results
    except BaseException as exc:
        save_json(destination / 'status.json', dict(state='failed', error=f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        lock.unlink(missing_ok=True)


def reevaluate(directory, catalog):
    """Rebuild metrics exclusively from frozen predictions, without training."""
    directory = Path(directory)
    result = {name: summarize(load_predictions(directory / f'{name}_predictions.npz'), catalog['trials'])
              for name in ('train', 'test')}
    save_json(directory / 'recomputed_metrics.json', result)
    return result


def restore_model(directory, device):
    checkpoint = load_torch(Path(directory) / 'model.pt', device)
    model = build_model(checkpoint['spec'], checkpoint['channels'], checkpoint['samples']).to(device)
    model.load_state_dict(checkpoint['model'])
    return model.eval()
