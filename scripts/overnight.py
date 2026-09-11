"""Focused two-dataset plan and bounded concurrent workers inside one GPU allocation."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bci_raspy_experiments.common import PROJECT, digest, profile, read_json, save_json
from bci_raspy_experiments.presets import BASE, resolve_task


def immutable(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Frozen configuration changed: {path}; choose a new output_root')
    else:
        save_json(path, value)


def variant(name, settings):
    cfg = deepcopy(BASE)
    if name.startswith('standard'):
        cfg['model']['family'] = 'standard'
    if name.endswith('_mil'):
        cfg['objective'] = 'mil'
    if name.endswith('_whole'):
        cfg.update(objective='whole', sampling='whole')
    if name == 'standard_deep':
        cfg['model']['extra'] = 2
    if name == 'no_augmentation':
        cfg['augmentation']['copies'] = 0
    if name in ('no_ica', 'no_csd'):
        cfg['preprocessing'] = name
    allowed = {'raspy_window', 'standard_window', 'standard_deep', 'raspy_mil',
               'standard_mil', 'raspy_whole', 'standard_whole', 'no_augmentation', 'no_ica', 'no_csd'}
    if name not in allowed:
        raise ValueError(f'Unknown variant: {name}')
    cfg['train']['max_epochs'] = settings['max_epochs']
    cfg['seed'] = settings['seed']
    return cfg


def plan(settings):
    root = Path(settings['output_root'])
    if not root.is_absolute() or not root.is_relative_to('/public/home/hugf2022'):
        raise ValueError('output_root must be absolute and under /public/home/hugf2022')
    for key in ('parallel_per_gpu', 'threads_per_process', 'max_epochs'):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if settings['training_hours'] <= 0:
        raise ValueError('training_hours must be positive')
    base = profile('server', PROJECT / settings['base_profile'])
    source = read_json(settings['source_manifest'])
    immutable(root / 'config.json', settings)
    for dataset in ('inhouse', 'bci2a'):
        catalogue = read_json(Path(base['cache']) / dataset / 'catalog.json')
        splits = {k: v for k, v in source['splits'].items() if k.startswith(dataset + '/')
                  and (v['kind'] == 'trial' if dataset == 'inhouse' else v['id'].endswith('-time-50'))}
        expected = 5 if dataset == 'inhouse' else 9
        if len(splits) != expected:
            raise ValueError(f'Expected {expected} {dataset} splits, found {len(splits)}')
        tasks = []
        for split in splits:
            for name in settings[dataset]:
                cfg = variant(name, settings)
                cfg.update(dataset=dataset, catalog_hash=catalogue['hash'], split=split)
                tasks.append(dict(id=dataset + '-' + digest(cfg)[:20], label=name,
                                  config=cfg, suites=['overnight']))
        if len({t['id'] for t in tasks}) != len(tasks):
            raise ValueError('Duplicate variants')
        destination = root / dataset
        p = dict(base, output=str(destination), device='cuda:0',
                 threads=settings['threads_per_process'], workers=0)
        immutable(destination / 'profile.json', p)
        immutable(destination / 'manifest.json', dict(version=2, suite='overnight', splits=splits,
                                                     tasks=tasks, count=len(tasks)))
        print(f'{dataset}: {len(splits)} splits x {len(settings[dataset])} variants = {len(tasks)} tasks; '
              f'{settings["parallel_per_gpu"]} processes on one GPU; output={destination}', flush=True)


def complete(directory):
    status = directory / 'status.json'
    return status.exists() and read_json(status).get('state') == 'complete'


def needs_explanation(task):
    return task['label'] in ('raspy_window', 'standard_window')


def worker(task, manifest, p):
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Allocation stopping: signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    from bci_raspy_experiments.runner import run_task
    from bci_raspy_experiments.engine import seed_all
    catalog = read_json(Path(p['cache']) / task['config']['dataset'] / 'catalog.json')
    resolved = resolve_task(task, manifest)
    run_task(resolved, catalog, p, resume=True)
    directory = Path(p['output']) / 'runs' / task['id']
    if needs_explanation(task) and not (directory / 'explanation/summary.json').exists():
        from bci_raspy_experiments.explain import explain
        seed_all(resolved['config']['seed'], p['threads'])
        explain(directory, catalog, resolved['config'], p)


def manage(args, settings, manifest, p):
    parallel = settings['parallel_per_gpu']
    if int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) < parallel * p['threads']:
        raise ValueError('Insufficient allocated CPUs for configured concurrency')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('This queue requires exactly one Slurm-visible CUDA GPU')
    print(f'GPU={torch.cuda.get_device_name(0)}, concurrent processes={parallel}', flush=True)
    root = Path(p['output'])
    lock = root / 'queue.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w') as stream:
        stream.write(f'{os.uname().nodename} pid={os.getpid()} job={os.environ["SLURM_JOB_ID"]}')
    pending = iter(manifest['tasks'][:args.limit_tasks] if args.limit_tasks else manifest['tasks'])
    active, failed = {}, []
    stopping = False
    exhausted = False
    signalled = False
    deadline = time.monotonic() + settings['training_hours'] * 3600
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stop)
    logs = root / 'worker_logs'
    logs.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    state_file = root / f'queue-{os.environ["SLURM_JOB_ID"]}-{stamp}.json'
    try:
        while active or not exhausted:
            stopping = stopping or time.monotonic() >= deadline
            if stopping and not signalled:
                signalled = True
                exhausted = True
                for process, _, _ in active.values():
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
            while not exhausted and len(active) < parallel:
                task = next(pending, None)
                if task is None:
                    exhausted = True
                    break
                log_path = logs / f'{task["id"]}-{stamp}.log'
                stream = log_path.open('x')
                env = dict(os.environ)
                for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                    env[key] = str(p['threads'])
                env['MPLCONFIGDIR'] = str(root / 'matplotlib' / task['id'])
                command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker',
                           '--config', str(args.config.resolve()), '--dataset', args.dataset, '--run-id', task['id']]
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
                active[task['id']] = (process, stream, time.monotonic())
                print(f'Start {task["id"]} {task["label"]} pid={process.pid} log={log_path}', flush=True)
            for run_id, (process, stream, started) in list(active.items()):
                code = process.poll()
                if code is not None:
                    stream.close()
                    del active[run_id]
                    if code:
                        failed.append(run_id)
                    print(f'End {run_id} exit={code} seconds={time.monotonic()-started:.1f}', flush=True)
            save_json(state_file, dict(active=list(active), failed=failed, stopped=stopping,
                                      expected=manifest['count'], updated=time.time()))
            if active:
                time.sleep(1)
    finally:
        for process, stream, _ in active.values():
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            stream.close()
        lock.unlink(missing_ok=True)
    if failed or stopping:
        raise SystemExit(1)


def report(manifest, p):
    from bci_raspy_experiments.reporting import report as existing_report
    from bci_raspy_experiments.runner import reevaluate
    root = Path(p['output'])
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = root / 'reports' / stamp
    destination.mkdir(parents=True, exist_ok=False)
    completed, missing, capped, explanations_missing = [], [], [], []
    catalog = read_json(Path(p['cache']) / manifest['tasks'][0]['config']['dataset'] / 'catalog.json')
    for task in manifest['tasks']:
        directory = root / 'runs' / task['id']
        if not complete(directory):
            missing.append(task['id'])
            continue
        completed.append(task['id'])
        if not (directory / 'recomputed_metrics.json').exists():
            reevaluate(directory, catalog)
        history = read_json(directory / 'inner/history.json')
        if len(history) == task['config']['train']['max_epochs']:
            capped.append(task['id'])
        if needs_explanation(task) and not (directory / 'explanation/summary.json').exists():
            explanations_missing.append(task['id'])
    print(existing_report(root, destination))
    save_json(destination / 'completion.json', dict(expected=manifest['count'], completed=len(completed),
              missing=missing, explanations_missing=explanations_missing, hit_epoch_limit=capped,
              provisional=bool(missing or explanations_missing), labels={t['id']: t['label'] for t in manifest['tasks']},
              note='One seed; strict internal selection, max 60 epochs by default. Inhouse folds are not independent subjects.'))
    if not completed:
        raise SystemExit('No completed tasks; completion report saved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'run', 'worker', 'report'])
    parser.add_argument('--config', type=Path, default=Path('config/overnight.json'))
    parser.add_argument('--dataset', choices=['inhouse', 'bci2a'])
    parser.add_argument('--run-id')
    parser.add_argument('--limit-tasks', type=int, help='Bounded validation only; reports retain the full expected count')
    args = parser.parse_args()
    if args.limit_tasks is not None and args.limit_tasks < 1:
        parser.error('--limit-tasks must be positive')
    settings = read_json(args.config)
    if args.stage == 'plan':
        plan(settings)
        return
    if not os.environ.get('SLURM_JOB_ID') or not args.dataset:
        parser.error('Computation requires a Slurm allocation and --dataset')
    root = Path(settings['output_root']) / args.dataset
    if read_json(root.parent / 'config.json') != settings:
        raise ValueError('Config differs from frozen plan')
    manifest, p = read_json(root / 'manifest.json'), read_json(root / 'profile.json')
    if args.stage == 'worker':
        tasks = [t for t in manifest['tasks'] if t['id'] == args.run_id]
        if len(tasks) != 1:
            parser.error('Unknown --run-id')
        worker(tasks[0], manifest, p)
    elif args.stage == 'run':
        manage(args, settings, manifest, p)
    else:
        report(manifest, p)


if __name__ == '__main__':
    main()
