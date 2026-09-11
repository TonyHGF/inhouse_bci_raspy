"""BME stage adapter; scientific configurations remain in presets.py."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bci_raspy_experiments.common import profile, read_json, save_json
from bci_raspy_experiments.__main__ import main as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'verify', 'prepare', 'smoke', 'run', 'evaluate', 'explain', 'report'])
    parser.add_argument('--profile-file', type=Path, required=True)
    parser.add_argument('--dataset', choices=['all', 'inhouse', 'bci2a', 'physionet'], default='all')
    parser.add_argument('--suite', choices=['all', 'protocol', 'single', 'interaction', 'extension'], default='all')
    parser.add_argument('--run-id')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    p = profile('server', args.profile_file)
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        parser.error('Require 0 <= shard-index < shards')
    datasets = ['inhouse', 'bci2a', 'physionet'] if args.dataset == 'all' else [args.dataset]
    base = ['--profile', 'server', '--profile-file', str(args.profile_file)]
    output = Path(p['output'])
    manifest = output / 'manifests' / 'all-all.json'
    for key in ('inhouse', 'bci2a', 'physionet'):
        if not Path(p[key]).is_dir():
            raise FileNotFoundError(f'{key}: {p[key]}')
    for key in ('cache', 'output'):
        if not Path(p[key]).is_relative_to('/public/home/hugf2022'):
            raise ValueError(f'{key} must be under /public/home/hugf2022')
    if args.stage == 'check':
        packages = ['torch', 'numpy', 'scipy', 'mne', 'h5py', 'scikit-learn', 'python-picard', 'matplotlib']
        print(json.dumps(dict(profile=p, python=sys.executable,
                              packages={k: importlib.metadata.version(k) for k in packages}), indent=2))
        return
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use run_experiments.sh server for computation')
    if args.stage in ('run', 'smoke', 'explain'):
        import torch
        if not p['device'].startswith('cuda') or not torch.cuda.is_available():
            raise RuntimeError('GPU stages require a CUDA profile and allocation')
        torch.set_num_threads(p['threads'])
    if args.stage == 'verify':
        import torch
        torch.set_num_threads(p['threads'])
        temp = Path(p['cache']).parent / 'verification-tmp'
        temp.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(temp)
        suite = unittest.defaultTestLoader.discover('src/bci_raspy_experiments/tests')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        return
    if args.stage == 'prepare':
        # One writer; complete all catalogues before producing the shared plan.
        experiment(['prepare', *base, '--dataset', 'all'])
        experiment(['manifest', *base, '--dataset', 'all', '--suite', 'all'])
        return
    selected = ['--dataset', args.dataset, '--suite', args.suite,
                '--manifest-file', str(manifest)]
    if args.run_id:
        selected += ['--run-id', args.run_id]
    if args.stage == 'smoke':
        for dataset in datasets:
            for objective in ('window', 'mil', 'whole'):
                command = ['run', *base, '--manifest-file', str(manifest), '--dataset', dataset,
                           '--smoke', '--smoke-preprocessing', 'full', '--smoke-objective', objective]
                experiment(command + (['--resume'] if args.resume else []))
        # Verify frozen GPU inference and actual PNG/PDF generation on tiny data.
        from bci_raspy_experiments.explain import explain
        for path in sorted((output / 'smoke').glob('*/task.json')):
            task = read_json(path)['task']
            if task['config']['dataset'] not in datasets:
                continue
            if (path.parent / 'explanation/summary.json').exists() and args.resume:
                continue
            catalog = read_json(Path(p['cache']) / task['config']['dataset'] / 'catalog.json')
            explain(path.parent, catalog, task['config'], p)
        return
    if args.stage == 'run':
        experiment(['run', *base, *selected, '--shards', str(args.shards),
                    '--shard-index', str(args.shard_index)] + (['--resume'] if args.resume else []))
        return
    from bci_raspy_experiments.presets import BASE, CORE, resolve_task
    plan = read_json(manifest)
    tasks = [t for t in plan['tasks'] if t['config']['dataset'] in datasets
             and (args.suite == 'all' or args.suite in t['suites'])
             and (not args.run_id or args.run_id == t['id'])]
    if not tasks:
        raise ValueError('No matching tasks')
    completed = [t for t in tasks if (output / 'runs' / t['id'] / 'status.json').exists()
                 and read_json(output / 'runs' / t['id'] / 'status.json')['state'] == 'complete']
    print(f'Completed {len(completed)}/{len(tasks)} selected formal tasks', flush=True)
    if not completed:
        raise RuntimeError('No complete formal runs; smoke does not count as formal evaluation')
    if args.stage == 'report':
        from bci_raspy_experiments.reporting import report
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        destination = output / 'reports' / (stamp + '-' + os.environ['SLURM_JOB_ID'])
        destination.mkdir(parents=True, exist_ok=False)
        print(report(output, destination))
        save_json(destination / 'manifest_completion.json', dict(
            manifest=str(manifest), expected=len(tasks), completed=len(completed),
            provisional=len(completed) != len(tasks), dataset=args.dataset, suite=args.suite,
            note='Report tables include all completed formal runs in this output root.',
            missing=[t['id'] for t in tasks if t not in completed]))
        return
    from bci_raspy_experiments.runner import reevaluate
    for index, task in enumerate(tasks):
        if index % args.shards != args.shard_index or task not in completed:
            continue
        directory = output / 'runs' / task['id']
        cfg = resolve_task(task, plan)['config']
        catalog = read_json(Path(p['cache']) / cfg['dataset'] / 'catalog.json')
        if args.stage == 'evaluate':
            if (directory / 'recomputed_metrics.json').exists():
                if args.resume:
                    continue
                raise FileExistsError('Recomputed metrics exist; use --resume to skip')
            reevaluate(directory, catalog)
        else:
            base_model = dict(BASE['model'], family=cfg['model']['family'], extra=cfg['model']['extra'])
            core = ((cfg['model']['family'], cfg['model']['extra']) in CORE and cfg['model'] == base_model
                    and cfg['objective'] == 'window' and cfg['sampling'] == 'dense' and cfg['loss'] == 'mse'
                    and cfg['preprocessing'] == 'full' and cfg['augmentation'] == BASE['augmentation']
                    and cfg['selection'] == 'strict')
            if not args.run_id and not core:
                continue
            if (directory / 'explanation/summary.json').exists() and args.resume:
                continue
            from bci_raspy_experiments.engine import seed_all
            seed_all(cfg['seed'], p['threads'])
            from bci_raspy_experiments.explain import explain
            explain(directory, catalog, cfg, p)


if __name__ == '__main__':
    main()
