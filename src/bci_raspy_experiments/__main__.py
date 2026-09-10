"""Portable entrypoint: python -m bci_raspy_experiments --help."""
import argparse
from copy import deepcopy
from pathlib import Path
from .common import digest, profile, read_json, save_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inspect', 'prepare', 'manifest', 'run', 'evaluate', 'report'])
    parser.add_argument('--profile', choices=['local', 'server'], default='local')
    parser.add_argument('--profile-file', type=Path)
    parser.add_argument('--dataset', choices=['all', 'inhouse', 'bci2a', 'physionet'], default='all')
    parser.add_argument('--suite', choices=['all', 'protocol', 'single', 'interaction', 'extension'], default='all')
    parser.add_argument('--run-id')
    parser.add_argument('--manifest-file', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--smoke', action='store_true', help='One tiny, two-epoch task; separate output namespace')
    parser.add_argument('--smoke-objective', choices=['window', 'mil', 'whole'])
    parser.add_argument('--smoke-preprocessing', choices=['filter_rms', 'full'], default='filter_rms')
    parser.add_argument('--explain', action='store_true', help='evaluate: frozen channel perturbation and kernel topomaps')
    args = parser.parse_args(argv)
    p = profile(args.profile, args.profile_file)
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        parser.error('Require 0 <= shard-index < shards')
    datasets = ['inhouse', 'bci2a', 'physionet'] if args.dataset == 'all' else [args.dataset]
    if args.stage == 'inspect':
        from .datasets import inventory
        import json
        for dataset in datasets:
            print(json.dumps(inventory(p[dataset], dataset), indent=2))
        return
    if args.stage == 'prepare':
        from .datasets import prepare
        for dataset in datasets:
            prepare(p[dataset], dataset, p['cache'])
        return
    if args.stage == 'report':
        from .reporting import report
        print(report(p['output']))
        return
    catalogues = {dataset: read_json(Path(p['cache']) / dataset / 'catalog.json') for dataset in datasets}
    manifest_path = args.manifest_file or Path(p['output']) / 'manifests' / (args.dataset + '-' + args.suite + '.json')
    if args.stage == 'manifest':
        from .presets import manifest
        values = manifest(catalogues, args.suite)
        if manifest_path.exists() and read_json(manifest_path) != values:
            raise FileExistsError('Different manifest already exists; choose --manifest-file')
        save_json(manifest_path, values)
        save_json(manifest_path.with_name(manifest_path.stem + '-summary.json'),
                  dict(tasks=values['count'], suites={s: sum(s in t['suites'] for t in values['tasks'])
                                                    for s in ('protocol', 'single', 'interaction', 'extension')},
                       note='Deduplicated configurations. Strict tasks run internal selection plus full refit. No training was launched.'))
        import csv
        with manifest_path.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
            fields = ['id', 'dataset', 'split', 'suites', 'seed', 'model', 'objective', 'sampling', 'loss', 'preprocessing', 'augmentation', 'selection']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            import json
            for task in values['tasks']:
                c = task['config']
                row = {k: c[k] for k in fields if k in c}
                row.update(id=task['id'], suites=','.join(task['suites']), model=json.dumps(c['model'], sort_keys=True), augmentation=json.dumps(c['augmentation'], sort_keys=True))
                writer.writerow(row)
        print(f"Manifest: {values['count']} tasks -> {manifest_path}")
        return
    from .presets import resolve_task
    plan = read_json(manifest_path)
    tasks = plan['tasks']
    tasks = [t for t in tasks if t['config']['dataset'] in datasets and (args.suite == 'all' or args.suite in t['suites'])]
    if args.run_id:
        tasks = [t for t in tasks if t['id'] == args.run_id]
    if not tasks:
        raise ValueError('No matching run ID/dataset/suite')
    if args.stage == 'evaluate':
        from .runner import reevaluate
        for unresolved in tasks:
            task = resolve_task(unresolved, plan)
            directory = Path(p['output']) / 'runs' / task['id']
            if not (directory / 'status.json').exists() or read_json(directory / 'status.json').get('state') != 'complete':
                continue
            reevaluate(directory, catalogues[task['config']['dataset']])
            if args.explain:
                from .presets import BASE, CORE
                c = task['config']
                base_model = dict(BASE['model'], family=c['model']['family'], extra=c['model']['extra'])
                core = ((c['model']['family'], c['model']['extra']) in CORE and c['model'] == base_model
                        and c['objective'] == 'window' and c['sampling'] == 'dense' and c['loss'] == 'mse'
                        and c['preprocessing'] == 'full' and c['augmentation'] == BASE['augmentation']
                        and c['selection'] == 'strict')
                if not args.run_id and not core:
                    continue
                from .explain import explain
                explain(directory, catalogues[task['config']['dataset']], task['config'], p)
        return
    if p['name'] == 'local' and not args.smoke:
        raise ValueError('Local profile is verification-only: use --smoke. Formal experiments require --profile server.')
    if args.smoke:
        task = resolve_task(tasks[0], plan)
        cfg = task['config']
        rows = {r['id']: r for r in catalogues[cfg['dataset']]['trials']}
        for part, count in [('train', 8), ('test', 2)]:
            ids = cfg['split'][part]
            cfg['split'][part] = [i for label in range(4) for i in [i for i in ids if rows[i]['label'] == label and not rows[i]['exclusion']][:count]]
        cfg['split']['kind'] = 'trial'
        cfg['split']['hash'] = digest([cfg['split']['train'], cfg['split']['test']])
        cfg['train']['max_epochs'] = 2
        cfg['preprocessing'] = args.smoke_preprocessing
        if args.smoke_objective:
            cfg['objective'] = args.smoke_objective
            cfg['sampling'] = 'whole' if args.smoke_objective == 'whole' else 'dense'
        cfg['smoke'] = True
        task['id'] = 'smoke-' + cfg['dataset'] + '-' + digest(cfg)[:12]
        tasks = [task]
    from .runner import run_task
    for index, task in enumerate(tasks):
        if index % args.shards == args.shard_index:
            print(f"Running {task['id']}", flush=True)
            task = resolve_task(task, plan)
            run_task(task, catalogues[task['config']['dataset']], p, resume=args.resume)


if __name__ == '__main__':
    main()
