"""Predeclared ablation suites. No result-dependent model selection."""
from copy import deepcopy
from .common import digest

PREPROCESS = {
    'full': dict(ica=True, csd=True, baseline=True, reject=True, car=True, rms=True),
    'no_ica': dict(ica=False, csd=True, baseline=True, reject=True, car=True, rms=True),
    'no_csd': dict(ica=True, csd=False, baseline=True, reject=True, car=True, rms=True),
    'no_baseline': dict(ica=True, csd=True, baseline=False, reject=True, car=True, rms=True),
    'no_reject': dict(ica=True, csd=True, baseline=True, reject=False, car=True, rms=True),
    'no_csd_car': dict(ica=True, csd=False, baseline=True, reject=True, car=False, rms=True),
    'filter_rms': dict(ica=False, csd=False, baseline=False, reject=False, car=False, rms=True),
    'filter_only': dict(ica=False, csd=False, baseline=False, reject=False, car=False, rms=False),
}
BASE = dict(model=dict(family='raspy', f1=8, d=2, f2=16, kernel=51, dropout=.5, extra=0),
            objective='window', sampling='dense', loss='mse', preprocessing='full',
            augmentation=dict(copies=4, strength=1., budget='matched'), selection='strict',
            train=dict(max_epochs=500, patience=10, learning_rate=.001, batch_trials=8))
CORE = [('raspy', 0), ('standard', 0), ('standard', 2)]


def variants(suite):
    values = []
    def add(**changes):
        value = deepcopy(BASE)
        for k, v in changes.items():
            if isinstance(v, dict):
                value[k].update(v)
            else:
                value[k] = v
        if value not in values:
            values.append(value)
    if suite == 'protocol':
        for family, extra in CORE:
            for selection in ('strict', 'test_selected'):
                add(model=dict(family=family, extra=extra), selection=selection)
    elif suite == 'single':
        add()
        for family in ('raspy', 'standard'):
            for extra in range(4):
                add(model=dict(family=family, extra=extra))
        for key, choices in [('f1', [4, 16]), ('d', [1, 4]), ('f2', [8, 32]), ('kernel', [25, 75]), ('dropout', [.25])]:
            for value in choices:
                add(model={'family': 'standard', key: value})
        for family in ('raspy', 'standard'):
            add(model=dict(family=family), loss='ce')
        for objective in ('window', 'mil', 'whole'):
            for sampling in (('whole',) if objective == 'whole' else ('center', 'sparse', 'medium', 'dense')):
                add(objective=objective, sampling=sampling)
        for copies, strength in [(0, 1.), (1, 1.), (4, .5), (4, 2.)]:
            add(augmentation=dict(copies=copies, strength=strength))
        for copies in (0, 4):
            add(augmentation=dict(copies=copies, budget='expanded'))
        for preprocessing in PREPROCESS:
            add(preprocessing=preprocessing)
    elif suite == 'interaction':
        for family, extra in CORE:
            for objective in ('window', 'mil'):
                for copies in (0, 4):
                    for preprocessing in ('full', 'filter_rms'):
                        add(model=dict(family=family, extra=extra), objective=objective,
                            augmentation=dict(copies=copies), preprocessing=preprocessing)
    elif suite == 'extension':
        for family, extra in CORE:
            for objective in ('window', 'mil'):
                add(model=dict(family=family, extra=extra), objective=objective)
    else:
        raise ValueError(f'Unknown suite: {suite}')
    return values


def manifest(catalogues, suite='all', seeds=(42, 43, 44)):
    from .protocols import outer_splits
    tasks, split_table = {}, {}
    suites = ('protocol', 'single', 'interaction', 'extension') if suite == 'all' else (suite,)
    for dataset, catalog in catalogues.items():
        for split in outer_splits(catalog['trials'], dataset):
            split_key = dataset + '/' + split['id']
            split_table[split_key] = split
            for stage in suites:
                if stage in ('single', 'interaction') and not split['core']:
                    continue
                if stage == 'extension' and (split['core'] or split['kind'] == 'subject'):
                    continue
                for variant in variants(stage):
                    for seed in seeds:
                        config = dict(dataset=dataset, catalog_hash=catalog['hash'], split=split_key,
                                      seed=seed, **variant)
                        run_id = dataset + '-' + digest(config)[:20]
                        if run_id not in tasks:
                            tasks[run_id] = dict(id=run_id, config=config, suites=[])
                        tasks[run_id]['suites'].append(stage)
    return dict(version=2, suite=suite, splits=split_table, tasks=list(tasks.values()), count=len(tasks))


def resolve_task(task, plan):
    task = deepcopy(task)
    if isinstance(task['config']['split'], str):
        task['config']['split'] = deepcopy(plan['splits'][task['config']['split']])
    return task
