"""Pure identity-based outer and inner split construction."""
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from .common import digest


def validate(rows, train, test, grouped=False):
    if not train or not test or set(train) & set(test):
        raise ValueError('Empty or overlapping partition')
    by_id = {r['id']: r for r in rows}
    for part in (train, test):
        if len(part) != len(set(part)) or set(by_id[i]['label'] for i in part) != {0, 1, 2, 3}:
            raise ValueError('Each partition must contain unique trials and all four classes')
    if grouped and {by_id[i]['subject'] for i in train} & {by_id[i]['subject'] for i in test}:
        raise ValueError('Subject leakage')


def outer_splits(rows, dataset):
    output = []
    subjects = sorted({r['subject'] for r in rows})

    def add(name, train, test, kind, core):
        validate(rows, train, test, kind == 'subject')
        output.append(dict(id=name, train=train, test=test, kind=kind, core=core,
                           hash=digest([train, test]), split_seed=42))

    if dataset != 'physionet':
        for subject in subjects:
            selected = [r for r in rows if r['subject'] == subject]
            ids = [r['id'] for r in selected]
            y = [r['label'] for r in selected]
            for fraction in (.5, .6, .7, .8):
                if dataset == 'bci2a' and fraction == .5:
                    train = [r['id'] for r in selected if r['session'] == 'T']
                    test = [r['id'] for r in selected if r['session'] == 'E']
                else:
                    cut = int(len(ids) * fraction)
                    train, test = ids[:cut], ids[cut:]
                add(f'{subject}-time-{int(fraction*100)}', train, test, 'time', dataset == 'bci2a' and fraction == .5)
            for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(ids, y)):
                add(f'{subject}-random-{fold}', [ids[i] for i in tr], [ids[i] for i in te], 'trial', dataset == 'inhouse')
    if dataset == 'bci2a':
        groups = [[s] for s in subjects]
    elif dataset == 'physionet':
        order = np.random.default_rng(42).permutation(subjects)
        groups = [list(g) for g in np.array_split(order, 5)]
    else:
        groups = []
    for fold, group in enumerate(groups):
        add(f'subject-{fold}', [r['id'] for r in rows if r['subject'] not in group],
            [r['id'] for r in rows if r['subject'] in group], 'subject', True)
    return output


def inner_split(rows, outer):
    by_id = {r['id']: r for r in rows}
    ids = list(outer['train'])
    if outer['kind'] == 'time':
        cut = int(len(ids) * .8)
        train, val = ids[:cut], ids[cut:]
    elif outer['kind'] == 'subject':
        subjects = sorted({by_id[i]['subject'] for i in ids})
        order = np.random.default_rng(42).permutation(subjects)
        held = set(order[:max(1, int(np.ceil(len(order) * .2)))])
        train, val = [i for i in ids if by_id[i]['subject'] not in held], [i for i in ids if by_id[i]['subject'] in held]
    else:
        train, val = train_test_split(ids, test_size=.2, stratify=[by_id[i]['label'] for i in ids], random_state=42)
    validate(rows, train, val, outer['kind'] == 'subject')
    return list(train), list(val)
