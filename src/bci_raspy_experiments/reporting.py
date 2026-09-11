"""Results tables with common-test-trial comparisons and subject weighting."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import numpy as np
from .common import digest, read_json, save_json
from .evaluation import load_predictions, probabilities, scores


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(output, destination=None):
    output = Path(output)
    destination = Path(destination) if destination is not None else output / 'reports'
    destination.mkdir(parents=True, exist_ok=True)
    records, groups = [], defaultdict(list)
    for path in sorted((output / 'runs').glob('*/metrics.json')):
        if read_json(path.parent / 'status.json')['state'] != 'complete':
            continue
        task = read_json(path.parent / 'task.json')['task']
        metrics, cfg = read_json(path), task['config']
        record = dict(run_id=task['id'], dataset=cfg['dataset'], split=cfg['split']['id'], seed=cfg['seed'],
                      selection=cfg['selection'], objective=cfg['objective'], sampling=cfg['sampling'],
                      model=json.dumps(cfg['model'], sort_keys=True), preprocessing=cfg['preprocessing'],
                      augmentation=json.dumps(cfg['augmentation'], sort_keys=True), loss=cfg['loss'],
                      trial_accuracy=metrics['test']['trial_probability']['accuracy'],
                      window_accuracy=metrics['test']['window']['accuracy'],
                      subject_mean_accuracy=metrics['test']['subject_mean_accuracy'],
                      macro_f1=metrics['test']['trial_probability']['macro_f1'], kappa=metrics['test']['trial_probability']['kappa'],
                      retained_test=metrics['retained_test'], requested_test=metrics['requested_test'],
                      retained_train=metrics['retained_train'], requested_train=metrics['requested_train'],
                      parameters=metrics['parameters'], seconds=metrics['seconds'], peak_cuda_bytes=metrics['peak_cuda_bytes'])
        records.append(record)
        groups[(cfg['dataset'], cfg['split']['hash'], cfg['seed'])].append((path.parent, task, record))
    write_csv(destination / 'runs.csv', records)
    common_rows, pairs, cohorts = [], [], []
    for key, values in groups.items():
        predictions = {task['id']: load_predictions(path / 'test_predictions.npz') for path, task, _ in values}
        common = sorted(set.intersection(*(set(p['ids']) for p in predictions.values())))
        cohorts.append(dict(dataset=key[0], split_hash=key[1], seed=key[2], run_ids=list(predictions), common_ids=common))
        comparisons = {}
        for path, task, record in values:
            p = predictions[task['id']]
            lookup = {i: j for j, i in enumerate(p['ids'])}
            indices = [lookup[i] for i in common]
            if not indices:
                continue
            y = p['y'][indices]
            predicted = probabilities(p['logits'][indices]).mean(1).argmax(-1)
            score = scores(y, predicted)
            subjects = np.array([i.split('/')[1] for i in common])
            macro_subject = float(np.mean([np.mean(predicted[subjects == s] == y[subjects == s]) for s in set(subjects)]))
            common_rows.append(dict(run_id=task['id'], common_trials=len(common), accuracy=score['accuracy'],
                                    subject_mean_accuracy=macro_subject, macro_f1=score['macro_f1'], kappa=score['kappa']))
            comparisons[task['id']] = (score['accuracy'], macro_subject)
        for _, task, _ in values:
            if task['id'] not in comparisons:
                continue
            for _, other, _ in values:
                if other['id'] not in comparisons or task['id'] >= other['id']:
                    continue
                a, b = task['config'], other['config']
                changed = [k for k in a if a[k] != b[k]]
                if len(changed) == 2 and set(changed) == {'objective', 'sampling'} and 'whole' in (a['objective'], b['objective']):
                    changed = ['whole_trial_vs_windowed']
                if len(changed) != 1:
                    continue
                pairs.append(dict(run_a=task['id'], run_b=other['id'], factor=changed[0],
                                  common_trials=len(common), delta_b_minus_a=comparisons[other['id']][0]-comparisons[task['id']][0],
                                  subject_delta_b_minus_a=comparisons[other['id']][1]-comparisons[task['id']][1],
                                  diagnostic_test_selection=changed[0] == 'selection'))
    write_csv(destination / 'common_trials.csv', common_rows)
    write_csv(destination / 'paired_differences.csv', pairs)
    save_json(destination / 'cohorts.json', cohorts)
    # Pool out-of-fold predictions per subject before averaging subjects; do not
    # count folds or seeds as independent subjects.
    pooled = defaultdict(list)
    definitions = {}
    for values in groups.values():
        for directory, task, _ in values:
            cfg = task['config']
            split = cfg['split']
            protocol = split['kind'] + ('-' + split['id'].split('-')[-1] if split['kind'] == 'time' else '')
            variant = {k: v for k, v in cfg.items() if k not in ('split', 'seed')}
            signature = digest([variant, protocol])
            definitions[signature] = dict(config=variant, protocol=protocol)
            p = load_predictions(directory / 'test_predictions.npz')
            prediction = probabilities(p['logits']).mean(1).argmax(-1)
            for i, trial in enumerate(p['ids']):
                pooled[(signature, cfg['dataset'], protocol, trial.split('/')[1], cfg['seed'])].append((trial, int(p['y'][i]), int(prediction[i])))
    subjects, seed_group = [], defaultdict(list)
    for (signature, dataset, protocol, subject, seed), rows in pooled.items():
        if len({r[0] for r in rows}) != len(rows):
            raise ValueError('Duplicate test trial while pooling out-of-fold predictions')
        metric = scores([r[1] for r in rows], [r[2] for r in rows])
        subjects.append(dict(variant=signature, dataset=dataset, protocol=protocol, subject=subject, seed=seed,
                             accuracy=metric['accuracy'], trials=len(rows)))
        seed_group[(signature, dataset, protocol, subject)].append(metric['accuracy'])
    write_csv(destination / 'subject_seed_results.csv', subjects)
    variants = defaultdict(list)
    for (signature, dataset, protocol, subject), values in seed_group.items():
        variants[(signature, dataset, protocol)].append(float(np.mean(values)))
    summary = [dict(variant=k[0], dataset=k[1], protocol=k[2], subjects=len(v),
                    mean_accuracy=float(np.mean(v)), between_subject_std=float(np.std(v))) for k, v in variants.items()]
    write_csv(destination / 'subject_summary.csv', summary)
    save_json(destination / 'variant_definitions.json', definitions)
    text = ['# EEG ablation results', '', f'Completed runs: {len(records)}', '',
            'Only completed formal runs are included. Smoke results are excluded.',
            'Test-selected runs are diagnostic results and cannot select algorithms.',
            'Common-trial comparisons use the intersection of currently completed variants for each split and seed; cohorts.json freezes membership.',
            'Reports are provisional until the declared manifest is complete; adding variants can change the common cohort.',
            'Subject summaries pool held-out trials per subject, average seeds within subject, then weight subjects equally.',
            'Training seeds, overlapping windows and subset repetitions are not independent subjects.',
            'Chronological ratios have different test cohorts and do not isolate training size alone.', '',
            '| Dataset | Split | Selection | Trial accuracy | Test retained |', '|---|---|---|---:|---:|']
    for row in records:
        text.append(f"| {row['dataset']} | {row['split']} | {row['selection']} | {row['trial_accuracy']:.4f} | {row['retained_test']}/{row['requested_test']} |")
    (destination / 'report.md').write_text('\n'.join(text), encoding='utf-8')
    if records:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for selection in ('strict', 'test_selected'):
            chosen = [r for r in records if r['selection'] == selection]
            ax.scatter([r['window_accuracy'] for r in chosen], [r['trial_accuracy'] for r in chosen], label=selection, alpha=.5)
        ax.set(xlabel='Window accuracy', ylabel='Trial mean-probability accuracy', xlim=(0, 1), ylim=(0, 1))
        ax.legend()
        fig.tight_layout()
        fig.savefig(destination / 'window_trial.png', dpi=160)
        plt.close(fig)
    return dict(completed_runs=len(records), directory=str(destination))
