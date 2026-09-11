"""Audit completed overnight report metadata without retraining or inference."""
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import numpy as np

ROOT = Path('/public/home/hugf2022/inhouse_bci_raspy/overnight-v1')
read = lambda p: json.loads(p.read_text())
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
target = ROOT / 'reviews' / stamp
target.mkdir(parents=True, exist_ok=False)
text = ['# Overnight results audit', '', 'Strict evaluation; seed 42; selection stage capped at 60 epochs.', '']
rows = []
for dataset, report_stamp in [('inhouse', '20260910T181604747237Z'), ('bci2a', '20260910T232126627110Z')]:
    root = ROOT / dataset
    report = root / 'reports' / report_stamp
    manifest = read(root / 'manifest.json')
    completion = read(report / 'completion.json')
    definitions = read(report / 'variant_definitions.json')
    subjects = list(csv.DictReader((report / 'subject_summary.csv').open()))
    records = list(csv.DictReader((report / 'runs.csv').open()))
    labels = completion['labels']
    grouped = defaultdict(list)
    details = []
    for task in manifest['tasks']:
        directory = root / 'runs' / task['id']
        training = read(directory / 'training.json')
        history = read(directory / 'inner/history.json')
        pipeline = read(directory / 'final/pipeline.json')
        detail = dict(dataset=dataset, label=task['label'], split=task['config']['split'],
                      run_id=task['id'], selection_epochs=len(history), refit_epochs=len(training['selected_schedule']),
                      ica_excluded=pipeline.get('ica_excluded_components'),
                      selection_first_loss=history[0]['selection_loss'],
                      selection_best_loss=min(h['selection_loss'] for h in history),
                      epoch_cap=task['id'] in completion['hit_epoch_limit'])
        details.append(detail)
        grouped[task['label']].append(detail)
    for row in subjects:
        cfg = definitions[row['variant']]['config']
        task = next(t for t in manifest['tasks'] if {k:v for k,v in t['config'].items() if k not in ('seed','split')} == cfg)
        name = task['label']; values = grouped[name]
        rec = dict(dataset=dataset, label=name, trial_accuracy=float(row['mean_accuracy']),
                   between_subject_std=float(row['between_subject_std']), runs=len(values),
                   selection_epochs=[v['selection_epochs'] for v in values],
                   refit_epochs=[v['refit_epochs'] for v in values],
                   epoch_caps=sum(v['epoch_cap'] for v in values))
        rows.append(rec)
    print(dataset, 'DETAILS', json.dumps(details))
    if dataset == 'inhouse':
        for split in manifest['splits']:
            pair = [next(t for t in manifest['tasks'] if t['label']==label and t['config']['split']==split)
                    for label in ('raspy_window','no_ica')]
            with np.load(root/'runs'/pair[0]['id']/'test_predictions.npz') as a, np.load(root/'runs'/pair[1]['id']/'test_predictions.npz') as b:
                print('ICA_IDENTITY',split, bool(np.array_equal(a['ids'],b['ids']) and np.array_equal(a['logits'],b['logits'])))
    else:
        by_subject = defaultdict(dict)
        for rec in records:
            by_subject[rec['split']][labels[rec['run_id']]] = float(rec['trial_accuracy'])
        for a,b in [('raspy_window','standard_window'),('raspy_window','raspy_mil'),
                    ('standard_window','standard_mil'),('raspy_window','raspy_whole'),('standard_window','standard_whole')]:
            differences = [v[b]-v[a] for v in by_subject.values()]
            print('PAIRED',a,b,'wins/ties/losses',sum(v>0 for v in differences),sum(v==0 for v in differences),sum(v<0 for v in differences),'mean_delta',statistics.mean(differences))
    (target/f'{dataset}_training_audit.json').write_text(json.dumps(details,indent=2))
    text += [f'## {dataset}', '', f'Complete: {completion["completed"]}/{completion["expected"]}; missing explanations: {len(completion["explanations_missing"])}.', '',
             '| Configuration | Trial accuracy | Selection epochs | Refit epochs | Hit epoch cap |',
             '|---|---:|---|---|---:|']
    for row in sorted((r for r in rows if r['dataset']==dataset),key=lambda r:-r['trial_accuracy']):
        text.append(f'| {row["label"]} | {row["trial_accuracy"]:.2%} | {row["selection_epochs"]} | {row["refit_epochs"]} | {row["epoch_caps"]} |')
    text += ['', 'Inhouse: pooled held-out trials across five folds. BCI: equal-weight mean across nine subjects.', '']
(target/'summary.json').write_text(json.dumps(rows,indent=2))
(target/'summary.md').write_text('\n'.join(text))
print('REVIEW',target)
