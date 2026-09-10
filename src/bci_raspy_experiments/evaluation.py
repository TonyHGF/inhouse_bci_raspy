"""Frozen-model predictions, predeclared aggregation, and subject-level metrics."""
from pathlib import Path
import csv
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from .common import save_json
from .engine import forward_bags
from .windows import loader


def scores(y, predicted):
    kappa = float(cohen_kappa_score(y, predicted, labels=[0, 1, 2, 3]))
    return dict(accuracy=float(accuracy_score(y, predicted)), balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                macro_f1=float(f1_score(y, predicted, labels=[0, 1, 2, 3], average='macro', zero_division=0)),
                kappa=kappa if np.isfinite(kappa) else None,
                confusion_matrix=confusion_matrix(y, predicted, labels=[0, 1, 2, 3]).tolist(), n=len(y))


def probabilities(z):
    e = np.exp(z-z.max(axis=-1, keepdims=True))
    return e/e.sum(axis=-1, keepdims=True)


def predict(model, bags, cfg, profile, device, zero_channel=None):
    ids, labels, logits, starts = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader(bags, cfg, profile):
            x = batch['x'].to(device)
            if zero_channel is not None:
                x[:, :, zero_channel] = 0
            z = forward_bags(model, x).cpu().numpy()
            logits.extend(z)
            labels.extend(batch['y'].tolist())
            ids.extend(batch['id'])
            starts.extend(batch['start'].numpy())
    return dict(ids=np.asarray(ids), y=np.asarray(labels), logits=np.asarray(logits), starts=np.asarray(starts))


def summarize(prediction, rows):
    z, y = prediction['logits'], prediction['y']
    p = probabilities(z)
    windows = p.argmax(-1)
    hard = np.eye(4)[windows].sum(axis=1).argmax(-1)
    result = dict(window=scores(np.repeat(y, z.shape[1]), windows.flatten()),
                  trial_probability=scores(y, p.mean(1).argmax(-1)),
                  trial_logits=scores(y, z.mean(1).argmax(-1)), trial_hard=scores(y, hard))
    by_id = {r['id']: r for r in rows}
    subjects = np.array([by_id[i]['subject'] for i in prediction['ids']])
    result['subjects'] = {}
    for subject in sorted(set(subjects)):
        chosen = subjects == subject
        result['subjects'][subject] = scores(y[chosen], p[chosen].mean(1).argmax(-1))
    result['subject_mean_accuracy'] = float(np.mean([v['accuracy'] for v in result['subjects'].values()]))
    return result


def aggregation(prediction, repeats=1000):
    p, y = probabilities(prediction['logits']), prediction['y']
    n, w, _ = p.shape
    result = dict(repetitions=repeats, random_seed=42,
                  interpretation='Random subset stability, not independent-sample confidence intervals',
                  random_subsets={}, positions=[], consecutive_four=[])
    for j in range(w):
        result['positions'].append(dict(offset=float(prediction['starts'][0, j]), accuracy=float(np.mean(p[:, j].argmax(-1) == y))))
    for k in (1, 2, 4, 8, 16, 31):
        if k > w:
            continue
        if k == 1:
            values = [float(np.mean(p.argmax(-1) == y[:, None]))]
        elif k == w:
            values = [float(np.mean(p.mean(1).argmax(-1) == y))]
        else:
            rng = np.random.default_rng(42 + k)
            values = []
            for _ in range(repeats):
                indices = np.argsort(rng.random((n, w)), axis=1)[:, :k]
                pred = p[np.arange(n)[:, None], indices].mean(1).argmax(-1)
                values.append(float(np.mean(pred == y)))
        result['random_subsets'][str(k)] = dict(mean=float(np.mean(values)), std=float(np.std(values)))
    if w == 31:
        result['four_nonoverlapping_accuracy'] = float(np.mean(p[:, [0, 10, 20, 30]].mean(1).argmax(-1) == y))
    for j in range(w-3):
        result['consecutive_four'].append(dict(offset=float(prediction['starts'][0, j]),
            accuracy=float(np.mean(p[:, j:j+4].mean(1).argmax(-1) == y))))
    return result


def save_predictions(path, prediction):
    path = Path(path)
    np.savez_compressed(path.with_suffix('.npz'), **prediction)
    p = probabilities(prediction['logits'])
    with path.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['trial_id', 'window_start', 'label', 'prediction'] + [f'logit_{i}' for i in range(4)] + [f'p_{i}' for i in range(4)])
        for i, trial in enumerate(prediction['ids']):
            for j, start in enumerate(prediction['starts'][i]):
                writer.writerow([trial, start, prediction['y'][i], p[i, j].argmax(), *prediction['logits'][i, j], *p[i, j]])
    with path.with_name(path.stem + '_trials.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['trial_id', 'label', 'prediction'] + [f'mean_logit_{i}' for i in range(4)] + [f'mean_p_{i}' for i in range(4)])
        for i, trial in enumerate(prediction['ids']):
            writer.writerow([trial, prediction['y'][i], p[i].mean(0).argmax(), *prediction['logits'][i].mean(0), *p[i].mean(0)])


def load_predictions(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
