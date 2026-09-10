"""Audit saved validation predictions and quantify window-to-trial aggregation.

Run from the project directory with PYTHONPATH=src:
    python -m inhouse_bci_raspy.analysis.aggregation
All comparisons reuse the same selected checkpoints and saved probabilities.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from ..io import save_json
from .presentation import write_outputs


def load_predictions(output: Path) -> dict:
    """Load trial-aligned probabilities [trial,time,class] and verify split/label integrity.

    Checks each trial appears in exactly one validation fold, its windows stay
    within that fold, all 31 start times are unique, and saved metrics reproduce.
    This audits exported predictions, not upstream preprocessing independence.
    """
    with h5py.File(output / 'prepared/clean_trials.h5', 'r') as source:
        labels = source['y'][:]
        trial_ids = source['trial_id'][:]
        recordings = source['recording_id'].asstr()[:]
        prepared_folds = [source[f'fold_{f}'][:] for f in range(5)]
    expected_times = np.linspace(1, 4, 31)
    probabilities = np.full((len(labels), 31, 4), np.nan)
    folds = np.full(len(labels), -1, dtype=int)
    hashes = {}
    for fold in range(5):
        directory = output / 'cv' / f'fold_{fold}'
        path = directory / 'validation_predictions.csv'
        hashes[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = np.genfromtxt(path, delimiter=',', names=True)
        split = json.loads((directory / 'split.json').read_text())
        train = set(split['train_trial_indices'])
        val = set(split['validation_trial_indices'])
        assert not train & val, 'Train/validation trial overlap'
        assert train | val == set(range(len(labels)))
        assert val == set(prepared_folds[fold]) == set(rows['trial_index'].astype(int))
        for trial in sorted(val):
            assert folds[trial] == -1, 'Trial repeated across validation folds'
            block = rows[rows['trial_index'] == trial]
            block = np.sort(block, order='window_start_s')
            assert len(block) == 31 and np.allclose(block['window_start_s'], expected_times)
            assert np.all(block['true_label'] == labels[trial])
            p = np.stack([block[f'p_{c}'] for c in range(4)], axis=1)
            assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
            assert np.allclose(p.sum(1), 1, atol=2e-7)
            assert np.array_equal(p.argmax(1), block['predicted_label'])
            probabilities[trial] = p
            folds[trial] = fold
        selected = folds == fold
        stored = json.loads((directory / 'metrics.json').read_text())
        measured = metric_row(probabilities[selected], labels[selected])
        assert np.isclose(measured['window_percent'], stored['validation_accuracy_percent'])
        assert np.isclose(measured['soft_percent'], stored['validation_trial_accuracy_percent'])
    assert (folds >= 0).all() and np.isfinite(probabilities).all()
    return dict(p=probabilities, y=labels, folds=folds, times=expected_times,
                trial_ids=trial_ids, recordings=recordings, prediction_sha256=hashes)


def metric_row(p: np.ndarray, y: np.ndarray) -> dict:
    """Compare window accuracy, mean-probability voting and hard plurality voting.

    Hard-vote ties use lowest class index (NumPy argmax); tie count is explicit.
    A winning class does not need more than half of votes in four classes.
    """
    votes = np.eye(4, dtype=int)[p.argmax(2)].sum(1)
    return dict(trials=len(y), window_percent=float(100 * (p.argmax(2) == y[:, None]).mean()),
                soft_percent=float(100 * (p.mean(1).argmax(1) == y).mean()),
                hard_percent=float(100 * (votes.argmax(1) == y).mean()),
                hard_ties=int(((votes == votes.max(1, keepdims=True)).sum(1) > 1).sum()))


def subset_curve(p: np.ndarray, y: np.ndarray, repeats: int, seed: int) -> list:
    """Average accuracy over random subsets of k windows without replacement.

    Each repetition draws a separate uniform subset for every trial. Percentiles
    reflect window-selection variability on these fixed trials, not a population
    confidence interval. k=1 expectation and k=31 result are computed exactly.
    """
    rng = np.random.default_rng(seed)
    results = []
    for k in [1, 2, 4, 8, 16, 31]:
        if k in (1, 31):
            value = metric_row(p, y)['window_percent' if k == 1 else 'soft_percent']
            scores = np.array([value])
        else:
            scores = np.empty(repeats)
            for i in range(repeats):
                indices = np.argsort(rng.random(p.shape[:2]), axis=1)[:, :k]
                means = np.take_along_axis(p, indices[:, :, None], axis=1).mean(1)
                scores[i] = 100 * (means.argmax(1) == y).mean()
        results.append(dict(windows=k, mean_percent=float(scores.mean()),
                            selection_p025=float(np.quantile(scores, .025)),
                            selection_p975=float(np.quantile(scores, .975))))
    return results


def analyze(data: dict, repeats: int = 1000, seed: int = 42) -> tuple[dict, list]:
    """Compute gain decomposition, vote disagreements, temporal and subgroup controls."""
    p, y = data['p'], data['y']
    wp = p.argmax(2)
    correct = wp == y[:, None]
    fraction = correct.mean(1)
    mean_p = p.mean(1)
    soft = mean_p.argmax(1)
    soft_ok = soft == y
    votes = np.eye(4, dtype=int)[wp].sum(1)
    hard = votes.argmax(1)
    hard_ok = hard == y
    margins = mean_p[np.arange(len(y)), y] - np.where(np.eye(4)[y], -np.inf, mean_p).max(1)
    local_four = [float(100 * (p[:,i:i+4].mean(1).argmax(1) == y).mean()) for i in range(28)]
    trial_rows = []
    for i in range(len(y)):
        trial_rows.append(dict(trial_index=i, recording_id=data['recordings'][i],
            source_trial_id=int(data['trial_ids'][i]), fold=int(data['folds'][i]),
            true_label=int(y[i]), window_correct=int(correct[i].sum()), window_count=31,
            window_accuracy_percent=float(100*fraction[i]), soft_prediction=int(soft[i]),
            soft_correct=bool(soft_ok[i]), hard_prediction=int(hard[i]), hard_correct=bool(hard_ok[i]),
            hard_tie=bool((votes[i] == votes[i].max()).sum() > 1),
            true_probability_margin=float(margins[i]),
            **{f'votes_{c}': int(votes[i,c]) for c in range(4)},
            **{f'mean_p_{c}': float(mean_p[i,c]) for c in range(4)}))
    summary = dict(pooled=metric_row(p, y),
        by_fold=[dict(fold=f, **metric_row(p[data['folds']==f], y[data['folds']==f])) for f in range(5)],
        by_class=[dict(label=c, **metric_row(p[y==c], y[y==c])) for c in range(4)],
        by_session=[dict(session=r, **metric_row(p[data['recordings']==r], y[data['recordings']==r]))
                    for r in sorted(set(data['recordings']))],
        trial_outcomes=dict(soft_correct=int(soft_ok.sum()), soft_wrong=int((~soft_ok).sum()),
            soft_correct_below_half_window_correct=int((soft_ok & (fraction < .5)).sum()),
            soft_wrong_above_half_window_correct=int((~soft_ok & (fraction > .5)).sum()),
            soft_correct_hard_wrong=int((soft_ok & ~hard_ok).sum()),
            soft_wrong_hard_correct=int((~soft_ok & hard_ok).sum()),
            all_windows_wrong=int((fraction==0).sum()), all_windows_right=int((fraction==1).sum()),
            correct_trial_mean_window_percent=float(100*fraction[soft_ok].mean()),
            wrong_trial_mean_window_percent=float(100*fraction[~soft_ok].mean()),
            recovered_wrong_windows_on_correct_trials=int((~correct[soft_ok]).sum()),
            lost_correct_windows_on_wrong_trials=int(correct[~soft_ok].sum()),
            positive_gain_pp=float(100*((1-fraction)*soft_ok).mean()),
            negative_gain_pp=float(100*(fraction*~soft_ok).mean())),
        confidence=dict(mean_window_max_probability=float(p.max(2).mean()),
            correct_window_mean_max_probability=float(p.max(2)[correct].mean()),
            wrong_window_mean_max_probability=float(p.max(2)[~correct].mean())),
        random_subset=dict(seed=seed, repeats=repeats, curve=subset_curve(p,y,repeats,seed)),
        temporal=dict(window_start_s=data['times'].tolist(), accuracy_percent=(100*correct.mean(0)).tolist(),
            consecutive_four=dict(start_s=data['times'][:28].tolist(), accuracy_percent=local_four,
                                  mean_percent=float(np.mean(local_four)),
                                  min_percent=min(local_four), max_percent=max(local_four)),
            prediction_agreement_by_lag={str(lag): float(100*(wp[:,:-lag]==wp[:,lag:]).mean()) for lag in [1,5,10,20,30]},
            nonoverlap_1_2_3_4_s=metric_row(p[:,[0,10,20,30]], y),
            first_1_0_to_1_3_s=metric_row(p[:,:4],y),
            last_3_7_to_4_0_s=metric_row(p[:,-4:],y)),
        audit=dict(passed=True, trials=len(y), windows=int(p.shape[0]*p.shape[1]), windows_per_trial=31,
            validation_fold_count=5, prediction_sha256=data['prediction_sha256'],
            scope='Saved prediction integrity, labels, split disjointness, coverage and metric reproduction; not upstream leakage audit'),
        limitations=['Same validation folds used for checkpoint selection; not independent test.',
            'Three sessions mixed across folds; session subgroup scores are not held-out-session performance.',
            'Adjacent 1-second windows have 90% overlap; 31 votes are not independent.',
            'Comparisons are post-hoc on fixed saved predictions; no new model was trained.',
            'No channel-importance topomap implementation found in local GitClone source.'])
    assert np.isclose(summary['pooled']['soft_percent'] - summary['pooled']['window_percent'],
        summary['trial_outcomes']['positive_gain_pp'] - summary['trial_outcomes']['negative_gain_pp'])
    return summary, trial_rows


def main() -> None:
    """Write isolated post-hoc reports under outputs/analysis/aggregation by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parents[3] / 'outputs')
    parser.add_argument('--repeats', type=int, default=1000)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    data = load_predictions(args.output_root)
    summary, rows = analyze(data, args.repeats)
    destination = args.output_root / 'analysis/aggregation'
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / 'summary.json', summary)
    with (destination / 'trial_diagnostics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_outputs(destination, summary, rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
