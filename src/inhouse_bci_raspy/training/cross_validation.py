"""Coordinate five independent models and verify complete out-of-fold trial coverage."""

import json
from pathlib import Path
from ..evaluation import summarize_folds
from ..io import save_json
from ..reporting import save_experiment
from ..runtime import setup_training
from .fold import train_one_fold


def train_folds(config: dict, output: Path, device_name: str, smoke: bool = False) -> None:
    """Train every validation fold once and write the aggregate summary.

    Owns round order and experiment-wide RNG initialization. Single-fold models are
    managed by train_one_fold, while the engine owns optimization and early stopping.
    Formal checkpoints are not overwritten. Smoke mode runs fold zero only.
    """
    destination = output / ('smoke' if smoke else 'cv')
    destination.mkdir(exist_ok=True)
    if not smoke and any(destination.glob('fold_*/best.pt')):
        raise FileExistsError(f'Existing checkpoints in {destination}; archive that directory before a new experiment')
    device = setup_training(config['seed'], device_name)
    save_experiment(destination, config, device)
    manifest = json.loads((output / 'prepared/folds.json').read_text(encoding='utf-8'))
    summary, all_true, all_pred, all_trials = [], [], [], []
    for fold in range(1 if smoke else config['num_folds']):
        metrics, actual, predicted, trials = train_one_fold(config, output, destination, manifest, fold, device, smoke)
        summary.append(metrics)
        all_true.extend(actual)
        all_pred.extend(predicted)
        all_trials.extend(trials)
    if not smoke and (len(all_trials) != manifest['shape'][0] or len(set(all_trials)) != len(all_trials)):
        raise ValueError('Validation must cover every trial exactly once across five folds')
    save_json(destination / 'summary.json', summarize_folds(summary, all_true, all_pred, smoke))
