"""Execute one fold: prepare arrays, train a fresh model, evaluate and save."""

from pathlib import Path
import numpy as np
from ..data.folds import build_fold
from ..data.loaders import make_loader, select_smoke_samples
from ..evaluation import fold_metrics, predict_probabilities
from ..io import save_json
from ..models.eegnet import EEGNet
from ..reporting import save_confusion, save_history, save_predictions
from .checkpoints import save_checkpoint
from .engine import train
from .logging import create_writer


def train_one_fold(config: dict, output: Path, destination: Path, manifest: dict,
                   fold: int, device, smoke: bool = False) -> tuple:
    """Run the original training engine for a single held-out fold.

    Args:
        config: Window, model and training settings.
        output: Root containing the prepared trial cache.
        destination: Parent directory for fold results (cv or smoke).
        manifest: Trial metadata, labels and channel names.
        fold: Zero-based validation fold; train on the other four folds.
        device: torch device selected once for the experiment.
        smoke: Use balanced tiny subsets and two epochs if true.

    Returns:
        Metrics, true window labels, predicted window labels and validated trial indices.
        The coordinator can aggregate these without retaining the fold's model or data.
    """
    fold_dir = destination / f'fold_{fold}'
    fold_dir.mkdir(exist_ok=True)
    training, validation, metadata = build_fold(output / 'prepared/clean_trials.h5', config, fold)
    settings = dict(config['training'])
    if smoke:
        for block in (training, validation):
            select_smoke_samples(block)
        settings['max_epochs'] = 2
    train_loader = make_loader(training, settings, 'train')
    val_loader = make_loader(validation, settings, 'val')
    model = EEGNet(config['model'], output_dim=len(manifest['label_names']),
                   n_electrodes=len(manifest['channel_names'])).to(device)
    writer = create_writer(fold_dir / 'tensorboard', smoke)
    print(f"Training fold {fold}: {len(training['y'])} train / {len(validation['y'])} validation windows", flush=True)
    try:
        best, stats, best_index, train_loss, val_loss, _, _ = train(
            model, train_loader, val_loader, settings, writer, device, verbosity=0)
    finally:
        writer.close()
    if not np.isfinite(train_loss + val_loss).all():
        raise RuntimeError(f'Nonfinite loss in fold {fold}')
    save_checkpoint(fold_dir / 'best.pt', best, config, manifest, metadata, fold, best_index)
    save_json(fold_dir / 'split.json', metadata)
    save_history(fold_dir, stats, train_loss, val_loss, best_index, smoke)
    probabilities = predict_probabilities(best, validation, device, settings['val_batch_size'])
    actual = validation['y']
    predicted = probabilities.argmax(axis=1)
    save_predictions(fold_dir / 'validation_predictions.csv', validation, probabilities)
    save_confusion(fold_dir, actual, predicted)
    metrics = fold_metrics(validation, probabilities, fold, best_index, stats, len(training['y']))
    save_json(fold_dir / 'metrics.json', metrics)
    print(f'Completed fold {fold}: {metrics}', flush=True)
    return metrics, actual.tolist(), predicted.tolist(), np.unique(validation['trial_index']).tolist()
