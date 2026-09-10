"""Reconstruct saved models and validation inputs without rebuilding training data."""

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from ..data.windows import make_windows
from ..models.eegnet import EEGNet


def read_json(path: Path) -> dict:
    """Read a UTF-8 experiment manifest."""
    return json.loads(path.read_text(encoding='utf-8'))


def load_model(output: Path, fold: int, device: torch.device) -> tuple:
    """Load strict checkpoint weights and validate its dimensions, labels and channel order."""
    checkpoint = torch.load(output / 'cv' / f'fold_{fold}' / 'best.pt',
                            map_location='cpu', weights_only=True)
    manifest = read_json(output / 'prepared/folds.json')
    if (checkpoint['validation_fold'] != fold or
            checkpoint['channel_names'] != manifest['channel_names'] or
            checkpoint['label_names'] != manifest['label_names'] or
            checkpoint['n_electrodes'] != len(checkpoint['channel_names']) or
            len(checkpoint['label_names']) != 4):
        raise ValueError(f'Fold {fold}: checkpoint/manifest mismatch')
    model = EEGNet(checkpoint['model_config'], output_dim=4,
                   n_electrodes=checkpoint['n_electrodes'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device).eval()
    return model, checkpoint


def load_validation(output: Path, fold: int, checkpoint: dict) -> dict:
    """Recreate only this fold's non-augmented windows with its saved RMS scales.

    Uses the frozen experiment.json rather than a potentially edited config file.
    Trial indices remain global prepared-data indices, for joining exported CSVs.
    """
    config = read_json(output / 'cv/experiment.json')
    split = read_json(output / 'cv' / f'fold_{fold}' / 'split.json')
    with h5py.File(output / 'prepared/clean_trials.h5', 'r') as source:
        data, labels = source['X'][:], source['y'][:]
        folds = [source[f'fold_{f}'][:] for f in range(5)]
        sfreq, tmin = float(source.attrs['sfreq']), float(source.attrs['tmin'])
    indices = folds[fold]
    if sorted(np.concatenate(folds).tolist()) != list(range(len(labels))):
        raise ValueError('Validation folds must cover every trial exactly once')
    train, val = set(split['train_trial_indices']), set(split['validation_trial_indices'])
    if train & val or val != set(indices) or train | val != set(range(len(labels))):
        raise ValueError(f'Fold {fold}: invalid train/validation split')
    scale = np.asarray(checkpoint['normalization_scales'], dtype=float)
    if (scale.shape != (data.shape[1],) or not np.isfinite(scale).all() or
            np.any(scale <= 0) or not np.array_equal(scale, split['normalization_scales'])):
        raise ValueError('Invalid or mismatched saved normalization scales')
    data /= scale[None, :, None]
    block = make_windows(data, labels, indices, sfreq, tmin, config['window'],
                         copies=1, rng=np.random.default_rng(42))
    if list(block['X'].shape) != split['validation_shape'] or np.any(block['augmentation_id']):
        raise ValueError('Reconstructed validation layout mismatch')
    return block


def verify_baseline(output: Path, fold: int, block: dict, probabilities: np.ndarray,
                    metrics: dict, atol: float = 1e-6) -> dict:
    """Match all saved probabilities/labels and both original accuracy metrics; fail closed."""
    directory = output / 'cv' / f'fold_{fold}'
    saved = np.genfromtxt(directory / 'validation_predictions.csv', delimiter=',', names=True)
    order = np.lexsort((block['window_start_s'], block['trial_index']))
    saved = np.sort(saved, order=['trial_index', 'window_start_s'])
    old_p = np.stack([saved[f'p_{k}'] for k in range(4)], axis=1)
    if (len(saved) != len(probabilities) or
            not np.array_equal(saved['trial_index'], block['trial_index'][order]) or
            not np.array_equal(saved['true_label'], block['y'][order]) or
            not np.allclose(saved['window_start_s'], block['window_start_s'][order], rtol=0, atol=1e-9)):
        raise ValueError(f'Fold {fold}: baseline row metadata mismatch')
    error = float(np.max(np.abs(probabilities[order] - old_p)))
    if (error > atol or not np.array_equal(probabilities[order].argmax(1), saved['predicted_label'])):
        raise ValueError(f'Fold {fold}: baseline prediction mismatch (max error {error:g})')
    original = read_json(directory / 'metrics.json')
    for key, old_key in [('window_accuracy_percent', 'validation_accuracy_percent'),
                         ('trial_accuracy_percent', 'validation_trial_accuracy_percent')]:
        if not np.isclose(metrics[key], original[old_key], atol=1e-10, rtol=0):
            raise ValueError(f'Fold {fold}: baseline {key} mismatch')
    return {'passed': True, 'probability_atol': atol, 'max_probability_error': error,
            'window_accuracy_percent': metrics['window_accuracy_percent'],
            'trial_accuracy_percent': metrics['trial_accuracy_percent']}
