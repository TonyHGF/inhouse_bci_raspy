"""Channel-zero ablation scored against one-hot targets, with no model fitting."""

import numpy as np
import torch


def infer(model, x: np.ndarray, batch_size: int = 32, channel: int | None = None) -> tuple:
    """Return logits and softmax [windows,4], zeroing only a copied channel if requested.

    Input is [windows,channels,time,1]. No caller arrays, parameters, buffers or
    normalization scales are modified. The caller must put the model in eval mode.
    """
    if model.training:
        raise ValueError('Analysis requires model.eval()')
    if batch_size < 1 or (channel is not None and not 0 <= channel < x.shape[1]):
        raise ValueError('Invalid batch size or channel')
    device = next(model.parameters()).device
    logits, probabilities = [], []
    with torch.inference_mode():
        for first in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[first:first + batch_size]).to(device)
            if channel is not None:
                batch = batch.clone()
                batch[:, channel, :, :] = 0
            z = model(batch)
            logits.append(z.cpu().numpy())
            probabilities.append(z.softmax(1).cpu().numpy())
    z, p = np.concatenate(logits), np.concatenate(probabilities)
    if not np.isfinite(z).all() or not np.isfinite(p).all():
        raise ValueError('Nonfinite model output')
    return z, p


def r2_outputs(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return one R² per output; explicitly reject constant targets and nonfinite input.

    Equivalent to sklearn r2_score(..., multioutput='raw_values', force_finite=False).
    No class weighting or clipping: negative R² and negative importance are valid.
    """
    y, z = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.ndim != 2 or z.shape != y.shape or len(y) < 2 or not (np.isfinite(y).all() and np.isfinite(z).all()):
        raise ValueError('R² requires matching finite [samples,outputs] arrays')
    denominator = ((y - y.mean(0)) ** 2).sum(0)
    if np.any(denominator <= 0):
        raise ValueError('R² undefined for constant target: include all validation classes')
    return 1 - ((y - z) ** 2).sum(0) / denominator


def summarize_outputs(block: dict, logits: np.ndarray, probabilities: np.ndarray) -> dict:
    """Aggregate logits for R² and, separately, probabilities for the original accuracy.

    Class-specific R² always uses every trial with a one-vs-rest target, not a
    constant-label subset. Every trial must contain the same unique window times.
    """
    ids = np.unique(block['trial_index'])
    trial_y, trial_z, trial_p = [], [], []
    expected_times = np.unique(block['window_start_s'])
    for trial in ids:
        selected = block['trial_index'] == trial
        y = block['y'][selected]
        times = np.sort(block['window_start_s'][selected])
        if not np.all(y == y[0]) or not np.array_equal(times, expected_times):
            raise ValueError(f'Trial {trial}: inconsistent labels or duplicate/missing windows')
        trial_y.append(int(y[0]))
        trial_z.append(logits[selected].astype(np.float64).mean(0))
        # Match the existing evaluation's float32 probability mean.
        trial_p.append(probabilities[selected].mean(0))
    trial_y, trial_z, trial_p = np.array(trial_y), np.array(trial_z), np.array(trial_p)
    window_r2 = r2_outputs(np.eye(4)[block['y']], logits)
    trial_r2 = r2_outputs(np.eye(4)[trial_y], trial_z)
    return {'trial_indices': ids, 'trial_y': trial_y, 'trial_logits': trial_z,
            'trial_probabilities': trial_p, 'trial_r2_per_class': trial_r2,
            'window_r2_per_class': window_r2, 'trial_r2': float(trial_r2.mean()),
            'window_r2': float(window_r2.mean()),
            'window_accuracy_percent': float(100 * (probabilities.argmax(1) == block['y']).mean()),
            'trial_accuracy_percent': float(100 * (trial_p.argmax(1) == trial_y).mean())}


def ablate_channels(model, block: dict, baseline: dict, batch_size: int = 32,
                    progress=None) -> list:
    """Evaluate each entire channel once and return outputs plus signed baseline-minus-R²."""
    results = []
    for channel in range(block['X'].shape[1]):
        if progress is not None:
            progress(channel)
        logits, p = infer(model, block['X'], batch_size, channel)
        result = summarize_outputs(block, logits, p)
        result['channel_index'] = channel
        for level in ['trial', 'window']:
            result[f'{level}_delta_r2_per_class'] = baseline[f'{level}_r2_per_class'] - result[f'{level}_r2_per_class']
            result[f'{level}_delta_r2'] = float(result[f'{level}_delta_r2_per_class'].mean())
            result[f'{level}_accuracy_drop_pp'] = baseline[f'{level}_accuracy_percent'] - result[f'{level}_accuracy_percent']
        results.append(result)
    return results


def weighted_moments(values: np.ndarray, weights: np.ndarray) -> tuple:
    """Return weighted mean and descriptive population SD across folds (not standard error)."""
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or len(weights) != len(values) or np.any(weights <= 0):
        raise ValueError('Positive fold weights required')
    mean = np.average(values, axis=0, weights=weights)
    sd = np.sqrt(np.average((values - mean) ** 2, axis=0, weights=weights))
    return mean, sd
