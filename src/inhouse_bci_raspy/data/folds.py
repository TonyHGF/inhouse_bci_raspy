"""Assemble one cross-validation round using training-only channel RMS."""

import h5py
import numpy as np
from .windows import make_windows


def build_fold(cache, config, validation_fold):
    """Read a trial cache and construct one round of training and validation arrays.
    
    Args:
        cache: Path to the prepared trial-level HDF5 file.
        config: Experiment settings containing folds, seed and window parameters.
        validation_fold: Zero-based held-out fold.
    Returns:
        Training block, validation block and split metadata including RMS scales.
    
    RMS is fitted on the four training folds before any windows/noise are generated.
    Validation has no noise copies; this function writes no files."""
    with h5py.File(cache, 'r') as f:
        data, y = f['X'][:], f['y'][:]
        val_idx = f[f'fold_{validation_fold}'][:]
        # Keep GitClone's fold concatenation order for the four training folds.
        train_idx = np.concatenate([f[f'fold_{i}'][:] for i in range(config['num_folds']) if i != validation_fold])
        sfreq, tmin = float(f.attrs['sfreq']), float(f.attrs['tmin'])
    if set(train_idx) & set(val_idx):
        raise ValueError('Trial overlap')
    flat = data[train_idx].transpose(0, 2, 1).reshape(-1, data.shape[1])
    scale = np.maximum(np.sqrt(np.mean(flat ** 2, axis=0)), np.finfo(float).eps)
    data /= scale[None, :, None]
    rng = np.random.default_rng(config['seed'] + validation_fold)
    training = make_windows(data, y, train_idx, sfreq, tmin, config['window'], config['window']['num_noise'] + 1, rng)
    validation = make_windows(data, y, val_idx, sfreq, tmin, config['window'], 1, rng)
    metadata = {'validation_fold': validation_fold, 'train_trial_indices': train_idx.tolist(),
                'validation_trial_indices': val_idx.tolist(),
                'normalization': 'per-channel RMS of this round training trials, full -2..5s epochs',
                'normalization_scales': scale.tolist(), 'augmentation_seed': config['seed'] + validation_fold,
                'train_shape': list(training['X'].shape), 'validation_shape': list(validation['X'].shape)}
    return training, validation, metadata
