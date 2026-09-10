"""Load clean MNE epochs, apply CSD, and persist trial-level data and folds."""

import hashlib
import h5py
import mne
import numpy as np
from ..io import save_json
from .partition import partition_data


def prepare(config, dataset, clean_root, output):
    """Load clean epochs, compute CSD, and save unnormalized trial data with fold indices.
    
    Args:
        config: Experiment settings with recordings, seed and num_folds.
        dataset: BIDS identifiers and ordered label mapping.
        clean_root: Derivatives root containing the subject/session EEG directories.
        output: Destination for clean_trials.h5 and folds.json.
    Returns:
        The manifest containing labels, source hashes, shapes and validation trial keys.
    Raises:
        ValueError: Layout mismatch, missing classes, duplicate trials or invalid folds."""
    output.mkdir(parents=True, exist_ok=True)
    names = list(dataset['event_id'])
    arrays, labels, records, ids, sources = [], [], [], [], []
    channels = sfreq = tmin = None
    for rec in config['recordings']:
        path = (clean_root / f"sub-{dataset['subject']}" / f"ses-{rec['session']}" / 'eeg' /
                f"sub-{dataset['subject']}_ses-{rec['session']}_task-{dataset['task']}_proc-clean_epo.fif")
        epochs = mne.read_epochs(path, preload=True, verbose='error')[names]
        inverse = {value: key for key, value in epochs.event_id.items()}
        conditions = [inverse[int(code)] for code in epochs.events[:, 2]]
        epochs = mne.preprocessing.compute_current_source_density(epochs, copy=True)
        picks = mne.pick_types(epochs.info, csd=True, eeg=True, exclude=[])
        current_channels = [epochs.ch_names[i] for i in picks]
        if channels is None:
            channels, sfreq, tmin = current_channels, epochs.info['sfreq'], epochs.tmin
        if (channels != current_channels or sfreq != epochs.info['sfreq'] or tmin != epochs.tmin):
            raise ValueError(f'Inconsistent channel/sample/time layout: {path}')
        arrays.append(epochs.get_data(picks=picks, copy=True))
        labels.extend(names.index(name) for name in conditions)
        records.extend([rec['id']] * len(epochs))
        ids.extend(epochs.selection.tolist())
        sources.append({'recording_id': rec['id'], 'path': str(path.resolve()),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'clean_trials': len(epochs)})
    x = np.concatenate(arrays)
    y = np.asarray(labels, dtype=np.int64)
    keys = [f'{r}:{i}' for r, i in zip(records, ids)]
    if len(set(keys)) != len(keys) or len(channels) != 16 or not np.isfinite(x).all():
        raise ValueError('Duplicate trial keys, nonfinite data, or unexpected EEG channels')
    if set(y) != set(range(4)) or min(np.bincount(y)) < config['num_folds']:
        raise ValueError('Need four imagery classes and at least one trial/class/fold')
    np.random.seed(config['seed'])
    folds = partition_data([(label, None) for label in y], config['num_folds'])
    if sorted(np.concatenate(folds).tolist()) != list(range(len(y))):
        raise ValueError('Folds must cover every trial exactly once')
    with h5py.File(output / 'clean_trials.h5', 'w') as f:
        f.create_dataset('X', data=x, compression='gzip')
        f.create_dataset('y', data=y)
        f.create_dataset('trial_id', data=ids)
        f.create_dataset('recording_id', data=records, dtype=h5py.string_dtype())
        f.attrs['sfreq'] = sfreq
        f.attrs['tmin'] = tmin
        for i, fold in enumerate(folds):
            f.create_dataset(f'fold_{i}', data=fold)
    manifest = {'seed': config['seed'], 'protocol': 'GitClone class-stratified trial-level 5-fold CV; no independent test set',
                'num_folds': config['num_folds'], 'shape': list(x.shape),
                'sfreq': sfreq, 'tmin': tmin, 'channel_names': channels, 'label_names': names,
                'sources': sources, 'class_counts': np.bincount(y).tolist(),
                'folds': [{'fold': i, 'validation_trial_indices': fold.tolist(),
                           'validation_trial_keys': [keys[j] for j in fold],
                           'class_counts': np.bincount(y[fold], minlength=4).tolist()}
                          for i, fold in enumerate(folds)]}
    save_json(output / 'folds.json', manifest)
    print('Prepared trial folds:', [len(fold) for fold in folds], flush=True)
    return manifest
