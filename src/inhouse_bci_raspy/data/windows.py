"""Generate one-second windows, training noise copies, and resampled arrays."""

import numpy as np
from scipy.signal import resample


def make_windows(data, y, indices, sfreq, tmin, settings, copies, rng):
    """Generate resampled windows and aligned trial/window/augmentation metadata.
    
    Args:
        data: Normalized epochs [trials, channels, samples].
        y: Integer class per epoch.
        indices: Epoch indices assigned to this split.
        sfreq: Original samples per second.
        tmin: Time in seconds of each epoch first sample.
        settings: Window start/step/duration and output sampling rate.
        copies: One for validation; five for training including the unmodified window.
        rng: Seeded NumPy Generator used only for added uniform noise.
    Returns:
        X[N,C,T,1], y[N], trial_index, window_start_s and augmentation_id.
    Raises:
        ValueError: Any requested window extends outside an epoch."""
    starts = np.arange(settings['start_s'], settings['last_start_s'] + 1e-9, settings['step_s'])
    samples = int(round(settings['duration_s'] * sfreq))
    target = int(round(settings['duration_s'] * settings['output_sfreq']))
    shape = (len(indices) * len(starts) * copies, data.shape[1], target, 1)
    x = np.empty(shape, dtype=np.float32)
    out_y = np.empty(shape[0], dtype=np.int64)
    trial_index = np.empty(shape[0], dtype=np.int64)
    window_start = np.empty(shape[0], dtype=np.float64)
    aug_id = np.empty(shape[0], dtype=np.int8)
    cursor = 0
    for i in indices:
        for start in starts:
            first = int(round((start - tmin) * sfreq))
            if first < 0 or first + samples > data.shape[2]:
                raise ValueError(f'Incomplete window: trial {i}, start {start}')
            window = data[i, :, first:first + samples]
            for aug in range(copies):
                current = window if aug == 0 else window + np.max(window) * rng.uniform(-0.5, 0.5, window.shape)
                x[cursor] = resample(current, target, axis=1).astype(np.float32)[..., None]
                out_y[cursor], trial_index[cursor] = y[i], i
                window_start[cursor], aug_id[cursor] = start, aug
                cursor += 1
    return {'X': x, 'y': out_y, 'trial_index': trial_index,
            'window_start_s': window_start, 'augmentation_id': aug_id}
