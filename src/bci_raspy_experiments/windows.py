"""Lazy trial bags and identity-seeded, frozen augmentation."""
import hashlib
import h5py
import numpy as np
import torch
from scipy.signal import resample
from torch.utils.data import Dataset, DataLoader


def starts_for(row, sampling):
    start, end = row['start'], row['end']
    if sampling == 'whole':
        return np.array([start]), end-start
    if sampling == 'center':
        return np.array([(start + end - 1) / 2]), 1.
    step = {'sparse': 1., 'medium': .5, 'dense': .1}[sampling]
    return np.round(np.arange(start, end-1+1e-8, step), 6), 1.


def identity_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')


def augment(window, trial_id, start, replica, seed, strength):
    if replica == 0:
        return window
    rng = np.random.default_rng(identity_seed(seed, trial_id, start, replica))
    return window + strength * np.max(window) * rng.uniform(-.5, .5, window.shape)


class Bags(Dataset):
    def __init__(self, path, ids, rows, config, training=False):
        self.path, self.ids, self.rows = str(path), list(ids), {r['id']: r for r in rows}
        self.config, self.training = config, training
        aug = config['augmentation']
        self.repeats = (5 if aug['budget'] == 'matched' else aug['copies']+1) if training else 1
        self.sampling = config['sampling'] if training or config['objective'] == 'whole' else 'dense'

    def __len__(self):
        return len(self.ids) * self.repeats

    def __getitem__(self, index):
        trial_id = self.ids[index // self.repeats]
        slot = index % self.repeats
        row, cfg = self.rows[trial_id], self.config
        aug = cfg['augmentation']
        replica = 0
        if self.training:
            if aug['budget'] == 'expanded':
                replica = slot
            else:
                replica = int(np.random.default_rng(identity_seed(cfg['seed'], trial_id, slot, 'choice')).integers(aug['copies'] + 1))
        with h5py.File(self.path, 'r') as f:
            x = f['trials/' + trial_id][:]
        starts, duration = starts_for(row, self.sampling)
        windows = []
        for start in starts:
            first = int(round((start - row['tmin']) * row['sfreq']))
            count = int(round(duration * row['sfreq']))
            window = x[:, first:first+count]
            if window.shape[1] != count:
                raise ValueError('Window extends outside valid trial')
            window = augment(window, trial_id, float(start), replica, cfg['seed'], aug['strength'])
            windows.append(resample(window, int(round(duration * 100)), axis=1).astype('float32')[..., None])
        return dict(x=np.stack(windows), y=row['label'], id=trial_id, start=starts, replica=replica)


def loader(bags, config, profile, epoch=0, shuffle=False):
    generator = torch.Generator().manual_seed(config['seed'] + epoch)
    return DataLoader(bags, batch_size=config['train']['batch_trials'], shuffle=shuffle,
                      num_workers=profile['workers'], drop_last=False, generator=generator)
