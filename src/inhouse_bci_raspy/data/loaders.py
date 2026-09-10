"""Convert prepared window blocks into the DataLoaders consumed by the training engine."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(block: dict, settings: dict, prefix: str) -> DataLoader:
    """Build a train/val loader using the original batch, shuffle and worker settings.

    ``block`` contains float32 X[N,C,T,1] and int64 y[N]. ``prefix`` is train or val.
    This function does not create or alter any trial partition.
    """
    workers = settings[f'{prefix}_num_workers']
    return DataLoader(TensorDataset(torch.from_numpy(block['X']), torch.from_numpy(block['y'])),
        batch_size=settings[f'{prefix}_batch_size'], shuffle=settings[f'{prefix}_shuffle'],
        drop_last=settings[f'{prefix}_drop_last'], num_workers=workers,
        prefetch_factor=settings[f'{prefix}_prefetch_factor'] if workers else None)


def select_smoke_samples(block: dict) -> None:
    """Keep the first eight windows of each class in-place for the two-epoch smoke check.

    Every metadata array is sliced with the same indices. Formal CV never calls this.
    """
    selected = np.concatenate([np.flatnonzero(block['y'] == label)[:8] for label in range(4)])
    for key in block:
        block[key] = block[key][selected]
