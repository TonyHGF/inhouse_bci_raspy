"""Configure output caches, subprocess environments, devices, and reproducibility."""

import os
from pathlib import Path
import random
import sys


def setup_runtime(root: Path) -> Path:
    """Set the project working directory and writable caches before importing MNE/torch.

    Returns the output directory. This sets process environment variables only;
    it never creates a conda environment or installs packages.
    """
    os.chdir(root)
    output = root / 'outputs'
    output.mkdir(exist_ok=True)
    for key, directory in [('MPLCONFIGDIR', 'matplotlib'), ('_MNE_FAKE_HOME_DIR', 'mne_home')]:
        cache = output / directory
        cache.mkdir(exist_ok=True)
        os.environ[key] = str(cache)
    os.environ.update(PYTHONUTF8='1', MPLBACKEND='Agg', PYVISTA_OFF_SCREEN='true',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8')
    if os.name == 'nt':
        prefix = Path(sys.prefix)
        os.environ['PATH'] = os.pathsep.join(map(str, (prefix, prefix / 'Library/bin', prefix / 'Scripts'))) + os.pathsep + os.environ['PATH']
    return output


def setup_training(seed: int, device_name: str):
    """Seed all training RNGs once and return the selected torch device.

    ``auto`` chooses CUDA when available. Seeds are not reset between folds,
    matching the existing training orchestration.
    """
    import numpy as np
    import torch

    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device_name == 'auto' else device_name)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    return device
