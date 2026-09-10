"""Locate the project and load experiment/data configuration without importing ML libraries."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_settings(root: Path = PROJECT_ROOT) -> tuple[dict, dict]:
    """Read experiment YAML and marker JSON; require the agreed five-fold protocol.

    Args:
        root: Directory containing ``config/``. Configuration paths remain relative to it.

    Returns:
        Experiment configuration and BrainVision/BIDS dataset configuration.

    Raises:
        ValueError: Either configuration requests a fold count other than five.
    """
    import yaml

    config = yaml.safe_load((root / 'config/experiment.yaml').read_text(encoding='utf-8'))
    dataset = json.loads((root / 'config/dataset.json').read_text(encoding='utf-8'))
    if config['num_folds'] != 5 or config['training']['num_folds'] != 5:
        raise ValueError('This experiment uses the GitClone five-fold protocol')
    return config, dataset
