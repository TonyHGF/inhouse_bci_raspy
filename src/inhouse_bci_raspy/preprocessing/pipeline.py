"""Run BrainVision conversion and MNE-BIDS-Pipeline once per configured recording."""

import os
from pathlib import Path
import subprocess
import sys

from ..io import save_json
from ..settings import PROJECT_ROOT


def run_logged(command: list[str], log: Path, env: dict, root: Path) -> None:
    """Run a preprocessing subprocess, capture its full log, and propagate failures."""
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Preprocessing failed; see {log}')


def preprocess(config: dict, dataset: dict, output: Path, root: Path = PROJECT_ROOT) -> None:
    """Create per-recording configs and execute conversion followed by official cleaning.

    Args:
        config: Experiment settings, including recording paths, sessions and counts.
        dataset: Shared marker mapping and BIDS settings.
        output: Experiment output root; logs/configs are saved in ``preprocessing/``.
        root: Base directory used to resolve configured source paths.

    Raw files are read only. A failed recording stops the pipeline with its log path.
    """
    logs = output / 'preprocessing'
    logs.mkdir(parents=True, exist_ok=True)
    module_dir = Path(__file__).resolve().parent
    for rec in config['recordings']:
        cfg = dict(dataset)
        cfg.update(source_vhdr=str((root / rec['source_vhdr']).resolve()), session=rec['session'],
                   expected_event_counts=dict(zip(dataset['event_id'], rec['counts'])))
        path = logs / f"{rec['id']}.json"
        save_json(path, cfg)
        env = os.environ.copy()
        env['EEG_DATASET_CONFIG'] = str(path)
        commands = {
            'convert': [sys.executable, str(module_dir / 'brainvision.py'), '--config', str(path)],
            'pipeline': [sys.executable, '-c', 'from mne_bids_pipeline._main import main; main()',
                         '--config', str(module_dir / 'mne_config.py')],
        }
        for step, command in commands.items():
            log = logs / f"{rec['id']}_{step}.log"
            print(f"Running {rec['id']} {step}: {log}", flush=True)
            run_logged(command, log, env, root)
