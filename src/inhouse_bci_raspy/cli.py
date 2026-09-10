"""Command-line parsing and stage dispatch; computations live in dedicated modules."""

import argparse
from pathlib import Path
from .runtime import setup_runtime
from .settings import PROJECT_ROOT, load_settings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a stage and optional existing-epoch root/device without side effects."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['preprocess', 'prepare', 'train', 'all', 'smoke'])
    parser.add_argument('--clean-root', type=Path, help='Existing MNE derivatives root containing sub-01')
    parser.add_argument('--device', default=None, help='auto, cpu, cuda, or cuda:0')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Initialize runtime and execute requested stages in pipeline order.

    Resolve CLI paths before changing directory. Import heavy libraries after cache
    setup, allowing --help to run without installed ML dependencies.
    """
    args = parse_args(argv)
    supplied_clean_root = args.clean_root.resolve() if args.clean_root else None
    output = setup_runtime(PROJECT_ROOT)
    config, dataset = load_settings()
    if args.stage in ('preprocess', 'all'):
        from .preprocessing.pipeline import preprocess
        preprocess(config, dataset, output)
    if args.stage in ('prepare', 'all'):
        from .data.preparation import prepare
        clean_root = supplied_clean_root or PROJECT_ROOT / dataset['bids_root'] / 'derivatives/mne-bids-pipeline'
        prepare(config, dataset, clean_root, output / 'prepared')
    if args.stage in ('train', 'all', 'smoke'):
        from .training.cross_validation import train_folds
        train_folds(config, output, args.device or config['device'], smoke=args.stage == 'smoke')
