"""Standalone entrypoint: python run_experiments.py --help."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from bci_raspy_experiments.__main__ import main
if __name__ == '__main__':
    main()
