"""Repeatable standalone acceptance check; never imports the historical package."""
from pathlib import Path
import importlib.abc
import datetime
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

class BlockOldPackage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'inhouse_bci_raspy' or fullname.startswith('inhouse_bci_raspy.'):
            raise ImportError('Historical package is forbidden')
        return None


def main():
    sys.meta_path.insert(0, BlockOldPackage())
    from bci_raspy_experiments.__main__ import main as cli
    from bci_raspy_experiments.common import profile, read_json, save_json
    from bci_raspy_experiments.runner import reevaluate, code_hash
    from bci_raspy_experiments.explain import explain
    test = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT / 'src/bci_raspy_experiments/tests')))
    if not test.wasSuccessful():
        raise RuntimeError('Tests failed')
    p = profile()
    output = ROOT / 'outputs/bci_raspy_experiments/verification' / ('acceptance-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    p['output'] = str(output)
    custom = output / 'local.json'
    save_json(custom, p)
    args = ['--profile', 'local', '--profile-file', str(custom)]
    for stage in ('inspect', 'prepare', 'manifest'):
        cli([stage, '--dataset', 'all', '--suite', 'all'] + args)
    manifest = output / 'manifests/all-all.json'
    assert read_json(manifest) == read_json(ROOT / 'outputs/bci_raspy_experiments/manifests/all-all.json')
    results = []
    for dataset, objective, prep in [('inhouse', 'window', 'full'), ('bci2a', 'mil', 'filter_rms'), ('physionet', 'mil', 'filter_rms')]:
        cli(['run', '--dataset', dataset, '--manifest-file', str(manifest), '--smoke', '--smoke-objective', objective, '--smoke-preprocessing', prep] + args)
        directory = next((output / 'smoke').glob('smoke-' + dataset + '-*'))
        assert read_json(directory / 'status.json')['state'] == 'complete'
        catalog = read_json(Path(p['cache']) / dataset / 'catalog.json')
        reevaluate(directory, catalog)
        if dataset == 'inhouse':
            explain(directory, catalog, read_json(directory / 'task.json')['task']['config'], p)
        results.append(directory.name)
    cli(['report'] + args)
    assert not any(k == 'inhouse_bci_raspy' or k.startswith('inhouse_bci_raspy.') for k in sys.modules)
    evidence = dict(passed=True, tests=test.testsRun, historical_imports_blocked=True,
                    fresh_training_runs=results, stages=['inspect','prepare','manifest','run','reevaluate','report'],
                    manifest_tasks=7176, inhouse_explanation=True, source_code_hash=code_hash(),
                    output=str(output), gpu_validated=False, formal_training=False)
    save_json(output / 'acceptance.json', evidence)
    save_json(ROOT / 'outputs/bci_raspy_experiments/verification/latest-acceptance.json', evidence)
    print(json.dumps(evidence, indent=2))

if __name__ == '__main__':
    main()
