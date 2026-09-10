"""Numerical output verification, without additional training."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from bci_raspy_experiments.common import save_json, read_json
from bci_raspy_experiments.presets import BASE
from bci_raspy_experiments.evaluation import summarize, aggregation, save_predictions, load_predictions
from bci_raspy_experiments.reporting import report


class ReportTests(unittest.TestCase):
    def test_roundtrip_and_predeclared_subsets(self):
        y = np.arange(8)%4
        ids = np.array([f'inhouse/01/01/01/{i:04}' for i in range(8)])
        logits = np.repeat(np.eye(4)[y, None, :], 31, axis=1).astype('float32')
        data = dict(ids=ids, y=y, logits=logits, starts=np.repeat(np.arange(31)[None, :]*.1, 8, 0))
        rows = [dict(id=i, subject='01') for i in ids]
        metrics = summarize(data, rows)
        self.assertEqual(metrics['trial_probability']['accuracy'], 1.)
        result = aggregation(data, 10)
        self.assertEqual(len(result['consecutive_four']), 28)
        self.assertEqual(result['four_nonoverlapping_accuracy'], 1.)
        self.assertTrue(all(v['mean']==1 for v in result['random_subsets'].values()))
        with tempfile.TemporaryDirectory() as tmp:
            save_predictions(Path(tmp)/'predictions', data)
            self.assertEqual(metrics, summarize(load_predictions(Path(tmp)/'predictions.npz'), rows))

    def test_common_cohort_and_subject_weighting(self):
        with tempfile.TemporaryDirectory() as tmp:
            for index, n in enumerate([8, 4]):
                directory = Path(tmp)/'runs'/str(index)
                directory.mkdir(parents=True)
                cfg = copy.deepcopy(BASE)
                cfg.update(dataset='inhouse', seed=42, catalog_hash='fixture',
                           split=dict(id='01-random-0', hash='fixed', kind='trial'))
                cfg['preprocessing'] = 'full' if index == 0 else 'filter_rms'
                ids = np.array([f'inhouse/01/01/01/{i:04}' for i in range(n)])
                y = np.arange(n)%4
                data = dict(ids=ids, y=y, logits=np.repeat(np.eye(4)[y,None,:],31,axis=1), starts=np.zeros((n,31)))
                metrics = summarize(data, [dict(id=i, subject='01') for i in ids])
                save_predictions(directory/'test_predictions', data)
                save_json(directory/'task.json', dict(task=dict(id=str(index), config=cfg)))
                save_json(directory/'status.json', dict(state='complete'))
                save_json(directory/'metrics.json', dict(test=metrics, retained_test=n, requested_test=8,
                          retained_train=32, requested_train=32, parameters=100, seconds=1., peak_cuda_bytes=0))
            value = report(tmp)
            self.assertEqual(value['completed_runs'], 2)
            self.assertEqual(len(read_json(Path(tmp)/'reports/cohorts.json')[0]['common_ids']), 4)
            self.assertIn('preprocessing', (Path(tmp)/'reports/paired_differences.csv').read_text())
            self.assertTrue((Path(tmp)/'reports/window_trial.png').exists())
