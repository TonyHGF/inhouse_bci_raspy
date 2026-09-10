"""Protocol and boundary checks; run with unittest, no additional test package."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inhouse_bci_raspy.data.folds import build_fold
from inhouse_bci_raspy.data.partition import partition_data


class ProtocolTests(unittest.TestCase):
    def test_partition_matches_gitclone(self):
        # Generated from the original checkout before archival; includes source SHA256.
        fixture = json.loads((ROOT / 'tests/fixtures/gitclone_partition_reference.json').read_text(encoding='utf-8'))
        labels = [(label, None) for label, count in enumerate(fixture['class_counts']) for _ in range(count)]
        expected = fixture['folds']
        np.random.seed(fixture['seed'])
        actual = partition_data(labels, fixture['num_folds'])
        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left, right)

    def test_disjoint_balanced_and_complete(self):
        labels = [(label, None) for label, count in enumerate([58, 58, 60, 59]) for _ in range(count)]
        np.random.seed(42)
        folds = partition_data(labels, 5)
        self.assertEqual(sorted(np.concatenate(folds).tolist()), list(range(len(labels))))
        y = np.asarray([label[0] for label in labels])
        counts = np.asarray([np.bincount(y[fold], minlength=4) for fold in folds])
        self.assertTrue(np.all(counts.max(axis=0) - counts.min(axis=0) <= 1))

    def test_validation_does_not_fit_rms_or_get_augmented(self):
        # A huge validation-only offset must never change the fitted training RMS.
        data = np.ones((20, 2, 701), dtype=np.float64)
        y = np.tile(np.arange(4), 5)
        folds = np.arange(20).reshape(5, 4)
        data[folds[0]] *= 1000
        config = {'seed': 42, 'num_folds': 5, 'window': {'start_s': 1, 'last_start_s': 4,
            'step_s': 0.1, 'duration_s': 1, 'output_sfreq': 100, 'num_noise': 4}}
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / 'trials.h5'
            with h5py.File(cache, 'w') as f:
                f['X'], f['y'] = data, y
                f.attrs['sfreq'], f.attrs['tmin'] = 100., -2.
                for i, fold in enumerate(folds):
                    f[f'fold_{i}'] = fold
            train, val, meta = build_fold(cache, config, 0)
        np.testing.assert_allclose(meta['normalization_scales'], [1, 1])
        self.assertEqual(train['X'].shape, (16 * 31 * 5, 2, 100, 1))
        self.assertEqual(val['X'].shape, (4 * 31, 2, 100, 1))
        self.assertEqual(set(val['augmentation_id']), {0})
        self.assertEqual(set(train['augmentation_id']), set(range(5)))
        self.assertFalse(set(train['trial_index']) & set(val['trial_index']))
        np.testing.assert_allclose(val['X'], 1000)

    def test_real_prepared_trials_if_present(self):
        path = ROOT / 'outputs/prepared/clean_trials.h5'
        if not path.is_file():
            self.skipTest('Run prepare to check real outputs')
        with h5py.File(path, 'r') as f:
            folds = [f[f'fold_{i}'][:] for i in range(5)]
            keys = list(zip(f['recording_id'].asstr()[:], f['trial_id'][:]))
            self.assertEqual(len(keys), len(set(keys)))
            self.assertEqual(sorted(np.concatenate(folds).tolist()), list(range(len(keys))))
            self.assertEqual(set(f['y'][:]), {0, 1, 2, 3})
            self.assertEqual(f['X'].shape[1], 16)
            self.assertTrue(np.isfinite(f['X'][:]).all())


if __name__ == '__main__':
    unittest.main()
