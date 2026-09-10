import copy
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch
import h5py
import numpy as np
import torch
from bci_raspy_experiments.common import profile, read_json
from bci_raspy_experiments.datasets import _bci_trials, _raw_trials
from bci_raspy_experiments.engine import fit, loss_for, load_torch, seed_all
from bci_raspy_experiments.models import build_model, describe
from bci_raspy_experiments.presets import BASE, manifest
from bci_raspy_experiments.protocols import outer_splits, inner_split, validate
from bci_raspy_experiments.preprocessing import Pipeline
from bci_raspy_experiments.runner import run_task
from bci_raspy_experiments.windows import Bags, augment, starts_for
from bci_raspy_experiments.evaluation import summarize, aggregation


def fixture(root, count=40):
    root = Path(root)
    folder = root / 'cache/inhouse'
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []
    with h5py.File(folder / 'raw.h5', 'w') as f:
        for i in range(count):
            row = dict(id=f'inhouse/01/01/01/{i:04}', subject='01', session='01', run='01', trial=i,
                       onset=float(i*10), label=i%4, start=0., end=4., tmin=-2., sfreq=100.,
                       channels=['C3', 'C4', 'F3', 'F4'], eeg_count=4, exclusion=None)
            rows.append(row)
            x = rng.normal(size=(4, 600))*1e-5
            x[i%4] += 1e-5*np.sin(np.arange(600)*.6)
            f[row['id']] = x
    catalog = dict(hash='synthetic', inventory=dict(dataset='inhouse'), trials=rows)
    cfg = copy.deepcopy(BASE)
    cfg.update(dataset='inhouse', catalog_hash='synthetic', seed=42, preprocessing='filter_rms', sampling='center', smoke=True,
               split=dict(id='fixture', kind='trial', core=True, hash='fixture',
                          train=[r['id'] for r in rows[:32]], test=[r['id'] for r in rows[32:]]))
    cfg['train']['max_epochs'] = 2
    cfg['augmentation']['copies'] = 0
    p = dict(name='local', cache=str(root/'cache'), output=str(root/'output'), threads=1, workers=0, device='cpu')
    return catalog, cfg, p


class ProtocolTests(unittest.TestCase):
    def test_chronology_and_random(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalog, cfg, _ = fixture(tmp)
            splits = outer_splits(catalog['trials'], 'inhouse')
            self.assertEqual(len(splits), 9)
            self.assertEqual([len(s['train']) for s in splits[:4]], [20, 24, 28, 32])
            covered = [i for s in splits[4:] for i in s['test']]
            self.assertEqual(len(covered), len(set(covered)))
            for split in splits:
                tr, val = inner_split(catalog['trials'], split)
                self.assertFalse(set(tr+val) & set(split['test']))

    def test_bci_session_boundary_and_subjects(self):
        rows = []
        for subject in range(9):
            for session, count in [('T', 40), ('E', 44)]:
                for i in range(count):
                    rows.append(dict(id=f'bci2a/A{subject:02}/{session}/{i}', subject=f'A{subject:02}', session=session, label=i%4))
        splits = outer_splits(rows, 'bci2a')
        self.assertEqual(len(splits), 90)
        boundary = next(s for s in splits if s['id'] == 'A00-time-50')
        self.assertTrue(all('/T/' in i for i in boundary['train']))
        self.assertTrue(all('/E/' in i for i in boundary['test']))
        for s in splits[-9:]:
            validate(rows, s['train'], s['test'], True)
            inner_split(rows, s)

    def test_manifest_stable_deduplicated(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, _, _ = fixture(tmp)
            first = manifest({'inhouse': c})
            second = manifest({'inhouse': c})
            self.assertEqual(first, second)
            self.assertEqual(first['count'], len({r['id'] for r in first['tasks']}))

    def test_server_requires_paths(self):
        with self.assertRaisesRegex(ValueError, 'no fallback'):
            profile('server')


class ModelAndWindowTests(unittest.TestCase):
    def test_models_all_channels_and_depths(self):
        seed_all(42, 1)
        for family in ('raspy', 'standard'):
            for channels in (16, 22, 64):
                for extra in range(4):
                    spec = dict(BASE['model'], family=family, extra=extra)
                    model = build_model(spec, channels)
                    y = model(torch.randn(2, channels, 100, 1))
                    self.assertEqual(tuple(y.shape), (2, 4))
                    y.square().mean().backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        for family in ('raspy', 'standard'):
            m = build_model(dict(BASE['model'], family=family, extra=3), 16, 400)
            self.assertEqual(tuple(m(torch.randn(2, 16, 400, 1)).shape), (2, 4))

    def test_raspy_identity(self):
        from bci_raspy_experiments.vendor.eegnet import EEGNet
        from bci_raspy_experiments.models import raspy_config
        seed_all(42, 1)
        original = EEGNet(raspy_config(BASE['model'], 100), output_dim=4, n_electrodes=16).eval()
        new = build_model(BASE['model'], 16).eval()
        new.load_state_dict(original.state_dict())
        x = torch.randn(3, 16, 100, 1)
        torch.testing.assert_close(original(x), new(x), rtol=0, atol=0)

    def test_mean_mil_gradient_permutation_and_distinct_loss(self):
        torch.manual_seed(7)
        z = torch.randn(2, 7, 4, requires_grad=True)
        y, weights = torch.tensor([1, 3]), torch.ones(4)
        bag_loss = loss_for(z, y, 'mil', 'mse', weights)
        instance_loss = loss_for(z, y, 'window', 'mse', weights)
        self.assertGreater(float(instance_loss.detach()), float(bag_loss.detach()))
        torch.testing.assert_close(bag_loss, loss_for(z[:, [6, 0, 1, 2, 3, 4, 5]], y, 'mil', 'mse', weights))
        bag_loss.backward()
        self.assertTrue((z.grad.abs().sum(-1) > 0).all())
        torch.testing.assert_close(z.grad[:, 0], z.grad[:, 1])

    def test_frozen_noise_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, cfg, p = fixture(tmp)
            pipe = Pipeline(c, p['cache'], 'filter_rms').fit(cfg['split']['train'])
            path = Path(tmp)/'processed.h5'
            kept = pipe.materialize(cfg['split']['train'], path)
            no = Bags(path, kept, c['trials'], cfg, training=True)
            other_cfg = copy.deepcopy(cfg)
            other_cfg['augmentation']['copies'] = 4
            yes = Bags(path, kept, c['trials'], other_cfg, training=True)
            self.assertEqual(len(no), len(yes))
            np.testing.assert_array_equal(yes[3]['x'], yes[3]['x'])
            evaluation = Bags(path, kept, c['trials'], other_cfg)
            self.assertEqual(evaluation[0]['replica'], 0)
            self.assertEqual(evaluation[0]['x'].shape[0], 31)
        x = np.arange(24).reshape(3, 8).astype(float)
        a = augment(x, 'id', 0., 1, 42, 1.)
        np.testing.assert_array_equal(a, augment(x, 'id', 0., 1, 42, 1.))
        self.assertLessEqual(np.abs(a-x).max(), x.max()*.5)
        np.testing.assert_allclose(augment(x, 'id', 0., 1, 42, .5)-x, .5*(a-x))
        row = dict(start=1., end=5.)
        self.assertEqual([len(starts_for(row, s)[0]) for s in ('center', 'sparse', 'medium', 'dense')], [1, 4, 7, 31])


class IsolationAndResumeTests(unittest.TestCase):
    def test_resume_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, cfg, p = fixture(tmp)
            pipe = Pipeline(c, p['cache'], 'filter_rms').fit(cfg['split']['train'])
            path = Path(tmp)/'processed.h5'
            ids = pipe.materialize(cfg['split']['train'], path)
            bags = Bags(path, ids, c['trials'], cfg, training=True)
            full, _ = fit(bags, None, cfg, p, Path(tmp)/'full', fixed_schedule=[.001, .0005])
            with self.assertRaises(InterruptedError):
                fit(bags, None, cfg, p, Path(tmp)/'resume', fixed_schedule=[.001, .0005], stop_after=1)
            resumed, _ = fit(bags, None, cfg, p, Path(tmp)/'resume', fixed_schedule=[.001, .0005], resume=True)
            for key in full.state_dict():
                torch.testing.assert_close(full.state_dict()[key], resumed.state_dict()[key], rtol=0, atol=0)

    def test_strict_test_labels_and_signals_do_not_affect_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, cfg, p = fixture(tmp)
            task = dict(id='isolation', config=cfg, suites=['test'])
            run_task(task, c, p)
            initial = Path(p['output'])/'smoke/isolation'
            reference = load_torch(initial/'model.pt')['model']
            training = read_json(initial/'training.json')['selected_schedule']
            scales = read_json(initial/'final/pipeline.json')['rms']
            for mutation in ('labels', 'signals'):
                changed = copy.deepcopy(c)
                for row in changed['trials'][32:]:
                    row['label'] = (row['label'] + 1) % 4 if mutation == 'labels' else row['label']
                mutation_cache = Path(tmp)/mutation/'cache'
                (mutation_cache/'inhouse').mkdir(parents=True)
                shutil.copyfile(Path(p['cache'])/'inhouse/raw.h5', mutation_cache/'inhouse/raw.h5')
                if mutation == 'signals':
                    with h5py.File(mutation_cache/'inhouse/raw.h5', 'a') as f:
                        for trial in cfg['split']['test']:
                            f[trial][:] *= 8
                runtime = dict(p, output=str(Path(tmp)/mutation/'output'), cache=str(mutation_cache))
                run_task(task, changed, runtime)
                result = Path(runtime['output'])/'smoke/isolation'
                for key, value in load_torch(result/'model.pt')['model'].items():
                    torch.testing.assert_close(reference[key], value, rtol=0, atol=0)
                self.assertEqual(training, read_json(result/'training.json')['selected_schedule'])
                self.assertEqual(scales, read_json(result/'final/pipeline.json')['rms'])

    def test_selection_can_choose_different_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, cfg, p = fixture(tmp)
            pipe = Pipeline(c, p['cache'], 'filter_rms').fit(cfg['split']['train'])
            path = Path(tmp)/'processed.h5'
            ids = pipe.materialize(cfg['split']['train'], path)
            bags = Bags(path, ids, c['trials'], cfg, training=True)
            val = Bags(path, ids, c['trials'], cfg)
            # Controlled external selection signals: same training, opposite
            # loss trajectory must select different actual model checkpoints.
            with patch('bci_raspy_experiments.engine.validation', side_effect=[(2., .25), (1., .25)]):
                a, ma = fit(bags, val, cfg, p, Path(tmp)/'a')
            with patch('bci_raspy_experiments.engine.validation', side_effect=[(1., .25), (2., .25)]):
                b, mb = fit(bags, val, cfg, p, Path(tmp)/'b')
            self.assertEqual((ma['best_epoch'], mb['best_epoch']), (1, 0))
            self.assertTrue(any(not torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()))


class RealAdaptersTests(unittest.TestCase):
    def test_inhouse_excludes_auxiliary_channels(self):
        path = Path(profile('local')['inhouse']) / 'hyz_1/hyz_1.vhdr'
        if not path.exists():
            self.skipTest('Inhouse raw data unavailable')
        row, x = next(_raw_trials(path, 'inhouse'))
        self.assertEqual(row['eeg_count'], 16)
        self.assertFalse({'x_dir', 'y_dir', 'z_dir'} & set(row['channels']))
        self.assertEqual(x.shape, (16, 1750))

    def test_public_samples_when_available(self):
        bci = Path('D:/ServerTransfer/PublicData/bci_competition_iv_2a/A01E.mat')
        physio = Path('D:/ServerTransfer/PublicData/physionet_mi/S001')
        if not bci.exists() or not physio.exists():
            self.skipTest('Local public samples unavailable')
        trials = list(_bci_trials(bci))
        self.assertEqual(len(trials), 288)
        self.assertEqual({r['label'] for r, _ in trials}, {0, 1, 2, 3})
        self.assertEqual(trials[0][1].shape, (25, 1500))
        self.assertLess(float(np.median(np.abs(trials[0][1]))), .001)
        mi = [item for run in (4, 6) for item in _raw_trials(physio/f'S001R{run:02}.edf', 'physionet')]
        self.assertEqual({r['label'] for r, _ in mi}, {0, 1, 2, 3})
        self.assertTrue(all(x.shape == (64, 960) for r, x in mi if x is not None))


if __name__ == '__main__':
    unittest.main()
