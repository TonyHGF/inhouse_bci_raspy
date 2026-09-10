"""Numerical and data-integrity tests for frozen-model channel explanations."""

import csv
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from inhouse_bci_raspy.analysis.channel_ablation import (
    infer, r2_outputs, summarize_outputs, ablate_channels, weighted_moments)
from inhouse_bci_raspy.analysis.channel_inputs import verify_baseline
from inhouse_bci_raspy.analysis.channel_kernels import extract_kernels
from inhouse_bci_raspy.models.eegnet import EEGNet


class ToyModel(torch.nn.Module):
    """Read four class scores directly from time coordinates in channel zero."""
    def __init__(self, distractor=False):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.distractor = distractor
    def forward(self, x):
        return self.scale * (x[:,0,:,0] + (x[:,1,:,0] if self.distractor else 0))


def toy_block():
    """Two windows per trial, eight balanced trials and two input channels."""
    labels = np.repeat(np.arange(8) % 4, 2)
    x = np.zeros((16,2,4,1), dtype=np.float32)
    x[:,0,:,0] = np.eye(4)[labels]
    return {'X':x,'y':labels,'trial_index':np.repeat(np.arange(8),2),
            'window_start_s':np.tile([1.,1.1],8)}


class AblationTests(unittest.TestCase):
    def test_cli_help_on_windows_legacy_console_encoding(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT/'src'), PYTHONIOENCODING='gbk')
        result = subprocess.run([sys.executable,'-m','inhouse_bci_raspy.visualize.topomap','--help'],
                                env=env,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr.decode('gbk'))
        self.assertIn(b'--plot-only',result.stdout)

    def test_r2_matches_sklearn_and_rejects_constant(self):
        y = np.eye(4)[np.arange(12)%4]
        z = np.random.default_rng(7).normal(size=y.shape)
        np.testing.assert_allclose(r2_outputs(y,z), r2_score(y,z,multioutput='raw_values',force_finite=False))
        with self.assertRaisesRegex(ValueError,'constant'):
            r2_outputs(np.ones((5,4)),np.zeros((5,4)))

    def test_known_useful_and_unused_channel(self):
        model = ToyModel().eval()
        block = toy_block()
        original = block['X'].copy()
        baseline = summarize_outputs(block,*infer(model,block['X'],3))
        results = ablate_channels(model,block,baseline,batch_size=5)
        self.assertEqual(baseline['trial_accuracy_percent'],100.)
        np.testing.assert_allclose(results[0]['trial_delta_r2_per_class'],4/3)
        self.assertEqual(results[1]['trial_delta_r2'],0.)
        np.testing.assert_array_equal(block['X'],original)
        self.assertEqual(model.scale.item(),1.)

    def test_harmful_channel_produces_negative_importance(self):
        block = toy_block()
        block['X'][:,1,:,0] = 2*np.roll(block['X'][:,0,:,0],1,axis=1)
        model = ToyModel(distractor=True).eval()
        baseline = summarize_outputs(block,*infer(model,block['X']))
        result = ablate_channels(model,block,baseline)[1]
        self.assertLess(result['trial_delta_r2'],0)
        self.assertEqual(result['trial_accuracy_percent'],100.)

    def test_probability_average_is_separate_from_mean_logits(self):
        # Average logits picks class 0; average probabilities picks class 1.
        block = {'y':np.repeat(np.arange(4),3), 'trial_index':np.repeat(np.arange(4),3),
                 'window_start_s':np.tile([1.,1.1,1.2],4)}
        z = np.tile(np.array([[100,0,-100,-100],[0,3,-100,-100],[0,3,-100,-100]],dtype=np.float32),(4,1))
        p = torch.from_numpy(z).softmax(1).numpy()
        result = summarize_outputs(block,z,p)
        np.testing.assert_array_equal(result['trial_logits'].argmax(1),np.zeros(4))
        np.testing.assert_array_equal(result['trial_probabilities'].argmax(1),np.ones(4))

    def test_duplicate_window_is_rejected(self):
        block = toy_block()
        z,p = infer(ToyModel().eval(),block['X'])
        block['window_start_s'][1] = 1.
        with self.assertRaisesRegex(ValueError,'duplicate/missing'):
            summarize_outputs(block,z,p)

    def test_weighted_sd_is_descriptive(self):
        mean,sd = weighted_moments(np.array([[1.,3.],[3.,7.]]),np.array([1.,3.]))
        np.testing.assert_allclose(mean,[2.5,6.])
        np.testing.assert_allclose(sd,np.sqrt([.75,3.]))

    def test_baseline_verification_detects_tampering(self):
        block = toy_block()
        z,p = infer(ToyModel().eval(),block['X'])
        metrics = summarize_outputs(block,z,p)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root/'cv/fold_0'
            directory.mkdir(parents=True)
            (directory/'metrics.json').write_text(json.dumps({'validation_accuracy_percent':100.,
                'validation_trial_accuracy_percent':100.}))
            with (directory/'validation_predictions.csv').open('w',newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['trial_index','window_start_s','true_label','predicted_label','p_0','p_1','p_2','p_3'])
                for i in range(len(p)):
                    writer.writerow([block['trial_index'][i],block['window_start_s'][i],block['y'][i],p[i].argmax(),*p[i]])
            self.assertTrue(verify_baseline(root,0,block,p,metrics)['passed'])
            damaged = p.copy()
            damaged[0,0] += .01
            with self.assertRaisesRegex(ValueError,'prediction mismatch'):
                verify_baseline(root,0,block,damaged,metrics)


class KernelTests(unittest.TestCase):
    def test_composed_kernel_matches_actual_linear_block_and_preserves_state(self):
        torch.manual_seed(31)
        config = {'sampling_frequency':100,'window_length':1000,'num_temporal_filters':2,
            'num_spatial_filters':2,'block1':{'conv':[1,5],'max_norm_value':1.,'eps':.01,'avg_pool':[1,3],'dropout':.5},
            'block2':{'sep_conv':[1,3],'avg_pool':[1,3],'dropout':.5,'max_norm_value':.25,'eps':.01}}
        model = EEGNet(config,output_dim=4,n_electrodes=3).eval()
        with torch.no_grad():
            model._batchnorm1.weight.copy_(torch.tensor([1.4,-.7]))
            model._batchnorm2.weight.copy_(torch.tensor([.5,1.2,-.8,1.1]))
            model._batchnorm1.running_var.copy_(torch.tensor([.8,1.7]))
            model._batchnorm2.running_var.copy_(torch.tensor([.5,1.2,1.8,.4]))
        before={k:v.clone() for k,v in model.state_dict().items()}
        result=extract_kernels(model)
        x=torch.randn(2,1,3,100)
        def linear(a):
            return model._batchnorm2(model._depthwise(model._batchnorm1(model._conv1(a))))
        with torch.no_grad():
            actual = linear(x)-linear(torch.zeros_like(x))
            expected = F.conv2d(x,torch.from_numpy(result['spatiotemporal'])[:,None],padding=(0,2))
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        self.assertTrue(np.all(result['channel_energy_fraction'] >= 0))
        self.assertAlmostEqual(result['channel_energy_fraction'].sum(),1.)
        for k,v in model.state_dict().items():
            self.assertTrue(torch.equal(before[k],v))
        energy=result['spatiotemporal'].astype(float)**2
        np.testing.assert_allclose(result['channel_energy'],energy.sum((0,2)))


if __name__ == '__main__':
    unittest.main()
