"""Protocol checks using synthetic, bounded CPU inputs."""
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from inhouse_comparison import aligned_splits, normalize_old, fit_model, read_json, PROJECT

class AlignmentTests(unittest.TestCase):
    def test_outer_training_is_entire_non_test_pool(self):
        splits=aligned_splits(np.tile(np.arange(4),60),42)
        tests=[]
        for s in splits:
            self.assertEqual(set(s),{'fold','outer_training','test'})
            self.assertEqual(set(s['outer_training']),set(range(240))-set(s['test']))
            self.assertFalse(set(s['outer_training'])&set(s['test']))
            self.assertEqual(len(s['outer_training']),192)
            tests+=s['test']
        self.assertEqual(sorted(tests),list(range(240)))

    def test_rms_does_not_use_held_out_signals(self):
        data={i:np.full((16,10),i+1.) for i in range(4)}
        a,ma=normalize_old(data,[0,1]); data[3]*=100
        b,mb=normalize_old(data,[0,1])
        self.assertEqual(ma,mb)
        np.testing.assert_array_equal(a[0],b[0])

    def test_fixed_training_has_no_selection_and_matches_first_epoch(self):
        rng=np.random.default_rng(7)
        block={'X':rng.normal(size=(8,16,100,1)).astype('float32'), 'y':np.tile(np.arange(4),2), 'trial_index':np.arange(8)}
        settings=read_json(PROJECT/'outputs/cv/experiment.json'); settings['training']['max_epochs']=2
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            _,selected=fit_model(block,block,settings,42,root/'selected',device_name='cpu')
            _,refit=fit_model(block,None,settings,42,root/'refit',fixed_schedule=selected['schedule'],device_name='cpu')
            self.assertEqual(selected['initial_state_hash'],refit['initial_state_hash'])
            self.assertEqual(selected['first_training_loss'],refit['first_training_loss'])
            self.assertEqual(refit['epochs'],len(selected['schedule']))
            self.assertNotIn('selection_loss',read_json(root/'refit'/'history.json')[0])
            with self.assertRaises(ValueError):
                fit_model(block,block,settings,42,root/'bad',fixed_schedule=[.001],device_name='cpu')
