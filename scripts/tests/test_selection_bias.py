"""Small synthetic checks for the matched-training selection diagnostic."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np

SCRIPTS=Path(__file__).parents[1]
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('selection_bias',SCRIPTS/'selection_bias.py')
bias=importlib.util.module_from_spec(spec); spec.loader.exec_module(bias)


class SelectionIsolationTests(unittest.TestCase):
    def test_outer_labels_do_not_change_inner_partition(self):
        y=np.tile(np.arange(4),5)
        folds=[np.arange(4*i,4*i+4) for i in range(5)]
        c=dict(inner_validation_fraction=.2,seed=42)
        a=bias.split_trials(y,folds,0,c)
        changed=y.copy(); changed[folds[0]]=(changed[folds[0]]+1)%4
        b=bias.split_trials(changed,folds,0,c)
        for x,z in zip(a,b): np.testing.assert_array_equal(x,z)
        self.assertEqual(set(np.concatenate(a)),set(range(20)))

    def test_outer_signals_do_not_change_rms_or_training_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            c=dict(output_root=tmp,seed=42)
            x=np.random.default_rng(8).normal(size=(20,4,700)).astype('float32')
            y=np.tile(np.arange(4),5)
            path=Path(tmp)/'clean_trials.h5'
            with h5py.File(path,'w') as f:
                f['X']=x; f['y']=y; f.attrs['sfreq']=100.; f.attrs['tmin']=-2.
            split=dict(fold=0,training=list(range(8,20)),validation=list(range(4,8)),test=list(range(4)))
            legacy=dict(window=dict(start_s=1.,last_start_s=4.,step_s=.1,duration_s=1.,output_sfreq=100,num_noise=4))
            a,scale_a=bias.make_blocks(c,split,legacy)
            with h5py.File(path,'r+') as f: f['X'][:4]=x[:4]*99+123
            b,scale_b=bias.make_blocks(c,split,legacy)
            np.testing.assert_array_equal(scale_a,scale_b)
            np.testing.assert_array_equal(a['training']['X'],b['training']['X'])
            np.testing.assert_array_equal(a['validation']['X'],b['validation']['X'])
            self.assertFalse(np.array_equal(a['test']['X'],b['test']['X']))
            self.assertEqual(len(a['training']['y']),12*31*5)


if __name__=='__main__': unittest.main()
