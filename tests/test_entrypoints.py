"""Check entry points still resolve their files after moving executable code into src."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from inhouse_bci_raspy.preprocessing.pipeline import preprocess
from inhouse_bci_raspy.settings import load_settings


class EntryPointTests(unittest.TestCase):
    def test_cli_help_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'run.py'), '--help'],
                                    cwd=directory, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('preprocess,prepare,train,all,smoke', result.stdout)

    def test_preprocessing_resolves_moved_scripts(self):
        config, dataset = load_settings()
        with tempfile.TemporaryDirectory() as directory:
            with patch('inhouse_bci_raspy.preprocessing.pipeline.run_logged') as run:
                preprocess(config, dataset, Path(directory))
            self.assertEqual(run.call_count, 2 * len(config['recordings']))
            for index, call in enumerate(run.call_args_list):
                command, _, env, cwd = call.args
                self.assertEqual(cwd, ROOT)
                self.assertTrue(Path(env['EEG_DATASET_CONFIG']).is_file())
                script = Path(command[1] if index % 2 == 0 else command[-1])
                self.assertTrue(script.is_file(), script)
                self.assertTrue(script.is_relative_to(ROOT / 'src'))


if __name__ == '__main__':
    unittest.main()
