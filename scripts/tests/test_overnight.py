"""Lightweight queue orchestration tests; no training or CUDA computation."""
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('overnight', Path(__file__).parents[1] / 'overnight.py')
night = importlib.util.module_from_spec(spec)
spec.loader.exec_module(night)


class QueueTests(unittest.TestCase):
    def test_bounded_concurrency_failure_continues_and_unlocks(self):
        live = set()
        created = []
        peak = []
        class Process:
            def __init__(self, command, **kwargs):
                self.pid = len(created) + 1
                self.returncode = None
                created.append(command)
                live.add(self.pid)
                peak.append(len(live))
            def poll(self):
                if self.returncode is None:
                    self.returncode = 1 if self.pid == 1 else 0
                    live.remove(self.pid)
                return self.returncode
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                               get_device_name=lambda i: 'fake GPU')
        with tempfile.TemporaryDirectory() as tmp:
            tasks = [dict(id=f'task-{i}', label='test') for i in range(3)]
            args = SimpleNamespace(limit_tasks=None, config=Path(tmp)/'config.json', dataset='inhouse')
            p = dict(output=tmp, threads=4)
            with patch.dict(os.environ, SLURM_JOB_ID='test', SLURM_CPUS_PER_TASK='8'), \
                 patch.dict('sys.modules', torch=SimpleNamespace(cuda=cuda)), \
                 patch.object(night.subprocess, 'Popen', Process), \
                 patch.object(night.signal, 'signal'), patch.object(night.time, 'sleep'):
                with self.assertRaises(SystemExit):
                    night.manage(args, dict(parallel_per_gpu=2, training_hours=1),
                                 dict(tasks=tasks, count=3), p)
            self.assertEqual(len(created), 3)
            self.assertEqual(max(peak), 2)
            self.assertFalse((Path(tmp)/'queue.lock').exists())
            self.assertEqual(len(list((Path(tmp)/'worker_logs').glob('*.log'))), 3)
            status = night.read_json(next(Path(tmp).glob('queue-*.json')))
            self.assertEqual(status['failed'], ['task-0'])

    def test_frozen_configuration_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'
            night.immutable(path, {'seed': 42})
            night.immutable(path, {'seed': 42})
            with self.assertRaises(ValueError):
                night.immutable(path, {'seed': 43})
            self.assertEqual(night.read_json(path), {'seed': 42})

    def test_queue_owner_is_exclusive(self):
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                               get_device_name=lambda i: 'fake GPU')
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp)/'queue.lock'
            lock.write_text('existing owner')
            with patch.dict(os.environ, SLURM_JOB_ID='test', SLURM_CPUS_PER_TASK='8'), \
                 patch.dict('sys.modules', torch=SimpleNamespace(cuda=cuda)):
                with self.assertRaises(FileExistsError):
                    night.manage(None, dict(parallel_per_gpu=2), {}, dict(output=tmp, threads=4))
            self.assertEqual(lock.read_text(), 'existing owner')


if __name__ == '__main__':
    unittest.main()
