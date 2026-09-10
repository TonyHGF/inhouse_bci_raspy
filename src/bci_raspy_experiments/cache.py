"""Shared split/transform caches with exclusive writers across task shards."""
from contextlib import contextmanager
from pathlib import Path
import os
import platform
import time
from .common import digest


@contextmanager
def writer_lock(path, timeout=3600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic()-started > timeout:
                raise TimeoutError(f'Cache locked: {path}; inspect owner before removing a stale lock')
            time.sleep(1)
    with os.fdopen(fd, 'w') as f:
        f.write(f'{platform.node()} pid={os.getpid()}')
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def materialize_cached(pipe, ids, directory):
    directory = Path(directory)
    path = directory / (digest(ids) + '.h5')
    with writer_lock(path.with_suffix('.lock')):
        kept = pipe.materialize(ids, path)
    return path, kept
