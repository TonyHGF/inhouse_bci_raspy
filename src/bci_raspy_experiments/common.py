"""Serialization, hashing, and portable execution profiles (no ML imports)."""
import hashlib
import json
import os
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent
_SOURCE_ROOT = _PACKAGE.parents[1] if _PACKAGE.parent.name == 'src' else Path.cwd()
PROJECT = Path(os.environ.get('BCI_EXPERIMENTS_HOME', _SOURCE_ROOT)).resolve()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding='utf-8')
    os.replace(temporary, path)


def profile(name='local', path=None):
    p = Path(path) if path else Path(__file__).parent / 'profiles' / (name + '.json')
    result = read_json(p)
    required = ('inhouse', 'bci2a', 'physionet', 'cache', 'output')
    missing = [k for k in required if not result.get(k)]
    if missing:
        raise ValueError(f'{name} profile requires paths: {", ".join(missing)}. Edit the profile or use --profile-file; no fallback.')
    for k in required:
        value = Path(result[k])
        result[k] = str((PROJECT / value).resolve() if not value.is_absolute() else value)
    result['name'] = name
    return result
