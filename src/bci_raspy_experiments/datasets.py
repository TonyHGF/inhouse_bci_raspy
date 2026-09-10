"""Read original recordings into an immutable, unfiltered trial cache in volts.

Catalogue entries survive signal rejection: splits use acquisition identities,
never a cleaned/reordered trial list. Only MI files are read for PhysioNet.
"""
from pathlib import Path
import re
import numpy as np
from .common import digest, file_hash, read_json, save_json

BCI_CHANNELS = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
                'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
MI_RUNS = (4, 6, 8, 10, 12, 14)


def sources(root, dataset):
    root = Path(root)
    if dataset == 'inhouse':
        return sorted(root.glob('hyz_*/*.vhdr'))
    if dataset == 'bci2a':
        return sorted(root.glob('A??[TE].mat'))
    return sorted(p for p in root.glob('S*/S*R*.edf') if int(p.stem[-2:]) in MI_RUNS)


def inventory(root, dataset):
    files = sources(root, dataset)
    if not files:
        raise FileNotFoundError(f'No {dataset} recordings in {root}')
    subjects = sorted({p.parent.name if dataset == 'physionet' else p.stem[:3] if dataset == 'bci2a' else '01' for p in files})
    missing = []
    if dataset == 'bci2a':
        missing = [f'A{s:02d}{session}.mat' for s in range(1, 10) for session in ('T', 'E')
                   if not (Path(root) / f'A{s:02d}{session}.mat').exists()]
    if dataset == 'physionet':
        missing = [f'{s}/{s}R{r:02d}.edf' for s in subjects for r in MI_RUNS
                   if not (Path(root) / s / f'{s}R{r:02d}.edf').exists()]
    if missing:
        raise ValueError(f'Incomplete {dataset} acquisition: {missing}')
    return {'dataset': dataset, 'subjects': subjects, 'files': [str(p) for p in files],
            'file_count': len(files), 'missing_subjects': [f'S{i:03d}' for i in range(1, 110)
                if f'S{i:03d}' not in subjects] if dataset == 'physionet' else []}


def _raw_trials(path, dataset):
    import mne
    if dataset == 'inhouse':
        raw = mne.io.read_raw_brainvision(path, preload=False, verbose='error')
        raw.pick('eeg')
        if len(raw.ch_names) != 16:
            raise ValueError(f'Expected 16 Inhouse EEG channels: {path}')
        pending, events = None, []
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
            match = re.fullmatch(r'Stimulus/S\s+(\d+)', str(desc))
            code = int(match[1]) if match else None
            if code in (5, 6, 7, 8):
                if pending is not None:
                    raise ValueError(f'Unpaired MI marker: {path}')
                pending = (float(onset), code - 5)
            elif code == 99 and pending is not None:
                events.append((*pending, float(onset) - pending[0]))
                pending = None
        if pending is not None:
            raise ValueError(f'Missing end marker: {path}')
        subject, session, run = '01', path.stem.split('_')[-1].zfill(2), '01'
        start, end = 1., 5.
        names = list(raw.ch_names)
    else:
        raw = mne.io.read_raw_edf(path, preload=False, verbose='error')
        names = [n.strip('.').upper().replace('Z', 'z').replace('FP', 'Fp') for n in raw.ch_names]
        r = int(path.stem[-2:])
        events = [(float(t), (0 if d == 'T1' else 1) + (2 if r in (6, 10, 14) else 0), float(duration))
                  for t, d, duration in zip(raw.annotations.onset, raw.annotations.description, raw.annotations.duration)
                  if d in ('T1', 'T2')]
        if set(raw.annotations.description) - {'T0', 'T1', 'T2'}:
            raise ValueError(f'Unknown EDF annotations: {path}')
        subject, session, run = path.parent.name, '01', f'{r:02d}'
        start, end = 0., 4.
    sf = float(raw.info['sfreq'])
    for i, (onset, label, duration) in enumerate(events):
        # Pre-onset context cannot overlap the preceding task.
        previous_end = events[i-1][0] + events[i-1][2] if i else 0.
        first = max(0, round((onset - 2) * sf), round(previous_end * sf))
        last = round((onset + end) * sf)
        reason = None
        if duration + 1e-6 < end or last > raw.n_times:
            reason = 'task_shorter_than_analysis_window'
        if first > round((onset - 2) * sf):
            reason = reason or 'insufficient_nonoverlapping_baseline_context'
        x = None if reason else raw.get_data(start=first, stop=last)
        yield dict(subject=subject, session=session, run=run, trial=i, onset=onset,
                   label=label, duration=duration, start=start, end=end, sfreq=sf, tmin=-2.,
                   channels=names, eeg_count=len(names), exclusion=reason), x


def _bci_trials(path):
    from scipy.io import loadmat
    runs = loadmat(path, simplify_cells=True).get('data')
    if isinstance(runs, dict):
        runs = [runs]
    if runs is None:
        raise ValueError(f'Missing data structure: {path}')
    for r, run in enumerate(runs):
        labels = np.asarray(run['y']).reshape(-1)
        starts = np.asarray(run['trial']).reshape(-1)
        if not len(labels):
            continue  # calibration blocks
        if len(labels) != len(starts) or not set(labels) <= {1, 2, 3, 4}:
            raise ValueError(f'Invalid labels/trials: {path}, run {r}')
        if list(run['classes']) != ['left hand', 'right hand', 'feet', 'tongue']:
            raise ValueError(f'Unexpected BCI label map: {path}')
        sf = float(run['fs'])
        # This MAT export is in microvolts; MATLAB trial indices are 1-based
        # fixation onsets, two seconds before the class cue.
        signals = np.asarray(run['X'], dtype=np.float64).T * 1e-6
        if signals.shape[0] != 25 or sf != 250:
            raise ValueError(f'Unexpected BCI layout: {path}')
        for i, (position, y) in enumerate(zip(starts, labels)):
            first = int(position) - 1
            cue = first + int(2 * sf)
            last = cue + int(4 * sf)
            reason = None
            if first < 0 or last > signals.shape[1] or (i + 1 < len(starts) and last > int(starts[i+1]) - 1):
                reason = 'incomplete_or_overlapping_trial'
            x = None if reason else signals[:, first:last]
            yield dict(subject=path.stem[:3], session=path.stem[-1], run=f'{r:02d}', trial=i,
                       onset=cue / sf, label=int(y)-1, duration=4., start=0., end=4., sfreq=sf,
                       tmin=-2., channels=BCI_CHANNELS + ['EOG1', 'EOG2', 'EOG3'], eeg_count=22,
                       source_artifact=bool(np.asarray(run['artifacts']).reshape(-1)[i]), exclusion=reason), x


def prepare(root, dataset, cache):
    import h5py
    inv = inventory(root, dataset)
    destination = Path(cache) / dataset
    catalog_path = destination / 'catalog.json'
    if catalog_path.exists():
        old = read_json(catalog_path)
        for p, h in old['source_hashes'].items():
            if file_hash(p) != h:
                raise ValueError(f'Source changed: {p}; use a new cache root')
        if not (destination / 'raw.h5').exists():
            raise ValueError('Incomplete cache: missing raw.h5')
        if dataset == 'inhouse' and old['trials'][0]['eeg_count'] != 16:
            raise ValueError('Old ablation cache contains auxiliary channels; use a new cache root')
        if 'raw_hash' not in old:
            old['raw_hash'] = file_hash(destination / 'raw.h5')
            old['hash'] = catalog_identity(old, root)
            save_json(catalog_path, old)
        return old
    destination.mkdir(parents=True, exist_ok=True)
    hashes, rows = {}, []
    temp = destination / 'raw.h5.tmp'
    with h5py.File(temp, 'w') as store:
        for path in map(Path, inv['files']):
            for source in ([path, path.with_suffix('.vmrk'), path.with_suffix('.eeg')] if dataset == 'inhouse' else [path]):
                hashes[str(source)] = file_hash(source)
            generator = _bci_trials(path) if dataset == 'bci2a' else _raw_trials(path, dataset)
            for row, x in generator:
                row['dataset'], row['source'] = dataset, str(path)
                row['id'] = f"{dataset}/{row['subject']}/{row['session']}/{row['run']}/{row['trial']:04d}"
                if x is not None:
                    if not np.isfinite(x).all():
                        raise ValueError(f'Nonfinite EEG: {row["id"]}')
                    store.create_dataset(row['id'], data=x.astype('float32'), compression='gzip')
                rows.append(row)
            print(f'Prepared {dataset}: {path.name} ({len(rows)} trials)', flush=True)
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('No trials or duplicate acquisition identity')
    order = lambda r: (r['subject'], {'T': 0, 'E': 1}.get(r['session'], r['session']), r['run'], r['onset'])
    rows.sort(key=order)
    if any((r['channels'], r['sfreq']) != (rows[0]['channels'], rows[0]['sfreq']) for r in rows):
        raise ValueError('Inconsistent dataset channel order or sample rate; explicitly resolve acquisition before running')
    temp.replace(destination / 'raw.h5')
    result = dict(inventory=inv, trials=rows, source_hashes=hashes, version=1)
    result['raw_hash'] = file_hash(destination / 'raw.h5')
    result['hash'] = catalog_identity(result, root)
    save_json(catalog_path, result)
    return result


def catalog_identity(catalog, root):
    rows = [{k: v for k, v in r.items() if k != 'source'} for r in catalog['trials']]
    source_hashes = {Path(k).relative_to(root).as_posix(): v for k, v in catalog['source_hashes'].items()}
    return digest([catalog['version'], rows, source_hashes])
