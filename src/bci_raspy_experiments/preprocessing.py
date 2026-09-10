"""Training-only fitted transformations; disk-backed processed trial stores."""
from pathlib import Path
import pickle
import h5py
import numpy as np
from .common import digest, save_json
from .presets import PREPROCESS


class Pipeline:
    def __init__(self, catalog, cache, name):
        import mne
        self.rows = {r['id']: r for r in catalog['trials']}
        self.raw_path = Path(cache) / catalog['inventory']['dataset'] / 'raw.h5'
        self.options = PREPROCESS[name].copy()
        self.name = name
        row = catalog['trials'][0]
        self.eeg_count = row['eeg_count']
        self.sfreq, self.tmin = row['sfreq'], row['tmin']
        self.info = mne.create_info(row['channels'], self.sfreq,
                                   ['eeg'] * self.eeg_count + ['eog'] * (len(row['channels']) - self.eeg_count))
        self.info.set_montage('standard_1020', match_case=False, on_missing='raise')
        self.ica = None
        self.scale = np.ones(self.eeg_count)
        self.csd = None
        if self.options['csd']:
            info = mne.pick_info(self.info, np.arange(self.eeg_count))
            basis = mne.io.RawArray(np.eye(self.eeg_count), info, verbose='error')
            self.csd = mne.preprocessing.compute_current_source_density(basis, copy=True).get_data()

    def fixed(self, trial_id):
        import mne
        row = self.rows[trial_id]
        if row['exclusion']:
            return None, row['exclusion']
        with h5py.File(self.raw_path, 'r') as f:
            x = f[trial_id][:].astype('float64')
        x = mne.filter.filter_data(x, self.sfreq, 1., 40., verbose='error')
        if self.options['car']:
            x[:self.eeg_count] -= x[:self.eeg_count].mean(axis=0, keepdims=True)
        if self.options['baseline']:
            n = int(round(-self.tmin * self.sfreq))
            x -= x[:, :n].mean(axis=1, keepdims=True)
        return x, None

    def unscaled(self, trial_id):
        import mne
        x, reason = self.fixed(trial_id)
        if reason:
            return x, reason
        if self.ica is not None:
            epoch = mne.EpochsArray(x[None], self.info.copy(), tmin=self.tmin, baseline=None, verbose='error')
            x = self.ica.apply(epoch, verbose='error').get_data(copy=True)[0]
        if self.options['reject'] and np.any(np.ptp(x[:self.eeg_count], axis=1) > 500e-6):
            return None, 'peak_to_peak_gt_500uV'
        x = x[:self.eeg_count]
        if self.csd is not None:
            x = self.csd @ x
        return x, None

    def fit(self, train_ids):
        import mne
        if len(set(train_ids)) != len(train_ids):
            raise ValueError('Duplicate fitting identity')
        self.fit_ids = list(train_ids)
        self.ica_fit_ids = []
        if self.options['ica']:
            # ICA uses all retained training epochs, never held-out epochs.
            candidates = train_ids
            epochs = []
            for trial_id in candidates:
                x, reason = self.fixed(trial_id)
                if reason or (self.options['reject'] and np.any(np.ptp(x[:self.eeg_count], axis=1) > 500e-6)):
                    continue
                epochs.append(x)
                self.ica_fit_ids.append(trial_id)
            if len(epochs) < 4:
                raise ValueError('Insufficient clean training epochs for ICA')
            ep = mne.EpochsArray(np.stack(epochs), self.info.copy(), tmin=self.tmin, baseline=None, verbose='error')
            self.ica = mne.preprocessing.ICA(n_components=10, method='picard', random_state=42, max_iter=500)
            self.ica.fit(ep, picks='eeg', decim=2, verbose='error')
            eog = [n for n in self.info.ch_names if n.startswith('EOG')]
            if not eog:
                eog = [n for n in self.info.ch_names if n.lower() == 'fp1']
            if not eog:
                raise ValueError('ICA artifact detection requires EOG or Fp1')
            excluded, _ = self.ica.find_bads_eog(ep, ch_name=eog, threshold=3., verbose='error')
            self.ica.exclude = excluded
        total, count = np.zeros(self.eeg_count), 0
        self.retained_fit_ids, self.excluded_fit = [], {}
        for trial_id in train_ids:
            x, reason = self.unscaled(trial_id)
            if reason:
                self.excluded_fit[trial_id] = reason
            else:
                self.retained_fit_ids.append(trial_id)
                total += np.square(x).sum(axis=1)
                count += x.shape[1]
        if not count:
            raise ValueError('No retained training data')
        if {self.rows[i]['label'] for i in self.retained_fit_ids} != {0, 1, 2, 3}:
            raise ValueError('Preprocessing removed a training class')
        if self.options['rms']:
            self.scale = np.maximum(np.sqrt(total / count), np.finfo(float).eps)
        return self

    def metadata(self):
        return dict(name=self.name, options=self.options, fit_ids=self.fit_ids,
                    ica_fit_ids=self.ica_fit_ids, ica_training_subsample=False,
                    ica_excluded_components=[] if self.ica is None else list(map(int, self.ica.exclude)),
                    retained_fit_ids=self.retained_fit_ids, excluded_fit=self.excluded_fit,
                    rms=self.scale.tolist(), channels=self.info.ch_names[:self.eeg_count])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            pickle.dump(self, stream)
        temporary.replace(path)
        save_json(path.with_suffix('.json'), self.metadata())

    def materialize(self, ids, path):
        path = Path(path)
        expected = digest([ids, self.metadata()])
        if path.exists():
            with h5py.File(path, 'r') as f:
                if f.attrs['identity'] != expected:
                    raise ValueError('Transformed cache identity mismatch')
                return [s.decode() if isinstance(s, bytes) else s for s in f['ids'][:]]
        path.parent.mkdir(parents=True, exist_ok=True)
        kept, rejected = [], {}
        temp = path.with_suffix('.tmp')
        with h5py.File(temp, 'w') as store:
            for trial_id in ids:
                x, reason = self.unscaled(trial_id)
                if reason:
                    rejected[trial_id] = reason
                    continue
                x = x / self.scale[:, None]
                if not np.isfinite(x).all():
                    raise ValueError('Nonfinite transformed data')
                store.create_dataset('trials/' + trial_id, data=x.astype('float32'), compression='gzip')
                kept.append(trial_id)
            store['ids'] = np.asarray(kept, dtype=h5py.string_dtype())
            store.attrs['identity'] = expected
        if not kept or {self.rows[i]['label'] for i in kept} != {0, 1, 2, 3}:
            temp.unlink()
            raise ValueError('Transform leaves an empty split or missing class')
        temp.replace(path)
        save_json(path.with_suffix('.json'), dict(requested_ids=ids, retained_ids=kept, exclusions=rejected,
                                                retention=len(kept)/len(ids), identity=expected))
        return kept
