"""Official MNE-BIDS-Pipeline preprocessing configuration."""

import json
import os
from pathlib import Path

_config_path = Path(os.environ.get("EEG_DATASET_CONFIG", "config/dataset.json"))
_cfg = json.loads(_config_path.read_text(encoding="utf-8"))

bids_root = Path(_cfg["bids_root"])
deriv_root = bids_root / "derivatives" / "mne-bids-pipeline"
subjects = [_cfg["subject"]]
sessions = [_cfg["session"]]
runs = [_cfg["run"]]
task = _cfg["task"]
data_type = "eeg"
ch_types = ["eeg"]
conditions = list(_cfg["event_id"])

interactive = False
n_jobs = 1
parallel_backend = "loky"
eeg_reference = "average"
eeg_template_montage = _cfg["montage"]
eog_channels = ["Fp1"]

l_freq = 1.0
h_freq = 40.0
notch_freq = None
raw_resample_sfreq = None

epochs_tmin = -2.0
epochs_tmax = 5.0
baseline = (-2.0, 0.0)
reject = {"eeg": 500e-6}

spatial_filter = "ica"
ica_reject = {"eeg": 500e-6}
ica_algorithm = "picard"
ica_n_components = 10
ica_decim = 2
ica_eog_threshold = 3.0

run_source_estimation = False
find_flat_channels_meg = False
find_noisy_channels_meg = False
on_error = "abort"
report_add_epochs_image_kwargs = None
