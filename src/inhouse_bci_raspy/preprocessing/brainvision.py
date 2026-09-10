#!/usr/bin/env python
"""Convert one BrainVision EEG recording to BIDS without changing the source."""

import argparse
import json
from pathlib import Path

import mne
from mne_bids import BIDSPath, write_raw_bids


def main() -> None:
    """Read --config, pair each selected onset with its first end marker, and write BIDS.
    
    Only configured imagery labels are kept. Missing/short trials and unexpected
    event counts raise errors; source BrainVision files are never modified."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/dataset.json"))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))

    source = Path(cfg["source_vhdr"])
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = mne.io.read_raw_brainvision(source, preload=False, verbose="error")
    raw.set_montage(cfg["montage"], match_case=False, on_missing="raise")
    raw.info["line_freq"] = float(cfg["line_frequency_hz"])

    event_map = cfg["event_map"]
    selected = []
    durations = []
    end_marker = cfg.get("end_marker")
    pending = None
    ignored_end_markers = 0
    for onset, description in zip(raw.annotations.onset, raw.annotations.description):
        description = str(description)
        if description in event_map:
            if pending is not None:
                raise ValueError(f"Trial at {pending[0]} s has no end marker")
            if end_marker is None:
                selected.append((onset, event_map[description]))
                durations.append(float(cfg["event_duration_s"]))
            else:
                pending = (onset, event_map[description])
        elif end_marker is not None and description == end_marker:
            if pending is None:
                ignored_end_markers += 1
                continue
            duration = float(onset - pending[0])
            if duration < float(cfg["event_duration_s"]):
                raise ValueError(f"Trial at {pending[0]} s ends before the configured analysis window")
            selected.append(pending)
            durations.append(duration)
            pending = None
    if pending is not None:
        raise ValueError(f"Trial at {pending[0]} s has no end marker")
    counts = {condition: 0 for condition in cfg["event_id"]}
    for _, condition in selected:
        counts[condition] += 1
    expected = cfg.get("expected_event_counts")
    if expected is not None and counts != expected:
        raise RuntimeError(f"Unexpected event counts: {counts}; expected {expected}")

    raw.set_annotations(mne.Annotations(
        onset=[onset for onset, _ in selected],
        duration=durations,
        description=[condition for _, condition in selected],
    ))
    bids_path = BIDSPath(
        root=Path(cfg["bids_root"]), subject=cfg["subject"],
        session=cfg["session"], task=cfg["task"], run=cfg["run"], datatype="eeg",
    )
    write_raw_bids(raw, bids_path, event_id=cfg["event_id"], overwrite=True,
                   format="BrainVision", verbose=True)
    print(f"Wrote {len(selected)} events to {cfg['bids_root']}; counts={counts}")
    if end_marker is not None:
        print(f"Used first end marker per trial; ignored {ignored_end_markers} unpaired/repeated ends")


if __name__ == "__main__":
    main()
