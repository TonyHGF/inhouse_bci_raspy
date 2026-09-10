"""Run frozen-fold analyses and persist reproducible numbers independently of plotting."""

import csv
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
import torch

from ..io import save_json
from .channel_inputs import load_model, load_validation, read_json, verify_baseline
from .channel_ablation import infer, summarize_outputs, ablate_channels, weighted_moments
from .channel_kernels import extract_kernels


def sha256(path: Path) -> str:
    """Hash input artifacts in streaming chunks without changing them."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list) -> None:
    """Save explicitly ordered, UTF-8 rows for downstream scientific analysis."""
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_ablation(directory: Path, baseline: dict, results: list, channels: list) -> tuple:
    """Save every baseline/ablated trial output and return [channel,overall+4 classes] maps."""
    output_rows, score_rows = [], []
    for name, item in [('baseline', baseline)] + [(channels[r['channel_index']], r) for r in results]:
        for i, trial in enumerate(item['trial_indices']):
            output_rows.append({'condition': name, 'trial_index': int(trial),
                'true_label': int(item['trial_y'][i]),
                **{f'mean_logit_{k}': float(item['trial_logits'][i,k]) for k in range(4)},
                **{f'mean_probability_{k}': float(item['trial_probabilities'][i,k]) for k in range(4)}})
        for level in ['trial', 'window']:
            for k in range(5):
                score_rows.append({'condition': name, 'level': level,
                    'output': 'overall' if k == 0 else str(k-1),
                    'r2': item[f'{level}_r2'] if k == 0 else float(item[f'{level}_r2_per_class'][k-1]),
                    'delta_r2': 0.0 if name == 'baseline' else (item[f'{level}_delta_r2'] if k == 0 else float(item[f'{level}_delta_r2_per_class'][k-1])),
                    'accuracy_percent': item[f'{level}_accuracy_percent'],
                    'accuracy_drop_pp': 0.0 if name == 'baseline' else item[f'{level}_accuracy_drop_pp']})
    write_csv(directory / 'trial_outputs.csv', output_rows)
    write_csv(directory / 'ablation_scores.csv', score_rows)
    maps = []
    for level in ['trial', 'window']:
        maps.append(np.array([[r[f'{level}_delta_r2'], *r[f'{level}_delta_r2_per_class']] for r in results]))
    np.savez_compressed(directory / 'ablation_arrays.npz',
        trial_indices=baseline['trial_indices'], trial_y=baseline['trial_y'],
        baseline_trial_logits=baseline['trial_logits'], baseline_trial_probabilities=baseline['trial_probabilities'],
        ablated_trial_logits=np.stack([r['trial_logits'] for r in results]),
        ablated_trial_probabilities=np.stack([r['trial_probabilities'] for r in results]),
        trial_delta_r2=maps[0], window_delta_r2=maps[1])
    return tuple(maps)


def compare_methods(ablation: np.ndarray, kernel: np.ndarray, channels: list) -> dict:
    """Compare channel rankings descriptively without mixing unlike importance scales."""
    def ranking(values):
        return [{'channel': channels[i], 'value': float(values[i]), 'rank': float(rankdata(-values)[i])}
                for i in np.argsort(-values, kind='stable')]
    rho = None
    if np.ptp(ablation) > 0 and np.ptp(kernel) > 0:
        rho = float(spearmanr(ablation, kernel).statistic)
    return {'spearman_rho': rho, 'ablation_ranking': ranking(ablation), 'kernel_ranking': ranking(kernel)}


def run_analysis(output: Path, destination: Path, method: str, device: str = 'cpu',
                 batch_size: int = 32, overwrite: bool = False) -> dict:
    """Run all five saved folds, checking immutable model state and baseline fidelity.

    Writes only destination; source checkpoints, data and training configuration
    are read-only. An existing analysis requires explicit --overwrite. A progress
    JSON records running/failed/computed status for long executions.
    """
    if method not in ('both', 'ablation', 'kernel'):
        raise ValueError('Unknown analysis method')
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f'Analysis exists: {destination}; use --overwrite to recompute')
    destination.mkdir(parents=True, exist_ok=True)
    progress_path = destination / 'status.json'
    save_json(progress_path, {'status': 'running', 'method': method})
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    target_device = torch.device(device)
    manifest = read_json(output / 'prepared/folds.json')
    channels = manifest['channel_names']
    metadata = {'method': method, 'device': str(target_device), 'batch_size': batch_size,
        'channel_names': channels, 'label_names': manifest['label_names'], 'montage': 'standard_1020',
        'input_space': 'CSD channels normalized by checkpoint training-fold RMS',
        'ablation': {'replacement': 0, 'target': 'four-dimensional one-hot', 'prediction': 'pre-softmax logits',
            'primary': 'R2 of trial-mean logits; uniform mean over four outputs',
            'class_scope': 'one-vs-rest on all validation trials',
            'fold_mean': 'validation-trial-count weighted mean of fold delta R2, NOT pooled R2',
            'fold_std': 'weighted population SD, NOT standard error or confidence interval',
            'accuracy': 'argmax of mean window softmax probabilities'},
        'kernel': {'formula': 'H[j,c,t]=BN2[j]*W_effective[j,c]*BN1[parent(j)]*T[parent(j),t]',
            'normalization': 'sum(H^2 over filter,time) divided by total energy within each fold',
            'fold_mean': 'equal-weight fold mean', 'fold_std': 'population SD (ddof=0)',
            'scope': 'linear coefficients before first ELU; no bias or downstream classifier'},
        'versions': {name: importlib.metadata.version(name) for name in ['torch','numpy','mne','scipy','scikit-learn','matplotlib']},
        'source_sha256': {}, 'folds': [], 'limitations': [
            'Selected validation folds are not an independent test set.',
            'Session preprocessing preceded folding; folds mix all three sessions.',
            'Zeroing a normalized CSD channel is a specified perturbation, not physical electrode removal.',
            'Kernel energy is not final decision attribution or brain source localization.',
            'Overlapping windows and overlapping training folds are not independent replicates.']}
    sources = [output/'prepared/folds.json', output/'prepared/clean_trials.h5', output/'cv/experiment.json']
    sources += [output/'cv'/f'fold_{f}'/name for f in range(5)
                for name in ['best.pt','split.json','metrics.json','validation_predictions.csv']]
    for path in sources:
        metadata['source_sha256'][str(path.relative_to(output))] = sha256(path)
    trial_maps, window_maps, kernel_maps, counts = [], [], [], []
    try:
        for fold in range(5):
            print(f'[fold {fold}/4] Loading saved model', flush=True)
            directory = destination / f'fold_{fold}'
            directory.mkdir(exist_ok=True)
            model, checkpoint = load_model(output, fold, target_device)
            before = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            info = {'fold': fold, 'best_epoch_index': checkpoint['best_epoch_index']}
            if method in ('both', 'ablation'):
                block = load_validation(output, fold, checkpoint)
                z, p = infer(model, block['X'], batch_size)
                baseline = summarize_outputs(block, z, p)
                info['baseline_verification'] = verify_baseline(output, fold, block, p, baseline)
                info['baseline_trial_r2'] = baseline['trial_r2']
                info['baseline_window_r2'] = baseline['window_r2']
                counts.append(len(baseline['trial_indices']))
                info['validation_trials'] = counts[-1]
                print(f'[fold {fold}/4] Baseline verified: trial accuracy {baseline["trial_accuracy_percent"]:.2f}%', flush=True)
                def progress(c):
                    print(f'[fold {fold}/4] Ablating {channels[c]} ({c+1}/{len(channels)}); overall {fold*len(channels)+c+1}/{5*len(channels)}', flush=True)
                    save_json(progress_path, {'status':'running','method':method,'fold':fold,'channel':channels[c],
                                               'completed_channels':fold*len(channels)+c,'total_channels':5*len(channels)})
                results = ablate_channels(model, block, baseline, batch_size, progress)
                trial, window = save_ablation(directory, baseline, results, channels)
                trial_maps.append(trial)
                window_maps.append(window)
            if method in ('both', 'kernel'):
                kernels = extract_kernels(model)
                np.savez_compressed(directory / 'kernels.npz', **kernels)
                kernel_maps.append(kernels['channel_energy_fraction'])
                write_csv(directory / 'kernel_channels.csv', [{'channel':c, 'energy':float(kernels['channel_energy'][i]),
                    'energy_fraction':float(kernel_maps[-1][i])} for i,c in enumerate(channels)])
                info['kernel_fraction_sum'] = float(kernel_maps[-1].sum())
            if any(not torch.equal(before[k], v.detach().cpu()) for k,v in model.state_dict().items()):
                raise RuntimeError('Analysis mutated model parameters or BatchNorm buffers')
            info['model_state_unchanged'] = True
            metadata['folds'].append(info)
            save_json(directory / 'metadata.json', info)
            print(f'[fold {fold}/4] Complete', flush=True)
        arrays = {}
        summary_rows = []
        if trial_maps:
            weights = np.asarray(counts, dtype=float)
            arrays['trial_counts'] = weights.astype(int)
            for level, maps in [('trial',trial_maps),('window',window_maps)]:
                arrays[f'{level}_fold_delta_r2'] = np.stack(maps)
                mean, sd = weighted_moments(np.stack(maps), weights)
                arrays[f'{level}_mean_delta_r2'], arrays[f'{level}_std_delta_r2'] = mean, sd
                for i,c in enumerate(channels):
                    for k in range(5):
                        summary_rows.append({'method':'ablation','level':level,'channel':c,
                            'output':'overall' if k==0 else str(k-1), 'mean':float(mean[i,k]), 'std':float(sd[i,k])})
        if kernel_maps:
            arrays['kernel_fold_fraction'] = np.stack(kernel_maps)
            arrays['kernel_mean_fraction'] = np.mean(kernel_maps, axis=0)
            arrays['kernel_std_fraction'] = np.std(kernel_maps, axis=0)
            for i,c in enumerate(channels):
                summary_rows.append({'method':'kernel','level':'coefficients','channel':c,'output':'overall',
                    'mean':float(arrays['kernel_mean_fraction'][i]),'std':float(arrays['kernel_std_fraction'][i])})
        if method == 'both':
            metadata['comparison'] = {'overall':compare_methods(arrays['trial_mean_delta_r2'][:,0], arrays['kernel_mean_fraction'], channels),
                'by_fold':[dict(fold=f, **compare_methods(trial_maps[f][:,0],kernel_maps[f],channels)) for f in range(5)]}
        # Confirm source checkpoint hashes still match after all inference.
        for fold in range(5):
            path = output/'cv'/f'fold_{fold}'/'best.pt'
            if sha256(path) != metadata['source_sha256'][str(path.relative_to(output))]:
                raise RuntimeError('A source checkpoint changed during analysis')
        np.savez_compressed(destination / 'summary_arrays.npz', **arrays)
        write_csv(destination / 'channel_summary.csv', summary_rows)
        save_json(destination / 'metadata.json', metadata)
        save_json(progress_path, {'status':'computed','method':method,'folds_complete':5})
    except Exception as error:
        save_json(progress_path, {'status':'failed','method':method,'error':str(error)})
        raise
    return metadata
