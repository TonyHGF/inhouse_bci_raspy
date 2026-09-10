"""Frozen input-channel perturbations and effective first-block kernels."""
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import r2_score
from .common import save_json, read_json
from .evaluation import predict, probabilities, load_predictions
from .runner import restore_model
from .windows import Bags
from .reporting import write_csv


def kernel_energy(model):
    m = getattr(model, 'base', model)
    if hasattr(m, 'temporal'):
        temporal, spatial, bn1, bn2 = m.temporal, m.spatial, m.bn1, m.bn2
        weight = spatial.weight.detach()
        norm = weight.norm(2, dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        weight = weight * (1. / norm).clamp(max=1.)
    else:
        temporal, spatial, bn1, bn2 = m._conv1, m._depthwise, m._batchnorm1, m._batchnorm2
        weight = spatial._max_norm(spatial.weight).detach()
    temporal = temporal.weight.detach()[:, 0, 0, :]
    weight = weight[:, 0, :, 0]
    d = len(weight) // len(temporal)
    slope1 = bn1.weight.detach() / (bn1.running_var + bn1.eps).sqrt()
    slope2 = bn2.weight.detach() / (bn2.running_var + bn2.eps).sqrt()
    parent = torch.arange(len(weight), device=weight.device) // d
    effective = weight[:, :, None] * temporal[parent, None, :] * slope1[parent, None, None] * slope2[:, None, None]
    energy = effective.square().sum(dim=(0, 2))
    return (energy/energy.sum()).cpu().numpy(), effective.cpu().numpy()


def explain(directory, catalog, cfg, profile):
    directory = Path(directory)
    target = directory / 'explanation'
    if (target / 'summary.json').exists():
        raise FileExistsError('Explanation already exists')
    target.mkdir(exist_ok=True)
    frozen = load_predictions(directory / 'test_predictions.npz')
    bags = Bags(read_json(directory / 'test_data.json')['path'], frozen['ids'].tolist(), catalog['trials'], cfg)
    model = restore_model(directory, profile['device'])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    baseline = predict(model, bags, cfg, profile, profile['device'])
    if not np.allclose(baseline['logits'], frozen['logits'], rtol=1e-5, atol=1e-6):
        raise ValueError('Frozen predictions do not reproduce')
    target_y = np.eye(4)[baseline['y']]
    base_r2 = r2_score(target_y, baseline['logits'].mean(1), multioutput='raw_values')
    base_window_r2 = r2_score(np.repeat(target_y, baseline['logits'].shape[1], axis=0), baseline['logits'].reshape(-1, 4), multioutput='raw_values')
    channels = catalog['trials'][0]['channels'][:catalog['trials'][0]['eeg_count']]
    energy, kernels = kernel_energy(model)
    rows = []
    outputs = {}
    for channel, name in enumerate(channels):
        value = predict(model, bags, cfg, profile, profile['device'], zero_channel=channel)
        z = value['logits']
        delta = base_r2-r2_score(target_y, z.mean(1), multioutput='raw_values')
        window_delta = base_window_r2-r2_score(np.repeat(target_y, z.shape[1], axis=0), z.reshape(-1, 4), multioutput='raw_values')
        rows.append(dict(channel=name, trial_delta_r2=float(delta.mean()), window_delta_r2=float(window_delta.mean()),
                         accuracy=float(np.mean(probabilities(z).mean(1).argmax(-1) == baseline['y'])),
                         kernel_energy=float(energy[channel]), **{f'class_{i}_delta_r2': float(delta[i]) for i in range(4)}))
        outputs[name] = z.mean(1)
    if any(not torch.equal(before[k], v) for k, v in model.state_dict().items()):
        raise RuntimeError('Explanation mutated model state')
    write_csv(target / 'channels.csv', rows)
    np.savez_compressed(target / 'arrays.npz', kernels=kernels, ids=baseline['ids'], y=baseline['y'], **outputs)
    import mne
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    info = mne.create_info(channels, 100., 'eeg')
    info.set_montage('standard_1020', match_case=False)
    for key in ('trial_delta_r2', 'window_delta_r2', 'kernel_energy') + tuple(f'class_{i}_delta_r2' for i in range(4)):
        fig, ax = plt.subplots()
        values = np.array([r[key] for r in rows])
        limit = max(float(np.max(np.abs(values))), 1e-8)
        image, _ = mne.viz.plot_topomap(values, info, axes=ax, show=False, names=channels,
                                       vlim=(0, limit) if key == 'kernel_energy' else (-limit, limit),
                                       cmap='Reds' if key == 'kernel_energy' else 'RdBu_r', image_interp='linear')
        fig.colorbar(image, ax=ax)
        ax.set_title(key)
        fig.savefig(target / (key + '.png'), dpi=180)
        fig.savefig(target / (key + '.pdf'))
        plt.close(fig)
    save_json(target / 'summary.json', dict(channels=rows, selection=cfg['selection'],
              interpretation='Post-transform input zeroing; not electrode removal or source localization. Kernel energy is not final decision attribution.'))
