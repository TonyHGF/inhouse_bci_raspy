"""Extract effective first-block kernels; these are coefficients, not decision attribution."""

import numpy as np
import torch


def extract_kernels(model) -> dict:
    """Fold eval BatchNorm slopes into the temporal/spatial linear kernels before ELU.

    H[j,c,t] = BN2[j] * effective_spatial[j,c] * BN1[parent(j)] * temporal[parent(j),t].
    Bias terms are intentionally excluded. Effective spatial weights use exactly
    the model's forward max-norm function, including its nonstandard norm axis.
    Energy sums H² over filter/time and normalizes across input channels. It does
    not incorporate downstream nonlinearities, pooling or the classifier.
    """
    if model.training:
        raise ValueError('Kernel extraction requires eval mode')
    with torch.no_grad():
        spatial = model._depthwise._max_norm(model._depthwise.weight)[:, 0, :, 0]
        temporal = model._conv1.weight[:, 0, 0, :]
        bn1 = model._batchnorm1.weight / torch.sqrt(model._batchnorm1.running_var + model._batchnorm1.eps)
        bn2 = model._batchnorm2.weight / torch.sqrt(model._batchnorm2.running_var + model._batchnorm2.eps)
        parent = torch.arange(spatial.shape[0], device=spatial.device) // model.D
        folded_spatial = spatial * (bn2 * bn1[parent])[:, None]
        h = folded_spatial[:, :, None] * temporal[parent, None, :]
        energy = h.double().square().sum((0, 2))
        if not torch.isfinite(h).all() or energy.sum() <= 0:
            raise ValueError('Undefined kernel distribution: nonfinite or all-zero energy')
        result = dict(effective_spatial=spatial, folded_spatial=folded_spatial,
                      temporal=temporal, parent_temporal_filter=parent,
                      spatiotemporal=h, channel_energy=energy,
                      channel_energy_fraction=energy / energy.sum())
    return {name: tensor.detach().cpu().numpy() for name, tensor in result.items()}
