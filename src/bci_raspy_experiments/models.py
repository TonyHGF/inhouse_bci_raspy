"""Raspy-compatible and conventional EEGNet backbones with extra blocks."""
import copy
import torch
from torch import nn


def raspy_config(spec, samples):
    return dict(num_temporal_filters=spec['f1'], num_spatial_filters=spec['d'],
                sampling_frequency=100, window_length=samples * 10,
                block1=dict(conv=[1, spec['kernel']], max_norm_value=1, eps=.01, avg_pool=[1, 3], dropout=spec['dropout']),
                block2=dict(sep_conv=[1, 16], max_norm_value=.25, eps=.01, avg_pool=[1, 16], dropout=spec['dropout']))


class ExtraBlock(nn.Sequential):
    def __init__(self, channels, dropout):
        super().__init__(nn.Conv2d(channels, channels, (1, 16), groups=channels, padding='same', bias=False),
                         nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels),
                         nn.ELU(), nn.Dropout(dropout))


class ConstrainedConv(nn.Conv2d):
    def forward(self, x):
        weight = self.weight
        norms = weight.norm(2, dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
        return self._conv_forward(x, weight * (1. / norms).clamp(max=1.), self.bias)


class ConstrainedLinear(nn.Linear):
    def forward(self, x):
        norm = self.weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return nn.functional.linear(x, self.weight * (.25 / norm).clamp(max=1.), self.bias)


class StandardEEGNet(nn.Module):
    def __init__(self, spec, channels, samples):
        super().__init__()
        f1, d, f2 = spec['f1'], spec['d'], spec['f2']
        self.temporal = nn.Conv2d(1, f1, (1, spec['kernel']), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial = ConstrainedConv(f1, f1*d, (channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1*d)
        self.first = nn.Sequential(nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(spec['dropout']))
        self.separable = nn.Sequential(nn.Conv2d(f1*d, f1*d, (1, 16), groups=f1*d, padding='same', bias=False),
                                       nn.Conv2d(f1*d, f2, 1, bias=False), nn.BatchNorm2d(f2), nn.ELU())
        self.extra = nn.Sequential(*(ExtraBlock(f2, spec['dropout']) for _ in range(spec['extra'])))
        self.last = nn.Sequential(nn.AvgPool2d((1, 8)), nn.Dropout(spec['dropout']), nn.Flatten())
        self.dense = ConstrainedLinear(f2 * (samples // 4 // 8), 4)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.first(self.bn2(self.spatial(self.bn1(self.temporal(x)))))
        return self.dense(self.last(self.extra(self.separable(x))))


class RaspyExtended(nn.Module):
    def __init__(self, spec, channels, samples):
        super().__init__()
        from .vendor.eegnet import EEGNet
        self.base = EEGNet(raspy_config(spec, samples), output_dim=4, n_electrodes=channels)
        self.extra = nn.Sequential(*(ExtraBlock(spec['f1']*spec['d'], spec['dropout']) for _ in range(spec['extra'])))

    def forward(self, x):
        m = self.base
        x = m.reshape_input(x)
        x = m._dropout1(m._avg_pool1(nn.functional.elu(m._batchnorm2(m._depthwise(m._batchnorm1(m._conv1(x)))))))
        x = nn.functional.elu(m._batchnorm3(m._seperable(x)))
        x = m._dropout2(m._avg_pool2(self.extra(x)))
        return m._dense(m._ff(torch.flatten(x, 1)))


def build_model(spec, channels, samples=100):
    if spec['family'] == 'standard':
        return StandardEEGNet(spec, channels, samples)
    if spec['extra']:
        return RaspyExtended(spec, channels, samples)
    from .vendor.eegnet import EEGNet
    return EEGNet(raspy_config(spec, samples), output_dim=4, n_electrodes=channels)


def describe(model, channels, samples):
    shapes, handles = {}, []
    for name, module in model.named_modules():
        if name and not list(module.children()):
            handles.append(module.register_forward_hook(lambda m, args, out, n=name: shapes.update({n: list(out.shape)})))
    training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(2, channels, samples, 1, device=next(model.parameters()).device))
    for handle in handles:
        handle.remove()
    model.train(training)
    return dict(parameters=sum(p.numel() for p in model.parameters()), shapes=shapes)

