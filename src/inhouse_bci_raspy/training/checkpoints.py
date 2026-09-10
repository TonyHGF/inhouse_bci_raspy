"""Serialize a selected model with the metadata needed to reconstruct its input pipeline."""

from pathlib import Path
import torch


def save_checkpoint(path: Path, model, config: dict, manifest: dict,
                    fold_metadata: dict, fold: int, best_index: int) -> None:
    """Save CPU state_dict, model dimensions, labels, channel order and this fold's RMS.

    The engine has already selected the model by minimum validation loss. This function
    makes no selection decisions and does not move the supplied model to another device.
    """
    checkpoint = {'state_dict': {name: value.detach().cpu() for name, value in model.state_dict().items()},
                  'model_config': config['model'], 'n_electrodes': len(manifest['channel_names']),
                  'label_names': manifest['label_names'], 'channel_names': manifest['channel_names'],
                  'normalization_scales': fold_metadata['normalization_scales'],
                  'validation_fold': fold, 'best_epoch_index': int(best_index)}
    torch.save(checkpoint, path)
