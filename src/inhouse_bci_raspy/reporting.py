"""Persist experiment metadata, predictions and plots; does not select or train models."""

import csv
import importlib.metadata as md
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from .io import save_json


def save_experiment(destination: Path, config: dict, device) -> None:
    """Record actual runtime versions and full experiment settings before training."""
    save_json(destination / 'environment.json', {'python': sys.version, 'device': str(device),
        'versions': {name: md.version(name) for name in ['numpy', 'scipy', 'mne', 'torch', 'h5py']}})
    save_json(destination / 'experiment.json', config)


def save_predictions(path: Path, validation: dict, probabilities) -> None:
    """Write one CSV row per window, including trial index, time and four probabilities."""
    actual = validation['y']
    predicted = probabilities.argmax(axis=1)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['trial_index', 'window_start_s', 'true_label', 'predicted_label'] + [f'p_{i}' for i in range(4)])
        for i in range(len(actual)):
            writer.writerow([int(validation['trial_index'][i]), float(validation['window_start_s'][i]),
                             int(actual[i]), int(predicted[i]), *probabilities[i].tolist()])


def save_confusion(directory: Path, actual, predicted) -> None:
    """Save a fixed four-class confusion matrix as JSON and PNG, including absent predictions."""
    cm = confusion_matrix(actual, predicted, labels=list(range(4)))
    save_json(directory / 'confusion_matrix.json', cm.tolist())
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=['left hand', 'right hand', 'both hands', 'both feet']).plot(ax=ax, colorbar=False)
    fig.tight_layout()
    fig.savefig(directory / 'confusion_matrix.png', dpi=150)
    plt.close(fig)


def save_history(directory: Path, stats: dict, train_loss: list, val_loss: list,
                 best_index: int, smoke: bool) -> None:
    """Save unchanged engine loss/accuracy sequences and draw their epoch curves."""
    save_json(directory / 'history.json', {
        'train_accuracy_percent': stats['acc'], 'validation_accuracy_percent': stats['val_acc'],
        'training_loss': train_loss, 'validation_loss': val_loss,
        'best_epoch_index': int(best_index), 'smoke_test_only': smoke})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, values, title in ((axes[0], (train_loss, val_loss), 'Loss'),
                              (axes[1], (stats['acc'], stats['val_acc']), 'Accuracy (%)')):
        ax.plot(values[0], label='train')
        ax.plot(values[1], label='validation')
        ax.set_title(title)
        ax.set_xlabel('Epoch (zero based)')
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / 'history.png', dpi=150)
    plt.close(fig)
