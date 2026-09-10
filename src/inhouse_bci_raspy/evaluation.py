"""Window inference and window/trial metrics; no training or filesystem output."""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix


def predict_probabilities(model, block: dict, device, batch_size: int) -> np.ndarray:
    """Return softmax probabilities [N,4] in the original window/metadata order.

    Uses evaluation mode without gradients. Unlike a shuffled validation loader,
    this order can be joined directly with trial_index and window_start_s.
    """
    model.eval()
    probabilities = []
    with torch.no_grad():
        for first in range(0, len(block['y']), batch_size):
            logits = model(torch.from_numpy(block['X'][first:first + batch_size]).to(device))
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def fold_metrics(validation: dict, probabilities: np.ndarray, fold: int,
                 best_index: int, stats: dict, training_windows: int) -> dict:
    """Compute window accuracy and trial accuracy after averaging each trial's probabilities.

    Accuracy values are percentages. Validation participates in model selection;
    these metrics do not describe an independent test set.
    """
    actual = validation['y']
    predicted = probabilities.argmax(axis=1)
    trial_true, trial_pred = [], []
    for trial in np.unique(validation['trial_index']):
        selected = validation['trial_index'] == trial
        trial_true.append(int(actual[selected][0]))
        trial_pred.append(int(probabilities[selected].mean(axis=0).argmax()))
    return {'fold': fold, 'best_epoch_index': int(best_index),
            'training_accuracy_percent': float(stats['acc'][best_index]),
            'validation_accuracy_percent': 100 * accuracy_score(actual, predicted),
            'validation_trial_accuracy_percent': 100 * accuracy_score(trial_true, trial_pred),
            'training_windows': training_windows, 'validation_windows': len(actual)}


def summarize_folds(metrics: list[dict], actual: list, predicted: list, smoke: bool) -> dict:
    """Aggregate fold percentages and pooled window predictions using the existing definitions."""
    return {'smoke_test_only': smoke, 'folds': metrics,
        'mean_validation_accuracy_percent': float(np.mean([m['validation_accuracy_percent'] for m in metrics])),
        'std_validation_accuracy_percent': float(np.std([m['validation_accuracy_percent'] for m in metrics])),
        'pooled_validation_accuracy_percent': 100 * accuracy_score(actual, predicted),
        'pooled_confusion_matrix': confusion_matrix(actual, predicted, labels=list(range(4))).tolist(),
        'evaluation': 'Validation used for early stopping and checkpoint selection; not an independent test set'}
