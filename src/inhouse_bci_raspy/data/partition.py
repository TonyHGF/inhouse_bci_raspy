"""Class-stratified trial partitions; preserves the GitClone fold protocol."""

import numpy as np


def partition_data(labels, num_folds):
    """Return disjoint class-stratified trial index arrays using the GitClone algorithm.
    
    Args:
        labels: Sequence of (class_id, per_sample_labels) pairs; only class_id is read.
        num_folds: Number of folds. The caller seeds the global NumPy RNG.
    Returns:
        A list of NumPy index arrays, concatenated in the original class order."""
    ids = np.arange(len(labels))
    labels = [label[0] for label in labels]
    label_set = list(set(labels))
    sub_ids = []
    for label in label_set:
        selected_ids = [l[0] for l in zip(ids, labels) if l[1] == label]
        np.random.shuffle(selected_ids)
        sub_ids_folds = np.array_split(selected_ids, num_folds)
        sub_ids.append(sub_ids_folds)
    return [np.concatenate([subgroup[i] for subgroup in sub_ids])
            for i in range(num_folds)]
