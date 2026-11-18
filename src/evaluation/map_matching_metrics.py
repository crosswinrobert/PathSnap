import numpy as np


def _group_observation_ids(observation_ids):
    """Create one group ID per unique observation."""
    ids = np.asarray(observation_ids)

    if ids.ndim == 1:
        keys = ids.tolist()
    elif ids.ndim == 2:
        keys = [tuple(row) for row in ids.tolist()]
    else:
        raise ValueError("Observation IDs must be 1D or 2D.")

    group_map = {}
    groups = np.empty(len(keys), dtype=int)

    for i, key in enumerate(keys):
        if key not in group_map:
            group_map[key] = len(group_map)
        groups[i] = group_map[key]

    return groups


def observation_top1_accuracy(y_true, y_pred, observation_ids):
    """
    Measure how often the highest-scoring candidate is correct
    for each GPS observation.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = _group_observation_ids(observation_ids)

    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError("All inputs must have equal lengths.")

    if len(y_true) == 0:
        return 0.0

    correct = 0
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        best_index = indices[np.argmax(y_pred[indices])]
        correct += int(y_true[best_index] == 1)

    return correct / len(np.unique(groups))


def candidate_miss_rate(y_true, observation_ids):
    """Measure observations with no correct candidate in their candidate set."""
    y_true = np.asarray(y_true)
    groups = _group_observation_ids(observation_ids)

    if len(y_true) != len(groups):
        raise ValueError("Inputs must have equal lengths.")

    if len(y_true) == 0:
        return 0.0

    missed = 0
    unique_groups = np.unique(groups)

    for group in unique_groups:
        indices = np.flatnonzero(groups == group)
        missed += int(not np.any(y_true[indices] == 1))

    return missed / len(unique_groups)
