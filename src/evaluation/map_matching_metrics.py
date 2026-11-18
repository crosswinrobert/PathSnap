
import numpy as np


def observation_top1_accuracy(
    y_true,
    y_pred,
    observation_ids
):
    """
    Measure how often the highest-ranked candidate
    is correct for each GPS observation.

    Assumes y_pred contains predicted scores or probabilities
    for candidate rows, and y_true contains their true labels.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    observation_ids = np.asarray(observation_ids)

    if not (
        len(y_true) == len(y_pred) == len(observation_ids)
    ):
        raise ValueError("All inputs must have equal lengths.")

    correct = 0
    total = 0

    for observation_id in np.unique(observation_ids):
        indices = np.where(observation_ids == observation_id)[0]

        best_index = indices[np.argmax(y_pred[indices])]

        if y_true[best_index] == 1:
            correct += 1

        total += 1

    return correct / total if total else 0.0


def candidate_miss_rate(
    y_true,
    observation_ids
):
    """
    Measure the fraction of GPS observations for which
    the correct road candidate is absent.

    Assumes y_true uses 1 for the correct candidate
    and 0 for incorrect candidates.
    """

    y_true = np.asarray(y_true)
    observation_ids = np.asarray(observation_ids)

    if len(y_true) != len(observation_ids):
        raise ValueError("Inputs must have equal lengths.")

    missed = 0
    total = 0

    for observation_id in np.unique(observation_ids):
        indices = np.where(observation_ids == observation_id)[0]

        if not np.any(y_true[indices] == 1):
            missed += 1

        total += 1

    return missed / total if total else 0.0

