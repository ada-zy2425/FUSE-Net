"""Dataset-specific regression and ordinal metrics; all accuracies/F1 are fractions."""

import numpy as np
from sklearn.metrics import f1_score


def simsv2_classes(values, classes):
    boundaries = {2: (0.0,), 3: (-0.1, 0.1), 5: (-0.7, -0.1, 0.1, 0.7)}
    if classes not in boundaries:
        raise ValueError("SIMSv2 supports 2, 3, or 5 classes")
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float64)
    # The official protocol places a threshold itself in the interval to its left.
    return np.searchsorted(np.asarray(boundaries[classes], dtype=values.dtype), np.clip(values, -1, 1), side="left")


def calculate_metrics(targets, predictions, dataset):
    if dataset not in {"mosi", "mosei", "simsv2"}:
        raise ValueError("Unknown dataset: {}".format(dataset))
    targets, predictions = np.asarray(targets).reshape(-1), np.asarray(predictions).reshape(-1)
    if targets.size == 0 or targets.shape != predictions.shape:
        raise ValueError("Expected equally sized, nonempty predictions and targets")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Metrics require finite inputs")
    limit = 1 if dataset == "simsv2" else 3
    if (np.abs(targets) > limit + 1e-6).any():
        raise ValueError("Targets exceed the dataset's label range")
    if dataset == "simsv2":
        # Upstream SIMSv2 clips before both regression and classification metrics.
        targets, predictions = np.clip(targets, -1, 1), np.clip(predictions, -1, 1)
    result = {"mae": float(np.abs(targets - predictions).mean()), "pearson": None}
    if len(targets) > 1 and np.ptp(targets) > 0 and np.ptp(predictions) > 0:
        result["pearson"] = float(np.corrcoef(targets, predictions)[0, 1])
    if dataset == "simsv2":
        for n in (2, 3, 5):
            truth, pred = simsv2_classes(targets, n), simsv2_classes(predictions, n)
            result["acc" + str(n)] = float((truth == pred).mean())
            if n == 2:
                result["f1"] = float(f1_score(truth, pred, average="weighted", zero_division=0))
        return result
    result["acc7"] = float((np.rint(np.clip(targets, -3, 3)) == np.rint(np.clip(predictions, -3, 3))).mean())
    for name, selected, comparator in (
        ("nonneg_neg", np.ones(len(targets), dtype=bool), lambda x: x >= 0),
        ("pos_neg", targets != 0, lambda x: x > 0),
    ):
        truth, pred = comparator(targets[selected]), comparator(predictions[selected])
        result["acc2_" + name] = float((truth == pred).mean()) if truth.size else None
        result["f1_" + name] = float(f1_score(truth, pred, average="weighted", zero_division=0)) if truth.size else None
    return result
