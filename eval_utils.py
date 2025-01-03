import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

def sensitivity(tn, fp, fn, tp):
    return tp / (tp + fn)


def specificity(tn, fp, fn, tp):
    return tn / (tn + fp)


def ppv(tn, fp, fn, tp):
    return tp / (tp + fp)


def npv(tn, fp, fn, tp):
    return tn / (tn + fn)


def f1p(tn, fp, fn, tp):
    _ppv = ppv(tn, fp, fn, tp)
    _sens = sensitivity(tn, fp, fn, tp)

    f1 = 2 * (_ppv * _sens) / (_ppv + _sens)

    return f1


def f1n(tn, fp, fn, tp):
    _npv = npv(tn, fp, fn, tp)
    _spec = specificity(tn, fp, fn, tp)

    f1 = 2 * (_npv * _spec) / (_npv + _spec)

    return f1


def get_classification_metrics(y_true, y_score):
    thresholds = np.linspace(0.01, 0.99, 100)
    confusion_matrices = [confusion_matrix(y_true, y_score > t).ravel() for t in thresholds]

    metrics = {
        "sensitivity": [sensitivity(*cm) for cm in confusion_matrices],
        "specificity": [specificity(*cm) for cm in confusion_matrices],
        "ppv": [ppv(*cm) for cm in confusion_matrices],
        "npv": [npv(*cm) for cm in confusion_matrices],
        "f1p": [f1p(*cm) for cm in confusion_matrices],
        "f1n": [f1n(*cm) for cm in confusion_matrices],
        "roc": roc_curve(y_true, y_score),
        "pr": precision_recall_curve(y_true, y_score),
    }

    return metrics
