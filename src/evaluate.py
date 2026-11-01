"""Evaluation metrics that stay comparable across splits with different fraud rates."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REF_FRAUD_RATE = 5621 / 2646363  # training-set fraud rate (0.212%)


def recall_at_fpr(y_true, score, fpr_target):
    """Share of fraud caught while flagging at most fpr_target of genuine rows."""
    fpr, tpr, _ = roc_curve(y_true, score)
    i = np.searchsorted(fpr, fpr_target, side="right") - 1
    return float(tpr[i])


def prevalence_weights(y_true, ref_rate=REF_FRAUD_RATE):
    """Weight genuine rows so the split behaves as if its fraud rate were ref_rate."""
    y = np.asarray(y_true)
    fraud, genuine = (y == 1).sum(), (y == 0).sum()
    w_genuine = (fraud / genuine) / (ref_rate / (1 - ref_rate))
    return np.where(y == 1, 1.0, w_genuine)


def evaluate(y_true, score, ref_rate=REF_FRAUD_RATE):
    y = np.asarray(y_true)
    w = prevalence_weights(y, ref_rate)
    return {
        "roc_auc": roc_auc_score(y, score),
        "pr_auc_raw": average_precision_score(y, score),
        "pr_auc_baseline": float(y.mean()),
        "pr_auc_adj": average_precision_score(y, score, sample_weight=w),
        "recall@0.1%FPR": recall_at_fpr(y, score, 0.001),
        "recall@1%FPR": recall_at_fpr(y, score, 0.01),
    }