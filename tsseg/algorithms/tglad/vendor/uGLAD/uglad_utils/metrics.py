import numpy as np
from sklearn import metrics


def get_auc(y, scores):
    """Compute ROC-AUC and AUPR for binary edge detection."""
    y = np.array(y).astype(int)
    fpr, tpr, _ = metrics.roc_curve(y, scores)
    roc_auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(y, scores)
    return roc_auc, aupr


def _safe_div(num, den):
    return num / den if den else 0.0


def reportMetrics(trueG, G, beta=1):
    """Compute precision/recall style metrics for precision matrices."""
    trueG = trueG.real
    G = G.real
    d = G.shape[-1]

    G_binary = np.where(G != 0, 1, 0)
    trueG_binary = np.where(trueG != 0, 1, 0)

    indices_triu = np.triu_indices(d, 1)
    trueEdges = trueG_binary[indices_triu]
    predEdges = G_binary[indices_triu]

    predEdges_auc = G[indices_triu]
    auc, aupr = get_auc(trueEdges, np.abs(predEdges_auc))

    TP = np.sum(trueEdges * predEdges)
    mismatches = np.logical_xor(trueEdges, predEdges)
    FP = np.sum(mismatches * predEdges)
    P = np.sum(predEdges)
    T = np.sum(trueEdges)
    F = len(trueEdges) - T
    SHD = np.sum(mismatches)
    FN = np.sum(mismatches * trueEdges)

    FDR = _safe_div(FP, P)
    TPR = _safe_div(TP, T)
    FPR = _safe_div(FP, F)
    precision = _safe_div(TP, TP + FP)
    recall = _safe_div(TP, TP + FN)

    num = (1 + beta ** 2) * TP
    den = (1 + beta ** 2) * TP + beta ** 2 * FN + FP
    Fbeta = _safe_div(num, den)

    return {
        "FDR": FDR,
        "TPR": TPR,
        "FPR": FPR,
        "SHD": SHD,
        "nnzTrue": T,
        "nnzPred": P,
        "precision": precision,
        "recall": recall,
        "Fbeta": Fbeta,
        "aupr": aupr,
        "auc": auc,
    }
