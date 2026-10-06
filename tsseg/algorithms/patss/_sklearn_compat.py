"""scikit-learn compatibility for PaTSS's logistic-regression step.

PaTSS fits a one-vs-rest logistic regression with an elastic-net penalty
(``l1_ratio=0.5``, ``saga`` solver). The original code spelled it
``LogisticRegression(multi_class="ovr", penalty="elasticnet", ...)``, but
scikit-learn deprecated ``multi_class`` in 1.5 and removed it in 1.8, and 1.8
deprecates ``penalty`` (the elastic net is then set by ``l1_ratio`` alone).
``OneVsRestClassifier`` around a binary ``LogisticRegression`` fits the same
per-class models and normalises ``predict_proba`` the same way.
"""

from __future__ import annotations

import re

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

__all__ = ["ovr_elasticnet_logistic_regression"]


def _sklearn_version() -> tuple[int, int]:
    major, minor = re.findall(r"\d+", sklearn.__version__)[:2]
    return int(major), int(minor)


def ovr_elasticnet_logistic_regression(C: float) -> OneVsRestClassifier:
    """Return PaTSS's unfitted one-vs-rest elastic-net logistic regression.

    Parameters
    ----------
    C : float
        Inverse regularisation strength (PaTSS's ``regularization``).

    Returns
    -------
    OneVsRestClassifier
        One-vs-rest wrapper around ``LogisticRegression(C=C, solver="saga",
        l1_ratio=0.5, random_state=0)`` with an elastic-net penalty, spelled
        for the installed scikit-learn version.
    """
    kwargs = {"C": C, "solver": "saga", "l1_ratio": 0.5, "random_state": 0}
    if _sklearn_version() < (1, 8):
        kwargs["penalty"] = "elasticnet"
    return OneVsRestClassifier(LogisticRegression(**kwargs))
