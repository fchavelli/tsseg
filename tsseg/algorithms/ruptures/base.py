"""Base classes mirroring the original `ruptures` API."""

from __future__ import annotations

import abc

from .utils import pairwise


class BaseEstimator(metaclass=abc.ABCMeta):
    """Base class for all change point detection estimators."""

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Fit the estimator to data."""

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """Predict change points for previously seen data."""

    @abc.abstractmethod
    def fit_predict(self, *args, **kwargs):
        """Convenience method combining :meth:`fit` and :meth:`predict`."""


class BaseCost(metaclass=abc.ABCMeta):
    """Base class for segment cost implementations."""

    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        """Prepare any cached statistics for the supplied signal."""

    @abc.abstractmethod
    def error(self, start: int, end: int) -> float:
        """Return the approximation cost for the segment ``[start, end)``."""

    def sum_of_costs(self, bkps: list[int]) -> float:
        """Return the total cost for a segmentation defined by ``bkps``."""

        return sum(self.error(s, e) for s, e in pairwise([0] + bkps))

    @property
    @abc.abstractmethod
    def model(self) -> str:
        """Identifier used by :func:`cost_factory`."""
