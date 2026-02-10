from abc import ABC, abstractmethod
from typing import Any, List, Dict
import numpy as np

class BaseMetric(ABC):
    """Base class for all metrics."""

    def __init__(self, **kwargs):
        """
        Initializes the metric with optional parameters.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Computes the value of the metric.

        Args:
            y_true: Ground truth labels or change points.
            y_pred: Predicted labels or change points.
            **kwargs: Additional arguments for metric computation.

        Returns:
            A dictionary containing metric names and their values.
        """
        pass

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """Shortcut to call the compute method."""
        return self.compute(y_true, y_pred, **kwargs)