"""
This module enables to perform a semantic segmentation, given the
pattern-based embedding. This means you can provide the :py:class:`PatternBasedEmbedding`
of a time series to a py:class:`Segmentor` in order to predict the
segment probabilities.
"""

from .Segmentor import Segmentor
from .LogisticRegressionSegmentor import LogisticRegressionSegmentor

__all__ = [
    'Segmentor',
    'LogisticRegressionSegmentor',
]
