"""
This module offers all functionality to create a pattern-based embedding! In essence,
a :py:class:`PatternBasedEmbedder` is used to transform a time series into a
:py:class:`PatternBasedEmbedding`.
"""


from .PatternBasedEmbedding import PatternBasedEmbedding
from .PatternBasedEmbedder import PatternBasedEmbedder
from .FrequentPatternMiningEmbedder import FrequentPatternMiningEmbedder

__all__ = [
    'PatternBasedEmbedding',
    'PatternBasedEmbedder',
    'FrequentPatternMiningEmbedder'
]
