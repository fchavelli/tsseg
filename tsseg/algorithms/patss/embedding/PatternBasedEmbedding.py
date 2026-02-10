
import numpy as np
import pandas as pd


class PatternBasedEmbedding:
    """
    A class to maintain all information of a pattern-based embedding. This
    includes the time series itself, the embedding matrix, as well as the
    corresponding patterns.

    Parameters
    ----------
    time_series : np.ndarray of shape (n_samples, n_attributes)
        The time series from which the patterns were mined, with ``n_samples``
        the number of observations in the time series, and ``n_attributes`` the
        number of attributes.
    embedding_matrix : np.ndarray of shape (n_patterns, n_samples)
        The computed pattern-based embedding matrix, with ``n_patterns`` the
        number of patterns used to compute the embedding and ``n_samples``
        the number of observations in the time series.
    patterns : pd.DataFrame of length n_patterns
        The patterns that where used to compute the pattern-based embedding.
        Each row corresponds to a pattern, and the columns represent meta-
        information regarding the patterns, such as exact pattern.
    """

    def __init__(self, time_series: np.ndarray, embedding_matrix: np.ndarray, patterns: pd.DataFrame):
        self.__time_series: np.ndarray = time_series
        self.__embedding_matrix: np.ndarray = embedding_matrix
        self.__patterns: pd.DataFrame = patterns

    @property
    def time_series(self) -> np.ndarray:
        return self.__time_series

    @property
    def embedding_matrix(self) -> np.ndarray:
        return self.__embedding_matrix

    @property
    def patterns(self) -> pd.DataFrame:
        return self.__patterns
