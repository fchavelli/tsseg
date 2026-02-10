import abc
import numpy as np
from .PatternBasedEmbedding import PatternBasedEmbedding


class PatternBasedEmbedder(abc.ABC):

    @abc.abstractmethod
    def fit(self, time_series: np.ndarray, y=None) -> 'PatternBasedEmbedder':
        """
        Fit this embedder to the given trend data, i.e., mining the patterns in the given time series

        Parameters
        ----------
        time_series : np.ndarray of shape (n_samples, n_attributes)
            The time series from which the patterns should be mined, with ``n_samples`` the number of
            observations in the time series, and ``n_attributes`` the number of attributes.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : PatternBasedEmbedder
            Returns the instance itself
        """
        raise NotImplementedError('Abstract method should be implemented in child!')

    @abc.abstractmethod
    def transform(self, time_series: np.ndarray) -> PatternBasedEmbedding:
        """
        Transforms the given trend data into a pattern-based embedding.

        Parameters
        ----------
        time_series : np.ndarray of shape (n_samples, n_attributes)
            The time series to transform into a pattern-based embedding, with
            `n_samples` the number of observations in the time series, and
            `n_attributes` the number of attributes.
            
        Returns
        -------
        embedding : PatternBasedEmbedding
            The computed pattern-based embedding.
        """
        raise NotImplementedError('Abstract method should be implemented in child!')

    def fit_transform(self, time_series: np.ndarray, y=None) -> PatternBasedEmbedding:
        """
        Fit the embedder on the given trend data and transform it to a pattern
        based embedding.

        Parameters
        ----------
        time_series : np.ndarray of shape (n_samples, n_attributes)
            The time series from which the patterns should be mined, with ``n_samples``
            the number of observations in the time series, and ``n_attributes`` the
            number of attributes.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        embedding : PatternBasedEmbedding
            The computed pattern-based embedding.
        """
        return self.fit(time_series, y).transform(time_series)
