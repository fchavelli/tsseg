
import abc
import numpy as np

from baselines.patss.embedding import PatternBasedEmbedding


class Segmentor(abc.ABC):

    @abc.abstractmethod
    def fit(self, pattern_based_embedding: PatternBasedEmbedding, y=None) -> 'Segmentor':
        """
        Fit this segmentor to the given embedding.

        Parameters
        ----------
        pattern_based_embedding : PatternBasedEmbedding
            The pattern-based embedding used for training this segmentor.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : Segmentor
            Returns the instance itself
        """
        raise NotImplementedError('Abstract method should be implemented in child!')

    @abc.abstractmethod
    def predict(self, pattern_based_embedding: PatternBasedEmbedding) -> np.ndarray:
        """
        Predicts the segment probabilities for the given pattern-based embedding.

        Parameters
        ----------
        pattern_based_embedding : PatternBasedEmbedding
            The pattern-based embedding to predict the segmentation for.

        Returns
        -------
        segmentation : np.ndarray of shape (n_segments, n_samples)
            The segmentation based on the given pattern-based embedding, which consists
            of ``n_segments`` different semantic segments for a time series with ´`n_samples``
            observations. The value ``segmentation[s, t]`` equals the probability of being
            in semantic segment ``s`` at time step ``t``.
        """
        raise NotImplementedError('Abstract method should be implemented in child!')

    def fit_predict(self, pattern_based_embedding: PatternBasedEmbedding, y=None) -> np.ndarray:
        """
        Fit this segmentor on the given pattern-based embedding and predict the
        semantic segmentation.

        Parameters
        ----------
        pattern_based_embedding : PatternBasedEmbedding
            The pattern-based embedding used for training this segmentor and predicting
            the semantic segmentation
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        segmentation : np.ndarray of shape (n_segments, n_samples)
            The segmentation based on the given pattern-based embedding, which consists
            of `n_segments` different semantic segments for a time series with ´n_samples
            observations. The value `segmentation[s, t]` equals the probability of being
            in semantic segment `s` at time step `t`.
        """
        return self.fit(pattern_based_embedding, y).predict(pattern_based_embedding)
