import multiprocessing

import numpy as np
import pandas as pd
from typing import List, Optional
from .embedding_matrix import format_embedding_with_overlapping_windows, create_univariate_embedding
from .pattern_mining import mine_patterns_univariate
from .pattern_filter import filter_maximum_variance, filter_jaccard_similarity_in_embedding, filter_pbad_embedding, filter_jaccard_similarity
from .PatternBasedEmbedder import PatternBasedEmbedder
from .PatternBasedEmbedding import PatternBasedEmbedding


class FrequentPatternMiningEmbedder(PatternBasedEmbedder):
    """
    Mines frequent sequential patterns from the time series data to
    construct a pattern-based embedding. This process happens in a
    few different steps.

    First, the time series is preprocessed by extracting fixed-size
    subsequences at multiple resolution from each attribute of the
    time series. These subsequences are consequently discretized
    through SAX. By employing multiple resolutions we enable to
    capture short-term and long-term behavior.

    Second, patterns are mined in every resolution and for each
    attribute independently. The patterns are then pruned using
    Jaccard similarity and the relative support to obtain a set
    of more interesting patterns.

    Third, The patterns are converted to an embedding matrix by
    computing at which interval the pattern and replacing those
    values in the matrix by the relative support of that pattern.
    If multiple subsequences overlap and therefore cover the same
    observations, the average value of each subsequence is taken.

    Parameters
    ----------
    window_sizes : Optional[List[int]], default=None
        The different window sizes or resolutions to use for extracting
        subsequences from the time series, from which patterns will be
        mined.
    stride : int, default = 1
        The amount with which the sliding windows will shift to extract
        fixed-size subsequences.
    word_size : int, default=16
        The word size for SAX discretization. This is the number of
        discrete symbols to maintain in the discretized word through
        PAA.
    alphabet_size : int, default=3
        The alphabet size for SAX discretization. This is th number
        of discrete symbols used to create words.
    binning_method : str, default=``'gloabl'``
        How to compute the discrete symbols with SAX. The possible
        options are:

            - ``'global'``: Consider all values in the time series
              and discretize them jointly;

            - ``'local'``: Discretize the values of each subsequence
              independently;

            - ``'k_means'``: Use K-Means clustering to cluster the
              values in the time series and assign a dicrete label
              to each cluster.

    top_k_patterns : int, default=25
        The number of patterns with maximum relative support to mine
        in each resolution.
    min_pattern_size : int, default=3
        The minimum length of a pattern before it should actually be
        considered relevant.
    max_pattern_size : int, default=10,
        The maximum length of a pattern to consider it.
    duration : float, default=1.2
        The constraint on the relative duration of a pattern before
        it covers a window. Specifically, for a pattern of length ``L``,
        there may be at most ``np.floor(L * duration)`` gaps. The value
        should therefore be larger than 1.
    do_mdl : bool, default=False
        Whether to prune the mined patterns based on the MDL-principle.
    min_r_support : float, default=0.0
        The minimum relative support of a pattern before it should be
        considered relevant.
    max_r_support : float, default=1.0
        The maximum relative support a pattern may have for it to still
        be interesting.
    jaccard_similarity_threshold : float, default=0.9
        The threshold on the Jaccard similarity for two patterns to be
        considered similar.
    nb_largest_variance : int, default=50
        The number of patterns with largest variance to maintain.
    do_pca : bool, default=False
        Whether PCA should be performed to maintain the number of linear
        combinations of patterns with largest variance.
    n_jobs : int, default=1
        The number of jobs that are allowed to run in parallel.
    """

    def __init__(self,
                 window_sizes: List[int] = None,
                 stride: int = 1,
                 word_size: int = 16,
                 alphabet_size: int = 3,
                 binning_method: str = 'global',
                 top_k_patterns: int = 25,
                 min_pattern_size: int = 3,
                 max_pattern_size: int = 10,
                 duration: float = 1.2,
                 do_mdl: bool = False,
                 min_r_support: float = 0.0,
                 max_r_support: float = 1.0,
                 jaccard_similarity_threshold: float = 0.9,
                 nb_largest_variance: int = 50,
                 do_pca: bool = False,
                 n_jobs: int = 1):
        self.window_sizes: Optional[List[int]] = window_sizes
        self.stride: int = stride
        self.word_size: int = word_size
        self.alphabet_size: int = alphabet_size
        self.binning_method: str = binning_method
        self.top_k_patterns: int = top_k_patterns
        self.min_pattern_size: int = min_pattern_size
        self.max_pattern_size: int = max_pattern_size
        self.duration: float = duration
        self.do_mdl: bool = do_mdl
        self.min_r_support: float = min_r_support
        self.max_r_support: float = max_r_support
        self.jaccard_similarity_threshold: float = jaccard_similarity_threshold
        self.nb_largest_variance: int = nb_largest_variance
        self.do_pca: bool = do_pca
        self.n_jobs: int = n_jobs

    def fit(self, time_series: np.ndarray, y=None) -> 'FrequentPatternMiningEmbedder':
        """
        Fitting and transforming a time series using the :py:class:`FrequentPatternMiningEmbedder`
        is closely tight together, and therefore the :py:func:`~FrequentPatternMiningEmbedder.fit()`
        method should not be used directly. Instead, use the :py:func:`~FrequentPatternMiningEmbedder.fit_transform()`
        method.

        Raises
        ------
        exception : Exception
            Upon calling this method.
        """
        raise Exception('The fit and transform steps of the `FrequentPatternMiner` are closely tied together.'
                        'Instead of using `.fit()` and `.transform()` independently, use the `.fit_transform()` method!')

    def transform(self, trend_data: np.ndarray) -> PatternBasedEmbedding:
        """
        Fitting and transforming a time series using the :py:class:`FrequentPatternMiningEmbedder`
        is closely tight together, and therefore the :py:func:`~FrequentPatternMiningEmbedder.transform()`
        method should not be used directly. Instead, use the :py:func:`~FrequentPatternMiningEmbedder.fit_transform()`
        method.

        Raises
        ------
        exception : Exception
            Upon calling this method.
        """
        raise Exception('The fit and transform steps of the `FrequentPatternMiner` are closely tied together.'
                        'Instead of using `.fit()` and `.transform()` independently, use the `.fit_transform()` method!')

    def fit_transform(self, time_series: np.ndarray, y=None) -> PatternBasedEmbedding:
        """
        Computes a pattern-based embedding for the given time series by mining
        frequent sequential patterns. This process happens in a few steps. (1)
        The time series is preprocessed by using multiple sliding windows of
        varying size to extract subsequences, which are discretized using SAX.
        (2) Frequent sequential patterns are mined within the discretized
        subsequences at each resolution and for each attribute. These patterns
        pruned to obtain a set of more informative patterns. (3) The embedding
        matrix is computed by checking at which location in the time series
        each pattern occurs.

        Parameters
        ----------
        time_series : np.ndarray of shape (n_samples, n_attributes)
            The time series from which the patterns should be mined, with ``n_samples`` the number of
            observations in the time series, and ``n_attributes`` the number of attributes.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        embedding : PatternBasedEmbedding
            The computed pattern-based embedding.
        """
        # Automatically infer the window sizes if they were not provided.
        window_sizes = self.window_sizes
        if window_sizes is None:
            start_exponent = int(np.log2(self.word_size))
            end_exponents = max(start_exponent + 1, min(11, int(np.log2(time_series.shape[0]))))
            window_sizes = [i for i in 2 ** np.arange(start_exponent, end_exponents)]

        # Construct the jobs for mining the patterns in every resolution independently
        jobs = [
            (
                pd.DataFrame(data={
                    'average_value': time_series[:, attribute_id],
                    'time': range(time_series.shape[0])
                }),
                attribute_id,
                window_size
            )
            for attribute_id in range(time_series.shape[1])
            for window_size in window_sizes
        ]

        # Execute the jobs in parallel if requested
        if self.n_jobs > 1:
            with multiprocessing.Pool(self.n_jobs) as pool:
                pool_results = pool.starmap(self._create_attribute_resolution_embedding, jobs)
        else:
            pool_results = []
            for args in jobs:
                results = self._create_attribute_resolution_embedding(*args)
                pool_results.append(results)

        individual_embeddings = {}
        all_patterns_combined = {}
        # Process the results of the various resolutions
        for (attribute_id, embedding, patterns) in pool_results:
            if patterns.shape[0] == 0:  # Skip if no patterns were found
                continue
            if attribute_id not in individual_embeddings.keys():
                individual_embeddings[attribute_id] = []
                all_patterns_combined[attribute_id] = []
            individual_embeddings[attribute_id].append(embedding)
            all_patterns_combined[attribute_id].append(patterns)

        # Concatenate the embeddings per attribute
        for attribute_id in individual_embeddings.keys():
            individual_embeddings[attribute_id] = np.concatenate(individual_embeddings[attribute_id])
            all_patterns_combined[attribute_id] = pd.concat(all_patterns_combined[attribute_id])
            all_patterns_combined[attribute_id].reset_index(drop=True, inplace=True)

        # Reduce the dimension of every embedding in every attribute
        nb_remaining_patterns = 0
        for attribute_id in individual_embeddings.keys():
            individual_embeddings[attribute_id], all_patterns_combined[attribute_id] = \
                filter_jaccard_similarity_in_embedding(individual_embeddings[attribute_id], all_patterns_combined[attribute_id], self.jaccard_similarity_threshold)
            individual_embeddings[attribute_id], all_patterns_combined[attribute_id] = \
                filter_maximum_variance(individual_embeddings[attribute_id], all_patterns_combined[attribute_id], self.nb_largest_variance, self.do_pca)
            nb_remaining_patterns += individual_embeddings[attribute_id].shape[0]

        # Creat one embedding and pattern matrix
        embedding = np.concatenate(list(individual_embeddings.values()))
        patterns = pd.concat(list(all_patterns_combined.values()))
        patterns.reset_index(drop=True, inplace=True)

        return PatternBasedEmbedding(time_series, embedding, patterns)

    def _create_attribute_resolution_embedding(self, data: pd.DataFrame, attribute_id, interval):
        """
        Create an embedding for the given data with the given resolution

        :param data: The (univariate) time series, a pandas DataFrame with the columns 'average_value' and 'time'
        :param attribute_id: The ID of the given univariate time series in a multivariate time series (only for
                             administrative aspects and bookkeeping)
        :param interval: The interval length or window size that the subsequences should have
        :return:
        """
        # Mine the patterns in the time series
        patterns, pbad_embedding, _, _ = mine_patterns_univariate(
            data=data,
            files_prefix='temp/' + str(attribute_id) + '_' + str(interval) + '/',
            interval=interval,
            stride=self.stride,
            nb_symbols=self.word_size,
            nb_bins=self.alphabet_size,
            binning_method=self.binning_method,
            top_k_patterns=self.top_k_patterns,
            min_pattern_size=self.min_pattern_size,
            max_pattern_size=self.max_pattern_size,
            pattern_duration=self.duration,
            do_mdl=self.do_mdl
        )
        if patterns.shape[0] == 0:  # Return if no patterns have been found
            return attribute_id, np.ndarray([]), patterns
        patterns['interval'] = interval

        # Filter the patterns in this specific resolution
        nb_original_patterns = patterns.shape[0]
        if 0.0 < self.min_r_support < self.max_r_support:
            patterns = patterns.loc[patterns['rsupport'] >= self.min_r_support]
        if self.min_r_support < self.max_r_support <= 1.0:
            patterns = patterns.loc[patterns['rsupport'] <= self.max_r_support]
        patterns = filter_jaccard_similarity(patterns, self.jaccard_similarity_threshold)
        # Format pbad embedding to remove filtered patterns
        if nb_original_patterns > patterns.shape[0]:
            pbad_embedding = filter_pbad_embedding(patterns, pbad_embedding)

        # Create the embedding for the resolution
        new_embedding = create_univariate_embedding(pbad_embedding, patterns.shape[0])
        new_embedding = format_embedding_with_overlapping_windows(new_embedding, interval, self.stride)

        # Reset index of patterns, thus pattern i corresponds to row i in the embedding
        patterns.reset_index(drop=True, inplace=True)

        return attribute_id, new_embedding, patterns
