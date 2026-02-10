
import pandas as pd
import pathlib

try:
    from npbad.symbolic_representation import create_segments, create_windows, discretise_segments_equal_distance_bins_local, discretise_segments_kmeans
    from npbad.symbolic_representation import discretise_segments_equal_distance_bins_global
    from npbad.preprocess_timeseries import min_max_norm

    from npbad.ipbad.main import save_transactions
    from npbad.ipbad.pattern_mining_petsc import mine_patterns_and_create_embedding
    from npbad.ipbad.minimum_description_length import post_process_mdl, _compute_occurrences_for_pattern
    _HAS_NPBAD = True
except ImportError:
    _HAS_NPBAD = False


def mine_patterns_univariate(
        data: pd.DataFrame,
        files_prefix: str = '../temp/',
        # Windowing
        interval: int = 24,
        stride: int = 24,
        # Discrete
        nb_symbols: int = 10,
        nb_bins: int = 5,
        binning_method: str = 'global',
        # Pattern Mining
        top_k_patterns: int = 10000,
        min_pattern_size: int = 4,
        max_pattern_size: int = 10,
        pattern_duration: float = 1.2,
        do_mdl: bool = True
):
    """
    Mine the frequent sequential patterns in the given time series.

    :param data: A univariate time series, a DataFrame with two columns: 'time' and 'average_value'
    :param files_prefix: The prefix for the temporary files, used to avoid rage conditions
    :param interval: The interval length of the subsequences
    :param stride: The stride to use for extracting subsequences
    :param nb_symbols: The word size to use for SAX
    :param nb_bins: The alphabet size to use for SAX
    :param binning_method: The method used to assign symbols: 'global', 'local' (per subsquence) or 'k_means'
    :param top_k_patterns: The number of patterns with highest frequency to mine
    :param min_pattern_size: The minimal length of the patterns
    :param max_pattern_size: The maximum pattern size
    :param pattern_duration: The constraint on the relative duration of a pattern
    :param do_mdl: Whether to filter the patterns using MDL

    :return: A DataFrame containing the patterns, the PBAD embedding corresponding to these patterns,
             the windows with raw values and the discrete segments
    """
    if not _HAS_NPBAD:
        raise ImportError("npbad is not installed. Please install it to use PaTSS.")

    transaction_file_name = files_prefix + 'sequences.txt'
    pattern_file_name = files_prefix + 'patterns.txt'
    embedding_file_name = files_prefix + 'embedding.txt'

    # Create a symbolic representation of the time series
    if binning_method == 'global':
        data_normalized = min_max_norm(data)
        windows = create_windows(data_normalized, interval=interval, stride=stride)
        segments = create_segments(data_normalized, windows)
        segments_discrete = discretise_segments_equal_distance_bins_global(segments, no_symbols=nb_symbols, no_bins=nb_bins)
    else:
        windows = create_windows(data, interval=interval, stride=stride)
        segments = create_segments(data, windows)
        if binning_method == 'local':
            segments_discrete = discretise_segments_equal_distance_bins_local(segments, no_symbols=nb_symbols, no_bins=nb_bins)
        elif binning_method == 'kmeans':
            segments_discrete = discretise_segments_kmeans(data, segments, no_symbols=nb_symbols, no_bins=nb_bins)
        else:
            raise Exception("Invalid binning method given: %s!" % binning_method)

    # Save the transactions to the correct file, which can be used for mining the patterns
    #print(transaction_file_name)
    #print(segments_discrete)
    save_transactions(transaction_file_name, segments_discrete)
    nb_transactions = len(segments_discrete)

    # Effectively mine the patterns
    patterns_df, embedding = mine_patterns_and_create_embedding(
        transaction_file_name,
        no_transactions=nb_transactions,
        top_k=top_k_patterns,
        min_size=min_pattern_size,
        max_size=max_pattern_size,
        duration=pattern_duration,
        patterns_fname=pattern_file_name,
        embedding_fname=embedding_file_name)

    # Filter the patterns using the Minimum Description Length
    if do_mdl:
        patterns_df, embedding = post_process_mdl(patterns_df, embedding, segments_discrete, no_symbols=nb_symbols, duration=pattern_duration)
    else:
        _compute_occurrences_for_pattern(patterns_df, embedding)

    return patterns_df, embedding, windows, segments_discrete
