
import numpy as np


def create_univariate_embedding(pbad_embedding, nb_patterns: int):
    """
    Wrapper function in case you want to embed the time series differently
    using the patterns.

    :param pbad_embedding: The PBAD embedding that is computed by the pattern
                           mining. This is a list of tuples, containing the ID
                           of the pattern and the support of that pattern
    :param nb_patterns: The total number of patterns mined

    :return: A 2D numpy array with the embedding of the time series
    """
    return __create_support_embedding_matrix(pbad_embedding, nb_patterns)


def format_embedding_with_overlapping_windows(overlapping_windows_embedding, interval, stride):
    """
    Format the embedding such that each individual time unit is embedded in case there are
    overlapping windows due to a stride smaller than the interval length

    :param overlapping_windows_embedding: The embedding of the time series per subsequence, thus
                                          each time unit is embedded in multiple subsequences
    :param interval: The interval length of a subsequence used for mining patterns
    :param stride: The stride used in extracting the subsequences with a rolling window

    :return: An embedding such that each time unit has a single feature representation
    """
    nb_windows_overlap = int(np.ceil(interval / stride))
    new_embedding = np.zeros((overlapping_windows_embedding.shape[0], overlapping_windows_embedding.shape[1] + nb_windows_overlap))

    # Average the first few windows that have less than 'nb_windows_overlap' windows overlapping
    running_sum = np.zeros((overlapping_windows_embedding.shape[0]))
    for i in range(nb_windows_overlap):
        running_sum += overlapping_windows_embedding[:, i]
        new_embedding[:, i] = running_sum / (i+1)
    # Average the windows that have exactly 'nb_windows_overlap' windows overlapping
    for i in range(nb_windows_overlap, overlapping_windows_embedding.shape[1]):
        running_sum += overlapping_windows_embedding[:, i]
        running_sum -= overlapping_windows_embedding[:, i-nb_windows_overlap]
        new_embedding[:, i] = running_sum / nb_windows_overlap
    # Average the first few windows that have more than 'nb_windows_overlap' windows overlapping
    for i in range(nb_windows_overlap):
        new_window_id = overlapping_windows_embedding.shape[1] + i
        running_sum -= overlapping_windows_embedding[:, new_window_id-nb_windows_overlap]
        new_embedding[:, new_window_id] = running_sum / (nb_windows_overlap - i)

    return new_embedding


def __create_support_embedding_matrix(pbad_embedding, nb_patterns: int) -> np.ndarray:
    """
    Compute the embedding matrix by replacing the occurence of a pattern by its relative support
    and otherwise a zero.

    :param pbad_embedding: A list of tuples containing the embedding mined by PBAD
    :param nb_patterns: The total number of patterns that were mined.

    :return: A 2D numpy array containing the embedding matrix
    """
    support_embedding_matrix = np.zeros((nb_patterns, len(pbad_embedding)))
    translate = {}  # a mapping of the index in the PBAD embedding to the index in the new embedding
    translate_index = 0
    for window_id in range(len(pbad_embedding)):
        for embedding_value in pbad_embedding[window_id]:
            if embedding_value[0] not in translate.keys():
                translate[embedding_value[0]] = translate_index
                translate_index += 1
            support_embedding_matrix[translate[embedding_value[0]], window_id] = embedding_value[1]

    return support_embedding_matrix
