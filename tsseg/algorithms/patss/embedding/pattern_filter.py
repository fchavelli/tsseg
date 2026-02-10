
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def filter_jaccard_similarity(patterns: pd.DataFrame, threshold):
    """
    Filter the given patterns using the Jaccard similarity

    :param patterns: The patterns to filter
    :param threshold: The upper threshold for the Jaccard index

    :return: The patterns that remain after filtering
    """
    if threshold is None or threshold <= 0 or threshold >= 1:
        return patterns

    index_patterns_to_return = []
    occurrences_of_patterns_to_return = []

    # Iterate over the patterns, and if the Jaccard similarity of some pattern exceeds
    # the threshold, than the new pattern is not needed
    for pattern in patterns.index:
        temp = set(patterns.loc[pattern, 'instances'])
        add_pattern = True
        for j in range(len(index_patterns_to_return)):
            nb_intersection = len(temp.intersection(occurrences_of_patterns_to_return[j]))
            nb_union = len(temp.union(occurrences_of_patterns_to_return[j]))
            if nb_intersection / nb_union > threshold:
                add_pattern = False
                break
        if add_pattern:
            index_patterns_to_return.append(pattern)
            occurrences_of_patterns_to_return.append(temp)

    return patterns.loc[index_patterns_to_return]


def filter_jaccard_similarity_in_embedding(embedding: np.ndarray, patterns: pd.DataFrame, threshold):
    """
    Filter the patterns with the Jaccard similarity, but in the embedding space

    :param embedding: The embedding matrix that should be filtered with the jaccard similarity
    :param patterns: The patterns that should also be filtered, to still match the embedding
    :param threshold: The threshold used in the Jaccard similarity

    :return: Both the embedding and patterns, but without those instances that did not satisfy
             the given threshold in the Jaccard index
    """
    if threshold is None or threshold <= 0 or threshold >= 1:
        return embedding, patterns

    indices_to_keep = []
    occurrences_of_indices_to_keep = []

    # Iterate over the patterns, and if the Jaccard similarity of some pattern exceeds
    # the threshold, than the new pattern is not needed
    for i in range(embedding.shape[0]):
        occurrences_i = np.where(embedding[i, :] > 0, True, False)
        add_pattern = True
        for j in range(len(indices_to_keep)):
            # A pattern occurs if its embedding value is larger than zero. This means that the
            # intersection equals the AND of the rows in the embedding matrix and the union is
            # the OR of the rows
            nb_intersection = np.count_nonzero(np.logical_and(occurrences_i, occurrences_of_indices_to_keep[j]))
            nb_union = np.count_nonzero(np.logical_or(occurrences_i, occurrences_of_indices_to_keep[j]))
            if nb_intersection / nb_union > threshold:
                add_pattern = False
                break
        if add_pattern:
            indices_to_keep.append(i)
            occurrences_of_indices_to_keep.append(occurrences_i)

    return embedding[indices_to_keep, :], patterns.loc[indices_to_keep, :].reset_index()


def filter_maximum_variance(embedding: np.ndarray, patterns: pd.DataFrame, nb_patterns: int, do_pca: bool):
    """
    Filter the embedding by taking the maximum variance

    :param embedding: The embedding to filter
    :param patterns: The patterns that also should be filtered
    :param nb_patterns: The number of patterns to keep after filtering
    :param do_pca: Whether or not to do PCA. If this is True, than the patterns are first transformed
                   to a new space using a linear transformation, afterwhich the features with maximum
                   variance are selected.

    :return: Both the embedding and the patterns, but with the given number of features.
    """
    if 0 < nb_patterns < embedding.shape[0]:
        if do_pca:
            pca = PCA(n_components=nb_patterns)
            embedding = pca.fit_transform(embedding.transpose()).transpose()
            patterns = pd.DataFrame()  # The patterns have no meaning after PCA
        else:
            indices_to_keep = np.sort(np.argpartition(np.var(embedding, axis=1), -nb_patterns)[-nb_patterns:])
            embedding = embedding[indices_to_keep, :]
            patterns = patterns.loc[indices_to_keep, :]
    return embedding, patterns


def filter_pbad_embedding(filtered_patterns, pbad_embedding):
    return [
        [pbad_embedding_value for pbad_embedding_value in pbad_window_embedding if pbad_embedding_value[0] in filtered_patterns.id.values]
        for pbad_window_embedding in pbad_embedding
    ]
