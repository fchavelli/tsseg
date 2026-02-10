
import time
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from baselines.patss.embedding.embedding_matrix import create_univariate_embedding, format_embedding_with_overlapping_windows
from baselines.patss.embedding.pattern_mining import mine_patterns_univariate
from baselines.patss.embedding.pattern_filter import filter_pbad_embedding, filter_jaccard_similarity, filter_jaccard_similarity_in_embedding, filter_maximum_variance

from baselines.patss.workflows.Logger import Logger


# The default parameters used in PaTSS
__DEFAULTS_HYPER_PARAMETERS = {
    'pattern_mining': {
        # Parameters regarding the segmentation
        'resolution': -1,
        'stride': 1,
        'max_resolution_exponent': 10,
        # Parameters regarding sax
        'word_size': 16,
        'alphabet_size': 3,
        'binning_method': 'global',
        # Parameters regarding effective pattern mining
        'top_k_patterns': 25,
        'min_pattern_size': 3,
        'max_pattern_size': 10,
        'duration': 1.2,
        'do_mdl': False
    },

    'pattern_filter': {
        'min_r_support': 0.0,
        'max_r_support': 1.0,
        'jaccard_similarity_threshold': 0.9,
        'nb_largest_variance': 50,
        'do_pca': False,
    },

    'segmentation' : {
        'algorithm' : "logistic_regression",
        'regularization': 0.1,
        'n_clusters' : list(range(2, 10)),
    }
}


def run_patss(directory, multivariate_time_series, length_time_series: int, config, logger: Logger):
    """
    Run PaTSS on the given time series to identify a semantic segmentation with gradual state transitions.

    :param directory: The directory containing the experiment that is being executed. This path will be used
                      to save intermediate files
    :param multivariate_time_series: A list of pandas DataFrames, and each DataFrame consists
                                   of two columns: 'average_value' and 'time'.
    :param length_time_series: The length of the time series, that is the number of measurements
    :param config: A dictionary containing the settings to use within PaTSS
    :param logger: A Logger object used for logging the progress of PaTSS

    :return: A 2D-numpy area containing the probability distribution over the various semantic segments, a dictionary
             containing the embedding of every attribute with as key the attribute ID (index in multivariate_time_series
             list), and a dictionary containing the mined patterns that were used in the embedding with similar keys. The
             patterns for a certain attribute are ordered according to the embedding matrix at the same attribute. The
             embeddings were concatenated for segmentation, but we separate them such that we can verify on which parts
             PaTSS focuses within each attribute.
    """

    # All the hyper parameters that are not filled in are set to default values
    __add_default_parameters(config)

    # If no window size is given, then the smallest window size equals the word size of SAX
    if config['pattern_mining']['resolution'] == -1:
        start_interval_exponents = int(np.log2(config['pattern_mining']['word_size']))
    else:
        start_interval_exponents = int(np.log2(config['pattern_mining']['resolution']))

    # Construct the jobs for mining the patterns in every resolution independently
    end_interval_exponents = max(start_interval_exponents + 1, min(config['pattern_mining']['max_resolution_exponent'] + 1, int(np.log2(length_time_series))))
    jobs = [
        (
            multivariate_time_series[attribute_id],
            directory,
            attribute_id,
            interval,
            config
        )
        for attribute_id in range(len(multivariate_time_series))
        for interval in 2**np.arange(start_interval_exponents, end_interval_exponents)
    ]

    # Execute the jobs, in parallel if requested by the config
    start = time.time()
    n_jobs = config['n_jobs'] if 'n_jobs' in config.keys() else 1
    if n_jobs > 1:
        logger.write('>>> Mining the patterns with %d parallel jobs\n' % config['n_jobs'])
        with multiprocessing.Pool(config['n_jobs']) as pool:
            pool_results = pool.starmap(__create_attribute_resolution_embedding, jobs)
        log_information = True
    else:
        logger.write('>>> Mining the patterns through one process\n')
        pool_results = []
        for args in jobs:
            results = __create_attribute_resolution_embedding(*args)
            logger.write(results[-1])
            pool_results.append(results)
        log_information = False
    logger.write('>>> Mined the patterns\n' +
                 'Total time: %f seconds\n\n' % (time.time() - start))

    start = time.time()
    individual_embeddings = {}
    all_patterns_combined = {}
    # Process the results of the various resolutions
    for (attribute_id, embedding, patterns, logging_information) in pool_results:
        if log_information:
            logger.write(logging_information)
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
    logger.write('>>> Created the embedding\n' +
                 ''.join(['nb patterns remaining for attribute %d: %s\n' % (attribute_id, individual_embeddings[attribute_id].shape) for attribute_id in all_patterns_combined.keys()]) +
                 'Total time: %f seconds\n\n' % (time.time() - start))

    # Reduce the dimension of every embedding in every attribute
    start = time.time()
    for attribute_id in individual_embeddings.keys():
        individual_embeddings[attribute_id], all_patterns_combined[attribute_id] = \
            filter_jaccard_similarity_in_embedding(individual_embeddings[attribute_id], all_patterns_combined[attribute_id], config['pattern_filter']['jaccard_similarity_threshold'])
        individual_embeddings[attribute_id], all_patterns_combined[attribute_id] = \
            filter_maximum_variance(individual_embeddings[attribute_id], all_patterns_combined[attribute_id], config['pattern_filter']['nb_largest_variance'], config['pattern_filter']['do_pca'])
    logger.write('>>> Post-processed embedding\n' +
                 ''.join(['dimension embedding %d: %s\n' % (attribute_id, individual_embeddings[attribute_id].shape) for attribute_id in all_patterns_combined.keys()]) +
                 'Total time: %f seconds\n\n' % (time.time() - start))

    # Segment the time series, in parallel if requested
    start = time.time()
    logger.write('>>> Segmenting the time series')
    if n_jobs > 1:
        segmentation, logging_information_segmentation = __segment_embedding_parallel(individual_embeddings, config['segmentation'], n_jobs)
        logger.write(logging_information_segmentation)
    else:
        segmentation = __segment_embedding(individual_embeddings, config['segmentation'], logger)
    logger.write('Segment labels: %s\n' % segmentation +
                 'Total time: %f seconds\n\n' % (time.time() - start))

    return segmentation, individual_embeddings, all_patterns_combined


def __add_default_parameters(config):
    """
    Add the default parameters (shown at the top of this file) to the given configuration

    :param config: The config file to which the default parameters should be added.

    :return: Nothing is returned, the given dictionary is adjusted in place.
    """
    __add_defaults_from_dictionary(config, __DEFAULTS_HYPER_PARAMETERS)


def __add_defaults_from_dictionary(config, defaults):
    """
    A recursive function to add all the values from the given dictionary to
    the other dictionary.

    :param config: The dictionary to which the values should be added
    :param defaults: The dictionary from which the values should be added

    :return: Nothing is returned, the given dictionary is adjusted in place.
    """
    for key in defaults.keys():
        if key not in config.keys():
            config[key] = defaults[key]
        elif type(defaults[key]) == dict:
            __add_defaults_from_dictionary(config[key], defaults[key])


def __create_attribute_resolution_embedding(data: pd.DataFrame, directory, attribute_id, interval, config):
    """
    Create an embedding for the given data with the given resolution

    :param data: The (univariate) time series, a pandas DataFrame with the columns 'average_value' and 'time'
    :param directory: The directory in which temporary files can be saved
    :param attribute_id: The ID of the given univariate time series in a multivariate time series (only for
                         administrative aspects and bookkeeping)
    :param interval: The interval length or window size that the subsequences should have
    :param config: A dictionary with the hyper parameters to use

    :return:
    """
    logging_information = ''

    # Mine the patterns in the time series
    start = time.time()
    patterns, pbad_embedding, _, _ = mine_patterns_univariate(
        data=data,
        files_prefix=directory + 'patss/temp/' + str(attribute_id) + '_' + str(interval) + '/',
        interval=interval,
        stride=config['pattern_mining']['stride'],
        nb_symbols=config['pattern_mining']['word_size'],
        nb_bins=config['pattern_mining']['alphabet_size'],
        binning_method=config['pattern_mining']['binning_method'],
        top_k_patterns=config['pattern_mining']['top_k_patterns'],
        min_pattern_size=config['pattern_mining']['min_pattern_size'],
        max_pattern_size=config['pattern_mining']['max_pattern_size'],
        pattern_duration=config['pattern_mining']['duration'],
        do_mdl=config['pattern_mining']['do_mdl']
    )
    logging_information += ('>>> Mined patterns in attribute %d with [interval, stride]=[%d, %d]\n' % (attribute_id, interval, config['pattern_mining']['stride']) +
                            'nb patterns: %d\n' % patterns.shape[0])
    if patterns.shape[0] == 0:  # Return if no patterns have been found
        logging_information += 'Skip this interval length for this attribute because no patterns were found'
        return attribute_id, np.ndarray([]), patterns, logging_information
    logging_information += ('max relative support: %f\n' % np.max(patterns.rsupport.values) +
                            'min relative support: %f\n' % np.min(patterns.rsupport.values) +
                            'Total time: %f seconds\n\n' % (time.time() - start))
    patterns['interval'] = interval

    # Filter the patterns in this specific resolution
    start = time.time()
    nb_original_patterns = patterns.shape[0]
    if 0.0 < config['pattern_filter']['min_r_support'] < config['pattern_filter']['max_r_support']:
        patterns = patterns.loc[patterns['rsupport'] >= config['pattern_filter']['min_r_support']]
    if config['pattern_filter']['min_r_support'] < config['pattern_filter']['max_r_support'] <= 1.0:
        patterns = patterns.loc[patterns['rsupport'] <= config['pattern_filter']['max_r_support']]
    patterns = filter_jaccard_similarity(patterns, config['pattern_filter']['jaccard_similarity_threshold'])
    # Format pbad embedding to remove filtered patterns
    if nb_original_patterns > patterns.shape[0]:
        pbad_embedding = filter_pbad_embedding(patterns, pbad_embedding)
    logging_information += ('>>> Filtered patterns in attribute %d with [interval, stride]=[%d, %d]\n' % (attribute_id, interval, config['pattern_mining']['stride']) +
                            'nb patterns remaining: %s\n' % patterns.shape[0] +
                            'Total time: %f seconds\n\n' % (time.time() - start))

    # Create the embedding for the resolution
    start = time.time()
    new_embedding = create_univariate_embedding(pbad_embedding, patterns.shape[0])
    new_embedding = format_embedding_with_overlapping_windows(new_embedding, interval, config['pattern_mining']['stride'])
    logging_information += ('>>> Created the embedding in attribute %d with [interval, stride]=[%d, %d]\n' % (attribute_id, interval, config['pattern_mining']['stride']) +
                            'dimension: (%s, %s)\n' % new_embedding.shape +
                            'Total time: %f seconds\n\n' % (time.time() - start))

    # Reset index of patterns, thus pattern i corresponds to row i in the embedding
    patterns.reset_index(drop=True, inplace=True)

    return attribute_id, new_embedding, patterns, logging_information


def __segment_embedding(embedding, parameters, logger):
    """
    Segment the given embedding using the given hyper parameters.

    :param embedding: A dictionary containing the embedding for each attribute in a multivariate time series
    :param parameters: The hyper parameters to use for semantic segmentation
    :param logger: The logger object used for logging the process of semantic segmenting the embedding

    :return: The best segmentation found according to the silhouette method
    """
    if parameters['algorithm'] not in ['logistic_regression', 'gmm', 'k_means']:
        raise Exception("Invalid segmentation technique given '%s'" % parameters['algorithm'])

    # Compute a semantic segmentation for every number of clusters
    formatted_embedding = np.concatenate(list(embedding.values())).transpose()
    best_silhouette_avg = -1.5  # A bit smaller than -1 (minimum silhouette score) such that the first segmentation is always better
    best_segmentation = None
    for n_segments in sorted(parameters['n_clusters']):
        n_segments, silhouette_avg, segmentation = __sub_segmentation(n_segments, formatted_embedding, parameters['algorithm'], parameters['regularization'])
        logger.write('Silhouette score for %d clusters: %f\n' % (n_segments, silhouette_avg))
        # Select the segmentation with the largest silhouette score
        if silhouette_avg > best_silhouette_avg:
            best_silhouette_avg = silhouette_avg
            best_segmentation = segmentation

    # Perform an additional logistic regression step to learn the probability distribution
    if parameters['algorithm'] == 'logistic_regression':
        model = LogisticRegression(C=parameters['regularization'], multi_class='ovr', penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=0)
        best_segmentation = __predict_probabilities(model.fit(formatted_embedding, best_segmentation), formatted_embedding)

    return best_segmentation


def __segment_embedding_parallel(embedding, parameters, n_jobs):
    """
    Segment the embedding using multiple parallel jobs. Each job will compute the silhouette score for
    a certain number of clusters. If n_jobs == 1, you should use __segment_embedding(...)

    :param embedding: A dictionary containing the embedding of the various attributes of a multivariate
                      time series.
    :param parameters: The hyper parameters to use for semantic segmentation
    :param n_jobs: The number of jobs to run in parallel

    :return: The probability distribution over the various semantic segments and the information about the
             semantic segmentation process
    """
    if parameters['algorithm'] not in ['logistic_regression', 'gmm', 'k_means']:
        raise Exception("Invalid segmentation technique given '%s'" % parameters['algorithm'])

    logging_information = '\n'

    # Concatenate the embedding of the various attributes
    formatted_embedding = np.concatenate(list(embedding.values())).transpose()

    # Compute the semantic segmentation in parallel
    args = [(n_segments, formatted_embedding, parameters['algorithm'], parameters['regularization']) for n_segments in parameters['n_clusters']]
    with multiprocessing.Pool(n_jobs) as pool:
        pool_results = pool.starmap(__sub_segmentation, args)

    # Find the best segmentation, the one with largest silhouette score
    best_silhouette_avg = -1.5  # A bit smaller than -1 (minimum silhouette score) such that the first segmentation is always better
    best_segmentation = None
    for (n_segments, silhouette_avg, segmentation) in pool_results:
        logging_information += 'Silhouette score for %d clusters: %f\n' % (n_segments, silhouette_avg)
        if silhouette_avg > best_silhouette_avg:
            best_silhouette_avg = silhouette_avg
            best_segmentation = segmentation

    # Apply an additional logistic regression step
    if parameters['algorithm'] == 'logistic_regression':
        model = LogisticRegression(C=parameters['regularization'], multi_class='ovr', penalty='elasticnet', solver='saga', l1_ratio=0.5, random_state=0)
        best_segmentation = __predict_probabilities(model.fit(formatted_embedding, best_segmentation), formatted_embedding)

    return best_segmentation, logging_information


def __sub_segmentation(n_segments, formatted_embedding, algorithm, regularization):
    """
    Segment the embedding with the given number of segments

    :param n_segments: The number of semantic segments to use for segmentation
    :param formatted_embedding: The embedding to use for segmentation. It must be formatted in the sense that
                                you need a single embedding for all the different attributes.
    :param algorithm: The algorithm/technique to use for semantic segmentation
    :param regularization: The regularization factor to use

    :return: The number of segments used (the given parameter), the average silhouette
              score, and the segmentation obtained
    """

    # Cluster the embedding
    if algorithm == 'gmm':
        clustering_algo = GaussianMixture(n_components=n_segments, reg_covar=regularization, random_state=0)
    else:
        clustering_algo = KMeans(n_clusters=n_segments, n_init='auto', init='k-means++', random_state=0)
    segmentation = clustering_algo.fit_predict(formatted_embedding)

    # Compute silhouette score
    if len(set(segmentation)) != n_segments:
        silhouette_avg = -1
    else:
        n = formatted_embedding.shape[0]
        sample_size = n if n < 2000 else 2000 + int(0.1 * (n - 2000))
        silhouette_avg = silhouette_score(formatted_embedding, segmentation, sample_size=sample_size)

    # Modify the segmentation in case of GMM
    if algorithm == 'gmm':
        segmentation = __predict_probabilities(clustering_algo, formatted_embedding)

    return n_segments, silhouette_avg, segmentation


def __predict_probabilities(solver, embedding):
    """
    Extract the probabilities of the given solver for the given embedding

    :param solver: The solver to predict the probabilities with
    :param embedding: The embedding for which the probabilities must be extracted

    :return: A 2D numpy array containing the probabilities for the various classes for each element in the embedding
    """
    return solver.predict_proba(embedding).transpose()
