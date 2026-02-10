
import numpy as np
import time
from baselines.patss.evaluation.utility import convert_to_borders
from baselines.patss.workflows.Logger import Logger
from claspy.segmentation import BinaryClaSPSegmentation


def run_clasp(univariate_time_series, ground_truth, config, logger: Logger):
    """
    Run ClaSP on the given time series to identify a semantic segmentation.

    :param univariate_time_series: A list of pandas DataFrames, and each DataFrame consists
                                   of two columns: 'average_value' and 'time'. Here, the list
                                   should consist of one DataFrame, thus a univariate time series
    :param ground_truth: A dictionary containing the ground truth window size and ground truth
                         number of segment boundaries
    :param config: A dictionary containing the settings to use within ClaSP
    :param logger: A Logger object used for logging the progress of ClaSP

    :return: A numpy array containing the identified segment boundaries, and the ClaSP object
             used to segment the time series.
    """
    # ClaSP can not cope with multivariate time series
    if len(univariate_time_series) > 1:
        raise Exception('ClaSP only handles univariate time series (05/04/2023)!')

    # We only need to provide the raw time series values to ClaSP
    data = np.array(univariate_time_series[0]['average_value'])

    # Whether to use ground truth number of segments or not
    n_segments = 'learn'
    if 'use_ground_truth_n_segments' in config.keys() and config['use_ground_truth_n_segments']:
        n_segments = len(convert_to_borders(ground_truth['segmentation'], univariate_time_series)) + 1

    # Whether to use the ground truth window size or not
    window_size = 'suss'
    if 'use_ground_truth_window_size' in config.keys() and config['use_ground_truth_window_size']:
        window_size = ground_truth['segment_length']
        # If the window size is too large (i.e., the window size multiplied with the exclusion radius is
        # exceeds half of the time series), then no proper classification problem can be constructed. In
        # this case we set the window size to just fit this constraint.
        if data.shape[0] < 2 * window_size * 5:  # 5 is the default exclusion radius
            window_size = data.shape[0] // 10

    # Start ClaSP
    start = time.time()
    clasp = BinaryClaSPSegmentation(n_segments=n_segments, window_size=window_size)
    try:
        # The first execution of ClaSP results in a Reference error from Numba
        # I assume this is due to some part of the code not yet being compiled
        # If ClaSP is executed again, then there is no problem
        # (this is similar for future ClaSP runs within the same python call)
        borders = clasp.fit_predict(data)

    except ReferenceError:
        # Log the exception
        logger.write('>>> Exception in ClaSP\n' +
                     'A Reference error occurred during execution, restarting ClaSP.\n' +
                     'Total time (waisted): %f seconds\n\n' % (time.time() - start))

        # Restart the timer and try to predict the borders again
        start = time.time()
        borders = clasp.fit_predict(data)

    # Log successful completion of the segmentation
    logger.write('>>> Fitted ClaSP\n' +
                 'Borders: %s\n' % str(borders) +
                 'Total time: %f seconds\n\n' % (time.time() - start))

    # Return the segment boundaries and the ClaSP object
    return borders, clasp
