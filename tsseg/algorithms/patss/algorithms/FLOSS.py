
import time
import numpy as np
import stumpy
from baselines.patss.evaluation.utility import convert_to_borders
from workflows.Logger import Logger


def run_floss(univariate_time_series, ground_truth, config, logger: Logger):
    """
    Run FLOSS on the given time series to identify a semantic segmentation.

    :param univariate_time_series: A list of pandas DataFrames, and each DataFrame consists
                                   of two columns: 'average_value' and 'time'. Here, the list
                                   should consist of one DataFrame, thus a univariate time series
    :param ground_truth: A dictionary containing the ground truth window size and ground truth
                         number of segment boundaries
    :param config: A dictionary containing the settings to use within FLOSS
    :param logger: A Logger object used for logging the progress of FLOSS

    :return: A numpy array containing the identified segment boundaries, and the corrected arc
             crossing curve
    """
    if len(univariate_time_series) > 1:
        raise Exception('FLOSS: Do not focus on multivariate time series!')

    # We only focus on the raw data within the time series
    data = univariate_time_series[0]['average_value']

    # Get the ground truth number of segments (n_regimes) and the ground truth window size
    n_regimes = len(convert_to_borders(ground_truth['segmentation'], univariate_time_series)) + 1
    window_size = ground_truth['segment_length']

    # Start the execution of FLOSS
    # Definitely not the most efficient way to do this, but it works
    start = time.time()
    temporal_constraint = window_size * config['temporal_constraint_factor']
    exclusion_zone = window_size
    index_array = np.zeros(data.shape[0], dtype=np.int64)
    for t in range(data.shape[0]):
        # Check whether we can extract a valid window at the current index
        window = data[t - window_size:t]
        if len(window) < window_size:
            continue

        # Compute the nearest segment within the temporal constraint that occurs before the window
        index_before = None
        distance_before = np.infty
        start_segment_before = max(0, t - temporal_constraint)
        end_segment_before = max(0, t - exclusion_zone)
        segment_before = data[start_segment_before:end_segment_before]
        if len(segment_before) > window_size:
            matrix_profile_before = stumpy.stump(T_A=window, T_B=segment_before, m=window_size, ignore_trivial=False)
            index_before = matrix_profile_before[:, 1] + start_segment_before
            distance_before = matrix_profile_before[:, 0]

        # Compute the nearest segment within the temporal constraint that occurs after the window
        index_after = None
        distance_after = np.infty
        start_segment_after = min(data.shape[0], t + exclusion_zone)
        end_segment_after = min(data.shape[0], t + temporal_constraint)
        segment_after = data[start_segment_after:end_segment_after]
        if len(segment_after) > window_size:
            matrix_profile_after = stumpy.stump(T_A=window, T_B=segment_after, m=window_size, ignore_trivial=False)
            index_after = matrix_profile_after[:, 1] + start_segment_after
            distance_after = matrix_profile_after[:, 0]

        # Now compare the two segments and choose the one with the smallest distance
        if distance_before < distance_after:
            index_array[t] = index_before
        else:
            index_array[t] = index_after

    # Compute the idealized corrected arc crossing curve
    a, b, c = calc_parabola_vertex(0, 0, data.shape[0], 0, data.shape[0]/2, 0.5*data.shape[0])
    x = np.arange(0, data.shape[0])
    iac = a * x * x + b * x + c  # The standard iac, without temporal constraint
    if temporal_constraint < iac.shape[0]/2: # If the temporal constraint is too large, there is no adjustment needed
        iac[temporal_constraint:-temporal_constraint] = iac[temporal_constraint]
    corrected_arc_crossings, regime_locations = stumpy.fluss(index_array, L=window_size, n_regimes=n_regimes, custom_iac=iac)

    logger.write('>>> Computed regime locations with FLOSS\n' +
                 'Regime locations: %s\n' % str(regime_locations) +
                 'Total time: %f seconds\n\n' % (time.time() - start))

    return regime_locations, corrected_arc_crossings


def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    """
    I found this here: https://chris35wills.github.io/parabola_python/
    Which was adapted from here: https://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    """

    denominator = (x1 - x2) * (x1 - x3) * (x2 - x3)
    a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denominator
    b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denominator
    c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denominator

    return a, b, c