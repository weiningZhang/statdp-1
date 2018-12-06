from inspect import isfunction
import numpy as np
import logging
import functools

logger = logging.getLogger(__name__)


def _evaluate_input(input_triplet, algorithm, iterations, search_space):
    d1, d2, kwargs = input_triplet
    result_d1 = np.fromiter((algorithm(d1, **kwargs) for _ in range(iterations)), dtype=np.float64)
    result_d2 = np.fromiter((algorithm(d2, **kwargs) for _ in range(iterations)), dtype=np.float64)

    if search_space is None:
        # determine the search space based on the return type
        # a subset of results to determine return type
        combined_result = np.concatenate((result_d1, result_d2))
        unique = np.unique(combined_result)

        # categorical output
        if len(unique) < iterations * 0.02 * 0.1:
            search_space = tuple(key for key in unique)
        else:
            combined_result.sort()
            average = combined_result.mean()
            idx = np.searchsorted(combined_result, average, side='left')
            # find the densest 70% range
            search_min = int(idx - 0.35 * len(combined_result)) if int(idx - 0.4 * len(combined_result)) > 0 else 0
            search_max = int(0.7 * len(combined_result) - (idx - search_min))

            search_space = tuple((-float('inf'), alpha) for alpha in
                                 np.linspace(combined_result[search_min], combined_result[search_max], num=25))

        logger.info('search space is set to {0}'.format(search_space))

    # bind the result_d1 and result_d2 to _evaluate_event, leaving out `event` argument to be filled
    counts = []
    for event in search_space:
        cx = np.count_nonzero(result_d1 == event if isinstance(event, (int, float)) else
                              np.logical_and(result_d1 > event[0], result_d1 < event[1]))
        cy = np.count_nonzero(result_d2 == event if isinstance(event, (int, float)) else
                              np.logical_and(result_d2 > event[0], result_d2 < event[1]))
        counts.append((cx, cy) if cx > cy else (cy, cx))
    return counts, search_space


def select_event(algorithm, input_list, epsilon, iterations=100000, search_space=None, process_pool=None):
    """
    :param algorithm: The algorithm to run on
    :param input_list: list of (d1, d2, kwargs) input pair for the algorithm to run
    :param epsilon: Test epsilon value
    :param iterations: The iterations to run algorithms
    :param search_space: The result search space to run on, auto-determine based on return type if None
    :param process_pool: The process pool to use, run with single process if None
    :return: (d1, d2, kwargs, event) pair which has minimum p value from search space
    """
    assert isfunction(algorithm)
    from .hypotest import test_statistics

    input_event_pairs = []
    p_values = []

    for (d1, d2, kwargs) in input_list:
        counts, search_space = _evaluate_input((d1, d2, kwargs), algorithm, iterations, search_space)
        threshold = 0.001 * iterations * np.exp(epsilon)

        input_p_values = [test_statistics(cx, cy, epsilon, iterations, process_pool=process_pool)
                          if cx + cy > threshold else float('inf') for (cx, cy) in counts]

        for (s, (cx, cy), p) in zip(search_space, counts, input_p_values):
            logger.debug('d1: %s | d2: %s | event: %s | p: %f | cx: %d | cy: %d | ratio: %f' %
                         (d1, d2, s, p, cx, cy, float(cy) / cx if cx != 0 else float('inf')))

        input_event_pairs.extend([(d1, d2, kwargs, event) for event in search_space])
        p_values.extend(input_p_values)

    # find an (d1, d2, kwargs, event) pair which has minimum p value from search space
    return min(zip(input_event_pairs, p_values), key=lambda zipped: zipped[1])[0]
