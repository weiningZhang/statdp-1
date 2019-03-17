# MIT License
#
# Copyright (c) 2018 Yuxin (Ryan) Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import functools
import itertools
import logging
import math
import multiprocessing as mp

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def _hypergeometric(cx, cy, iterations):
    return 1 - stats.hypergeom.cdf(cx, 2 * iterations, iterations, cx + cy)


def test_statistics(cx, cy, epsilon, iterations, process_pool=None):
    """ Calculate p-value based on observed results.
    :param cx: The observed count of running algorithm with database 1 that falls into the event
    :param cy:The observed count of running algorithm with database 2 that falls into the event
    :param epsilon: The epsilon to test for.
    :param iterations: The total iterations for running algorithm.
    :param process_pool: The process pool to run on, run in single core if None
    :return: p-value
    """
    # average p value
    sample_num = 200
    if process_pool is None:
        return np.fromiter((_hypergeometric(cx, cy, iterations)
                            for cx in np.random.binomial(cx, 1.0 / (np.exp(epsilon)), sample_num)),
                           dtype=np.float64, count=sample_num).mean()
    else:
        # bind cy and iterations to _hypergeometric function and feed different cx into it
        return np.fromiter(process_pool.imap_unordered(functools.partial(_hypergeometric, cy=cy, iterations=iterations),
                                                       np.random.binomial(cx, 1.0 / (np.exp(epsilon)), sample_num),
                                                       chunksize=int(sample_num / mp.cpu_count())),
                           dtype=np.float64, count=sample_num).mean()


def run_algorithm(algorithm, d1, d2, kwargs, event, iterations):
    """ Run the algorithm for :iteration: times, count and return the number of iterations in :event:,
    event search space is auto-generated if not specified.
    :param algorithm: The algorithm to run.
    :param d1: The D1 input to run.
    :param d2: The D2 input to run.
    :param kwargs: The keyword arguments for the algorithm.
    :param event: The event to test, auto generate event search space if None.
    :param iterations: The iterations to run.
    :return: [(cx, cy), ...], [(d1, d2, kwargs, event), ...]
    """
    np.random.seed()
    # support multiple return values, each return value is stored as a row in result_d1 / result_d2
    # e.g if an algorithm returns (1, 1), result_d1 / result_d2 would be like
    # [
    #   [x, x, x, ..., x],
    #   [x, x, x, ..., x]
    # ]

    # get return type by a sample run
    sample_result = algorithm(d1, **kwargs)
    if np.issubdtype(type(sample_result), np.number):
        result_d1 = (np.fromiter((algorithm(d1, **kwargs) for _ in range(iterations)),
                                 dtype=type(sample_result), count=iterations), )
        result_d2 = (np.fromiter((algorithm(d2, **kwargs) for _ in range(iterations)),
                                 dtype=type(sample_result), count=iterations), )
    elif isinstance(sample_result, (tuple, list)):
        # run the algorithm and store the corresponding return value into vanilla python list first
        result_d1, result_d2 = tuple([] for _ in range(len(sample_result))),  tuple([] for _ in range(len(sample_result)))
        for _ in range(iterations):
            out_1 = algorithm(d1, **kwargs)
            out_2 = algorithm(d2, **kwargs)
            for row, (value_1, value_2) in enumerate(zip(out_1, out_2)):
                result_d1[row].append(value_1)
                result_d2[row].append(value_2)

        # convert the python list to numpy array
        result_d1 = tuple(np.asarray(row) for row in result_d1)
        result_d2 = tuple(np.asarray(row) for row in result_d2)
    else:
        raise ValueError('Unsupported return type: {}'.format(type(sample_result)))

    # get desired search space for each return value
    event_search_space = []
    if event is None:
        for row in range(len(result_d1)):
            # determine the event search space based on the return type
            combined_result = np.concatenate((result_d1[row], result_d2[row]))
            unique = np.unique(combined_result)

            # categorical output
            if len(unique) < iterations * 0.002:
                event_search_space.append(tuple(key for key in unique))
            else:
                combined_result.sort()
                # find the densest 70% range
                search_range = int(0.7 * len(combined_result))
                search_max = min(range(search_range, len(combined_result)),
                                 key=lambda x: combined_result[x] - combined_result[x - search_range])
                search_min = search_max - search_range

                event_search_space.append(
                    tuple((-float('inf'), alpha) for alpha in
                          np.linspace(combined_result[search_min], combined_result[search_max], num=10)))

        logger.debug('search space is set to {}'.format(' × '.join(str(event) for event in event_search_space)))
    else:
        # if `event` is given, it should have the corresponding events for each return value
        if len(event) != len(result_d1):
            raise ValueError('Given event should have the same dimension as return value.')
        # here if the event is given, we carefully construct the search space in the following format:
        # [first_event] × [second_event] × [third_event] × ... × [last_event]
        # so that when the search begins, only one possible combination can happen which is the given event
        event_search_space = ((separate_event, ) for separate_event in event)

    counts, input_event_pairs = [], []
    for event in itertools.product(*event_search_space):
        cx_check, cy_check = np.full(iterations, True, dtype=np.bool), np.full(iterations, True, dtype=np.bool)
        # check for all events in the return values
        for row in range(len(result_d1)):
            if np.issubdtype(type(event[row]), np.number):
                cx_check = np.logical_and(cx_check, result_d1[row] == event[row])
                cy_check = np.logical_and(cy_check, result_d2[row] == event[row])
            else:
                cx_check = np.logical_and(cx_check, np.logical_and(result_d1[row] > event[row][0],
                                                                   result_d1[row] < event[row][1]))
                cy_check = np.logical_and(cy_check, np.logical_and(result_d2[row] > event[row][0],
                                                                   result_d2[row] < event[row][1]))

        cx, cy = np.count_nonzero(cx_check), np.count_nonzero(cy_check)
        counts.append((cx, cy) if cx > cy else (cy, cx))
        input_event_pairs.append((d1, d2, kwargs, event))
    return counts, input_event_pairs


def hypothesis_test(algorithm, d1, d2, kwargs, event, epsilon, iterations, report_p2=True, process_pool=None):
    """ Run hypothesis tests on given input and events.
    :param algorithm: The algorithm to run on
    :param kwargs: The keyword arguments the algorithm needs
    :param d1: Database 1
    :param d2: Database 2
    :param event: The event set
    :param iterations: Number of iterations to run
    :param epsilon: The epsilon value to test for
    :param report_p2: The boolean to whether report p2 or not
    :param process_pool: The process pool to use, run with single process if None
    :return: p values
    """
    if process_pool is None:
        ((cx, cy), *_), _ = run_algorithm(algorithm, d1, d2, kwargs, event, iterations)
        if report_p2:
            return test_statistics(cx, cy, epsilon, iterations), test_statistics(cy, cx, epsilon, iterations)
        else:
            return test_statistics(cx, cy, epsilon, iterations)
    else:
        process_iterations = [int(math.floor(float(iterations) / mp.cpu_count())) for _ in range(mp.cpu_count())]
        # add the remaining iterations to the last index
        process_iterations[mp.cpu_count() - 1] += iterations % process_iterations[mp.cpu_count() - 1]

        # start the pool to run the algorithm and collects the statistics
        cx, cy = 0, 0
        for ((local_cx, local_cy), *_), _ in process_pool.imap_unordered(
                functools.partial(run_algorithm, algorithm, d1, d2, kwargs, event), process_iterations):
            cx += local_cx
            cy += local_cy
        cx, cy = (cx, cy) if cx > cy else (cy, cx)

        # calculate and return p value
        if report_p2:
            return test_statistics(cx, cy, epsilon, iterations, process_pool), \
                   test_statistics(cy, cx, epsilon, iterations, process_pool)
        else:
            return test_statistics(cx, cy, epsilon, iterations, process_pool)
