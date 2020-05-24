""" simple performance tracker  
imports:
    from lib.timing import timing, runtime_summary

Usage: add "@timing" decorator to wrap function
    @timing
    def function_to_be_timed():
        ...

OR directly apply the wrapper:
    timing(function_to_be_timed)()

report for accumulated run-times:
    runtime_summary()
"""

import time
import functools
from lib.logger import get_logger
log = get_logger("Timing")

_RT_START = time.perf_counter()

_RT_BINS = dict()
_RT_FUNCS = dict()


def timing(_func=None, *, bin=None):
    """ decorator for timing functions
    Usage:
        @timing
        def function_to_be_timed():
            ...

    OR directly apply the wrapper:
        timing(function_to_be_timed)()

    Optional: specify a bin to collect accumulated run-time
        @timing(bin="Tree Search")
        def function_to_be_timed():
            ...
        timing(function_to_be_timed)(bin="Tree Search")
    """
    def decorator(func):
        @functools.wraps(func)
        def wrap(*args, **kwargs):  # pylint: disable=no-value-for-parameter
            # pylint: disable = E, W, R, C
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            run_time = time.perf_counter() - start_time
            name = func.__name__ + "()"
            log.info('{:s} took {:s}'.format(name, prettify(run_time)))
            # store run_time of func
            if name not in _RT_FUNCS:
                _RT_FUNCS[name] = run_time
            else:
                _RT_FUNCS[name] += run_time
            # store run_time in bin
            if bin is not None:
                if bin not in _RT_BINS:
                    _RT_BINS[bin] = run_time
                else:
                    _RT_BINS[bin] += run_time
            return value
        return wrap

    if _func is None:               # call with argument
        return decorator
    return decorator(_func)         # call without argument


def runtime_summary():
    """ logs the accumulated run-times for functions and bins
    """
    def summary_write(_dict):
        total = sum(_dict.values())
        elapsed_since_import = time.perf_counter() - _RT_START
        for item in sorted(_dict.items(), key=lambda item: item[1], reverse=True):
            _bin = item[0]
            run_time = item[1]
            perc = int(100 * run_time / total)
            perc_import = int(100 * run_time / elapsed_since_import)
            log.info("{}: {}% total: {} ({:.4f} seconds) [{}%]".format(
                _bin, perc, prettify(run_time), run_time, perc_import))

    log.info("================== Runtime Summary ==================")
    if _RT_BINS:
        log.info("Bins:")
        summary_write(_RT_BINS)
    log.info("Functions:")
    summary_write(_RT_FUNCS)
    log.info("[runtime % since import]")


def prettify(seconds):
    """ converts seconds to human readable time
    Args:
        seconds (float)
    Returns:
        (string): time in pretty output format
    """
    hours = int(seconds / 3600)
    mins = int((seconds % 3600) / 60)
    sec = int((seconds % 60))
    if hours > 0:
        return "{}h {}m {}s".format(hours, mins, sec)
    elif mins > 0:
        return "{}m {}s".format(mins, sec)
    elif sec > 0:
        return "{:.2f}s".format(seconds)
    else:
        return "{}ms".format(int(seconds * 1000))
