""" simple performance tracker
import:
from lib.timing import timing

then add "@timing" to wrap any function that should be timed:
@timing
def function_to_be_timed():
    ...
"""

import time


def timing(func):
    """ wrapper for timing functions
    Usage:
    @timing
    def function_to_be_timed():
        ...
    """
    def wrap(*args, **kwargs):
        start_time = time.perf_counter()
        ret = func(*args, **kwargs)
        total = time.perf_counter() - start_time
        print('{:s}() took {:s}'.format(func.__name__, prettify(total)))
        return ret
    return wrap


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
        return "{:.2f}s".format(time)
    else:
        return "{:d}ms".format(time * 1000)
