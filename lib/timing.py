""" module for time tests
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
    usage:
    @timing
    def function_to_be_timed():
        ...
    """
    def wrap(*args, **kwargs):
        time1 = time.process_time()
        ret = func(*args, **kwargs)
        total = time.process_time() - time1
        hours = int(total / 3600)
        minutes = int((total % 3600) / 60)
        seconds = int((total % 60))
        if hours is not 0:
            print('{:s}() took {}h {}m {}s'.format(
                func.__name__, hours, minutes, seconds))
        elif minutes is not 0:
            print('{:s}() took {}m {}s'.format(
                func.__name__, minutes, seconds))
        elif seconds is not 0:
            print('{:s}() took {:.2f}s'.format(func.__name__, total))
        else:
            print('{:s}() took {:.2f}ms'.format(func.__name__, total * 1000))
        return ret
    return wrap
