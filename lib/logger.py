""" 
Usage:
    from lib.logger import get_logger
    log = get_logger(__name__)

    log.info("info test message")
"""

import os
import sys
import time
import logging


# config
_LOG_DIR = "logs"
_LOG_FILE_FMT = "%Y-%m-%d_%H.%M.%S.log"

_LOG_FMT = '%(asctime)s %(levelname)s @ %(name)s : %(message)s'
_LOG_DATEFMT = '%H:%M:%S'

_LOG_STDOUT = True


# setup
if not os.path.isdir(_LOG_DIR):
    os.mkdir(_LOG_DIR)


def file_handler(file):
    """ creates a new file handler
    """
    handler = logging.FileHandler(file)
    handler.setFormatter(__LOG_FORMATTER)
    return handler


__LOG_FILE = os.path.join(_LOG_DIR, time.strftime(_LOG_FILE_FMT))
__LOG_FORMATTER = logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT)

__LOG_STREAM_HANDLER = logging.StreamHandler(sys.stdout)
__LOG_STREAM_HANDLER.setFormatter(__LOG_FORMATTER)

__LOG_FILE_HANDLER = file_handler(__LOG_FILE)


def get_logger(name="std", file=None):
    """
    Args:
        name (str): name of the logger
        file (str), optional: if given, will output log to a new file
    Returns:
        logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if _LOG_STDOUT:
        logger.addHandler(__LOG_STREAM_HANDLER)

    if file is not None:
        logger.addHandler(file_handler(file))
    else:
        logger.addHandler(__LOG_FILE_HANDLER)
    return logger
