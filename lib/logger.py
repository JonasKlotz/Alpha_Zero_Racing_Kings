"""
Usage:
    from lib.logger import get_logger
    log = get_logger(__name__)

    log.info("info test message")

Enforce level for all existing and future loggers:
    from lib.logger import set_level_global
    set_level_global("INFO", enforce=True)

Make console only show "CRITICAL" messages:
    from lib.logger import set_level_stdout
    set_level_stdout("CRITICAL")

Note: console won't print messages that are filtered by logger or globally
"""

import os
import sys
import time
import logging


# config
LOG_DIR = "logs"
_LOG_FILE_FMT = "%Y-%m-%d_%H.%M.%S.log"

_LOG_FMT = '%(asctime)s %(levelname)s @ %(name)s : %(message)s'
_LOG_DATEFMT = '%H:%M:%S'

_LOG_LEVEL_STDOUT = logging.DEBUG    # level used for log prints to console
# used for new loggers at log_level "DEFAULT" or when _GLOBAL_LEVEL_ENFORCED
_LOG_LEVEL_GLOBAL = logging.DEBUG
_GLOBAL_LEVEL_ENFORCED = False


# setup
__loggers = []

__LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}
__LOG_LEVELS_STR = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL"
}

if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)


def __file_handler(file):
    """ creates a new file handler """
    handler = logging.FileHandler(file)
    handler.setFormatter(__LOG_FORMATTER)
    return handler


__LOG_FILE = os.path.join(LOG_DIR, time.strftime(_LOG_FILE_FMT))
__LOG_FORMATTER = logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT)

__STDOUT_HANDLER = logging.StreamHandler(sys.stdout)
__STDOUT_HANDLER.setFormatter(__LOG_FORMATTER)
__STDOUT_HANDLER.setLevel(_LOG_LEVEL_STDOUT)

__LOG_FILE_HANDLER = __file_handler(__LOG_FILE)


def __get_log_level(level):
    """ Returns: log level in logging (int) format """
    assert level in __LOG_LEVELS, "Unrecognized log level " + level
    return __LOG_LEVELS[level]


def get_logger(name, level="DEFAULT", file=None):
    """
    Args:
        name (str): name of the logger
        file (str), optional: if given, will output log to a separate file
    Returns:
        logger
    """
    logger = logging.getLogger(name)
    logger.addHandler(__STDOUT_HANDLER)

    if file is not None:
        logger.addHandler(__file_handler(file))
    else:
        logger.addHandler(__LOG_FILE_HANDLER)

    if level == "DEFAULT":
        log_level = _LOG_LEVEL_GLOBAL
    else:
        log_level = __get_log_level(level)
        if _GLOBAL_LEVEL_ENFORCED and not log_level == _LOG_LEVEL_GLOBAL:
            log_level = _LOG_LEVEL_GLOBAL
            log.info("log level was enforced globally to %s for @ %s",
                     __LOG_LEVELS_STR[_LOG_LEVEL_GLOBAL], logger.name)

    logger.setLevel(log_level)

    __loggers.append(logger)

    return logger


def set_level_global(level, enforce=False):
    """ Sets the global log level for all loggers
    Args:
        level (str): log level, one of: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        enforce (bool): if set to True, the global log level will also
                        be enforced for all future loggers
    """
    global _GLOBAL_LEVEL_ENFORCED
    _GLOBAL_LEVEL_ENFORCED = enforce

    global _LOG_LEVEL_GLOBAL
    _LOG_LEVEL_GLOBAL = __get_log_level(level)

    for logger in __loggers:
        logger.setLevel(_LOG_LEVEL_GLOBAL)

    log.info("Global log level set to %s",
             __LOG_LEVELS_STR[_LOG_LEVEL_GLOBAL])


def set_level_stdout(level):
    """ Sets the log level for console output
    Args:
        level (str): log level, one of: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    """
    global _LOG_LEVEL_STDOUT
    _LOG_LEVEL_STDOUT = __get_log_level(level)

    __STDOUT_HANDLER.setLevel(logging.INFO)
    log.info("Log level for console set to %s", level)
    __STDOUT_HANDLER.setLevel(_LOG_LEVEL_STDOUT)


log = get_logger("Log")

if __name__ == "__main__":

    # Tests

    log2 = get_logger("Log2", level="CRITICAL")
    log2.info("XXXXX You won't see this, Log2 doesn't log info messages")

    set_level_global("DEBUG", enforce=True)

    log2.info("YYYYYY You see this, because global log level was changed")

    log3 = get_logger("Log3", level="CRITICAL")
    log3.debug(
        "YYYYYY This should now still be seen, because global is enforced for new loggers")

    set_level_stdout("CRITICAL")
    set_level_global("INFO")
    log.warning("ZZZZZZ This won't be seen on console, but in logs")
