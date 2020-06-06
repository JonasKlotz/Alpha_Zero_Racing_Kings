""" Have a look at the logger.py's __main__ for detailed examples on usage;
Usage:
    from lib.logger import get_logger
    log = get_logger(__name__)

    log.info("info test message")

logger will print message to console and to unique log file in LOG_DIR (/logs)

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


# setup variables
__loggers = []

__LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "SILENCE": 100
}
__LOG_LEVELS_STR = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
    100: "SILENCE"
}

__LOG_FILE = os.path.join(LOG_DIR, time.strftime(_LOG_FILE_FMT))
__LOG_FORMATTER = logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT)

if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)


# setup handlers
__STDOUT_HANDLER = logging.StreamHandler(sys.stdout)
__STDOUT_HANDLER.setFormatter(__LOG_FORMATTER)
__STDOUT_HANDLER.setLevel(_LOG_LEVEL_STDOUT)


def __file_handler(file):
    """ creates a new file handler """
    handler = logging.FileHandler(file)
    handler.setFormatter(__LOG_FORMATTER)
    return handler


# create special logger for logger.py
__LOG_FILE_HANDLER = __file_handler(__LOG_FILE)

log = logging.getLogger("Log")
log.addHandler(__STDOUT_HANDLER)
log.addHandler(__LOG_FILE_HANDLER)
log.setLevel(logging.DEBUG)
log.info(time.strftime("Date: %d.%m.%Y"))


def __get_log_level(level):
    """ Returns: log level in logging (int) format """
    assert level in __LOG_LEVELS, "Unrecognized log level " + level
    return __LOG_LEVELS[level]


def __file_handler(file):
    """ creates a new file handler """
    handler = logging.FileHandler(file)
    handler.setFormatter(__LOG_FORMATTER)
    return handler


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
        level (str): log level, one of: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "SILENCE"
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
        level (str): log level, one of: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "SILENCE"
    """
    global _LOG_LEVEL_STDOUT
    _LOG_LEVEL_STDOUT = __get_log_level(level)

    __STDOUT_HANDLER.setLevel(logging.INFO)
    log.info("Log level for console set to %s", level)
    __STDOUT_HANDLER.setLevel(_LOG_LEVEL_STDOUT)


def silence_all():
    """ Silences all existing loggers, can be undone by setting global level """
    set_level_global("SILENCE")


def remove_all():
    """ Effectively removes all existing loggers and makes them useless """
    global __loggers
    for logger in __loggers:
        logger.handlers.clear()
    __loggers = []
    log.info("All existing loggers removed")


if __name__ == "__main__":

    # Tests

    log2 = get_logger("Log2", level="CRITICAL")
    log2.info("XXXXX You won't see this, since Log2 doesn't log info messages")

    set_level_global("DEBUG", enforce=True)

    log2.info("YYYYYY Now you see this, because global log level was changed")

    log3 = get_logger("Log3", level="CRITICAL")
    log3.debug(
        "YYYYYY This will still be seen, "
        "because global is enforced for new loggers")

    set_level_global("INFO")
    log3.warning("ZZZZZZ This will be seen again, global changed. "
                 "Also, global level is not enforced anymore for new loggers "
                 "(enforce=False is default)")

    set_level_stdout("CRITICAL")
    log3.warning("ZZZZZZ This won't be seen on console, but in logs")

    set_level_stdout("INFO")
    set_level_global("CRITICAL")
    log3.warning(
        "ZZZZZZ This won't be seen on console OR in logs, "
        "since it is filtered first by logger's level")

    silence_all()
    log3.info("AAAAA helllllp")
    log3.critical("AAAAA No one can hear this")

    set_level_global("INFO")
    log3.critical("AAAAA back again")

    remove_all()
    log2.info("This logger has become useless")
    log3.debug("And it won't ever log again")
