"""
Logging helper methods
"""

from logging import StreamHandler, basicConfig, DEBUG, getLogger, Formatter

"""
we need to setup logger with a standard logging path
then add :
logger = getLogger(__name__)

in every class we want to log
"""


def setup_logger(log_filename):
    """
    setup a logger that logs lol
    :arg log_filename : path to logfile
    """
    format_str = '%(asctime)s@%(name)s %(levelname)s # %(message)s'
    basicConfig(filename=log_filename, level=DEBUG, format=format_str)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter(format_str))
    getLogger().addHandler(stream_handler)


if __name__ == '__main__':
    setup_logger("log.log")
    logger = getLogger("testing")
    logger.info("worked")
