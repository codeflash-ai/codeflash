VERBOSE_LOGGING_FORMAT = (
    "%(asctime)s %(levelname)s [%(pathname)s:%(lineno)s in function %(funcName)s] %(message)s"
)
LOGGING_FORMAT = "[%(levelname)s] %(message)s"
BARE_LOGGING_FORMAT = "%(message)s"


def set_level(level: int, *, echo_setting: bool = True) -> None:
    import logging
    import sys
    import time

    logging.basicConfig(format=LOGGING_FORMAT, stream=sys.stdout, force=True)
    logging.getLogger().setLevel(level)

    if echo_setting:
        if level == logging.DEBUG:
            logging.Formatter.converter = time.gmtime
            logging.basicConfig(format=VERBOSE_LOGGING_FORMAT, stream=sys.stdout, force=True)
            logging.debug("Verbose DEBUG logging enabled")
        else:
            logging.info("Logging level set to INFO")
