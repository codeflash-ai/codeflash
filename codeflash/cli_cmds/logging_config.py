LOGGING_FORMAT = "[%(levelname)s] %(message)s"


def set(level: int) -> None:
    import logging
    import sys

    if level == logging.DEBUG:
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, stream=sys.stdout)
        logging.debug("Verbose DEBUG logging enabled")
    else:
        logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, stream=sys.stdout)
        logging.info("Logging level set to INFO")
