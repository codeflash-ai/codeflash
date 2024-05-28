LOGGING_FORMAT = "[%(levelname)s] %(message)s"


def set(level: int) -> None:
    import logging
    import sys

    logging.basicConfig(format=LOGGING_FORMAT, stream=sys.stdout)
    logging.getLogger().setLevel(level)

    if level == logging.DEBUG:
        logging.debug("Verbose DEBUG logging enabled")
    else:
        logging.info("Logging level set to INFO")
