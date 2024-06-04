LOGGING_FORMAT = "[%(levelname)s] %(message)s"
BARE_LOGGING_FORMAT = "%(message)s"


def set_level(level: int, *, echo_setting: bool = True) -> None:
    import logging
    import sys

    logging.basicConfig(format=LOGGING_FORMAT, stream=sys.stdout)
    logging.getLogger().setLevel(level)

    if echo_setting:
        if level == logging.DEBUG:
            logging.debug("Verbose DEBUG logging enabled")
        else:
            logging.info("Logging level set to INFO")
