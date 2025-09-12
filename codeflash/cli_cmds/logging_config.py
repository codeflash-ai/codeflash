VERBOSE_LOGGING_FORMAT = "%(asctime)s [%(pathname)s:%(lineno)s in fn %(funcName)s] %(message)s"
LOGGING_FORMAT = "[%(levelname)s] %(message)s"
BARE_LOGGING_FORMAT = "%(message)s"


def set_level(level: int, *, echo_setting: bool = True) -> None:
    import logging
    import time

    from rich.logging import RichHandler

    from codeflash.cli_cmds.console import console

    class RuleAfterLogHandler(RichHandler):
        def emit(self, record: logging.LogRecord) -> None:
            super().emit(record)
            console.rule()

    logging.basicConfig(
        level=level,
        handlers=[
            RuleAfterLogHandler(rich_tracebacks=True, markup=False, console=console, show_path=False, show_time=False)
        ],
        format=BARE_LOGGING_FORMAT,
    )
    logging.getLogger().setLevel(level)
    if echo_setting:
        if level == logging.DEBUG:
            logging.Formatter.converter = time.gmtime
            logging.basicConfig(
                format=VERBOSE_LOGGING_FORMAT,
                handlers=[
                    RuleAfterLogHandler(
                        rich_tracebacks=True, markup=False, console=console, show_path=False, show_time=False
                    )
                ],
                force=True,
            )
            logging.info("Verbose DEBUG logging enabled")
        else:
            logging.info("Logging level set to INFO")
