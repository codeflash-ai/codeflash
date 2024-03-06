import logging
import sys

LOGGING_FORMAT = "[%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOGGING_FORMAT, stream=sys.stdout)
