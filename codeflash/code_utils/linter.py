import logging
import subprocess


def lint_code(path) -> str:
    logging.info("Formatting code with black...")
    # black currently does not have a stable public API, so we are using the CLI
    # the main problem is custom config parsing https://github.com/psf/black/issues/779
    result = subprocess.run(["black", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        logging.info("OK")
    else:
        logging.error("Failed to format")
    with open(path, "r") as f:
        new_code = f.read()
    return new_code
