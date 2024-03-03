import logging
import os.path
import subprocess


def format_code(formatter_cmd: str, path: str) -> str:
    # TODO: Only allow a particular whitelist of formatters here to prevent arbitrary code execution
    if formatter_cmd.lower() == "disabled":
        with open(path, "r", encoding="utf8") as f:
            new_code = f.read()
        return new_code
    formatter_cmd_list = [chunk for chunk in formatter_cmd.split(" ") if chunk != ""]
    logging.info(f"Formatting code with {formatter_cmd} ...")
    # black currently does not have a stable public API, so we are using the CLI
    # the main problem is custom config parsing https://github.com/psf/black/issues/779
    assert os.path.exists(path), f"File {path} does not exist. Cannot format the file. Exiting..."
    result = subprocess.run(
        formatter_cmd_list + [path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode == 0:
        logging.info("OK")
    else:
        logging.error(f"Failed to format code with {formatter_cmd}")
    with open(path, "r", encoding="utf8") as f:
        new_code = f.read()
    return new_code
