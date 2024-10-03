from codeflash.cli_cmds.console import logger
import os.path
import shlex
import subprocess

import isort


def format_code(
    formatter_cmds: list[str],
    path: str,
) -> str:
    # TODO: Only allow a particular whitelist of formatters here to prevent arbitrary code execution
    if not os.path.exists(path):
        logger.error(f"File {path} does not exist. Cannot format the file.")
        return None
    if formatter_cmds[0].lower() == "disabled":
        with open(path, encoding="utf8") as f:
            new_code = f.read()
        return new_code
    file_token = "$file"

    for command in formatter_cmds:
        formatter_cmd_list = shlex.split(command, posix=os.name != "nt")
        formatter_cmd_list = [path if chunk == file_token else chunk for chunk in formatter_cmd_list]
        logger.info(f"Formatting code with {' '.join(formatter_cmd_list)} ...")

        try:
            result = subprocess.run(
                formatter_cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except Exception as e:
            logger.exception(f"Failed to format code with {' '.join(formatter_cmd_list)}: {e}")
            return None
        if result.returncode == 0:
            logger.info("FORMATTING OK")
        else:
            logger.error(f"Failed to format code with {' '.join(formatter_cmd_list)}")

    with open(path, encoding="utf8") as f:
        new_code = f.read()

    return new_code


def sort_imports(code: str) -> str:
    try:
        # Deduplicate and sort imports, modify the code in memory, not on disk
        sorted_code = isort.code(code)
    except Exception:
        logger.exception("Failed to sort imports with isort.")
        return code  # Fall back to original code if isort fails

    return sorted_code
