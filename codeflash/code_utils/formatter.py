import logging
import os.path
import subprocess

import isort


def format_code(
    formatter_cmd: list[str],
    should_sort_imports: bool,
    path: str,
) -> str:
    # TODO: Only allow a particular whitelist of formatters here to prevent arbitrary code execution
    assert os.path.exists(
        path,
    ), f"File {path} does not exist. Cannot format the file. Exiting..."
    if formatter_cmd[0].lower() == "disabled":
        if should_sort_imports:
            return sort_imports(should_sort_imports, path)

        with open(path, encoding="utf8") as f:
            new_code = f.read()
        return new_code
    file_token = "$file"

    for command in formatter_cmd:
        formatter_cmd_list = [chunk for chunk in command.split(" ") if chunk != ""]
        formatter_cmd_list = [path if chunk == file_token else chunk for chunk in formatter_cmd_list]
        logging.info(f"Formatting code with {' '.join(formatter_cmd_list)} ...")

        try:
            result = subprocess.run(
                formatter_cmd_list,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except Exception as e:
            logging.exception(f"Failed to format code with {' '.join(formatter_cmd_list)}: {e}")
            return None
        if result.returncode == 0:
            logging.info("FORMATTING OK")
        else:
            logging.error(f"Failed to format code with {' '.join(formatter_cmd_list)}")

    if should_sort_imports:
        return sort_imports(should_sort_imports, path)

    with open(path, encoding="utf8") as f:
        new_code = f.read()

    return new_code


def sort_imports(should_sort_imports: bool, path: str) -> str:
    try:
        with open(path, encoding="utf8") as f:
            code = f.read()

        if not should_sort_imports:
            return code

        # Deduplicate and sort imports, modify the code in memory, not on disk
        sorted_code = isort.code(code)
    except Exception as e:
        logging.exception(f"Failed to sort imports with isort for {path}: {e}")
        return code  # Fall back to original code if isort fails

    return sorted_code
