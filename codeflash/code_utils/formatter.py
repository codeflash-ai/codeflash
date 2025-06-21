from __future__ import annotations

import difflib
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

import isort

from codeflash.cli_cmds.console import console, logger


def generate_unified_diff(original: str, modified: str, from_file: str, to_file: str) -> str:
    line_pattern = re.compile(r"(.*?(?:\r\n|\n|\r|$))")

    def split_lines(text: str) -> list[str]:
        lines = [match[0] for match in line_pattern.finditer(text)]
        if lines and lines[-1] == "":
            lines.pop()
        return lines

    original_lines = split_lines(original)
    modified_lines = split_lines(modified)

    diff_output = []
    for line in difflib.unified_diff(original_lines, modified_lines, fromfile=from_file, tofile=to_file, n=5):
        if line.endswith("\n"):
            diff_output.append(line)
        else:
            diff_output.append(line + "\n")
            diff_output.append("\\ No newline at end of file\n")

    return "".join(diff_output)


def apply_formatter_cmds(
    cmds: list[str],
    path: Path,
    test_dir_str: Optional[str],
    print_status: bool,  # noqa
    exit_on_failure: bool = True,  # noqa
) -> tuple[Path, str]:
    # TODO: Only allow a particular whitelist of formatters here to prevent arbitrary code execution
    formatter_name = cmds[0].lower()
    should_make_copy = False
    file_path = path

    if test_dir_str:
        should_make_copy = True
        file_path = Path(test_dir_str) / "temp.py"

    if not cmds or formatter_name == "disabled":
        return path, path.read_text(encoding="utf8")

    if not path.exists():
        msg = f"File {path} does not exist. Cannot apply formatter commands."
        raise FileNotFoundError(msg)

    if should_make_copy:
        shutil.copy2(path, file_path)

    file_token = "$file"  # noqa: S105

    for command in cmds:
        formatter_cmd_list = shlex.split(command, posix=os.name != "nt")
        formatter_cmd_list = [file_path.as_posix() if chunk == file_token else chunk for chunk in formatter_cmd_list]
        try:
            result = subprocess.run(formatter_cmd_list, capture_output=True, check=False)
            if result.returncode == 0:
                if print_status:
                    console.rule(f"Formatted Successfully with: {command.replace('$file', path.name)}")
            else:
                logger.error(f"Failed to format code with {' '.join(formatter_cmd_list)}")
        except FileNotFoundError as e:
            from rich.panel import Panel
            from rich.text import Text

            panel = Panel(
                Text.from_markup(f"⚠️  Formatter command not found: {' '.join(formatter_cmd_list)}", style="bold red"),
                expand=False,
            )
            console.print(panel)
            if exit_on_failure:
                raise e from None

    return file_path, file_path.read_text(encoding="utf8")


def get_diff_lines_count(diff_output: str) -> int:
    lines = diff_output.split("\n")

    def is_diff_line(line: str) -> bool:
        return line.startswith(("+", "-")) and not line.startswith(("+++", "---"))

    diff_lines = [line for line in lines if is_diff_line(line)]
    return len(diff_lines)


def format_code(
    formatter_cmds: list[str],
    path: Union[str, Path],
    optimized_function: str = "",
    check_diff: bool = False,  # noqa
    print_status: bool = True,  # noqa
    exit_on_failure: bool = True,  # noqa
) -> str:
    with tempfile.TemporaryDirectory() as test_dir_str:
        if isinstance(path, str):
            path = Path(path)

        original_code = path.read_text(encoding="utf8")
        original_code_lines = len(original_code.split("\n"))

        if check_diff and original_code_lines > 50:
            # we dont' count the formatting diff for the optimized function as it should be well-formatted
            original_code_without_opfunc = original_code.replace(optimized_function, "")

            original_temp = Path(test_dir_str) / "original_temp.py"
            original_temp.write_text(original_code_without_opfunc, encoding="utf8")

            formatted_temp, formatted_code = apply_formatter_cmds(
                formatter_cmds, original_temp, test_dir_str, print_status=False
            )

            diff_output = generate_unified_diff(
                original_code_without_opfunc, formatted_code, from_file=str(original_temp), to_file=str(formatted_temp)
            )
            diff_lines_count = get_diff_lines_count(diff_output)

            max_diff_lines = min(int(original_code_lines * 0.3), 50)

            if diff_lines_count > max_diff_lines and max_diff_lines != -1:
                logger.debug(
                    f"Skipping formatting {path}: {diff_lines_count} lines would change (max: {max_diff_lines})"
                )
                return original_code
        # TODO : We can avoid formatting the whole file again and only formatting the optimized code standalone and replace in formatted file above.
        _, formatted_code = apply_formatter_cmds(
            formatter_cmds, path, test_dir_str=None, print_status=print_status, exit_on_failure=exit_on_failure
        )
        logger.debug(f"Formatted {path} with commands: {formatter_cmds}")
        return formatted_code


def sort_imports(code: str) -> str:
    try:
        # Deduplicate and sort imports, modify the code in memory, not on disk
        sorted_code = isort.code(code)
    except Exception:
        logger.exception("Failed to sort imports with isort.")
        return code  # Fall back to original code if isort fails

    return sorted_code
