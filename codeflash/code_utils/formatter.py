from __future__ import annotations

import os
import shlex
import subprocess
from typing import TYPE_CHECKING, Optional

import isort

from codeflash.cli_cmds.console import console, logger

if TYPE_CHECKING:
    from pathlib import Path


def get_diff_output_by_black(filepath: str, unformatted_content: str) -> Optional[str]:
    try:
        from black import Mode, format_file_contents, output, report

        formatted_content = format_file_contents(src_contents=unformatted_content, fast=True, mode=Mode())
        return output.diff(unformatted_content, formatted_content, a_name=filepath, b_name=filepath)
    except (ImportError, report.NothingChanged):
        return None


def get_diff_lines_count(diff_output: str) -> int:
    # Use a generator expression to avoid creating an intermediate list
    return sum(line.startswith(("+", "-")) and not line.startswith(("+++", "---")) for line in diff_output.split("\n"))


def is_safe_to_format(filepath: str, content: str, max_diff_lines: int = 100) -> bool:
    diff_changes_str = None

    diff_changes_str = get_diff_output_by_black(filepath, unformatted_content=content)

    if diff_changes_str is None:
        logger.warning("Looks like black formatter not found, make sure it is installed.")
        return False

    diff_lines_count = get_diff_lines_count(diff_changes_str)
    if diff_lines_count > max_diff_lines:
        logger.debug(f"Skipping formatting {filepath}: {diff_lines_count} lines would change (max: {max_diff_lines})")
        return False

    return True


def format_code(formatter_cmds: list[str], path: Path, print_status: bool = True) -> str:  # noqa
    # TODO: Only allow a particular whitelist of formatters here to prevent arbitrary code execution
    formatter_name = formatter_cmds[0].lower()
    if not path.exists():
        msg = f"File {path} does not exist. Cannot format the file."
        raise FileNotFoundError(msg)
    file_content = path.read_text(encoding="utf8")
    if formatter_name == "disabled" or not is_safe_to_format(filepath=str(path), content=file_content):
        return file_content

    file_token = "$file"  # noqa: S105
    for command in formatter_cmds:
        formatter_cmd_list = shlex.split(command, posix=os.name != "nt")
        formatter_cmd_list = [path.as_posix() if chunk == file_token else chunk for chunk in formatter_cmd_list]
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

            raise e from None

    return path.read_text(encoding="utf8")


def sort_imports(code: str) -> str:
    try:
        # Deduplicate and sort imports, modify the code in memory, not on disk
        sorted_code = isort.code(code)
    except Exception:
        logger.exception("Failed to sort imports with isort.")
        return code  # Fall back to original code if isort fails

    return sorted_code
