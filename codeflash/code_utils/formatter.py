from __future__ import annotations

import os
import shlex
import subprocess
from typing import TYPE_CHECKING, Optional

import isort

from codeflash.cli_cmds.console import console, logger

if TYPE_CHECKING:
    from pathlib import Path


def get_nth_line(text: str, n: int) -> str | None:
    for i, line in enumerate(text.splitlines(), start=1):
        if i == n:
            return line
    return None


def get_diff_output(cmd: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip() or None
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        if isinstance(e, subprocess.CalledProcessError):
            # ruff returns 1 when the file needs formatting, and 0 when it is already formatted
            is_ruff = cmd[0] == "ruff"
            if e.returncode == 0 and is_ruff:
                return ""
            if e.returncode == 1 and is_ruff:
                return e.stdout.strip() or None
        return None


def get_diff_lines_output_by_black(filepath: str) -> Optional[str]:
    try:
        import black  # type: ignore

        return get_diff_output(["black", "--diff", filepath])
    except ImportError:
        return None


def get_diff_lines_output_by_ruff(filepath: str) -> Optional[str]:
    try:
        import ruff  # type: ignore

        return get_diff_output(["ruff", "format", "--diff", filepath])
    except ImportError:
        print("can't import ruff")
        return None


def get_diff_lines_count(diff_output: str) -> int:
    # Count lines that are diff changes (start with '+' or '-', but not '+++' or '---')
    count = 0
    for line in diff_output.split("\n"):
        if line and (line[0] in ("+", "-")) and not (line[:3] == "+++" or line[:3] == "---"):
            count += 1
    return count


def is_safe_to_format(filepath: str, max_diff_lines: int = 100) -> bool:
    diff_changes_stdout = None

    diff_changes_stdout = get_diff_lines_output_by_black(filepath)

    if diff_changes_stdout is None:
        logger.warning("black formatter not found, trying ruff instead...")
        diff_changes_stdout = get_diff_lines_output_by_ruff(filepath)
        if diff_changes_stdout is None:
            logger.warning("Both ruff, black formatters not found, skipping formatting diff check.")
            return False

    diff_lines_count = get_diff_lines_count(diff_changes_stdout)

    if diff_lines_count > max_diff_lines:
        logger.debug(f"Skipping {filepath}: {diff_lines_count} lines would change (max: {max_diff_lines})")
        return False
    return True


def format_code(formatter_cmds: list[str], path: Path, print_status: bool = True) -> str:  # noqa
    # TODO: Only allow a particular whitelist of formatters here to prevent arbitrary code execution
    formatter_name = formatter_cmds[0].lower()
    if not path.exists():
        msg = f"File {path} does not exist. Cannot format the file."
        raise FileNotFoundError(msg)
    if formatter_name == "disabled" or not is_safe_to_format(str(path)):
        return path.read_text(encoding="utf8")

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
