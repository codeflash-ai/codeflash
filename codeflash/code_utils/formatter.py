from __future__ import annotations

import os
import shlex
import subprocess
import sys
from typing import TYPE_CHECKING

import isort

from codeflash.cli_cmds.console import console, logger

if TYPE_CHECKING:
    from pathlib import Path


def format_code(formatter_cmds: list[str], path: Path) -> str:
    known_formatters = ["ruff", "black", "autopep8", "yapf", "isort", "disabled"]
    formatter_name = formatter_cmds[0].split()[0].lower()
    if not any(formatter_name in formatter for formatter in known_formatters):
        msg = f"The formatter command must start with one of {known_formatters}, but got {formatter_name}."

    if not path.exists():
        msg = f"File {path} does not exist. Cannot format the file."
        raise FileNotFoundError(msg)
    if formatter_name == "disabled":
        return path.read_text(encoding="utf8")
    file_token = "$file"  # noqa: S105

    for command in formatter_cmds:
        formatter_cmd_list = shlex.split(command, posix=os.name != "nt")
        formatter_cmd_list = [str(path) if chunk == file_token else chunk for chunk in formatter_cmd_list]
        console.rule(f"Formatting code with {' '.join(formatter_cmd_list)} ...")
        try:
            result = subprocess.run(formatter_cmd_list, capture_output=True, check=False)
            if result.returncode == 0:
                logger.info("FORMATTING OK")
            logger.error(f"Failed to format code with {' '.join(formatter_cmd_list)}")
        except FileNotFoundError:
            from rich.panel import Panel
            from rich.text import Text

            panel = Panel(
                Text.from_markup(f"⚠️  Formatter command not found: {' '.join(formatter_cmd_list)}", style="bold red"),
                expand=False,
            )
            console.print(panel)
            sys.exit(1)
        except Exception:
            logger.exception(f"Failed to format code with {' '.join(formatter_cmd_list)}")
            # Fall back to original code if formatter fails
            return path.read_text(encoding="utf8")

    return None


def sort_imports(code: str) -> str:
    try:
        # Deduplicate and sort imports, modify the code in memory, not on disk
        sorted_code = isort.code(code)
    except Exception:
        logger.exception("Failed to sort imports with isort.")
        return code  # Fall back to original code if isort fails

    return sorted_code
