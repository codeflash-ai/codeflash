from __future__ import annotations

import os
import shlex
import subprocess
from typing import TYPE_CHECKING

import isort

from codeflash.cli_cmds.console import console, logger

if TYPE_CHECKING:
    from pathlib import Path


def should_format_file(filepath, max_lines_changed=100):
        try:
            # check if black is installed
            subprocess.run(['black', '--version'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            result = subprocess.run(
                ['black', '--diff', filepath], 
                capture_output=True, 
                text=True
            )
                
            diff_lines = [line for line in result.stdout.split('\n') 
                        if line.startswith(('+', '-')) and not line.startswith(('+++', '---'))]
            
            changes_count = len(diff_lines)
            
            if changes_count > max_lines_changed:
                logger.debug(f"Skipping {filepath}: {changes_count} lines would change (max: {max_lines_changed})")
                return False
            
            return True
            
        except subprocess.CalledProcessError:
            logger.warning(f"black --diff command failed for {filepath}")
            return False
        except FileNotFoundError:
            logger.warning("black formatter is not installed. Skipping formatting diff check.")
            return False



def format_code(formatter_cmds: list[str], path: Path, print_status: bool = True) -> str:  # noqa
    # TODO: Only allow a particular whitelist of formatters here to prevent arbitrary code execution
    formatter_name = formatter_cmds[0].lower()
    if not path.exists():
        msg = f"File {path} does not exist. Cannot format the file."
        raise FileNotFoundError(msg)
    if formatter_name == "disabled" or not should_format_file(path):
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
