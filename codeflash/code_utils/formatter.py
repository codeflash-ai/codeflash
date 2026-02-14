from __future__ import annotations

import difflib
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

import isort

from codeflash.cli_cmds.console import console, logger
from codeflash.lsp.helpers import is_LSP_enabled

# Whitelist of allowed formatter executables to prevent arbitrary command execution
ALLOWED_FORMATTER_EXECUTABLES = frozenset({
    # Python formatters
    "black",
    "ruff",
    "isort",
    # JavaScript package manager (used to run formatters)
    "npx",
    # JavaScript formatters (may be called directly or via npx)
    "prettier",
    "eslint",
    # Future: Java formatters
    # "google-java-format",
})

# Tools that can be run via npx
ALLOWED_NPX_TOOLS = frozenset({
    "prettier",
    "eslint",
})


def validate_formatter_command(command: str) -> None:
    """Validate that a formatter command uses only whitelisted executables.

    Args:
        command: Formatter command string (e.g., "black $file", "npx prettier --write $file")

    Raises:
        ValueError: If the command uses a non-whitelisted executable

    Security: This prevents arbitrary command execution by validating that only
    known, safe formatter tools are used.
    """
    if not command or command.strip() == "":
        raise ValueError("Formatter command cannot be empty")

    # Parse the command safely
    try:
        tokens = shlex.split(command, posix=os.name != "nt")
    except ValueError as e:
        raise ValueError(f"Invalid formatter command syntax: {command}") from e

    if not tokens:
        raise ValueError(f"Formatter command has no executable: {command}")

    # Extract the executable name (first token)
    executable = tokens[0]

    # Check if executable is in whitelist
    if executable not in ALLOWED_FORMATTER_EXECUTABLES:
        allowed_list = ", ".join(sorted(ALLOWED_FORMATTER_EXECUTABLES))
        raise ValueError(
            f"Formatter executable '{executable}' is not allowed. "
            f"Only whitelisted formatters are permitted: {allowed_list}. "
            f"Rejected command: {command}"
        )

    # Special validation for npx - ensure the tool being run is also whitelisted
    if executable == "npx" and len(tokens) > 1:
        npx_tool = tokens[1]
        if npx_tool not in ALLOWED_NPX_TOOLS:
            allowed_npx = ", ".join(sorted(ALLOWED_NPX_TOOLS))
            raise ValueError(
                f"NPX tool '{npx_tool}' is not allowed. "
                f"Only whitelisted npx tools are permitted: {allowed_npx}. "
                f"Rejected command: {command}"
            )


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
    cmds: list[str], path: Path, test_dir_str: Optional[str], print_status: bool, exit_on_failure: bool = True
) -> tuple[Path, str, bool]:
    from codeflash.languages.registry import get_language_support

    if not path.exists():
        msg = f"File {path} does not exist. Cannot apply formatter commands."
        raise FileNotFoundError(msg)

    file_path = path
    lang_support = get_language_support(path)
    if test_dir_str:
        file_path = Path(test_dir_str) / ("temp" + lang_support.default_file_extension)
        shutil.copy2(path, file_path)

    file_token = "$file"  # noqa: S105

    changed = False
    for command in cmds:
        # Validate command against whitelist before execution (security check)
        try:
            validate_formatter_command(command)
        except ValueError as e:
            logger.error(f"Security: Rejected formatter command - {e}")
            if exit_on_failure:
                raise
            continue
        formatter_cmd_list = shlex.split(command, posix=os.name != "nt")
        formatter_cmd_list = [file_path.as_posix() if chunk == file_token else chunk for chunk in formatter_cmd_list]
        try:
            result = subprocess.run(formatter_cmd_list, capture_output=True, check=False)
            if result.returncode == 0:
                if print_status:
                    console.rule(f"Formatted Successfully with: {command.replace('$file', path.name)}")
                changed = True
            else:
                logger.error(f"Failed to format code with {' '.join(formatter_cmd_list)}")
        except FileNotFoundError as e:
            from rich.panel import Panel

            command_str = " ".join(str(part) for part in formatter_cmd_list)
            panel = Panel(f"⚠️  Formatter command not found: {command_str}", expand=False, border_style="yellow")
            console.print(panel)
            if exit_on_failure:
                raise e from None

    return file_path, file_path.read_text(encoding="utf8"), changed


def get_diff_lines_count(diff_output: str) -> int:
    lines = diff_output.split("\n")

    def is_diff_line(line: str) -> bool:
        return line.startswith(("+", "-")) and not line.startswith(("+++", "---"))

    diff_lines = [line for line in lines if is_diff_line(line)]
    return len(diff_lines)


def format_generated_code(generated_test_source: str, formatter_cmds: list[str], language: str = "python") -> str:
    from codeflash.languages.registry import get_language_support

    # Skip formatting if no formatter configured
    if not formatter_cmds or formatter_cmds[0] in ("disabled", ""):
        return re.sub(r"\n{2,}", "\n\n", generated_test_source)

    # Validate formatter commands (security check)
    for cmd in formatter_cmds:
        try:
            validate_formatter_command(cmd)
        except ValueError as e:
            logger.warning(f"Security: Skipping invalid formatter - {e}")
            return re.sub(r"\n{2,}", "\n\n", generated_test_source)
    with tempfile.TemporaryDirectory() as test_dir_str:
        # try running formatter, if nothing changes (could be due to formatting failing or no actual formatting needed) return code with 2 or more newlines substituted with 2 newlines
        lang_support = get_language_support(language)
        original_temp = Path(test_dir_str) / ("original_temp" + lang_support.default_file_extension)
        original_temp.write_text(generated_test_source, encoding="utf8")
        _, formatted_code, changed = apply_formatter_cmds(
            formatter_cmds, original_temp, test_dir_str, print_status=False, exit_on_failure=False
        )
        if not changed:
            return re.sub(r"\n{2,}", "\n\n", formatted_code)
    return formatted_code


def format_code(
    formatter_cmds: list[str],
    path: Union[str, Path],
    optimized_code: str = "",
    check_diff: bool = False,
    print_status: bool = True,
    exit_on_failure: bool = True,
) -> str:
    from codeflash.languages.registry import get_language_support

    if is_LSP_enabled():
        exit_on_failure = False

    if isinstance(path, str):
        path = Path(path)

    # Validate formatter commands against whitelist (security check)
    # Skip validation if no formatter configured
    if not formatter_cmds or formatter_cmds[0] in ("disabled", ""):
        return path.read_text(encoding="utf8")

    # Validate all formatter commands before proceeding
    for cmd in formatter_cmds:
        try:
            validate_formatter_command(cmd)
        except ValueError as e:
            logger.error(f"Security: Invalid formatter configuration - {e}")
            if exit_on_failure:
                raise
            return path.read_text(encoding="utf8")

    with tempfile.TemporaryDirectory() as test_dir_str:
        original_code = path.read_text(encoding="utf8")
        original_code_lines = len(original_code.split("\n"))

        if check_diff and original_code_lines > 50:
            # we don't count the formatting diff for the optimized function as it should be well-formatted
            original_code_without_opfunc = original_code.replace(optimized_code, "")

            lang_support = get_language_support(path)
            original_temp = Path(test_dir_str) / ("original_temp" + lang_support.default_file_extension)
            original_temp.write_text(original_code_without_opfunc, encoding="utf8")

            formatted_temp, formatted_code, changed = apply_formatter_cmds(
                formatter_cmds, original_temp, test_dir_str, print_status=False, exit_on_failure=exit_on_failure
            )

            if not changed:
                logger.warning(
                    f"No changes detected in {path} after formatting, are you sure you have valid formatter commands?"
                )
                return original_code

            diff_output = generate_unified_diff(
                original_code_without_opfunc, formatted_code, from_file=str(original_temp), to_file=str(formatted_temp)
            )
            diff_lines_count = get_diff_lines_count(diff_output)

            max_diff_lines = min(int(original_code_lines * 0.3), 50)

            if diff_lines_count > max_diff_lines:
                logger.warning(
                    f"Skipping formatting {path}: {diff_lines_count} lines would change (max: {max_diff_lines})"
                )
                return original_code

        # TODO : We can avoid formatting the whole file again and only formatting the optimized code standalone and replace in formatted file above.
        _, formatted_code, changed = apply_formatter_cmds(
            formatter_cmds, path, test_dir_str=None, print_status=print_status, exit_on_failure=exit_on_failure
        )

        if not changed:
            logger.warning(
                f"No changes detected in {path} after formatting, are you sure you have valid formatter commands?"
            )
            return original_code

        logger.debug(f"Formatted {path} with commands: {formatter_cmds}")
        return formatted_code


def sort_imports(code: str, **kwargs: Any) -> str:
    try:
        # Deduplicate and sort imports, modify the code in memory, not on disk
        sorted_code = isort.code(code, **kwargs)
    except Exception:  # this will also catch the FileSkipComment exception, use this fn everywhere
        logger.exception("Failed to sort imports with isort.")
        return code  # Fall back to original code if isort fails

    return sorted_code
