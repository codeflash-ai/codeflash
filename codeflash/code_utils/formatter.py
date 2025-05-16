from __future__ import annotations

import os
import shlex
import subprocess
from typing import TYPE_CHECKING

import isort
import libcst as cst

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.code_replacer import OptimFunctionCollector
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.models import CodeOptimizationContext 

if TYPE_CHECKING:
    from pathlib import Path


def format_code(formatter_cmds: list[str], path: Path) -> str:
    # TODO: Only allow a particular whitelist of formatters here to prevent arbitrary code execution
    formatter_name = formatter_cmds[0].lower()
    if not path.exists():
        msg = f"File {path} does not exist. Cannot format the file."
        raise FileNotFoundError(msg)
    if formatter_name == "disabled":
        return path.read_text(encoding="utf8")
    file_token = "$file"  # noqa: S105
    for command in set(formatter_cmds):
        formatter_cmd_list = shlex.split(command, posix=os.name != "nt")
        formatter_cmd_list = [path.as_posix() if chunk == file_token else chunk for chunk in formatter_cmd_list]
        try:
            result = subprocess.run(formatter_cmd_list, capture_output=True, check=False)
            if result.returncode == 0:
                console.rule(f"Formatted Successfully with: {formatter_name.replace('$file', path.name)}")
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

# TODO(zomglings): Write unit tests.
def get_modification_code_ranges(
    modified_code: str,
    fto: FunctionToOptimize,
    code_context: CodeOptimizationContext,
) -> list[tuple[int, int]]:
    """
    Returns the line number of modified and new functions in a string containing containing the code in a fully modified file.
    """
    modified_functions = set()
    modified_functions.add(fto.qualified_name)
    for helper_function in code_context.helper_functions:
        if helper_function.jedi_definition.type != "class":
            modified_functions.add(helper_function.qualified_name)
    
    parsed_function_names = set()
    for original_function_name in modified_functions:
        if original_function_name.count(".") == 0:
            class_name, function_name = None, original_function_name
        elif original_function_name.count(".") == 1:
            class_name, function_name = original_function_name.split(".")
        else:
            msg = f"Unable to find {original_function_name}. Returning unchanged source code."
            logger.error(msg)
            continue
        parsed_function_names.add((class_name, function_name))

    module = cst.metadata.MetadataWrapper(cst.parse_module(modified_code))
    visitor = OptimFunctionCollector(code_context.preexisting_objects, parsed_function_names)
    module.visit(visitor)
    return visitor.modification_code_range_lines
