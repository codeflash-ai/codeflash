"""Language-agnostic code replacement utilities.

Used by non-Python language optimizers to replace function definitions
via the LanguageSupport protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.languages.base import LanguageSupport
    from codeflash.models.models import CodeStringsMarkdown


def get_optimized_code_for_module(relative_path: Path, optimized_code: CodeStringsMarkdown) -> str:
    file_to_code_context = optimized_code.file_to_path()
    module_optimized_code = file_to_code_context.get(str(relative_path))
    if module_optimized_code is None:
        # Fallback: if there's only one code block with None file path,
        # use it regardless of the expected path (the AI server doesn't always include file paths)
        if "None" in file_to_code_context and len(file_to_code_context) == 1:
            module_optimized_code = file_to_code_context["None"]
            logger.debug(f"Using code block with None file_path for {relative_path}")
        else:
            # Avoid expensive string formatting when logging is not enabled
            if logger.isEnabledFor(logger.level):
                logger.warning(
                    f"Optimized code not found for {relative_path} In the context\n-------\n{optimized_code}\n-------\n"
                    "re-check your 'markdown code structure'"
                    f"existing files are {file_to_code_context.keys()}"
                )
            module_optimized_code = ""
    return module_optimized_code


def replace_function_definitions_for_language(
    function_names: list[str],
    optimized_code: CodeStringsMarkdown,
    module_abspath: Path,
    project_root_path: Path,
    lang_support: LanguageSupport,
    function_to_optimize: FunctionToOptimize | None = None,
) -> bool:
    """Replace function definitions using the LanguageSupport protocol.

    Works for any language that implements LanguageSupport.replace_function
    and LanguageSupport.discover_functions_from_source.
    """
    original_source_code: str = module_abspath.read_text(encoding="utf8")
    code_to_apply = get_optimized_code_for_module(module_abspath.relative_to(project_root_path), optimized_code)

    original_source_code = lang_support.add_global_declarations(
        optimized_code=code_to_apply, original_source=original_source_code, module_abspath=module_abspath
    )

    if (
        function_to_optimize
        and function_to_optimize.starting_line
        and function_to_optimize.ending_line
        and function_to_optimize.file_path == module_abspath
    ):
        optimized_func = _extract_function_from_code(
            lang_support, code_to_apply, function_to_optimize.function_name, module_abspath
        )
        if optimized_func:
            new_code = lang_support.replace_function(original_source_code, function_to_optimize, optimized_func)
        else:
            new_code = lang_support.replace_function(original_source_code, function_to_optimize, code_to_apply)
    else:
        new_code = original_source_code
        modified = False

        functions_to_replace = list(function_names)

        for func_name in functions_to_replace:
            current_functions = lang_support.discover_functions_from_source(new_code, module_abspath)

            func = None
            for f in current_functions:
                if func_name in (f.qualified_name, f.function_name):
                    func = f
                    break

            if func is None:
                continue

            optimized_func = _extract_function_from_code(
                lang_support, code_to_apply, func.function_name, module_abspath
            )
            if optimized_func:
                new_code = lang_support.replace_function(new_code, func, optimized_func)
                modified = True

        if not modified:
            logger.warning(f"Could not find function {function_names} in {module_abspath}")
            return False

    if original_source_code.strip() == new_code.strip():
        return False

    module_abspath.write_text(new_code, encoding="utf8")
    return True


def _extract_function_from_code(
    lang_support: LanguageSupport, source_code: str, function_name: str, file_path: Path | None = None
) -> str | None:
    """Extract a specific function's source code from a code string.

    Includes JSDoc/docstring comments if present.
    """
    try:
        functions = lang_support.discover_functions_from_source(source_code, file_path)
        for func in functions:
            if func.function_name == function_name:
                lines = source_code.splitlines(keepends=True)
                effective_start = func.doc_start_line or func.starting_line
                if effective_start and func.ending_line and effective_start <= len(lines):
                    func_lines = lines[effective_start - 1 : func.ending_line]
                    return "".join(func_lines)
    except Exception as e:
        logger.debug(f"Error extracting function {function_name}: {e}")

    return None
