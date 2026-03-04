"""Language-agnostic code replacement utilities.

Used by non-Python language optimizers to replace function definitions
via the LanguageSupport protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.languages.base import FunctionFilterCriteria, Language

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.languages.base import LanguageSupport
    from codeflash.models.models import CodeStringsMarkdown

# Permissive criteria for discovering functions in code snippets (no export/return filtering)
_SOURCE_CRITERIA = FunctionFilterCriteria(require_return=False, require_export=False)


def get_optimized_code_for_module(relative_path: Path, optimized_code: CodeStringsMarkdown) -> str:
    from codeflash.languages.current import is_python

    file_to_code_context = optimized_code.file_to_path()
    relative_path_str = str(relative_path)
    module_optimized_code = file_to_code_context.get(relative_path_str)
    if module_optimized_code is None:
        # Fallback: if there's only one code block with None file path,
        # use it regardless of the expected path (the AI server doesn't always include file paths)
        if "None" in file_to_code_context and len(file_to_code_context) == 1:
            module_optimized_code = file_to_code_context["None"]
            logger.debug(f"Using code block with None file_path for {relative_path}")
        else:
            # Fallback: try to match by just the filename (for Java/JS where the AI
            # might return just the class name like "Algorithms.java" instead of
            # the full path like "src/main/java/com/example/Algorithms.java")
            target_filename = relative_path.name
            for file_path_str, code in file_to_code_context.items():
                if file_path_str:
                    if file_path_str.endswith(target_filename) and (
                        len(file_path_str) == len(target_filename)
                        or file_path_str[-len(target_filename) - 1] in ("/", "\\")
                    ):
                        module_optimized_code = code
                        logger.debug(f"Matched {file_path_str} to {relative_path} by filename")
                        break

            if module_optimized_code is None:
                # Also try matching if there's only one code file, but ONLY for non-Python
                # languages where path matching is less strict.
                if len(file_to_code_context) == 1 and not is_python():
                    only_key = next(iter(file_to_code_context.keys()))
                    module_optimized_code = file_to_code_context[only_key]
                    logger.debug(f"Using only code block {only_key} for {relative_path}")
                else:
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
    and LanguageSupport.discover_functions.
    """
    original_source_code: str = module_abspath.read_text(encoding="utf8")
    code_to_apply = get_optimized_code_for_module(module_abspath.relative_to(project_root_path), optimized_code)

    if not code_to_apply.strip():
        return False

    original_source_code = lang_support.add_global_declarations(
        optimized_code=code_to_apply, original_source=original_source_code, module_abspath=module_abspath
    )

    language = lang_support.language

    if (
        function_to_optimize
        and function_to_optimize.starting_line
        and function_to_optimize.ending_line
        and function_to_optimize.file_path == module_abspath
    ):
        # For Java, we need to pass the full optimized code so replace_function can
        # extract and add any new class members (static fields, helper methods).
        # For other languages, we extract just the target function.
        if language == Language.JAVA:
            new_code = lang_support.replace_function(original_source_code, function_to_optimize, code_to_apply)
        else:
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
            current_functions = lang_support.discover_functions(new_code, module_abspath, _SOURCE_CRITERIA)

            func = None
            for f in current_functions:
                if func_name in (f.qualified_name, f.function_name):
                    func = f
                    break

            if func is None:
                continue

            # For Java, pass the full optimized code to handle class member insertion.
            # For other languages, extract just the target function.
            if language == Language.JAVA:
                new_code = lang_support.replace_function(new_code, func, code_to_apply)
                modified = True
            else:
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
    lang_support: LanguageSupport, source_code: str, function_name: str, file_path: Path
) -> str | None:
    """Extract a specific function's source code from a code string.

    Includes JSDoc/docstring comments if present.
    """
    try:
        functions = lang_support.discover_functions(source_code, file_path, _SOURCE_CRITERIA)
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
