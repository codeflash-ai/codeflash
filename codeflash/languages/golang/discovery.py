from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from codeflash.languages.golang.parser import GoAnalyzer

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.languages.base import FunctionFilterCriteria
    from codeflash.languages.golang.parser import GoFunctionNode, GoMethodNode
    from codeflash.models.function_types import FunctionToOptimize


logger = logging.getLogger(__name__)

_SKIP_FUNCTION_NAMES = frozenset({"init", "main"})


def discover_functions(
    file_path: Path, filter_criteria: FunctionFilterCriteria | None = None, analyzer: GoAnalyzer | None = None
) -> list[FunctionToOptimize]:
    try:
        source = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        logger.warning("Failed to read Go file: %s", file_path)
        return []
    return discover_functions_from_source(source, file_path, filter_criteria, analyzer)


def discover_functions_from_source(
    source: str,
    file_path: Path,
    filter_criteria: FunctionFilterCriteria | None = None,
    analyzer: GoAnalyzer | None = None,
) -> list[FunctionToOptimize]:
    from codeflash.models.function_types import FunctionParent, FunctionToOptimize

    if analyzer is None:
        analyzer = GoAnalyzer()

    results: list[FunctionToOptimize] = []

    functions = analyzer.find_functions(source)
    for func in functions:
        if not _should_include_function(func, filter_criteria, file_path):
            continue
        results.append(
            FunctionToOptimize(
                function_name=func.name,
                file_path=file_path,
                parents=[],
                starting_line=func.starting_line,
                ending_line=func.ending_line,
                starting_col=func.starting_col,
                ending_col=func.ending_col,
                is_async=False,
                is_method=False,
                language="go",
                doc_start_line=func.doc_start_line,
            )
        )

    methods = analyzer.find_methods(source)
    for method in methods:
        if not _should_include_method(method, filter_criteria, file_path):
            continue
        results.append(
            FunctionToOptimize(
                function_name=method.name,
                file_path=file_path,
                parents=[FunctionParent(name=method.receiver_name, type="StructDef")],
                starting_line=method.starting_line,
                ending_line=method.ending_line,
                starting_col=method.starting_col,
                ending_col=method.ending_col,
                is_async=False,
                is_method=True,
                language="go",
                doc_start_line=method.doc_start_line,
            )
        )

    return results


def _should_include_function(func: GoFunctionNode, criteria: FunctionFilterCriteria | None, file_path: Path) -> bool:
    if file_path.name.endswith("_test.go"):
        return False

    if func.name in _SKIP_FUNCTION_NAMES:
        return False

    if criteria is None:
        return True

    if criteria.require_export and not func.is_exported:
        return False

    if criteria.require_return and not func.has_return_type:
        return False

    if criteria.matches_exclude_patterns(func.name):
        return False

    if not criteria.matches_include_patterns(func.name):
        return False

    line_count = func.ending_line - func.starting_line + 1
    if criteria.min_lines is not None and line_count < criteria.min_lines:
        return False
    if criteria.max_lines is not None and line_count > criteria.max_lines:
        return False

    return True


def _should_include_method(method: GoMethodNode, criteria: FunctionFilterCriteria | None, file_path: Path) -> bool:
    if file_path.name.endswith("_test.go"):
        return False

    if criteria is None:
        return True

    if not criteria.include_methods:
        return False

    if criteria.require_export and not method.is_exported:
        return False

    if criteria.require_return and not method.has_return_type:
        return False

    if criteria.matches_exclude_patterns(method.name):
        return False

    if not criteria.matches_include_patterns(method.name):
        return False

    line_count = method.ending_line - method.starting_line + 1
    if criteria.min_lines is not None and line_count < criteria.min_lines:
        return False
    if criteria.max_lines is not None and line_count > criteria.max_lines:
        return False

    return True
