"""Benchmark libcst visitor performance across many files.

Exercises the visitor-heavy codepaths that benefit from the libcst dispatch
table cache: discover_functions + get_code_optimization_context on multiple
real source files.
"""

from __future__ import annotations

from pathlib import Path

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.python.context.code_context_extractor import get_code_optimization_context
from codeflash.languages.python.support import PythonSupport
from codeflash.models.models import FunctionParent

# Real source files from the codeflash codebase, chosen for size and visitor diversity.
_CODEFLASH_ROOT = Path(__file__).parent.parent.parent.resolve() / "codeflash"

_SOURCE_FILES: list[Path] = [
    _CODEFLASH_ROOT / "languages" / "function_optimizer.py",
    _CODEFLASH_ROOT / "languages" / "python" / "context" / "code_context_extractor.py",
    _CODEFLASH_ROOT / "languages" / "python" / "support.py",
    _CODEFLASH_ROOT / "languages" / "python" / "static_analysis" / "code_extractor.py",
    _CODEFLASH_ROOT / "languages" / "python" / "static_analysis" / "code_replacer.py",
    _CODEFLASH_ROOT / "code_utils" / "instrument_existing_tests.py",
    _CODEFLASH_ROOT / "benchmarking" / "compare.py",
    _CODEFLASH_ROOT / "models" / "models.py",
    _CODEFLASH_ROOT / "discovery" / "discover_unit_tests.py",
    _CODEFLASH_ROOT / "languages" / "base.py",
]

# For each file, pick one top-level function to extract context for.
# (class, function_name) — class=None means module-level.
_TARGETS: list[tuple[Path, str | None, str]] = [
    (_SOURCE_FILES[0], "FunctionOptimizer", "replace_function_and_helpers_with_optimized_code"),
    (_SOURCE_FILES[1], None, "get_code_optimization_context"),
    (_SOURCE_FILES[2], "PythonSupport", "discover_functions"),
    (_SOURCE_FILES[3], None, "add_global_assignments"),
    (_SOURCE_FILES[4], None, "replace_functions_in_file"),
    (_SOURCE_FILES[5], None, "inject_profiling_into_existing_test"),
    (_SOURCE_FILES[6], None, "compare_branches"),
    (_SOURCE_FILES[7], None, "get_comment_prefix"),
    (_SOURCE_FILES[8], None, "discover_unit_tests"),
    (_SOURCE_FILES[9], None, "convert_parents_to_tuple"),
]


def _discover_all() -> None:
    """Run discover_functions on all source files."""
    ps = PythonSupport()
    for file_path in _SOURCE_FILES:
        source = file_path.read_text()
        ps.discover_functions(source=source, file_path=file_path)


def _extract_all_contexts() -> None:
    """Run get_code_optimization_context on every target function."""
    project_root = _CODEFLASH_ROOT.parent
    for file_path, class_name, func_name in _TARGETS:
        parents = [FunctionParent(name=class_name, type="ClassDef")] if class_name else []
        fto = FunctionToOptimize(
            function_name=func_name, file_path=file_path, parents=parents, starting_line=None, ending_line=None
        )
        get_code_optimization_context(fto, project_root)


def test_benchmark_discover_functions_multi_file(benchmark) -> None:
    """Discover functions across 10 source files."""
    benchmark(_discover_all)


def test_benchmark_extract_context_multi_file(benchmark) -> None:
    """Extract code optimization context for 10 functions across 10 files."""
    benchmark(_extract_all_contexts)
