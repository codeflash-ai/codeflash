"""Benchmark the full libcst-heavy pipeline on a single file.

Runs discover → extract context → replace functions → add global assignments
in sequence, exercising ~15 distinct visitor/transformer classes in one pass.
"""

from __future__ import annotations

from pathlib import Path

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.python.context.code_context_extractor import get_code_optimization_context
from codeflash.languages.python.static_analysis.code_extractor import add_global_assignments
from codeflash.languages.python.static_analysis.code_replacer import replace_functions_in_file
from codeflash.languages.python.support import PythonSupport

_CODEFLASH_ROOT = Path(__file__).parent.parent.parent.resolve() / "codeflash"
_PROJECT_ROOT = _CODEFLASH_ROOT.parent

# Target: a real, non-trivial file with classes and module-level functions.
_TARGET_FILE = _CODEFLASH_ROOT / "languages" / "python" / "static_analysis" / "code_extractor.py"
_TARGET_FUNC = "add_global_assignments"

# A second file to serve as "optimized" source for replace/merge steps.
_SECOND_FILE = _CODEFLASH_ROOT / "languages" / "python" / "static_analysis" / "code_replacer.py"


def _run_pipeline() -> None:
    """Simulate a single-file optimization pass through the full visitor pipeline."""
    source = _TARGET_FILE.read_text(encoding="utf-8")
    source2 = _SECOND_FILE.read_text(encoding="utf-8")

    # 1. Discover functions (FunctionVisitor + MetadataWrapper)
    ps = PythonSupport()
    functions = ps.discover_functions(source=source, file_path=_TARGET_FILE)

    # 2. Extract code optimization context (multiple collectors + dependency resolver)
    fto = FunctionToOptimize(
        function_name=_TARGET_FUNC, file_path=_TARGET_FILE, parents=[], starting_line=None, ending_line=None
    )
    get_code_optimization_context(fto, _PROJECT_ROOT)

    # 3. Replace functions (GlobalFunctionCollector + GlobalFunctionTransformer)
    # Use a class method from discovered functions if available, else module-level.
    func_names = [_TARGET_FUNC]
    replace_functions_in_file(
        source_code=source, original_function_names=func_names, optimized_code=source2, preexisting_objects=set()
    )

    # 4. Add global assignments (6 visitors/transformers)
    add_global_assignments(source2, source)


def test_benchmark_full_pipeline(benchmark) -> None:
    """Full discover → extract → replace → merge pipeline on one file."""
    benchmark(_run_pipeline)
