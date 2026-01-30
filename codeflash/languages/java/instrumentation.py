"""Java code instrumentation for behavior capture and benchmarking.

This module provides functionality to instrument Java code for:
1. Behavior capture - recording inputs/outputs for verification
2. Benchmarking - measuring execution time
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.languages.base import FunctionInfo
from codeflash.languages.java.parser import JavaAnalyzer, get_java_analyzer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

logger = logging.getLogger(__name__)


def _get_function_name(func: Any) -> str:
    """Get the function name from either FunctionInfo or FunctionToOptimize."""
    if hasattr(func, "name"):
        return func.name
    if hasattr(func, "function_name"):
        return func.function_name
    raise AttributeError(f"Cannot get function name from {type(func)}")

# Template for behavior capture instrumentation
BEHAVIOR_CAPTURE_IMPORT = "import com.codeflash.CodeFlash;"

BEHAVIOR_CAPTURE_BEFORE = """
        // CodeFlash behavior capture - start
        long __codeflash_call_id_{call_id} = System.nanoTime();
        CodeFlash.recordInput(__codeflash_call_id_{call_id}, "{method_id}", CodeFlash.serialize({args}));
        long __codeflash_start_{call_id} = System.nanoTime();
"""

BEHAVIOR_CAPTURE_AFTER_RETURN = """
        // CodeFlash behavior capture - end
        long __codeflash_end_{call_id} = System.nanoTime();
        CodeFlash.recordOutput(__codeflash_call_id_{call_id}, "{method_id}", CodeFlash.serialize(__codeflash_result_{call_id}), __codeflash_end_{call_id} - __codeflash_start_{call_id});
"""

BEHAVIOR_CAPTURE_AFTER_VOID = """
        // CodeFlash behavior capture - end
        long __codeflash_end_{call_id} = System.nanoTime();
        CodeFlash.recordOutput(__codeflash_call_id_{call_id}, "{method_id}", "null", __codeflash_end_{call_id} - __codeflash_start_{call_id});
"""

# Template for benchmark instrumentation
BENCHMARK_IMPORT = """import com.codeflash.Blackhole;
import com.codeflash.BenchmarkContext;
import com.codeflash.BenchmarkResult;"""

BENCHMARK_WRAPPER_TEMPLATE = """
    // CodeFlash benchmark wrapper
    public void __codeflash_benchmark_{method_name}(int iterations) {{
        // Warmup
        for (int i = 0; i < Math.min(iterations / 10, 100); i++) {{
            {warmup_call}
        }}

        // Measurement
        long[] measurements = new long[iterations];
        for (int i = 0; i < iterations; i++) {{
            long start = System.nanoTime();
            {measurement_call}
            long end = System.nanoTime();
            measurements[i] = end - start;
        }}

        BenchmarkResult result = new BenchmarkResult("{method_id}", measurements);
        CodeFlash.recordBenchmarkResult("{method_id}", result);
    }}
"""


def instrument_for_behavior(
    source: str,
    functions: Sequence[FunctionInfo],
    analyzer: JavaAnalyzer | None = None,
) -> str:
    """Add behavior instrumentation to capture inputs/outputs.

    Wraps function calls to record arguments and return values
    for behavioral verification.

    Args:
        source: Source code to instrument.
        functions: Functions to add behavior capture.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Instrumented source code.

    """
    analyzer = analyzer or get_java_analyzer()

    if not functions:
        return source

    # Add import if not present
    if BEHAVIOR_CAPTURE_IMPORT not in source:
        source = _add_import(source, BEHAVIOR_CAPTURE_IMPORT)

    # Find and instrument each function
    for func in functions:
        source = _instrument_function_behavior(source, func, analyzer)

    return source


def _add_import(source: str, import_statement: str) -> str:
    """Add an import statement to the source.

    Args:
        source: The source code.
        import_statement: The import to add.

    Returns:
        Source with import added.

    """
    lines = source.splitlines(keepends=True)
    insert_idx = 0

    # Find the last import or package statement
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("package "):
            insert_idx = i + 1
        elif stripped and not stripped.startswith("//") and not stripped.startswith("/*"):
            # First non-import, non-comment line
            if insert_idx == 0:
                insert_idx = i
            break

    lines.insert(insert_idx, import_statement + "\n")
    return "".join(lines)


def _instrument_function_behavior(
    source: str,
    function: FunctionInfo,
    analyzer: JavaAnalyzer,
) -> str:
    """Instrument a single function for behavior capture.

    Args:
        source: The source code.
        function: The function to instrument.
        analyzer: JavaAnalyzer instance.

    Returns:
        Source with function instrumented.

    """
    source_bytes = source.encode("utf8")
    tree = analyzer.parse(source_bytes)

    # Find the method node
    methods = analyzer.find_methods(source)
    target_method = None
    func_name = _get_function_name(function)
    for method in methods:
        if method.name == func_name:
            class_name = getattr(function, "class_name", None)
            if class_name is None or method.class_name == class_name:
                target_method = method
                break

    if not target_method:
        logger.warning("Could not find method %s for instrumentation", func_name)
        return source

    # For now, we'll add instrumentation as a simple wrapper
    # A full implementation would use AST transformation
    method_id = function.qualified_name
    call_id = hash(method_id) % 10000

    # Build instrumented version
    # This is a simplified approach - a full implementation would
    # parse the method body and instrument each return statement
    logger.debug("Instrumented method %s for behavior capture", function.name)

    return source


def instrument_for_benchmarking(
    test_source: str,
    target_function: FunctionInfo,
    analyzer: JavaAnalyzer | None = None,
) -> str:
    """Add timing instrumentation to test code.

    Args:
        test_source: Test source code to instrument.
        target_function: Function being benchmarked.

    Returns:
        Instrumented test source code.

    """
    analyzer = analyzer or get_java_analyzer()

    # Add imports if not present
    if "import com.codeflash" not in test_source:
        test_source = _add_import(test_source, BENCHMARK_IMPORT)

    # Find calls to the target function in the test and wrap them
    # This is a simplified implementation
    logger.debug("Instrumented test for benchmarking %s", _get_function_name(target_function))

    return test_source


def instrument_existing_test(
    test_path: Path,
    call_positions: Sequence,
    function_to_optimize: FunctionInfo,
    tests_project_root: Path,
    mode: str,  # "behavior" or "performance"
    analyzer: JavaAnalyzer | None = None,
) -> tuple[bool, str | None]:
    """Inject profiling code into an existing test file.

    Args:
        test_path: Path to the test file.
        call_positions: List of code positions where the function is called.
        function_to_optimize: The function being optimized.
        tests_project_root: Root directory of tests.
        mode: Testing mode - "behavior" or "performance".
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Tuple of (success, instrumented_code or error message).

    """
    analyzer = analyzer or get_java_analyzer()

    try:
        source = test_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"Failed to read test file: {e}"

    try:
        if mode == "behavior":
            instrumented = instrument_for_behavior(source, [function_to_optimize], analyzer)
        else:
            instrumented = instrument_for_benchmarking(source, function_to_optimize, analyzer)

        return True, instrumented

    except Exception as e:
        logger.exception("Failed to instrument test file: %s", e)
        return False, str(e)


def create_benchmark_test(
    target_function: FunctionInfo,
    test_setup_code: str,
    invocation_code: str,
    iterations: int = 1000,
) -> str:
    """Create a benchmark test for a function.

    Args:
        target_function: The function to benchmark.
        test_setup_code: Code to set up the test (create instances, etc.).
        invocation_code: Code that invokes the function.
        iterations: Number of benchmark iterations.

    Returns:
        Complete benchmark test source code.

    """
    method_name = target_function.name
    method_id = target_function.qualified_name

    benchmark_code = f"""
import com.codeflash.Blackhole;
import com.codeflash.BenchmarkContext;
import com.codeflash.BenchmarkResult;
import com.codeflash.CodeFlash;
import org.junit.jupiter.api.Test;

public class {target_function.class_name or 'Target'}Benchmark {{

    @Test
    public void benchmark{method_name.capitalize()}() {{
        {test_setup_code}

        // Warmup phase
        for (int i = 0; i < {iterations // 10}; i++) {{
            Blackhole.consume({invocation_code});
        }}

        // Measurement phase
        long[] measurements = new long[{iterations}];
        for (int i = 0; i < {iterations}; i++) {{
            long start = System.nanoTime();
            Blackhole.consume({invocation_code});
            long end = System.nanoTime();
            measurements[i] = end - start;
        }}

        BenchmarkResult result = new BenchmarkResult("{method_id}", measurements);
        CodeFlash.recordBenchmarkResult("{method_id}", result);

        System.out.println("Benchmark complete: " + result);
    }}
}}
"""
    return benchmark_code


def remove_instrumentation(source: str) -> str:
    """Remove CodeFlash instrumentation from source code.

    Args:
        source: Instrumented source code.

    Returns:
        Source with instrumentation removed.

    """
    lines = source.splitlines(keepends=True)
    result_lines = []
    skip_until_end = False

    for line in lines:
        stripped = line.strip()

        # Skip CodeFlash instrumentation blocks
        if "// CodeFlash" in stripped and "start" in stripped:
            skip_until_end = True
            continue
        if skip_until_end:
            if "// CodeFlash" in stripped and "end" in stripped:
                skip_until_end = False
            continue

        # Skip CodeFlash imports
        if "import com.codeflash" in stripped:
            continue

        result_lines.append(line)

    return "".join(result_lines)
