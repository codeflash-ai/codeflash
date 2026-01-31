"""Java code instrumentation for behavior capture and benchmarking.

This module provides functionality to instrument Java code for:
1. Behavior capture - recording inputs/outputs for verification
2. Benchmarking - measuring execution time

Timing instrumentation adds System.nanoTime() calls around the function being tested
and prints timing markers in a format compatible with Python/JS implementations:
  Start: !$######testModule:testClass:funcName:loopIndex:iterationId######$!
  End:   !######testModule:testClass:funcName:loopIndex:iterationId:durationNs######!

This allows codeflash to extract timing data from stdout for accurate benchmarking.
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


def _get_qualified_name(func: Any) -> str:
    """Get the qualified name from either FunctionInfo or FunctionToOptimize."""
    if hasattr(func, "qualified_name"):
        return func.qualified_name
    # Build qualified name from function_name and parents
    if hasattr(func, "function_name"):
        parts = []
        if hasattr(func, "parents") and func.parents:
            for parent in func.parents:
                if hasattr(parent, "name"):
                    parts.append(parent.name)
        parts.append(func.function_name)
        return ".".join(parts)
    return str(func)


def instrument_for_behavior(
    source: str,
    functions: Sequence[FunctionInfo],
    analyzer: JavaAnalyzer | None = None,
) -> str:
    """Add behavior instrumentation to capture inputs/outputs.

    For Java, we don't modify the test file for behavior capture.
    Instead, we rely on JUnit test results (pass/fail) to verify correctness.
    The test file is returned unchanged.

    Args:
        source: Source code to instrument.
        functions: Functions to add behavior capture.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Source code (unchanged for Java).

    """
    # For Java, we don't need to instrument tests for behavior capture.
    # The JUnit test results (pass/fail) serve as the verification mechanism.
    if functions:
        func_name = _get_function_name(functions[0])
        logger.debug("Java behavior testing for %s - using JUnit pass/fail results", func_name)
    return source


def instrument_for_benchmarking(
    test_source: str,
    target_function: FunctionInfo,
    analyzer: JavaAnalyzer | None = None,
) -> str:
    """Add timing instrumentation to test code.

    For Java, we rely on Maven Surefire's timing information rather than
    modifying the test code. The test file is returned unchanged.

    Args:
        test_source: Test source code to instrument.
        target_function: Function being benchmarked.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Test source code (unchanged for Java).

    """
    func_name = _get_function_name(target_function)
    logger.debug("Java benchmarking for %s - using Maven Surefire timing", func_name)
    return test_source


def instrument_existing_test(
    test_path: Path,
    call_positions: Sequence,
    function_to_optimize: Any,  # FunctionInfo or FunctionToOptimize
    tests_project_root: Path,
    mode: str,  # "behavior" or "performance"
    analyzer: JavaAnalyzer | None = None,
    output_class_suffix: str | None = None,  # Suffix for renamed class
) -> tuple[bool, str | None]:
    """Inject profiling code into an existing test file.

    For Java, this:
    1. Renames the class to match the new file name (Java requires class name = file name)
    2. Adds timing instrumentation to test methods (for performance mode)

    Args:
        test_path: Path to the test file.
        call_positions: List of code positions where the function is called.
        function_to_optimize: The function being optimized.
        tests_project_root: Root directory of tests.
        mode: Testing mode - "behavior" or "performance".
        analyzer: Optional JavaAnalyzer instance.
        output_class_suffix: Optional suffix for the renamed class.

    Returns:
        Tuple of (success, modified_source).

    """
    try:
        source = test_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("Failed to read test file %s: %s", test_path, e)
        return False, f"Failed to read test file: {e}"

    func_name = _get_function_name(function_to_optimize)

    # Get the original class name from the file name
    original_class_name = test_path.stem  # e.g., "AlgorithmsTest"

    # Determine the new class name based on mode
    if mode == "behavior":
        new_class_name = f"{original_class_name}__perfinstrumented"
    else:
        new_class_name = f"{original_class_name}__perfonlyinstrumented"

    # Rename the class declaration in the source
    # Pattern: "public class ClassName" or "class ClassName"
    pattern = rf'\b(public\s+)?class\s+{re.escape(original_class_name)}\b'
    replacement = rf'\1class {new_class_name}'
    modified_source = re.sub(pattern, replacement, source)

    # For performance mode, add timing instrumentation to test methods
    # Use original class name (without suffix) in timing markers for consistency with Python
    if mode == "performance":
        modified_source = _add_timing_instrumentation(
            modified_source,
            original_class_name,  # Use original name in markers, not the renamed class
            func_name,
        )

    logger.debug(
        "Java %s testing for %s: renamed class %s -> %s",
        mode,
        func_name,
        original_class_name,
        new_class_name,
    )

    return True, modified_source


def _add_timing_instrumentation(source: str, class_name: str, func_name: str) -> str:
    """Add timing instrumentation to test methods.

    For each @Test method, this adds:
    1. Start timing marker printed at the beginning
    2. End timing marker printed at the end (in a finally block)

    Timing markers format:
      Start: !$######testModule:testClass:funcName:loopIndex:iterationId######$!
      End:   !######testModule:testClass:funcName:loopIndex:iterationId:durationNs######!

    Args:
        source: The test source code.
        class_name: Name of the test class.
        func_name: Name of the function being tested.

    Returns:
        Instrumented source code.

    """
    # Find all @Test methods and add timing around their bodies
    # Pattern matches: @Test (with optional parameters) followed by method declaration
    # We process line by line for cleaner handling

    lines = source.split('\n')
    result = []
    i = 0
    iteration_counter = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Look for @Test annotation
        if stripped.startswith('@Test'):
            result.append(line)
            i += 1

            # Collect any additional annotations
            while i < len(lines) and lines[i].strip().startswith('@'):
                result.append(lines[i])
                i += 1

            # Now find the method signature and opening brace
            method_lines = []
            while i < len(lines):
                method_lines.append(lines[i])
                if '{' in lines[i]:
                    break
                i += 1

            # Add the method signature lines
            for ml in method_lines:
                result.append(ml)
            i += 1

            # We're now inside the method body
            iteration_counter += 1
            iter_id = iteration_counter

            # Detect indentation from method signature line (line with opening brace)
            method_sig_line = method_lines[-1] if method_lines else ""
            base_indent = len(method_sig_line) - len(method_sig_line.lstrip())
            indent = " " * (base_indent + 4)  # Add one level of indentation

            # Add timing start code
            # Note: CODEFLASH_LOOP_INDEX must always be set - no null check, crash if missing
            # Start marker is printed BEFORE timing starts
            # System.nanoTime() immediately precedes try block with test code
            timing_start_code = [
                f"{indent}// Codeflash timing instrumentation",
                f'{indent}int _cf_loop{iter_id} = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));',
                f"{indent}int _cf_iter{iter_id} = {iter_id};",
                f'{indent}String _cf_mod{iter_id} = "{class_name}";',
                f'{indent}String _cf_cls{iter_id} = "{class_name}";',
                f'{indent}String _cf_fn{iter_id} = "{func_name}";',
                f'{indent}System.out.println("!$######" + _cf_mod{iter_id} + ":" + _cf_cls{iter_id} + ":" + _cf_fn{iter_id} + ":" + _cf_loop{iter_id} + ":" + _cf_iter{iter_id} + "######$!");',
                f"{indent}long _cf_start{iter_id} = System.nanoTime();",
                f"{indent}try {{",
            ]
            result.extend(timing_start_code)

            # Collect method body until we find matching closing brace
            brace_depth = 1
            body_lines = []

            while i < len(lines) and brace_depth > 0:
                body_line = lines[i]
                # Count braces (simple approach - doesn't handle strings/comments perfectly)
                for ch in body_line:
                    if ch == '{':
                        brace_depth += 1
                    elif ch == '}':
                        brace_depth -= 1

                if brace_depth > 0:
                    body_lines.append(body_line)
                    i += 1
                else:
                    # This line contains the closing brace, but we've hit depth 0
                    # Add indented body lines
                    for bl in body_lines:
                        result.append("    " + bl)

                    # Add finally block
                    method_close_indent = " " * base_indent  # Same level as method signature
                    timing_end_code = [
                        f"{indent}}} finally {{",
                        f"{indent}    long _cf_end{iter_id} = System.nanoTime();",
                        f"{indent}    long _cf_dur{iter_id} = _cf_end{iter_id} - _cf_start{iter_id};",
                        f'{indent}    System.out.println("!######" + _cf_mod{iter_id} + ":" + _cf_cls{iter_id} + ":" + _cf_fn{iter_id} + ":" + _cf_loop{iter_id} + ":" + _cf_iter{iter_id} + ":" + _cf_dur{iter_id} + "######!");',
                        f"{indent}}}",
                        f"{method_close_indent}}}",  # Method closing brace
                    ]
                    result.extend(timing_end_code)
                    i += 1
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)


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
    method_name = _get_function_name(target_function)
    method_id = _get_qualified_name(target_function)
    class_name = getattr(target_function, "class_name", None) or "Target"

    benchmark_code = f"""
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

/**
 * Benchmark test for {method_name}.
 * Generated by CodeFlash.
 */
public class {class_name}Benchmark {{

    @Test
    @DisplayName("Benchmark {method_name}")
    public void benchmark{method_name.capitalize()}() {{
        {test_setup_code}

        // Warmup phase
        for (int i = 0; i < {iterations // 10}; i++) {{
            {invocation_code};
        }}

        // Measurement phase
        long startTime = System.nanoTime();
        for (int i = 0; i < {iterations}; i++) {{
            {invocation_code};
        }}
        long endTime = System.nanoTime();

        long totalNanos = endTime - startTime;
        long avgNanos = totalNanos / {iterations};

        System.out.println("CODEFLASH_BENCHMARK:{method_id}:total_ns=" + totalNanos + ",avg_ns=" + avgNanos + ",iterations={iterations}");
    }}
}}
"""
    return benchmark_code


def remove_instrumentation(source: str) -> str:
    """Remove CodeFlash instrumentation from source code.

    For Java, since we don't add instrumentation, this is a no-op.

    Args:
        source: Source code.

    Returns:
        Source unchanged.

    """
    return source


def instrument_generated_java_test(
    test_code: str,
    function_name: str,
    qualified_name: str,
    mode: str,  # "behavior" or "performance"
) -> str:
    """Instrument a generated Java test for behavior or performance testing.

    Args:
        test_code: The generated test source code.
        function_name: Name of the function being tested.
        qualified_name: Fully qualified name of the function.
        mode: "behavior" for behavior capture or "performance" for timing.

    Returns:
        Instrumented test source code.

    """
    # Extract class name from the test code
    class_match = re.search(r'\bclass\s+(\w+)', test_code)
    if not class_match:
        logger.warning("Could not find class name in generated test")
        return test_code

    original_class_name = class_match.group(1)

    # Rename class based on mode
    if mode == "behavior":
        new_class_name = f"{original_class_name}__perfinstrumented"
    else:
        new_class_name = f"{original_class_name}__perfonlyinstrumented"

    # Rename the class in the source
    modified_code = re.sub(
        rf'\b(public\s+)?class\s+{re.escape(original_class_name)}\b',
        rf'\1class {new_class_name}',
        test_code,
    )

    # For performance mode, add timing instrumentation
    # Use original class name (without suffix) in timing markers for consistency with Python
    if mode == "performance":
        modified_code = _add_timing_instrumentation(
            modified_code,
            original_class_name,  # Use original name in markers, not the renamed class
            function_name,
        )

    logger.debug("Instrumented generated Java test for %s (mode=%s)", function_name, mode)
    return modified_code


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
