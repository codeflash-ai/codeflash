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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.languages.java.parser import JavaAnalyzer

logger = logging.getLogger(__name__)


def _get_function_name(func: Any) -> str:
    """Get the function name from FunctionToOptimize."""
    if hasattr(func, "function_name"):
        return func.function_name
    if hasattr(func, "name"):
        return func.name
    msg = f"Cannot get function name from {type(func)}"
    raise AttributeError(msg)


# Pattern to detect primitive array types in assertions
_PRIMITIVE_ARRAY_PATTERN = re.compile(r"new\s+(int|long|double|float|short|byte|char|boolean)\s*\[\s*\]")


def _infer_array_cast_type(line: str) -> str | None:
    """Infer the array cast type needed for assertion methods.

    When a line contains an assertion like assertArrayEquals with a primitive array
    as the first argument, we need to cast the captured Object result back to
    that primitive array type.

    Args:
        line: The source line containing the assertion.

    Returns:
        The cast type (e.g., "int[]") if needed, None otherwise.

    """
    # Only apply to assertion methods that take arrays
    assertion_methods = ("assertArrayEquals", "assertArrayNotEquals")
    if not any(method in line for method in assertion_methods):
        return None

    # Look for primitive array type in the line (usually the first/expected argument)
    match = _PRIMITIVE_ARRAY_PATTERN.search(line)
    if match:
        primitive_type = match.group(1)
        return f"{primitive_type}[]"

    return None


def _get_qualified_name(func: Any) -> str:
    """Get the qualified name from FunctionToOptimize."""
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
    source: str, functions: Sequence[FunctionToOptimize], analyzer: JavaAnalyzer | None = None
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
    test_source: str, target_function: FunctionToOptimize, analyzer: JavaAnalyzer | None = None
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
    function_to_optimize: Any,  # FunctionToOptimize or FunctionToOptimize
    tests_project_root: Path,
    mode: str,  # "behavior" or "performance"
    analyzer: JavaAnalyzer | None = None,
    output_class_suffix: str | None = None,  # Suffix for renamed class
) -> tuple[bool, str | None]:
    """Inject profiling code into an existing test file.

    For Java, this:
    1. Renames the class to match the new file name (Java requires class name = file name)
    2. For behavior mode: adds timing instrumentation that writes to SQLite
    3. For performance mode: adds timing instrumentation with stdout markers

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
        logger.exception("Failed to read test file %s: %s", test_path, e)
        return False, f"Failed to read test file: {e}"

    func_name = _get_function_name(function_to_optimize)

    # Get the original class name from the file name
    original_class_name = test_path.stem  # e.g., "AlgorithmsTest"

    # Determine the new class name based on mode
    if mode == "behavior":
        new_class_name = f"{original_class_name}__perfinstrumented"
    else:
        new_class_name = f"{original_class_name}__perfonlyinstrumented"

    # Rename all references to the original class name in the source.
    # Uses tree-sitter to find identifier/type_identifier AST nodes,
    # which correctly excludes matches inside string literals and comments.
    if analyzer is None:
        from codeflash.languages.java.parser import get_java_analyzer
        analyzer = get_java_analyzer()

    refs = analyzer.find_identifier_references(source, original_class_name)
    if refs:
        source_bytes = source.encode("utf8")
        new_name_bytes = new_class_name.encode("utf8")
        for start, end in reversed(refs):
            source_bytes = source_bytes[:start] + new_name_bytes + source_bytes[end:]
        modified_source = source_bytes.decode("utf8")
    else:
        modified_source = source

    # Add timing instrumentation to test methods
    # Use original class name (without suffix) in timing markers for consistency with Python
    if mode == "performance":
        modified_source = _add_timing_instrumentation(
            modified_source,
            original_class_name,  # Use original name in markers, not the renamed class
            func_name,
        )
    else:
        # Behavior mode: add timing instrumentation that also writes to SQLite
        modified_source = _add_behavior_instrumentation(modified_source, original_class_name, func_name)

    logger.debug("Java %s testing for %s: renamed class %s -> %s", mode, func_name, original_class_name, new_class_name)

    return True, modified_source


def _add_behavior_instrumentation(source: str, class_name: str, func_name: str) -> str:
    """Add behavior instrumentation to test methods.

    Uses tree-sitter to find @Test methods, their body boundaries, and
    method invocations of func_name (with lambda-awareness via parent-chain walk).

    For behavior mode, this adds:
    1. SQL imports for SQLite database writes
    2. SQLite database connection setup
    3. Function call wrapping to capture return values
    4. SQLite insert with serialized return values

    Args:
        source: The test source code.
        class_name: Name of the test class.
        func_name: Name of the function being tested.

    Returns:
        Instrumented source code.

    """
    from codeflash.languages.java.parser import get_java_analyzer

    analyzer = get_java_analyzer()

    # ── Step 1: Add imports ──────────────────────────────────────────────
    import_statements = [
        "import java.sql.Connection;",
        "import java.sql.DriverManager;",
        "import java.sql.PreparedStatement;",
    ]

    lines = source.split("\n")
    result_lines: list[str] = []
    imports_added = False
    idx = 0

    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        if not imports_added:
            if stripped.startswith("import "):
                result_lines.append(line)
                idx += 1
                while idx < len(lines) and lines[idx].strip().startswith("import "):
                    result_lines.append(lines[idx])
                    idx += 1
                for imp in import_statements:
                    if imp not in source:
                        result_lines.append(imp)
                imports_added = True
                continue
            # Use tree-sitter class detection: any class/interface/enum keyword
            if stripped.startswith(("public class", "class", "public final class",
                                   "final class", "abstract class", "public abstract class")):
                result_lines.extend(import_statements)
                result_lines.append("")
                imports_added = True

        result_lines.append(line)
        idx += 1

    source = "\n".join(result_lines)

    # ── Step 2: Find @Test methods and wrap their bodies ─────────────────
    source_bytes = source.encode("utf8")
    test_methods = analyzer.find_test_methods(source)

    if not test_methods:
        return source

    lines = source.split("\n")
    replacements: list[tuple[int, int, str]] = []

    for iter_id, method_info in enumerate(test_methods, start=1):
        body_node = method_info.body_node

        # Extract body lines (between { and })
        body_start_line = body_node.start_point[0]
        body_end_line = body_node.end_point[0]
        body_lines = lines[body_start_line + 1 : body_end_line]

        brace_line = lines[body_start_line]
        base_indent = len(brace_line) - len(brace_line.lstrip())
        indent = " " * (base_indent + 4)

        # ── Find method invocations via tree-sitter ──────────────────────
        invocations = analyzer.find_method_invocations(body_node, source_bytes, func_name)

        # Group invocations by their 0-indexed source line
        invocations_by_line: dict[int, list] = {}
        for inv in invocations:
            inv_line = inv.node.start_point[0]
            invocations_by_line.setdefault(inv_line, []).append(inv)

        # ── Wrap function calls per body line ────────────────────────────
        call_counter = 0
        wrapped_body_lines: list[str] = []

        for local_idx, body_line in enumerate(body_lines):
            source_line_idx = body_start_line + 1 + local_idx
            line_invocations = invocations_by_line.get(source_line_idx, [])

            # Filter to non-lambda invocations on a single line
            actionable = [
                inv for inv in line_invocations
                if not inv.in_lambda and inv.node.start_point[0] == inv.node.end_point[0]
            ]

            if actionable:
                line_indent = len(body_line) - len(body_line.lstrip())
                line_indent_str = " " * line_indent

                # Process matches in reverse column order to preserve positions
                actionable.sort(key=lambda inv: inv.node.start_point[1], reverse=True)
                new_line = body_line
                last_var_name = None
                last_var_with_cast = None

                for inv in actionable:
                    call_counter += 1
                    var_name = f"_cf_result{iter_id}_{call_counter}"
                    last_var_name = var_name

                    cast_type = _infer_array_cast_type(body_line)
                    var_with_cast = f"({cast_type}){var_name}" if cast_type else var_name
                    last_var_with_cast = var_with_cast

                    start_col = inv.node.start_point[1]
                    end_col = inv.node.end_point[1]
                    new_line = new_line[:start_col] + var_with_cast + new_line[end_col:]

                    capture_line = f"{line_indent_str}var {var_name} = {inv.full_text};"
                    wrapped_body_lines.append(capture_line)

                    serialize_line = (
                        f"{line_indent_str}_cf_serializedResult{iter_id} = "
                        f"com.codeflash.Serializer.serialize((Object) {var_name});"
                    )
                    wrapped_body_lines.append(serialize_line)

                # Skip line if it collapsed to just a bare variable reference
                stripped_new = new_line.strip().rstrip(";").strip()
                if stripped_new and stripped_new not in (last_var_name, last_var_with_cast):
                    wrapped_body_lines.append(new_line)
            else:
                wrapped_body_lines.append(body_line)

        # ── Build replacement body ───────────────────────────────────────
        method_close_indent = " " * base_indent
        behavior_lines = [
            "{",
            f"{indent}// Codeflash behavior instrumentation",
            f'{indent}int _cf_loop{iter_id} = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));',
            f"{indent}int _cf_iter{iter_id} = {iter_id};",
            f'{indent}String _cf_mod{iter_id} = "{class_name}";',
            f'{indent}String _cf_cls{iter_id} = "{class_name}";',
            f'{indent}String _cf_fn{iter_id} = "{func_name}";',
            f'{indent}String _cf_outputFile{iter_id} = System.getenv("CODEFLASH_OUTPUT_FILE");',
            f'{indent}String _cf_testIteration{iter_id} = System.getenv("CODEFLASH_TEST_ITERATION");',
            f'{indent}if (_cf_testIteration{iter_id} == null) _cf_testIteration{iter_id} = "0";',
            f'{indent}System.out.println("!$######" + _cf_mod{iter_id} + ":" + _cf_cls{iter_id} + ":" + _cf_fn{iter_id} + ":" + _cf_loop{iter_id} + ":" + _cf_iter{iter_id} + "######$!");',
            f"{indent}long _cf_start{iter_id} = System.nanoTime();",
            f"{indent}byte[] _cf_serializedResult{iter_id} = null;",
            f"{indent}try {{",
        ]

        for bl in wrapped_body_lines:
            behavior_lines.append("    " + bl)

        behavior_lines.extend([
            f"{indent}}} finally {{",
            f"{indent}    long _cf_end{iter_id} = System.nanoTime();",
            f"{indent}    long _cf_dur{iter_id} = _cf_end{iter_id} - _cf_start{iter_id};",
            f'{indent}    System.out.println("!######" + _cf_mod{iter_id} + ":" + _cf_cls{iter_id} + ":" + _cf_fn{iter_id} + ":" + _cf_loop{iter_id} + ":" + _cf_iter{iter_id} + ":" + _cf_dur{iter_id} + "######!");',
            f"{indent}    // Write to SQLite if output file is set",
            f"{indent}    if (_cf_outputFile{iter_id} != null && !_cf_outputFile{iter_id}.isEmpty()) {{",
            f"{indent}        try {{",
            f'{indent}            Class.forName("org.sqlite.JDBC");',
            f'{indent}            try (Connection _cf_conn{iter_id} = DriverManager.getConnection("jdbc:sqlite:" + _cf_outputFile{iter_id})) {{',
            f"{indent}                try (java.sql.Statement _cf_stmt{iter_id} = _cf_conn{iter_id}.createStatement()) {{",
            f'{indent}                    _cf_stmt{iter_id}.execute("CREATE TABLE IF NOT EXISTS test_results (" +',
            f'{indent}                        "test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, " +',
            f'{indent}                        "function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, " +',
            f'{indent}                        "runtime INTEGER, return_value BLOB, verification_type TEXT)");',
            f"{indent}                }}",
            f'{indent}                String _cf_sql{iter_id} = "INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";',
            f"{indent}                try (PreparedStatement _cf_pstmt{iter_id} = _cf_conn{iter_id}.prepareStatement(_cf_sql{iter_id})) {{",
            f"{indent}                    _cf_pstmt{iter_id}.setString(1, _cf_mod{iter_id});",
            f"{indent}                    _cf_pstmt{iter_id}.setString(2, _cf_cls{iter_id});",
            f'{indent}                    _cf_pstmt{iter_id}.setString(3, "{class_name}Test");',
            f"{indent}                    _cf_pstmt{iter_id}.setString(4, _cf_fn{iter_id});",
            f"{indent}                    _cf_pstmt{iter_id}.setInt(5, _cf_loop{iter_id});",
            f'{indent}                    _cf_pstmt{iter_id}.setString(6, _cf_iter{iter_id} + "_" + _cf_testIteration{iter_id});',
            f"{indent}                    _cf_pstmt{iter_id}.setLong(7, _cf_dur{iter_id});",
            f"{indent}                    _cf_pstmt{iter_id}.setBytes(8, _cf_serializedResult{iter_id});",
            f'{indent}                    _cf_pstmt{iter_id}.setString(9, "function_call");',
            f"{indent}                    _cf_pstmt{iter_id}.executeUpdate();",
            f"{indent}                }}",
            f"{indent}            }}",
            f"{indent}        }} catch (Exception _cf_e{iter_id}) {{",
            f'{indent}            System.err.println("CodeflashHelper: SQLite error: " + _cf_e{iter_id}.getMessage());',
            f"{indent}        }}",
            f"{indent}    }}",
            f"{indent}}}",
            f"{method_close_indent}}}",
        ])

        replacement = "\n".join(behavior_lines)
        replacements.append((body_node.start_byte, body_node.end_byte, replacement))

    # Apply replacements in reverse byte order
    for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
        source_bytes = source_bytes[:start] + replacement.encode("utf8") + source_bytes[end:]

    return source_bytes.decode("utf8")


def _add_timing_instrumentation(source: str, class_name: str, func_name: str) -> str:
    """Add timing instrumentation to test methods with inner loop for JIT warmup.

    Uses tree-sitter to find @Test methods and their body boundaries,
    then replaces each method body with a timing-wrapped version.

    For each @Test method, this adds:
    1. Inner loop that runs N iterations (controlled by CODEFLASH_INNER_ITERATIONS env var)
    2. Start timing marker printed at the beginning of each iteration
    3. End timing marker printed at the end of each iteration (in a finally block)

    The inner loop allows JIT warmup within a single JVM invocation, avoiding
    expensive Maven restarts. Post-processing uses min runtime across all iterations.

    Timing markers format:
      Start: !$######testModule:testClass:funcName:loopIndex:iterationId######$!
      End:   !######testModule:testClass:funcName:loopIndex:iterationId:durationNs######!

    Where iterationId is the inner iteration number (0, 1, 2, ..., N-1).

    Args:
        source: The test source code.
        class_name: Name of the test class.
        func_name: Name of the function being tested.

    Returns:
        Instrumented source code.

    """
    from codeflash.languages.java.parser import get_java_analyzer

    analyzer = get_java_analyzer()
    source_bytes = source.encode("utf8")
    test_methods = analyzer.find_test_methods(source)

    if not test_methods:
        return source

    lines = source.split("\n")

    # Build a replacement for each @Test method body.
    # iter_id is assigned in forward (source) order; replacements are applied in reverse.
    replacements: list[tuple[int, int, str]] = []

    for iter_id, method_info in enumerate(test_methods, start=1):
        body_node = method_info.body_node

        # Extract body lines (between { and }, exclusive of both brace lines)
        body_start_line = body_node.start_point[0]  # 0-indexed line of {
        body_end_line = body_node.end_point[0]  # 0-indexed line of }
        body_lines = lines[body_start_line + 1 : body_end_line]

        # Indentation from the line containing the opening brace
        brace_line = lines[body_start_line]
        base_indent = len(brace_line) - len(brace_line.lstrip())
        indent = " " * (base_indent + 4)
        inner_indent = " " * (base_indent + 8)
        method_close_indent = " " * base_indent

        # Build the replacement body (opening { through closing })
        timing_lines = [
            "{",
            f"{indent}// Codeflash timing instrumentation with inner loop for JIT warmup",
            f'{indent}int _cf_loop{iter_id} = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));',
            f'{indent}int _cf_innerIterations{iter_id} = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "100"));',
            f'{indent}String _cf_mod{iter_id} = "{class_name}";',
            f'{indent}String _cf_cls{iter_id} = "{class_name}";',
            f'{indent}String _cf_fn{iter_id} = "{func_name}";',
            "",
            f"{indent}for (int _cf_i{iter_id} = 0; _cf_i{iter_id} < _cf_innerIterations{iter_id}; _cf_i{iter_id}++) {{",
            f'{inner_indent}System.out.println("!$######" + _cf_mod{iter_id} + ":" + _cf_cls{iter_id} + ":" + _cf_fn{iter_id} + ":" + _cf_loop{iter_id} + ":" + _cf_i{iter_id} + "######$!");',
            f"{inner_indent}long _cf_start{iter_id} = System.nanoTime();",
            f"{inner_indent}try {{",
        ]

        # Original body lines, indented by 8 extra spaces (for loop + try)
        for bl in body_lines:
            timing_lines.append("        " + bl)

        timing_lines.extend([
            f"{inner_indent}}} finally {{",
            f"{inner_indent}    long _cf_end{iter_id} = System.nanoTime();",
            f"{inner_indent}    long _cf_dur{iter_id} = _cf_end{iter_id} - _cf_start{iter_id};",
            f'{inner_indent}    System.out.println("!######" + _cf_mod{iter_id} + ":" + _cf_cls{iter_id} + ":" + _cf_fn{iter_id} + ":" + _cf_loop{iter_id} + ":" + _cf_i{iter_id} + ":" + _cf_dur{iter_id} + "######!");',
            f"{inner_indent}}}",
            f"{indent}}}",  # Close for loop
            f"{method_close_indent}}}",  # Method closing brace
        ])

        replacement = "\n".join(timing_lines)
        replacements.append((body_node.start_byte, body_node.end_byte, replacement))

    # Apply replacements in reverse byte order to preserve earlier offsets
    for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
        source_bytes = source_bytes[:start] + replacement.encode("utf8") + source_bytes[end:]

    return source_bytes.decode("utf8")


def create_benchmark_test(
    target_function: FunctionToOptimize, test_setup_code: str, invocation_code: str, iterations: int = 1000
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

    return f"""
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

    For generated tests (AI-generated), this function:
    1. Removes assertions and captures function return values (for regression testing)
    2. Renames the class to include mode suffix
    3. Adds timing instrumentation for performance mode

    Args:
        test_code: The generated test source code.
        function_name: Name of the function being tested.
        qualified_name: Fully qualified name of the function.
        mode: "behavior" for behavior capture or "performance" for timing.

    Returns:
        Instrumented test source code.

    """
    from codeflash.languages.java.remove_asserts import transform_java_assertions

    # For behavior mode, remove assertions and capture function return values
    # This converts the generated test into a regression test that captures outputs
    if mode == "behavior":
        test_code = transform_java_assertions(test_code, function_name, qualified_name)

    # Extract class name from the test code using tree-sitter
    from codeflash.languages.java.parser import get_java_analyzer

    analyzer = get_java_analyzer()
    classes = analyzer.find_classes(test_code)
    if not classes:
        logger.warning("Could not find class name in generated test")
        return test_code

    original_class_name = classes[0].name

    # Rename class based on mode
    if mode == "behavior":
        new_class_name = f"{original_class_name}__perfinstrumented"
    else:
        new_class_name = f"{original_class_name}__perfonlyinstrumented"

    # Rename all identifier references to the class name using tree-sitter
    # (excludes matches inside string literals and comments)
    refs = analyzer.find_identifier_references(test_code, original_class_name)
    if refs:
        code_bytes = test_code.encode("utf8")
        new_name_bytes = new_class_name.encode("utf8")
        for start, end in reversed(refs):
            code_bytes = code_bytes[:start] + new_name_bytes + code_bytes[end:]
        modified_code = code_bytes.decode("utf8")
    else:
        modified_code = test_code

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



