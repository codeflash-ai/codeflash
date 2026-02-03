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
from codeflash.languages.java.parser import JavaAnalyzer

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
    pattern = rf"\b(public\s+)?class\s+{re.escape(original_class_name)}\b"
    replacement = rf"\1class {new_class_name}"
    modified_source = re.sub(pattern, replacement, source)

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
        modified_source = _add_behavior_instrumentation(
            modified_source,
            original_class_name,
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


def _add_behavior_instrumentation(source: str, class_name: str, func_name: str) -> str:
    """Add behavior instrumentation to test methods.

    For behavior mode, this adds:
    1. Gson import for JSON serialization
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
    # Add necessary imports at the top of the file
    # Note: We don't import java.sql.Statement because it can conflict with
    # other Statement classes (e.g., com.aerospike.client.query.Statement).
    # Instead, we use the fully qualified name java.sql.Statement in the code.
    # Note: We don't use Gson because it may not be available as a dependency.
    # Instead, we use String.valueOf() for serialization.
    import_statements = [
        "import java.sql.Connection;",
        "import java.sql.DriverManager;",
        "import java.sql.PreparedStatement;",
    ]

    # Find position to insert imports (after package, before class)
    lines = source.split("\n")
    result = []
    imports_added = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Add imports after the last existing import or before the class declaration
        if not imports_added:
            if stripped.startswith("import "):
                result.append(line)
                i += 1
                # Find end of imports
                while i < len(lines) and lines[i].strip().startswith("import "):
                    result.append(lines[i])
                    i += 1
                # Add our imports
                for imp in import_statements:
                    if imp not in source:
                        result.append(imp)
                imports_added = True
                continue
            if stripped.startswith("public class") or stripped.startswith("class"):
                # No imports found, add before class
                for imp in import_statements:
                    result.append(imp)
                result.append("")
                imports_added = True

        result.append(line)
        i += 1

    # Now add timing and SQLite instrumentation to test methods
    source = "\n".join(result)
    lines = source.split("\n")
    result = []
    i = 0
    iteration_counter = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Look for @Test annotation
        if stripped.startswith("@Test"):
            result.append(line)
            i += 1

            # Collect any additional annotations
            while i < len(lines) and lines[i].strip().startswith("@"):
                result.append(lines[i])
                i += 1

            # Now find the method signature and opening brace
            method_lines = []
            while i < len(lines):
                method_lines.append(lines[i])
                if "{" in lines[i]:
                    break
                i += 1

            # Add the method signature lines
            for ml in method_lines:
                result.append(ml)
            i += 1

            # We're now inside the method body
            iteration_counter += 1
            iter_id = iteration_counter

            # Detect indentation
            method_sig_line = method_lines[-1] if method_lines else ""
            base_indent = len(method_sig_line) - len(method_sig_line.lstrip())
            indent = " " * (base_indent + 4)

            # Collect method body until we find matching closing brace
            brace_depth = 1
            body_lines = []

            while i < len(lines) and brace_depth > 0:
                body_line = lines[i]
                for ch in body_line:
                    if ch == "{":
                        brace_depth += 1
                    elif ch == "}":
                        brace_depth -= 1

                if brace_depth > 0:
                    body_lines.append(body_line)
                    i += 1
                else:
                    # We've hit the closing brace
                    i += 1
                    break

            # Wrap function calls to capture return values
            # Look for patterns like: obj.funcName(args) or new Class().funcName(args)
            call_counter = 0
            wrapped_body_lines = []

            # Use regex to find method calls with the target function
            # Pattern matches: receiver.funcName(args) where receiver can be:
            # - identifier (counter, calc, etc.)
            # - new ClassName()
            # - new ClassName(args)
            # - this
            method_call_pattern = re.compile(
                rf"((?:new\s+\w+\s*\([^)]*\)|[a-zA-Z_]\w*))\s*\.\s*({re.escape(func_name)})\s*\(([^)]*)\)",
                re.MULTILINE
            )

            for body_line in body_lines:
                # Check if this line contains a call to the target function
                if func_name in body_line and "(" in body_line:
                    line_indent = len(body_line) - len(body_line.lstrip())
                    line_indent_str = " " * line_indent

                    # Find all matches in the line
                    matches = list(method_call_pattern.finditer(body_line))
                    if matches:
                        # Process matches in reverse order to maintain correct positions
                        new_line = body_line
                        for match in reversed(matches):
                            call_counter += 1
                            var_name = f"_cf_result{iter_id}_{call_counter}"
                            full_call = match.group(0)  # e.g., "new StringUtils().reverse(\"hello\")"

                            # Replace this occurrence with the variable
                            new_line = new_line[:match.start()] + var_name + new_line[match.end():]

                            # Insert capture line
                            capture_line = f"{line_indent_str}Object {var_name} = {full_call};"
                            wrapped_body_lines.append(capture_line)

                        wrapped_body_lines.append(new_line)
                    else:
                        wrapped_body_lines.append(body_line)
                else:
                    wrapped_body_lines.append(body_line)

            # Build the serialized return value expression
            # If we captured any calls, serialize the last one; otherwise serialize null
            # Note: We use String.valueOf() instead of Gson to avoid external dependencies
            if call_counter > 0:
                result_var = f"_cf_result{iter_id}_{call_counter}"
                serialize_expr = f"String.valueOf({result_var})"
            else:
                serialize_expr = '"null"'

            # Add behavior instrumentation code
            behavior_start_code = [
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
                f"{indent}String _cf_serializedResult{iter_id} = null;",
                f"{indent}try {{",
            ]
            result.extend(behavior_start_code)

            # Add the wrapped body lines with extra indentation
            for bl in wrapped_body_lines:
                result.append("    " + bl)

            # Add serialization after the body (before finally)
            result.append(f"{indent}    _cf_serializedResult{iter_id} = {serialize_expr};")

            # Add finally block with SQLite write
            method_close_indent = " " * base_indent
            behavior_end_code = [
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
                f'{indent}                        "runtime INTEGER, return_value TEXT, verification_type TEXT)");',
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
                f"{indent}                    _cf_pstmt{iter_id}.setString(8, _cf_serializedResult{iter_id});",  # Serialized return value
                f'{indent}                    _cf_pstmt{iter_id}.setString(9, "function_call");',
                f"{indent}                    _cf_pstmt{iter_id}.executeUpdate();",
                f"{indent}                }}",
                f"{indent}            }}",
                f"{indent}        }} catch (Exception _cf_e{iter_id}) {{",
                f'{indent}            System.err.println("CodeflashHelper: SQLite error: " + _cf_e{iter_id}.getMessage());',
                f"{indent}        }}",
                f"{indent}    }}",
                f"{indent}}}",
                f"{method_close_indent}}}",  # Method closing brace
            ]
            result.extend(behavior_end_code)
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


def _add_timing_instrumentation(source: str, class_name: str, func_name: str) -> str:
    """Add timing instrumentation to test methods with inner loop for JIT warmup.

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
    # Find all @Test methods and add timing around their bodies
    # Pattern matches: @Test (with optional parameters) followed by method declaration
    # We process line by line for cleaner handling

    lines = source.split("\n")
    result = []
    i = 0
    iteration_counter = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Look for @Test annotation
        if stripped.startswith("@Test"):
            result.append(line)
            i += 1

            # Collect any additional annotations
            while i < len(lines) and lines[i].strip().startswith("@"):
                result.append(lines[i])
                i += 1

            # Now find the method signature and opening brace
            method_lines = []
            while i < len(lines):
                method_lines.append(lines[i])
                if "{" in lines[i]:
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
            inner_indent = " " * (base_indent + 8)  # Two levels for inside inner loop
            inner_body_indent = " " * (base_indent + 12)  # Three levels for try block body

            # Add timing instrumentation with inner loop
            # Note: CODEFLASH_LOOP_INDEX must always be set - no null check, crash if missing
            # CODEFLASH_INNER_ITERATIONS controls inner loop count (default: 100)
            timing_start_code = [
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
            result.extend(timing_start_code)

            # Collect method body until we find matching closing brace
            brace_depth = 1
            body_lines = []

            while i < len(lines) and brace_depth > 0:
                body_line = lines[i]
                # Count braces (simple approach - doesn't handle strings/comments perfectly)
                for ch in body_line:
                    if ch == "{":
                        brace_depth += 1
                    elif ch == "}":
                        brace_depth -= 1

                if brace_depth > 0:
                    body_lines.append(body_line)
                    i += 1
                else:
                    # This line contains the closing brace, but we've hit depth 0
                    # Add indented body lines (inside try block, inside for loop)
                    for bl in body_lines:
                        result.append("        " + bl)  # 8 extra spaces for inner loop + try

                    # Add finally block and close inner loop
                    method_close_indent = " " * base_indent  # Same level as method signature
                    timing_end_code = [
                        f"{inner_indent}}} finally {{",
                        f"{inner_indent}    long _cf_end{iter_id} = System.nanoTime();",
                        f"{inner_indent}    long _cf_dur{iter_id} = _cf_end{iter_id} - _cf_start{iter_id};",
                        f'{inner_indent}    System.out.println("!######" + _cf_mod{iter_id} + ":" + _cf_cls{iter_id} + ":" + _cf_fn{iter_id} + ":" + _cf_loop{iter_id} + ":" + _cf_i{iter_id} + ":" + _cf_dur{iter_id} + "######!");',
                        f"{inner_indent}}}",
                        f"{indent}}}",  # Close for loop
                        f"{method_close_indent}}}",  # Method closing brace
                    ]
                    result.extend(timing_end_code)
                    i += 1
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


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
    # Use pattern that starts at beginning of line to avoid matching words in comments
    class_match = re.search(r"^(?:public\s+)?class\s+(\w+)", test_code, re.MULTILINE)
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
        rf"\b(public\s+)?class\s+{re.escape(original_class_name)}\b",
        rf"\1class {new_class_name}",
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
