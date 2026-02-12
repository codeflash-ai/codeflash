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

# Pattern to match @Test annotation exactly (not @TestOnly, @TestFactory, etc.)
_TEST_ANNOTATION_RE = re.compile(r"^@Test(?:\s*\(.*\))?(?:\s.*)?$")


def _is_test_annotation(stripped_line: str) -> bool:
    """Check if a stripped line is an @Test annotation (not @TestOnly, @TestFactory, etc.).

    Matches:
        @Test
        @Test(expected = ...)
        @Test(timeout = 5000)
    Does NOT match:
        @TestOnly
        @TestFactory
        @TestTemplate
    """
    return bool(_TEST_ANNOTATION_RE.match(stripped_line))


def _find_balanced_end(text: str, start: int) -> int:
    """Find the position after the closing paren that balances the opening paren at start.

    Args:
        text: The source text.
        start: Index of the opening parenthesis '('.

    Returns:
        Index one past the matching closing ')', or -1 if not found.

    """
    if start >= len(text) or text[start] != "(":
        return -1
    depth = 1
    pos = start + 1
    in_string = False
    string_char = None
    in_char = False
    while pos < len(text) and depth > 0:
        ch = text[pos]
        prev = text[pos - 1] if pos > 0 else ""
        if ch == "'" and not in_string and prev != "\\":
            in_char = not in_char
        elif ch == '"' and not in_char and prev != "\\":
            if not in_string:
                in_string = True
                string_char = ch
            elif ch == string_char:
                in_string = False
                string_char = None
        elif not in_string and not in_char:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
        pos += 1
    return pos if depth == 0 else -1


def _find_method_calls_balanced(line: str, func_name: str):
    """Find method calls to func_name with properly balanced parentheses.

    Handles nested parentheses in arguments correctly, unlike a pure regex approach.
    Returns a list of (start, end, full_call) tuples where start/end are positions
    in the line and full_call is the matched text (receiver.funcName(args)).

    Args:
        line: A single line of Java source code.
        func_name: The method name to look for.

    Returns:
        List of (start_pos, end_pos, full_call_text) tuples.

    """
    # First find all occurrences of .funcName( in the line using regex
    # to locate the method name, then use balanced paren finding for args
    prefix_pattern = re.compile(
        rf"((?:new\s+\w+\s*\([^)]*\)|[a-zA-Z_]\w*))\s*\.\s*{re.escape(func_name)}\s*\("
    )
    results = []
    search_start = 0
    while search_start < len(line):
        m = prefix_pattern.search(line, search_start)
        if not m:
            break
        # m.end() - 1 is the position of the opening paren
        open_paren_pos = m.end() - 1
        close_pos = _find_balanced_end(line, open_paren_pos)
        if close_pos == -1:
            # Unbalanced parens - skip this match
            search_start = m.end()
            continue
        full_call = line[m.start():close_pos]
        results.append((m.start(), close_pos, full_call))
        search_start = close_pos
    return results


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
    # This includes the class declaration, return types, constructor calls,
    # variable declarations, etc. We use word-boundary matching to avoid
    # replacing substrings of other identifiers.
    modified_source = re.sub(
        rf"\b{re.escape(original_class_name)}\b", new_class_name, source
    )

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
            if stripped.startswith(("public class", "class")):
                # No imports found, add before class
                result.extend(import_statements)
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
    helper_added = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Look for @Test annotation (not @TestOnly, @TestFactory, etc.)
        if _is_test_annotation(stripped):
            if not helper_added:
                helper_added = True
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
                # Count braces more efficiently using string methods
                open_count = body_line.count("{")
                close_count = body_line.count("}")
                brace_depth += open_count - close_count

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

            # Track lambda block nesting depth to avoid wrapping calls inside lambda bodies.
            # assertThrows/assertDoesNotThrow expect an Executable (void functional interface),
            # and wrapping the call in a variable assignment would turn the void-compatible
            # lambda into a value-returning lambda, causing a compilation error.
            # Handles both expression lambdas: () -> func()
            # and block lambdas: () -> { func(); }
            lambda_brace_depth = 0

            for body_line in body_lines:
                # Detect new block lambda openings: () -> {
                is_lambda_open = bool(re.search(r"\(\s*\)\s*->\s*\{", body_line))

                # Update lambda brace depth tracking for block lambdas
                if is_lambda_open or lambda_brace_depth > 0:
                    open_braces = body_line.count("{")
                    close_braces = body_line.count("}")
                    if is_lambda_open and lambda_brace_depth == 0:
                        # Starting a new lambda block - only count braces from this lambda
                        lambda_brace_depth = open_braces - close_braces
                    else:
                        lambda_brace_depth += open_braces - close_braces
                    # Ensure depth doesn't go below 0
                    lambda_brace_depth = max(0, lambda_brace_depth)

                inside_lambda = lambda_brace_depth > 0 or bool(re.search(r"\(\s*\)\s*->", body_line))

                # Check if this line contains a call to the target function
                if func_name in body_line and "(" in body_line:
                    # Skip wrapping if the function call is inside a lambda expression
                    if inside_lambda:
                        wrapped_body_lines.append(body_line)
                        continue

                    line_indent = len(body_line) - len(body_line.lstrip())
                    line_indent_str = " " * line_indent

                    # Find all matches using balanced parenthesis matching
                    # This correctly handles nested parens like:
                    #   obj.func(a, Rows.toRowID(frame.getIndex(), row))
                    matches = _find_method_calls_balanced(body_line, func_name)
                    if matches:
                        # Process matches in reverse order to maintain correct positions
                        new_line = body_line
                        for start_pos, end_pos, full_call in reversed(matches):
                            call_counter += 1
                            var_name = f"_cf_result{iter_id}_{call_counter}"

                            # Check if we need to cast the result for assertions with primitive arrays
                            # This handles assertArrayEquals(int[], int[]) etc.
                            cast_type = _infer_array_cast_type(body_line)
                            var_with_cast = f"({cast_type}){var_name}" if cast_type else var_name

                            # Replace this occurrence with the variable (with cast if needed)
                            new_line = new_line[:start_pos] + var_with_cast + new_line[end_pos:]

                            # Use 'var' instead of 'Object' to preserve the exact return type.
                            # This avoids boxing mismatches (e.g., assertEquals(int, Object) where
                            # Object is boxed Long but expected is boxed Integer). Requires Java 10+.
                            capture_line = f"{line_indent_str}var {var_name} = {full_call};"
                            wrapped_body_lines.append(capture_line)

                            # Immediately serialize the captured result while the variable
                            # is still in scope. This is necessary because the variable may
                            # be declared inside a nested block (while/for/if/try) and would
                            # be out of scope at the end of the method body.
                            serialize_line = (
                                f"{line_indent_str}_cf_serializedResult{iter_id} = "
                                f"com.codeflash.Serializer.serialize((Object) {var_name});"
                            )
                            wrapped_body_lines.append(serialize_line)

                        # Check if the line is now just a variable reference (invalid statement)
                        # This happens when the original line was just a void method call
                        # e.g., "BubbleSort.bubbleSort(original);" becomes "_cf_result1_1;"
                        stripped_new = new_line.strip().rstrip(";").strip()
                        if stripped_new and stripped_new not in (var_name, var_with_cast):
                            wrapped_body_lines.append(new_line)
                    else:
                        wrapped_body_lines.append(body_line)
                else:
                    wrapped_body_lines.append(body_line)

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
                f"{indent}byte[] _cf_serializedResult{iter_id} = null;",
                f"{indent}try {{",
            ]
            result.extend(behavior_start_code)

            # Add the wrapped body lines with extra indentation.
            # Serialization of captured results is already done inline (immediately
            # after each capture) so the _cf_serializedResult variable is always
            # assigned while the captured variable is still in scope.
            for bl in wrapped_body_lines:
                result.append("    " + bl)

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
                f"{indent}                    _cf_pstmt{iter_id}.setBytes(8, _cf_serializedResult{iter_id});",  # Kryo-serialized return value
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

        # Look for @Test annotation (not @TestOnly, @TestFactory, etc.)
        if _is_test_annotation(stripped):
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
            result.extend(method_lines)
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

    # Rename all references to the original class name in the source.
    # This includes the class declaration, return types, constructor calls, etc.
    modified_code = re.sub(
        rf"\b{re.escape(original_class_name)}\b", new_class_name, test_code
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
        if stripped.startswith(("import ", "package ")):
            insert_idx = i + 1
        elif stripped and not stripped.startswith("//") and not stripped.startswith("/*"):
            # First non-import, non-comment line
            if insert_idx == 0:
                insert_idx = i
            break

    lines.insert(insert_idx, import_statement + "\n")
    return "".join(lines)


