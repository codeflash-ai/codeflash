"""Java code instrumentation for behavior capture and benchmarking.

This module provides functionality to instrument Java code for:
1. Behavior capture - recording inputs/outputs for verification
2. Benchmarking - measuring execution time

Timing instrumentation adds System.nanoTime() calls around the function being tested
and prints timing markers in a format compatible with Python/JS implementations:
  Start: !$######testModule:testClass.testMethod:funcName:loopId:invocationId######$!
  End:   !######testModule:testClass.testMethod:funcName:loopId:invocationId:durationNs######!

Where:
  - loopId = outerLoopIndex * maxInnerIterations + innerIteration (CUDA-style composite)
  - invocationId = call position in test method (1, 2, 3, ... for multiple calls)

This allows codeflash to extract timing data from stdout for accurate benchmarking.
"""

from __future__ import annotations

import bisect
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.languages.java.parser import JavaAnalyzer

_WORD_RE = re.compile(r"^\w+$")

_ASSERTION_METHODS = ("assertArrayEquals", "assertArrayNotEquals")

logger = logging.getLogger(__name__)


def _get_function_name(func: Any) -> str:
    """Get the function name from FunctionToOptimize."""
    if hasattr(func, "function_name"):
        return str(func.function_name)
    if hasattr(func, "name"):
        return str(func.name)
    msg = f"Cannot get function name from {type(func)}"
    raise AttributeError(msg)


_METHOD_SIG_PATTERN = re.compile(
    r"\b(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*"
    r"(?:void|String|int|long|boolean|double|float|char|byte|short|\w+(?:\[\])?)\s+(\w+)\s*\("
)
_FALLBACK_METHOD_PATTERN = re.compile(r"\b(\w+)\s*\(")


def _extract_test_method_name(method_lines: list[str]) -> str:
    method_sig = " ".join(method_lines).strip()

    # Fast-path heuristic: if a common modifier or built-in return type appears,
    # try to extract the identifier immediately before the following '(' using
    # simple string operations which are much cheaper than regex on large inputs.
    # Fall back to the original regex-based logic if the heuristic doesn't
    # confidently produce a result.
    s = method_sig
    if s:
        # Look for common modifiers first; modifiers are strong signals of a method declaration
        for mod in ("public ", "private ", "protected "):
            idx = s.find(mod)
            if idx != -1:
                sub = s[idx:]
                paren = sub.find("(")
                if paren != -1:
                    left = sub[:paren].strip()
                    parts = left.split()
                    if parts:
                        candidate = parts[-1]
                        if _WORD_RE.match(candidate):
                            return candidate
                break  # if modifier was found but fast-path failed, avoid trying other modifiers

        # If no modifier found or modifier path didn't return, check common primitive/reference return types.
        # This helps with package-private methods declared like "void foo(", "int bar(", "String baz(", etc.
        for typ in ("void ", "String ", "int ", "long ", "boolean ", "double ", "float ", "char ", "byte ", "short "):
            idx = s.find(typ)
            if idx != -1:
                sub = s[idx + len(typ) :]  # start after the type token
                paren = sub.find("(")
                if paren != -1:
                    left = sub[:paren].strip()
                    parts = left.split()
                    if parts:
                        candidate = parts[-1]
                        if _WORD_RE.match(candidate):
                            return candidate
                break  # stop after first matching type token

    # Original behavior: fall back to the precompiled regex patterns.
    match = _METHOD_SIG_PATTERN.search(method_sig)
    if match:
        return match.group(1)
    fallback_match = _FALLBACK_METHOD_PATTERN.search(method_sig)
    if fallback_match:
        return fallback_match.group(1)
    return "unknown"


# Pattern to detect primitive array types in assertions
_PRIMITIVE_ARRAY_PATTERN = re.compile(r"new\s+(int|long|double|float|short|byte|char|boolean)\s*\[\s*\]")
# Pattern to extract type from variable declaration: Type varName = ...
_VAR_DECL_TYPE_PATTERN = re.compile(r"^\s*([\w<>[\],\s]+?)\s+\w+\s*=")

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
    if not stripped_line.startswith("@Test"):
        return False
    if len(stripped_line) == 5:
        return True
    next_char = stripped_line[5]
    return next_char in {" ", "("}


def _is_inside_lambda(node: Any) -> bool:
    """Check if a tree-sitter node is inside a lambda_expression."""
    current = node.parent
    while current is not None:
        t = current.type
        if t == "lambda_expression":
            return True
        if t == "method_declaration":
            return False
        current = current.parent
    return False


def _is_inside_complex_expression(node: Any) -> bool:
    """Check if a tree-sitter node is inside a complex expression that shouldn't be instrumented directly.

    This includes:
    - Cast expressions: (Long)list.get(2)
    - Ternary expressions: condition ? func() : other
    - Array access: arr[func()]
    - Binary operations: func() + 1

    Returns True if the node should not be directly instrumented.
    """
    current = node.parent
    while current is not None:
        # Stop at statement boundaries
        if current.type in {
            "method_declaration",
            "block",
            "if_statement",
            "for_statement",
            "while_statement",
            "try_statement",
            "expression_statement",
        }:
            return False

        # These are complex expressions that shouldn't have instrumentation inserted in the middle
        if current.type in {
            "cast_expression",
            "ternary_expression",
            "array_access",
            "binary_expression",
            "unary_expression",
            "parenthesized_expression",
            "instanceof_expression",
        }:
            logger.debug("Found complex expression parent: %s", current.type)
            return True

        current = current.parent
    return False


_TS_BODY_PREFIX = "class _D { void _m() {\n"
_TS_BODY_SUFFIX = "\n}}"
_TS_BODY_PREFIX_BYTES = _TS_BODY_PREFIX.encode("utf8")


def _generate_sqlite_write_code(
    iter_id: int,
    call_counter: int,
    indent: str,
    class_name: str,
    func_name: str,
    test_method_name: str,
    invocation_id: str = "",
) -> list[str]:
    """Generate SQLite write code for a single function call.

    Args:
        iter_id: Test method iteration ID
        call_counter: Call counter for unique variable naming
        indent: Base indentation string
        class_name: Test class name
        func_name: Function being tested
        test_method_name: Test method name
        invocation_id: The invocation ID string (e.g. "L15_1") to write into the DB and markers.
                        Falls back to str(call_counter) if empty.

    Returns:
        List of code lines for SQLite write in finally block.

    """
    inv_id_str = invocation_id or str(call_counter)
    inner_indent = indent + "    "
    id_pair = f"{iter_id}_{call_counter}"
    return [
        f"{indent}}} finally {{",
        f"{inner_indent}long _cf_end{id_pair}_finally = System.nanoTime();",
        f"{inner_indent}long _cf_dur{id_pair} = (_cf_end{id_pair} != -1 ? _cf_end{id_pair} : _cf_end{id_pair}_finally) - _cf_start{id_pair};",
        f'{inner_indent}System.out.println("!######" + _cf_mod{iter_id} + ":" + _cf_cls{iter_id} + "." + _cf_test{iter_id} + ":" + _cf_fn{iter_id} + ":" + _cf_loop{iter_id} + ":" + "{inv_id_str}" + "######!");',
        f"{inner_indent}// Write to SQLite if output file is set",
        f"{inner_indent}if (_cf_outputFile{iter_id} != null && !_cf_outputFile{iter_id}.isEmpty()) {{",
        f"{inner_indent}    try {{",
        f'{inner_indent}        Class.forName("org.sqlite.JDBC");',
        f'{inner_indent}        try (Connection _cf_conn{id_pair} = DriverManager.getConnection("jdbc:sqlite:" + _cf_outputFile{iter_id})) {{',
        f"{inner_indent}            try (java.sql.Statement _cf_stmt{id_pair} = _cf_conn{id_pair}.createStatement()) {{",
        f'{inner_indent}                _cf_stmt{id_pair}.execute("CREATE TABLE IF NOT EXISTS test_results (" +',
        f'{inner_indent}                    "test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, " +',
        f'{inner_indent}                    "function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, " +',
        f'{inner_indent}                    "runtime INTEGER, return_value BLOB, verification_type TEXT)");',
        f"{inner_indent}            }}",
        f'{inner_indent}            String _cf_sql{id_pair} = "INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";',
        f"{inner_indent}            try (PreparedStatement _cf_pstmt{id_pair} = _cf_conn{id_pair}.prepareStatement(_cf_sql{id_pair})) {{",
        f"{inner_indent}                _cf_pstmt{id_pair}.setString(1, _cf_mod{iter_id});",
        f"{inner_indent}                _cf_pstmt{id_pair}.setString(2, _cf_cls{iter_id});",
        f"{inner_indent}                _cf_pstmt{id_pair}.setString(3, _cf_test{iter_id});",
        f"{inner_indent}                _cf_pstmt{id_pair}.setString(4, _cf_fn{iter_id});",
        f"{inner_indent}                _cf_pstmt{id_pair}.setInt(5, _cf_loop{iter_id});",
        f'{inner_indent}                _cf_pstmt{id_pair}.setString(6, "{inv_id_str}");',
        f"{inner_indent}                _cf_pstmt{id_pair}.setLong(7, _cf_dur{id_pair});",
        f"{inner_indent}                _cf_pstmt{id_pair}.setBytes(8, _cf_serializedResult{id_pair});",
        f'{inner_indent}                _cf_pstmt{id_pair}.setString(9, "function_call");',
        f"{inner_indent}                _cf_pstmt{id_pair}.executeUpdate();",
        f"{inner_indent}            }}",
        f"{inner_indent}        }}",
        f"{inner_indent}    }} catch (Exception _cf_e{id_pair}) {{",
        f'{inner_indent}        System.err.println("CodeflashHelper: SQLite error: " + _cf_e{id_pair}.getMessage());',
        f"{inner_indent}    }}",
        f"{inner_indent}}}",
        f"{indent}}}",
    ]


def wrap_target_calls_with_treesitter(
    body_lines: list[str],
    func_name: str,
    iter_id: int,
    precise_call_timing: bool = False,
    class_name: str = "",
    test_method_name: str = "",
    body_start_line: int = 0,
    target_return_type: str = "",
) -> tuple[list[str], int]:
    """Replace target method calls in body_lines with capture + serialize using tree-sitter.

    Operates on the full body text with character offsets (not line-by-line) to correctly
    handle calls that span multiple lines. Processes calls back-to-front so earlier offsets
    remain valid after later replacements.

    For behavior mode (precise_call_timing=True), each call is wrapped in its own
    try-finally block with immediate SQLite write to prevent data loss from multiple calls.

    Returns (wrapped_body_lines, call_counter).
    """
    from codeflash.languages.java.parser import get_java_analyzer

    body_text = "\n".join(body_lines)
    if func_name not in body_text:
        return list(body_lines), 0

    analyzer = get_java_analyzer()
    body_bytes = body_text.encode("utf8")
    prefix_len = len(_TS_BODY_PREFIX_BYTES)

    wrapper_bytes = _TS_BODY_PREFIX_BYTES + body_bytes + _TS_BODY_SUFFIX.encode("utf8")
    tree = analyzer.parse(wrapper_bytes)

    # Collect all matching calls with their metadata
    calls: list[dict[str, Any]] = []
    _collect_calls(tree.root_node, wrapper_bytes, body_bytes, prefix_len, func_name, analyzer, calls)

    if not calls:
        return list(body_lines), 0

    # Build line byte-start offsets for mapping byte offsets to line indices
    line_byte_starts: list[int] = []
    offset = 0
    for line in body_lines:
        line_byte_starts.append(offset)
        offset += len(line.encode("utf8")) + 1  # +1 for \n from join

    # Filter out lambda and complex-expression calls, sort by start_byte ascending for counter assignment
    valid_calls = [c for c in calls if not c["in_lambda"] and not c.get("in_complex", False)]
    if not valid_calls:
        return list(body_lines), 0
    valid_calls.sort(key=lambda c: c["start_byte"])

    # Pre-compute character offsets and line info for each call (before any text modifications)
    for i, call in enumerate(valid_calls, 1):
        call["_counter"] = i
        call["_call_start_char"] = len(body_bytes[: call["start_byte"]].decode("utf8"))
        call["_call_end_char"] = len(body_bytes[: call["end_byte"]].decode("utf8"))
        if call["parent_type"] == "expression_statement":
            call["_es_start_char"] = len(body_bytes[: call["es_start_byte"]].decode("utf8"))
            call["_es_end_char"] = len(body_bytes[: call["es_end_byte"]].decode("utf8"))
        line_idx = _byte_to_line_index(call["start_byte"], line_byte_starts)
        call["_line_idx"] = line_idx
        call["_line_char_start"] = len(body_bytes[: line_byte_starts[line_idx]].decode("utf8"))

    # Process calls back-to-front so earlier character offsets stay valid
    for call in reversed(valid_calls):
        call_counter = call["_counter"]
        line_idx = call["_line_idx"]
        call_absolute_line = body_start_line + line_idx + 1
        inv_id = f"L{call_absolute_line}_{call_counter}"

        orig_line = body_lines[line_idx]
        line_indent_str = " " * (len(orig_line) - len(orig_line.lstrip()))

        var_name = f"_cf_result{iter_id}_{call_counter}"
        cast_type = _infer_array_cast_type(orig_line)
        if not cast_type and target_return_type and target_return_type != "void":
            cast_type = target_return_type
        var_with_cast = f"({cast_type}){var_name}" if cast_type else var_name

        capture_stmt_with_decl = f"var {var_name} = {call['full_call']};"
        capture_stmt_assign = f"{var_name} = {call['full_call']};"
        if precise_call_timing:
            serialize_stmt = f"_cf_serializedResult{iter_id}_{call_counter} = com.codeflash.Serializer.serialize((Object) {var_name});"
            start_stmt = f"_cf_start{iter_id}_{call_counter} = System.nanoTime();"
            end_stmt = f"_cf_end{iter_id}_{call_counter} = System.nanoTime();"
        else:
            serialize_stmt = f"_cf_serializedResult{iter_id} = com.codeflash.Serializer.serialize((Object) {var_name});"
            start_stmt = f"_cf_start{iter_id} = System.nanoTime();"
            end_stmt = f"_cf_end{iter_id} = System.nanoTime();"

        if call["parent_type"] == "expression_statement":
            es_start = call["_es_start_char"]
            es_end = call["_es_end_char"]
            if precise_call_timing:
                # No indent on first line — body_text[:es_start] already has leading whitespace.
                # Subsequent lines get line_indent_str.
                var_decls = [
                    f"Object {var_name} = null;",
                    f"long _cf_end{iter_id}_{call_counter} = -1;",
                    f"long _cf_start{iter_id}_{call_counter} = 0;",
                    f"byte[] _cf_serializedResult{iter_id}_{call_counter} = null;",
                ]
                start_marker = f'System.out.println("!$######" + _cf_mod{iter_id} + ":" + _cf_cls{iter_id} + "." + _cf_test{iter_id} + ":" + _cf_fn{iter_id} + ":" + _cf_loop{iter_id} + ":{inv_id}" + "######$!");'
                try_block = [
                    "try {",
                    f"    {start_stmt}",
                    f"    {capture_stmt_assign}",
                    f"    {end_stmt}",
                    f"    {serialize_stmt}",
                ]
                finally_block = _generate_sqlite_write_code(
                    iter_id, call_counter, "", class_name, func_name, test_method_name, invocation_id=inv_id
                )
                all_lines = [*var_decls, start_marker, *try_block, *finally_block]
                replacement = (
                    all_lines[0] + "\n" + "\n".join(f"{line_indent_str}{repl_line}" for repl_line in all_lines[1:])
                )
            else:
                replacement = f"{capture_stmt_with_decl} {serialize_stmt}"
            body_text = body_text[:es_start] + replacement + body_text[es_end:]
        else:
            # Embedded call: replace call with variable, then insert capture lines before the line
            call_start = call["_call_start_char"]
            call_end = call["_call_end_char"]
            line_char_start = call["_line_char_start"]

            if precise_call_timing:
                prefix_lines = [
                    f"{line_indent_str}Object {var_name} = null;",
                    f"{line_indent_str}long _cf_end{iter_id}_{call_counter} = -1;",
                    f"{line_indent_str}long _cf_start{iter_id}_{call_counter} = 0;",
                    f"{line_indent_str}byte[] _cf_serializedResult{iter_id}_{call_counter} = null;",
                    f'{line_indent_str}System.out.println("!$######" + _cf_mod{iter_id} + ":" + _cf_cls{iter_id} + "." + _cf_test{iter_id} + ":" + _cf_fn{iter_id} + ":" + _cf_loop{iter_id} + ":{inv_id}" + "######$!");',
                    f"{line_indent_str}try {{",
                    f"{line_indent_str}    {start_stmt}",
                    f"{line_indent_str}    {capture_stmt_assign}",
                    f"{line_indent_str}    {end_stmt}",
                    f"{line_indent_str}    {serialize_stmt}",
                ]
                finally_lines = _generate_sqlite_write_code(
                    iter_id,
                    call_counter,
                    line_indent_str,
                    class_name,
                    func_name,
                    test_method_name,
                    invocation_id=inv_id,
                )
                prefix_lines.extend(finally_lines)
            else:
                prefix_lines = [f"{line_indent_str}{capture_stmt_with_decl}", f"{line_indent_str}{serialize_stmt}"]

            # Step 1: Replace the call with the variable (at higher offset, safe to do first)
            body_text = body_text[:call_start] + var_with_cast + body_text[call_end:]
            # Step 2: Insert prefix lines before the line containing the call (at lower offset)
            prefix_text = "\n".join(prefix_lines) + "\n"
            body_text = body_text[:line_char_start] + prefix_text + body_text[line_char_start:]

    # Split back into lines, filtering out any lines that became empty from statement replacement
    wrapped = [line for line in body_text.split("\n") if line.strip()]
    return wrapped, len(valid_calls)


def _collect_calls(
    node: Any,
    wrapper_bytes: bytes,
    body_bytes: bytes,
    prefix_len: int,
    func_name: str,
    analyzer: JavaAnalyzer,
    out: list[dict[str, Any]],
) -> None:
    """Recursively collect method_invocation nodes matching func_name."""
    node_type = node.type
    if node_type == "method_invocation":
        name_node = node.child_by_field_name("name")
        if name_node and analyzer.get_node_text(name_node, wrapper_bytes) == func_name:
            start = node.start_byte - prefix_len
            end = node.end_byte - prefix_len
            body_len = len(body_bytes)
            if start >= 0 and end <= body_len:
                parent = node.parent
                parent_type = parent.type if parent else ""
                es_start = es_end = 0
                if parent_type == "expression_statement":
                    es_start = parent.start_byte - prefix_len
                    es_end = parent.end_byte - prefix_len
                out.append(
                    {
                        "start_byte": start,
                        "end_byte": end,
                        "full_call": analyzer.get_node_text(node, wrapper_bytes),
                        "parent_type": parent_type,
                        "in_lambda": _is_inside_lambda(node),
                        "in_complex": _is_inside_complex_expression(node),
                        "es_start_byte": es_start,
                        "es_end_byte": es_end,
                    }
                )
    for child in node.children:
        _collect_calls(child, wrapper_bytes, body_bytes, prefix_len, func_name, analyzer, out)


def _byte_to_line_index(byte_offset: int, line_byte_starts: list[int]) -> int:
    """Map a byte offset in body_text to a body_lines index."""
    idx = bisect.bisect_right(line_byte_starts, byte_offset) - 1
    return max(idx, 0)


def _infer_array_cast_type(line: str) -> str | None:
    """Infer the cast type needed when replacing function calls with result variables.

    When a line contains a variable declaration or assertion, we need to cast the
    captured Object result back to the original type.

    Examples:
        byte[] digest = Crypto.computeDigest(...) -> cast to (byte[])
        assertArrayEquals(new int[] {...}, func()) -> cast to (int[])

    Args:
        line: The source line containing the function call.

    Returns:
        The cast type (e.g., "byte[]", "int[]") if needed, None otherwise.

    """
    # Check for assertion methods that take arrays
    if "assertArrayEquals" in line or "assertArrayNotEquals" in line:
        match = _PRIMITIVE_ARRAY_PATTERN.search(line)
        if match:
            primitive_type = match.group(1)
            return f"{primitive_type}[]"

    # Check for variable declaration: Type varName = func()
    match = _VAR_DECL_TYPE_PATTERN.search(line)
    if match:
        type_str = match.group(1).strip()
        # Only add cast if it's not 'var' (which uses type inference) and not 'Object' (no cast needed)
        if type_str not in ("var", "Object"):
            return type_str

    return None


def _extract_return_type(function_to_optimize: Any) -> str:
    """Extract the return type of a Java function from its source file using tree-sitter."""
    file_path = getattr(function_to_optimize, "file_path", None)
    func_name = _get_function_name(function_to_optimize)
    if not file_path or not file_path.exists():
        return ""
    try:
        from codeflash.languages.java.parser import get_java_analyzer

        analyzer = get_java_analyzer()
        source_text = file_path.read_text(encoding="utf-8")
        methods = analyzer.find_methods(source_text)
        for method in methods:
            if method.name == func_name and method.return_type:
                return method.return_type
    except Exception:
        logger.debug("Could not extract return type for %s", func_name)
    return ""


def _get_qualified_name(func: Any) -> str:
    """Get the qualified name from FunctionToOptimize."""
    if hasattr(func, "qualified_name"):
        return str(func.qualified_name)
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
    test_string: str,
    function_to_optimize: Any,  # FunctionToOptimize or FunctionToOptimize
    mode: str,  # "behavior" or "performance"
    test_path: Path | None = None,
    test_class_name: str | None = None,
) -> tuple[bool, str | None]:
    """Inject profiling code into an existing test file.

    For Java, this:
    1. Renames the class to match the new file name (Java requires class name = file name)
    2. For behavior mode: adds timing instrumentation that writes to SQLite
    3. For performance mode: adds timing instrumentation with stdout markers

    Args:
        test_string: String to the test file.
        call_positions: List of code positions where the function is called.
        function_to_optimize: The function being optimized.
        tests_project_root: Root directory of tests.
        mode: Testing mode - "behavior" or "performance".
        analyzer: Optional JavaAnalyzer instance.
        output_class_suffix: Optional suffix for the renamed class.

    Returns:
        Tuple of (success, modified_source).

    """
    source = test_string
    func_name = _get_function_name(function_to_optimize)
    target_return_type = _extract_return_type(function_to_optimize)

    # Get the original class name from the file name
    if test_path:
        original_class_name = test_path.stem  # e.g., "AlgorithmsTest"
    elif test_class_name is not None:
        original_class_name = test_class_name
    else:
        raise ValueError("test_path or test_class_name must be provided")

    # Determine the new class name based on mode
    if mode == "behavior":
        new_class_name = f"{original_class_name}__perfinstrumented"
    else:
        new_class_name = f"{original_class_name}__perfonlyinstrumented"

    # Rename all references to the original class name in the source.
    # This includes the class declaration, return types, constructor calls,
    # variable declarations, etc. We use word-boundary matching to avoid
    # replacing substrings of other identifiers.
    modified_source = re.sub(rf"\b{re.escape(original_class_name)}\b", new_class_name, source)

    # Add @SuppressWarnings("CheckReturnValue") to the class declaration.
    # Projects using Error Prone (e.g. Guava) enforce CheckReturnValue as a compiler error.
    # Applied in both modes: performance mode strips assertions (creating discarded return values),
    # and behavior mode adds wrapper calls that may also discard return values.
    modified_source = _add_suppress_warnings_annotation(modified_source, new_class_name)

    # Add timing instrumentation to test methods
    # Use the new (instrumented) class name in markers so each test file has a unique
    # _cf_mod/_cf_cls. This is critical for disambiguating existing vs generated tests
    # that share the same original class name (e.g., both have "FibonacciTest").
    # For existing tests, the __perfinstrumented suffix is later replaced with
    # __existing_perfinstrumented, making markers unique per test file.
    if mode == "performance":
        modified_source = _add_timing_instrumentation(modified_source, new_class_name, func_name)
    else:
        # Behavior mode: add timing instrumentation that also writes to SQLite
        modified_source = _add_behavior_instrumentation(modified_source, new_class_name, func_name, target_return_type)

    logger.debug("Java %s testing for %s: renamed class %s -> %s", mode, func_name, original_class_name, new_class_name)
    # Why return True here?
    return True, modified_source


def _add_behavior_instrumentation(source: str, class_name: str, func_name: str, target_return_type: str = "") -> str:
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
    lines = result.copy()
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

            # Extract the test method name from the method signature
            test_method_name = _extract_test_method_name(method_lines)

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
            body_first_line_index = i  # 0-based line index of first body line in source

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

            # Wrap function calls to capture return values using tree-sitter AST analysis.
            # This correctly handles lambdas, try-catch blocks, assignments, and nested calls.
            # Each call gets its own try-finally block with immediate SQLite write.
            wrapped_body_lines, _call_counter = wrap_target_calls_with_treesitter(
                body_lines=body_lines,
                func_name=func_name,
                iter_id=iter_id,
                precise_call_timing=True,
                class_name=class_name,
                test_method_name=test_method_name,
                body_start_line=body_first_line_index,
                target_return_type=target_return_type,
            )

            # Add behavior instrumentation setup code (shared variables for all calls in the method)
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
                f'{indent}String _cf_test{iter_id} = "{test_method_name}";',
            ]
            result.extend(behavior_start_code)

            # Add the wrapped body lines without extra indentation.
            # Each call already has its own try-finally block with SQLite write from wrap_target_calls_with_treesitter().
            for bl in wrapped_body_lines:
                result.append(bl)

            # Add method closing brace
            method_close_indent = " " * base_indent
            result.append(f"{method_close_indent}}}")
        else:
            result.append(line)
            i += 1

    return "\n".join(result)


def _add_suppress_warnings_annotation(source: str, class_name: str) -> str:
    """Add @SuppressWarnings("CheckReturnValue") before the class declaration.

    Projects using Error Prone (e.g. Guava) enforce CheckReturnValue as a compiler error.
    Our instrumented tests intentionally discard return values after assertion stripping,
    which would fail compilation without this suppression.
    """
    class_decl_pattern = re.compile(
        rf"^((?:(?:public|protected|final|abstract)\s+)*class\s+{re.escape(class_name)}\b)", re.MULTILINE
    )
    match = class_decl_pattern.search(source)
    if not match:
        return source
    insert_pos = match.start()
    return source[:insert_pos] + '@SuppressWarnings("CheckReturnValue")\n' + source[insert_pos:]


def _add_timing_instrumentation(source: str, class_name: str, func_name: str) -> str:
    """Add timing instrumentation to test methods with inner loop for JIT warmup.

    For each @Test method, this adds:
    1. Inner loop that runs N iterations (controlled by CODEFLASH_INNER_ITERATIONS env var)
    2. Start timing marker printed at the beginning of each iteration
    3. End timing marker printed at the end of each iteration (in a finally block)

    The inner loop allows JIT warmup within a single JVM invocation, avoiding
    expensive Maven restarts. Post-processing uses min runtime across all iterations.

    Timing markers format:
      Start: !$######testModule:testClass:funcName:loopId:invocationId######$!
      End:   !######testModule:testClass:funcName:loopId:invocationId:durationNs######!

    Where:
      - loopId = outerLoopIndex * maxInnerIterations + innerIteration (0, 1, 2, ..., N-1)
      - invocationId = call position in test method (1, 2, 3, ... for multiple calls)

    Args:
        source: The test source code.
        class_name: Name of the test class.
        func_name: Name of the function being tested.

    Returns:
        Instrumented source code.

    """
    from codeflash.languages.java.parser import get_java_analyzer

    source_bytes = source.encode("utf8")
    analyzer = get_java_analyzer()
    tree = analyzer.parse(source_bytes)

    def has_test_annotation(method_node: Any) -> bool:
        modifiers = None
        for child in method_node.children:
            if child.type == "modifiers":
                modifiers = child
                break
        if not modifiers:
            return False
        for child in modifiers.children:
            if child.type not in {"annotation", "marker_annotation"}:
                continue
            # annotation text includes '@'
            annotation_text = analyzer.get_node_text(child, source_bytes).strip()
            if annotation_text.startswith("@"):
                name = annotation_text[1:].split("(", 1)[0].strip()
                if name == "Test" or name.endswith(".Test"):
                    return True
        return False

    def collect_test_methods(node: Any, out: list[tuple[Any, Any]]) -> None:
        stack = [node]
        while stack:
            current = stack.pop()
            if current.type == "method_declaration" and has_test_annotation(current):
                body_node = current.child_by_field_name("body")
                if body_node is not None:
                    out.append((current, body_node))
                continue
            if current.children:
                stack.extend(reversed(current.children))

    def collect_target_calls(node: Any, wrapper_bytes: bytes, func: str, out: list[Any]) -> None:
        stack = [node]
        while stack:
            current = stack.pop()
            if current.type == "method_invocation":
                name_node = current.child_by_field_name("name")
                if name_node and analyzer.get_node_text(name_node, wrapper_bytes) == func:
                    if not _is_inside_lambda(current) and not _is_inside_complex_expression(current):
                        out.append(current)
                    else:
                        logger.debug("Skipping instrumentation of %s inside lambda or complex expression", func)
            if current.children:
                stack.extend(reversed(current.children))

    def reindent_block(text: str, target_indent: str) -> str:
        lines = text.splitlines()
        non_empty = [line for line in lines if line.strip()]
        if not non_empty:
            return text
        min_leading = min(len(line) - len(line.lstrip(" ")) for line in non_empty)
        reindented: list[str] = []
        for line in lines:
            if not line.strip():
                reindented.append(line)
                continue
            # Normalize relative indentation and place block under target indent.
            reindented.append(f"{target_indent}{line[min_leading:]}")
        return "\n".join(reindented)

    def find_top_level_statement(node: Any, body_node: Any) -> Any:
        current = node
        while current is not None and current.parent is not None and current.parent != body_node:
            current = current.parent
        return current if current is not None and current.parent == body_node else None

    def split_var_declaration(stmt_node: Any, source_bytes_ref: bytes) -> tuple[str, str] | None:
        """Split a local_variable_declaration into a hoisted declaration and an assignment.

        When a target call is inside a variable declaration like:
            int len = Buffer.stringToUtf8(input, buf, 0);
        wrapping it in a for/try block would put `len` out of scope for subsequent code.

        This function splits it into:
            hoisted:    int len;
            assignment: len = Buffer.stringToUtf8(input, buf, 0);

        Returns (hoisted_decl, assignment_stmt) or None if not a local_variable_declaration.
        """
        if stmt_node.type != "local_variable_declaration":
            return None

        # Extract the type and declarator from the AST
        type_node = stmt_node.child_by_field_name("type")
        declarator_node = None
        for child in stmt_node.children:
            if child.type == "variable_declarator":
                declarator_node = child
                break
        if type_node is None or declarator_node is None:
            return None

        # Get the variable name and initializer
        name_node = declarator_node.child_by_field_name("name")
        value_node = declarator_node.child_by_field_name("value")
        if name_node is None or value_node is None:
            return None

        type_text = analyzer.get_node_text(type_node, source_bytes_ref)
        name_text = analyzer.get_node_text(name_node, source_bytes_ref)
        value_text = analyzer.get_node_text(value_node, source_bytes_ref)

        # Initialize with a default value to satisfy Java's definite assignment rules.
        # The variable is assigned inside a for/try block which Java considers
        # conditionally executed, so an uninitialized declaration would cause
        # "variable might not have been initialized" errors.
        primitive_defaults = {
            "byte": "0",
            "short": "0",
            "int": "0",
            "long": "0L",
            "float": "0.0f",
            "double": "0.0",
            "char": "'\\0'",
            "boolean": "false",
        }
        default_val = primitive_defaults.get(type_text, "null")
        hoisted = f"{type_text} {name_text} = {default_val};"
        assignment = f"{name_text} = {value_text};"
        return hoisted, assignment

    def build_instrumented_body(
        body_text: str,
        next_wrapper_id: int,
        base_indent: str,
        test_method_name: str = "unknown",
        body_start_line: int = 0,
    ) -> tuple[str, int]:
        body_bytes = body_text.encode("utf8")
        wrapper_bytes = _TS_BODY_PREFIX_BYTES + body_bytes + _TS_BODY_SUFFIX.encode("utf8")
        wrapper_tree = analyzer.parse(wrapper_bytes)
        wrapped_method = None
        stack = [wrapper_tree.root_node]
        while stack:
            node = stack.pop()
            if node.type == "method_declaration":
                wrapped_method = node
                break
            stack.extend(reversed(node.children))
        if wrapped_method is None:
            return body_text, next_wrapper_id
        wrapped_body = wrapped_method.child_by_field_name("body")
        if wrapped_body is None:
            return body_text, next_wrapper_id
        calls: list[Any] = []
        collect_target_calls(wrapped_body, wrapper_bytes, func_name, calls)

        indent = base_indent
        inner_indent = f"{indent}    "
        inner_body_indent = f"{inner_indent}    "

        if not calls:
            return body_text, next_wrapper_id

        # _TS_BODY_PREFIX is 1 line, so wrapper line 1 = body line 0
        wrapper_prefix_lines = 1

        # Map each call to its absolute line number and the containing statement range
        statement_ranges: list[tuple[int, int, Any, int]] = []  # (char_start, char_end, ast_node, call_absolute_line)
        for call in sorted(calls, key=lambda n: n.start_byte):
            stmt_node = find_top_level_statement(call, wrapped_body)
            if stmt_node is None:
                continue
            stmt_byte_start = stmt_node.start_byte - len(_TS_BODY_PREFIX_BYTES)
            stmt_byte_end = stmt_node.end_byte - len(_TS_BODY_PREFIX_BYTES)
            if not (0 <= stmt_byte_start <= stmt_byte_end <= len(body_bytes)):
                continue
            # Convert byte offsets to character offsets for correct Python str slicing.
            # Tree-sitter returns byte offsets but body_text is a Python str (Unicode),
            # so multi-byte UTF-8 characters (e.g., é, 世) cause misalignment if we
            # slice the str directly with byte offsets.
            stmt_start = len(body_bytes[:stmt_byte_start].decode("utf8"))
            stmt_end = len(body_bytes[:stmt_byte_end].decode("utf8"))
            # Compute absolute line: call's line in wrapper minus prefix lines, plus body_start_line, 1-indexed
            call_line_in_wrapper = call.start_point[0]
            call_absolute_line = body_start_line + (call_line_in_wrapper - wrapper_prefix_lines) + 1
            statement_ranges.append((stmt_start, stmt_end, stmt_node, call_absolute_line))

        # Deduplicate repeated calls within the same top-level statement.
        unique_ranges: list[tuple[int, int, Any, int]] = []
        seen_offsets: set[tuple[int, int]] = set()
        for stmt_start, stmt_end, stmt_node, call_abs_line in statement_ranges:
            key = (stmt_start, stmt_end)
            if key in seen_offsets:
                continue
            seen_offsets.add(key)
            unique_ranges.append((stmt_start, stmt_end, stmt_node, call_abs_line))
        if not unique_ranges:
            return body_text, next_wrapper_id

        if len(unique_ranges) == 1:
            stmt_start, stmt_end, stmt_ast_node, call_abs_line = unique_ranges[0]
            prefix = body_text[:stmt_start]
            target_stmt = body_text[stmt_start:stmt_end]
            suffix = body_text[stmt_end:]

            current_id = next_wrapper_id + 1
            inv_id = f"L{call_abs_line}_{current_id}"
            setup_lines = [
                f"{indent}// Codeflash timing instrumentation with inner loop for JIT warmup",
                f'{indent}int _cf_outerLoop{current_id} = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));',
                f'{indent}int _cf_maxInnerIterations{current_id} = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "10"));',
                f'{indent}int _cf_innerIterations{current_id} = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "10"));',
                f'{indent}String _cf_mod{current_id} = "{class_name}";',
                f'{indent}String _cf_cls{current_id} = "{class_name}";',
                f'{indent}String _cf_test{current_id} = "{test_method_name}";',
                f'{indent}String _cf_fn{current_id} = "{func_name}";',
                "",
            ]

            # If the target statement is a variable declaration (e.g., int len = func()),
            # hoist the declaration before the timing block so the variable stays in scope
            # for subsequent code that references it.
            var_split = split_var_declaration(stmt_ast_node, wrapper_bytes)
            if var_split is not None:
                hoisted_decl, assignment_stmt = var_split
                setup_lines.append(f"{indent}{hoisted_decl}")
                stmt_in_try = reindent_block(assignment_stmt, inner_body_indent)
            else:
                stmt_in_try = reindent_block(target_stmt, inner_body_indent)
            timing_lines = [
                f"{indent}for (int _cf_i{current_id} = 0; _cf_i{current_id} < _cf_innerIterations{current_id}; _cf_i{current_id}++) {{",
                f"{inner_indent}int _cf_loopId{current_id} = _cf_outerLoop{current_id} * _cf_maxInnerIterations{current_id} + _cf_i{current_id};",
                f'{inner_indent}System.out.println("!$######" + _cf_mod{current_id} + ":" + _cf_cls{current_id} + "." + _cf_test{current_id} + ":" + _cf_fn{current_id} + ":" + _cf_loopId{current_id} + ":" + "{inv_id}" + "######$!");',
                f"{inner_indent}long _cf_end{current_id} = -1;",
                f"{inner_indent}long _cf_start{current_id} = 0;",
                f"{inner_indent}try {{",
                f"{inner_body_indent}_cf_start{current_id} = System.nanoTime();",
                stmt_in_try,
                f"{inner_body_indent}_cf_end{current_id} = System.nanoTime();",
                f"{inner_indent}}} finally {{",
                f"{inner_body_indent}long _cf_end{current_id}_finally = System.nanoTime();",
                f"{inner_body_indent}long _cf_dur{current_id} = (_cf_end{current_id} != -1 ? _cf_end{current_id} : _cf_end{current_id}_finally) - _cf_start{current_id};",
                f'{inner_body_indent}System.out.println("!######" + _cf_mod{current_id} + ":" + _cf_cls{current_id} + "." + _cf_test{current_id} + ":" + _cf_fn{current_id} + ":" + _cf_loopId{current_id} + ":" + "{inv_id}" + ":" + _cf_dur{current_id} + "######!");',
                f"{inner_indent}}}",
                f"{indent}}}",
            ]

            normalized_prefix = prefix.rstrip(" \t")
            result_parts = ["\n" + "\n".join(setup_lines)]
            if normalized_prefix.strip():
                prefix_body = normalized_prefix.lstrip("\n")
                result_parts.append(f"{indent}\n")
                result_parts.append(prefix_body)
                if not prefix_body.endswith("\n"):
                    result_parts.append("\n")
            else:
                result_parts.append("\n")
            result_parts.append("\n".join(timing_lines))
            if suffix and not suffix.startswith("\n"):
                result_parts.append("\n")
            result_parts.append(suffix)
            return "".join(result_parts), current_id

        multi_result_parts: list[str] = []
        cursor = 0
        wrapper_id = next_wrapper_id

        for stmt_start, stmt_end, stmt_ast_node, call_abs_line in unique_ranges:
            prefix = body_text[cursor:stmt_start]
            target_stmt = body_text[stmt_start:stmt_end]
            multi_result_parts.append(prefix.rstrip(" \t"))

            wrapper_id += 1
            current_id = wrapper_id
            inv_id = f"L{call_abs_line}_{current_id}"

            setup_lines = [
                f"{indent}// Codeflash timing instrumentation with inner loop for JIT warmup",
                f'{indent}int _cf_outerLoop{current_id} = Integer.parseInt(System.getenv("CODEFLASH_LOOP_INDEX"));',
                f'{indent}int _cf_maxInnerIterations{current_id} = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "10"));',
                f'{indent}int _cf_innerIterations{current_id} = Integer.parseInt(System.getenv().getOrDefault("CODEFLASH_INNER_ITERATIONS", "10"));',
                f'{indent}String _cf_mod{current_id} = "{class_name}";',
                f'{indent}String _cf_cls{current_id} = "{class_name}";',
                f'{indent}String _cf_test{current_id} = "{test_method_name}";',
                f'{indent}String _cf_fn{current_id} = "{func_name}";',
                "",
            ]

            # Hoist variable declarations to avoid scoping issues (same as single-range branch)
            var_split = split_var_declaration(stmt_ast_node, wrapper_bytes)
            if var_split is not None:
                hoisted_decl, assignment_stmt = var_split
                setup_lines.append(f"{indent}{hoisted_decl}")
                stmt_in_try = reindent_block(assignment_stmt, inner_body_indent)
            else:
                stmt_in_try = reindent_block(target_stmt, inner_body_indent)

            timing_lines = [
                f"{indent}for (int _cf_i{current_id} = 0; _cf_i{current_id} < _cf_innerIterations{current_id}; _cf_i{current_id}++) {{",
                f"{inner_indent}int _cf_loopId{current_id} = _cf_outerLoop{current_id} * _cf_maxInnerIterations{current_id} + _cf_i{current_id};",
                f'{inner_indent}System.out.println("!$######" + _cf_mod{current_id} + ":" + _cf_cls{current_id} + "." + _cf_test{current_id} + ":" + _cf_fn{current_id} + ":" + _cf_loopId{current_id} + ":{inv_id}" + "######$!");',
                f"{inner_indent}long _cf_end{current_id} = -1;",
                f"{inner_indent}long _cf_start{current_id} = 0;",
                f"{inner_indent}try {{",
                f"{inner_body_indent}_cf_start{current_id} = System.nanoTime();",
                stmt_in_try,
                f"{inner_body_indent}_cf_end{current_id} = System.nanoTime();",
                f"{inner_indent}}} finally {{",
                f"{inner_body_indent}long _cf_end{current_id}_finally = System.nanoTime();",
                f"{inner_body_indent}long _cf_dur{current_id} = (_cf_end{current_id} != -1 ? _cf_end{current_id} : _cf_end{current_id}_finally) - _cf_start{current_id};",
                f'{inner_body_indent}System.out.println("!######" + _cf_mod{current_id} + ":" + _cf_cls{current_id} + "." + _cf_test{current_id} + ":" + _cf_fn{current_id} + ":" + _cf_loopId{current_id} + ":{inv_id}" + ":" + _cf_dur{current_id} + "######!");',
                f"{inner_indent}}}",
                f"{indent}}}",
            ]

            multi_result_parts.append("\n" + "\n".join(setup_lines))
            multi_result_parts.append("\n".join(timing_lines))
            cursor = stmt_end

        multi_result_parts.append(body_text[cursor:])
        return "".join(multi_result_parts), wrapper_id

    test_methods: list[tuple[Any, Any]] = []
    collect_test_methods(tree.root_node, test_methods)
    if not test_methods:
        return source

    replacements: list[tuple[int, int, bytes]] = []
    wrapper_id = 0
    for method_ordinal, (method_node, body_node) in enumerate(test_methods, start=1):
        body_start = body_node.start_byte + 1  # skip '{'
        body_end = body_node.end_byte - 1  # skip '}'
        body_text = source_bytes[body_start:body_end].decode("utf8")
        base_indent = " " * (method_node.start_point[1] + 4)
        # Extract test method name from AST
        name_node = method_node.child_by_field_name("name")
        test_method_name = analyzer.get_node_text(name_node, source_bytes) if name_node else "unknown"
        next_wrapper_id = max(wrapper_id, method_ordinal - 1)
        # body_node.start_point[0] is the 0-based line of the opening brace.
        # The body_text starts after the '{', so its first content line is on that same line or the next.
        # We pass the 0-based line index so build_instrumented_body can compute 1-indexed absolute lines.
        body_start_line_0based = body_node.start_point[0]
        new_body, new_wrapper_id = build_instrumented_body(
            body_text, next_wrapper_id, base_indent, test_method_name, body_start_line=body_start_line_0based
        )
        # Reserve one id slot per @Test method even when no instrumentation is added,
        # matching existing deterministic numbering expected by tests.
        wrapper_id = method_ordinal if new_wrapper_id == next_wrapper_id else new_wrapper_id
        replacements.append((body_start, body_end, new_body.encode("utf8")))

    updated = source_bytes
    for start, end, new_bytes in sorted(replacements, key=lambda item: item[0], reverse=True):
        updated = updated[:start] + new_bytes + updated[end:]
    return updated.decode("utf8")


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
    function_to_optimize: FunctionToOptimize,
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
    if not test_code or not test_code.strip():
        return test_code

    # Input is pre-stripped source (assertions already removed by caller)

    # Extract class name from the test code
    # Use pattern that starts at beginning of line to avoid matching words in comments
    class_match = re.search(r"^\s*(?:(?:public|static|final|abstract)\s+)*class\s+(\w+)", test_code, re.MULTILINE)
    if not class_match:
        logger.warning("Could not find class name in generated test")
        return test_code

    original_class_name = class_match.group(1)

    if mode == "performance":
        new_class_name = f"{original_class_name}__perfonlyinstrumented"

        # Rename all references to the original class name in the source.
        # This includes the class declaration, return types, constructor calls, etc.
        modified_code = re.sub(rf"\b{re.escape(original_class_name)}\b", new_class_name, test_code)

        # Suppress Error Prone's CheckReturnValue for generated performance tests
        modified_code = _add_suppress_warnings_annotation(modified_code, new_class_name)

        modified_code = _add_timing_instrumentation(modified_code, new_class_name, function_name)
    elif mode == "behavior":
        _, behavior_code = instrument_existing_test(
            test_string=test_code,
            mode=mode,
            function_to_optimize=function_to_optimize,
            test_class_name=original_class_name,
        )
        modified_code = behavior_code or test_code
    else:
        modified_code = test_code

    logger.debug("Instrumented generated Java test for %s (mode=%s)", function_name, mode)
    return modified_code
