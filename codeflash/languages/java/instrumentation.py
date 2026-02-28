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

_PROTECTED_RENAME_TYPES = frozenset(
    ("string_literal", "character_literal", "line_comment", "block_comment", "import_declaration")
)

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


def _has_test_annotation(method_node: Any, analyzer: JavaAnalyzer, source_bytes: bytes) -> bool:
    """Check if a method_declaration node has a @Test annotation."""
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
        annotation_text = analyzer.get_node_text(child, source_bytes).strip()
        if annotation_text.startswith("@"):
            name = annotation_text[1:].split("(", 1)[0].strip()
            if name == "Test" or name.endswith(".Test"):
                return True
    return False


def _collect_test_methods(
    node: Any, analyzer: JavaAnalyzer, source_bytes: bytes, out: list[tuple[Any, Any]]
) -> None:
    """Collect @Test methods as (method_node, body_node) pairs."""
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type == "method_declaration" and _has_test_annotation(current, analyzer, source_bytes):
            body_node = current.child_by_field_name("body")
            if body_node is not None:
                out.append((current, body_node))
            continue
        if current.children:
            stack.extend(reversed(current.children))


def _is_java_ident_byte(b: int) -> bool:
    """Check if a byte represents a Java identifier character (ASCII subset)."""
    return (
        (ord("a") <= b <= ord("z"))
        or (ord("A") <= b <= ord("Z"))
        or (ord("0") <= b <= ord("9"))
        or b == ord("_")
        or b == ord("$")
    )


def _collect_protected_ranges_for_rename(node: Any, out: list[tuple[int, int]]) -> None:
    """Collect byte ranges of AST nodes that should be excluded from class renaming.

    Protects string literals, character literals, comments, and import declarations
    so that renaming only affects actual Java identifiers in code.
    """
    # Use an explicit stack to avoid recursion overhead on deep trees.
    stack = [node]
    while stack:
        current = stack.pop()
        node_type = current.type
        if node_type in _PROTECTED_RENAME_TYPES:
            out.append((current.start_byte, current.end_byte))
            continue
        # Push children in reverse so they are processed in original left-to-right order
        children = current.children
        if children:
            stack.extend(reversed(children))


def _overlaps_protected(start: int, end: int, protected: list[tuple[int, int]]) -> bool:
    """Check if [start, end) overlaps with any sorted protected range using binary search."""
    lo, hi = 0, len(protected)
    while lo < hi:
        mid = (lo + hi) // 2
        if protected[mid][1] <= start:
            lo = mid + 1
        else:
            hi = mid
    return lo < len(protected) and protected[lo][0] < end


def _rename_class_treesitter(source: str, old_name: str, new_name: str) -> str:
    """Rename a class using tree-sitter to skip strings, comments, and import declarations.

    Unlike re.sub(r"\\bOldName\\b", new_name, source), this avoids renaming occurrences
    inside string literals, character literals, comments, and import paths, which would
    produce invalid Java (broken imports, corrupted strings).
    """
    from codeflash.languages.java.parser import get_java_analyzer

    analyzer = get_java_analyzer()
    source_bytes = source.encode("utf8")
    tree = analyzer.parse(source_bytes)

    protected_ranges: list[tuple[int, int]] = []
    _collect_protected_ranges_for_rename(tree.root_node, protected_ranges)
    protected_ranges.sort()

    old_bytes = old_name.encode("utf8")
    new_bytes = new_name.encode("utf8")
    old_len = len(old_bytes)
    src_len = len(source_bytes)

    replace_positions: list[int] = []
    search_start = 0
    while True:
        pos = source_bytes.find(old_bytes, search_start)
        if pos == -1:
            break
        end_pos = pos + old_len
        if (pos == 0 or not _is_java_ident_byte(source_bytes[pos - 1])) and (
            end_pos >= src_len or not _is_java_ident_byte(source_bytes[end_pos])
        ):
            if not _overlaps_protected(pos, end_pos, protected_ranges):
                replace_positions.append(pos)
        search_start = pos + 1

    if not replace_positions:
        return source

    result = bytearray(source_bytes)
    for pos in reversed(replace_positions):
        result[pos : pos + old_len] = new_bytes

    return bytes(result).decode("utf8")


_TS_BODY_PREFIX = "class _D { void _m() {\n"
_TS_BODY_SUFFIX = "\n}}"
_TS_BODY_PREFIX_BYTES = _TS_BODY_PREFIX.encode("utf8")


def _generate_sqlite_write_code(
    iter_id: int, call_counter: int, indent: str, class_name: str, func_name: str, test_method_name: str
) -> list[str]:
    """Generate SQLite write code for a single function call.

    Args:
        iter_id: Test method iteration ID
        call_counter: Call counter for unique variable naming
        indent: Base indentation string
        class_name: Test class name
        func_name: Function being tested
        test_method_name: Test method name

    Returns:
        List of code lines for SQLite write in finally block.

    """
    inner_indent = indent + "    "
    return [
        f"{indent}}} finally {{",
        f"{inner_indent}long _cf_end{iter_id}_{call_counter}_finally = System.nanoTime();",
        f"{inner_indent}long _cf_dur{iter_id}_{call_counter} = (_cf_end{iter_id}_{call_counter} != -1 ? _cf_end{iter_id}_{call_counter} : _cf_end{iter_id}_{call_counter}_finally) - _cf_start{iter_id}_{call_counter};",
        f'{inner_indent}System.out.println("!######" + _cf_mod{iter_id} + ":" + _cf_cls{iter_id} + "." + _cf_test{iter_id} + ":" + _cf_fn{iter_id} + ":" + _cf_loop{iter_id} + ":" + "{call_counter}" + "######!");',
        f"{inner_indent}// Write to SQLite if output file is set",
        f"{inner_indent}if (_cf_outputFile{iter_id} != null && !_cf_outputFile{iter_id}.isEmpty()) {{",
        f"{inner_indent}    try {{",
        f'{inner_indent}        Class.forName("org.sqlite.JDBC");',
        f'{inner_indent}        try (java.sql.Connection _cf_conn{iter_id}_{call_counter} = java.sql.DriverManager.getConnection("jdbc:sqlite:" + _cf_outputFile{iter_id})) {{',
        f"{inner_indent}            try (java.sql.Statement _cf_stmt{iter_id}_{call_counter} = _cf_conn{iter_id}_{call_counter}.createStatement()) {{",
        f'{inner_indent}                _cf_stmt{iter_id}_{call_counter}.execute("CREATE TABLE IF NOT EXISTS test_results (" +',
        f'{inner_indent}                    "test_module_path TEXT, test_class_name TEXT, test_function_name TEXT, " +',
        f'{inner_indent}                    "function_getting_tested TEXT, loop_index INTEGER, iteration_id TEXT, " +',
        f'{inner_indent}                    "runtime INTEGER, return_value BLOB, verification_type TEXT)");',
        f"{inner_indent}            }}",
        f'{inner_indent}            String _cf_sql{iter_id}_{call_counter} = "INSERT INTO test_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)";',
        f"{inner_indent}            try (java.sql.PreparedStatement _cf_pstmt{iter_id}_{call_counter} = _cf_conn{iter_id}_{call_counter}.prepareStatement(_cf_sql{iter_id}_{call_counter})) {{",
        f"{inner_indent}                _cf_pstmt{iter_id}_{call_counter}.setString(1, _cf_mod{iter_id});",
        f"{inner_indent}                _cf_pstmt{iter_id}_{call_counter}.setString(2, _cf_cls{iter_id});",
        f"{inner_indent}                _cf_pstmt{iter_id}_{call_counter}.setString(3, _cf_test{iter_id});",
        f"{inner_indent}                _cf_pstmt{iter_id}_{call_counter}.setString(4, _cf_fn{iter_id});",
        f"{inner_indent}                _cf_pstmt{iter_id}_{call_counter}.setInt(5, _cf_loop{iter_id});",
        f'{inner_indent}                _cf_pstmt{iter_id}_{call_counter}.setString(6, "{call_counter}");',
        f"{inner_indent}                _cf_pstmt{iter_id}_{call_counter}.setLong(7, _cf_dur{iter_id}_{call_counter});",
        f"{inner_indent}                _cf_pstmt{iter_id}_{call_counter}.setBytes(8, _cf_serializedResult{iter_id}_{call_counter});",
        f'{inner_indent}                _cf_pstmt{iter_id}_{call_counter}.setString(9, "function_call");',
        f"{inner_indent}                _cf_pstmt{iter_id}_{call_counter}.executeUpdate();",
        f"{inner_indent}            }}",
        f"{inner_indent}        }}",
        f"{inner_indent}    }} catch (Exception _cf_e{iter_id}_{call_counter}) {{",
        f'{inner_indent}        System.err.println("CodeflashHelper: SQLite error: " + _cf_e{iter_id}_{call_counter}.getMessage());',
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
    return_type: str | None = None,
) -> tuple[list[str], int]:
    """Replace target method calls in body_lines with capture + serialize using tree-sitter.

    Parses the method body with tree-sitter, walks the AST for method_invocation nodes
    matching func_name, and generates capture/serialize lines. Uses the parent node type
    to determine whether to keep or remove the original line after replacement.

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

    # Build line byte-start offsets for mapping calls to body_lines indices
    line_byte_starts = []
    offset = 0
    for line in body_lines:
        line_byte_starts.append(offset)
        offset += len(line.encode("utf8")) + 1  # +1 for \n from join

    # Group non-lambda and non-complex-expression calls by their line index
    calls_by_line: dict[int, list[dict[str, Any]]] = {}
    for call in calls:
        if call["in_lambda"] or call.get("in_complex", False):
            logger.debug("Skipping behavior instrumentation for call in lambda or complex expression")
            continue
        line_idx = _byte_to_line_index(call["start_byte"], line_byte_starts)
        calls_by_line.setdefault(line_idx, []).append(call)

    wrapped = []
    call_counter = 0

    for line_idx, body_line in enumerate(body_lines):
        if line_idx not in calls_by_line:
            wrapped.append(body_line)
            continue

        line_calls = sorted(calls_by_line[line_idx], key=lambda c: c["start_byte"], reverse=True)
        line_indent_str = " " * (len(body_line) - len(body_line.lstrip()))
        line_byte_start = line_byte_starts[line_idx]
        line_bytes = body_line.encode("utf8")

        new_line = body_line
        # Track cumulative char shift from earlier edits on this line
        char_shift = 0

        for call in line_calls:
            call_counter += 1
            var_name = f"_cf_result{iter_id}_{call_counter}"
            cast_type = _infer_array_cast_type(body_line)
            if not cast_type and return_type and return_type not in ("void", "Object"):
                cast_type = return_type
            var_with_cast = f"({cast_type}){var_name}" if cast_type else var_name

            # Use per-call unique variables (with call_counter suffix) for behavior mode
            # For behavior mode, we declare the variable outside try block, so use assignment not declaration here
            # For performance mode, use shared variables without call_counter suffix
            capture_stmt_with_decl = f"Object {var_name} = {call['full_call']};"
            capture_stmt_assign = f"{var_name} = {call['full_call']};"
            if precise_call_timing:
                # Behavior mode: per-call unique variables
                serialize_stmt = f"_cf_serializedResult{iter_id}_{call_counter} = com.codeflash.Serializer.serialize((Object) {var_name});"
                start_stmt = f"_cf_start{iter_id}_{call_counter} = System.nanoTime();"
                end_stmt = f"_cf_end{iter_id}_{call_counter} = System.nanoTime();"
            else:
                # Performance mode: shared variables without call_counter suffix
                serialize_stmt = (
                    f"_cf_serializedResult{iter_id} = com.codeflash.Serializer.serialize((Object) {var_name});"
                )
                start_stmt = f"_cf_start{iter_id} = System.nanoTime();"
                end_stmt = f"_cf_end{iter_id} = System.nanoTime();"

            if call["parent_type"] == "expression_statement":
                # Replace the expression_statement IN PLACE with capture+serialize.
                # This keeps the code inside whatever scope it's in (e.g. try block),
                # preventing calls from being moved outside try-catch blocks.
                es_start_byte = call["es_start_byte"] - line_byte_start
                es_end_byte = call["es_end_byte"] - line_byte_start
                es_start_char = len(line_bytes[:es_start_byte].decode("utf8"))
                es_end_char = len(line_bytes[:es_end_byte].decode("utf8"))
                if precise_call_timing:
                    # For behavior mode: wrap each call in its own try-finally with SQLite write.
                    # This ensures data from all calls is captured independently.
                    # Declare per-call variables
                    var_decls = [
                        f"Object {var_name} = null;",
                        f"long _cf_end{iter_id}_{call_counter} = -1;",
                        f"long _cf_start{iter_id}_{call_counter} = 0;",
                        f"byte[] _cf_serializedResult{iter_id}_{call_counter} = null;",
                    ]
                    # Start marker
                    start_marker = f'System.out.println("!$######" + _cf_mod{iter_id} + ":" + _cf_cls{iter_id} + "." + _cf_test{iter_id} + ":" + _cf_fn{iter_id} + ":" + _cf_loop{iter_id} + ":{call_counter}" + "######$!");'
                    # Try block with capture (use assignment, not declaration, since variable is declared above)
                    try_block = [
                        "try {",
                        f"    {start_stmt}",
                        f"    {capture_stmt_assign}",
                        f"    {end_stmt}",
                        f"    {serialize_stmt}",
                    ]
                    # Finally block with SQLite write
                    finally_block = _generate_sqlite_write_code(
                        iter_id, call_counter, "", class_name, func_name, test_method_name
                    )

                    replacement_lines = [*var_decls, start_marker, *try_block, *finally_block]
                    # Don't add indent to first line (it's placed after existing indent), but add to subsequent lines
                    if replacement_lines:
                        replacement = (
                            replacement_lines[0]
                            + "\n"
                            + "\n".join(f"{line_indent_str}{line}" for line in replacement_lines[1:])
                        )
                    else:
                        replacement = ""
                else:
                    replacement = f"{capture_stmt_with_decl} {serialize_stmt}"
                adj_start = es_start_char + char_shift
                adj_end = es_end_char + char_shift
                new_line = new_line[:adj_start] + replacement + new_line[adj_end:]
                char_shift += len(replacement) - (es_end_char - es_start_char)
            else:
                # The call is embedded in a larger expression (assignment, assertion, etc.)
                # Emit capture+serialize before the line, then replace the call with the variable.
                if precise_call_timing:
                    # For behavior mode: wrap in try-finally with SQLite write.
                    # For calls inside assertions (assertEquals, etc.), the assertion line is
                    # dropped after capture — behavior comparison uses serialized return values
                    # instead. This avoids Object auto-boxing type mismatches (e.g., Long vs
                    # Integer) that break JUnit assertEquals for primitive types.
                    in_assertion = call.get("in_assertion", False)
                    wrapped.append(f"{line_indent_str}Object {var_name} = null;")
                    wrapped.append(f"{line_indent_str}long _cf_end{iter_id}_{call_counter} = -1;")
                    wrapped.append(f"{line_indent_str}long _cf_start{iter_id}_{call_counter} = 0;")
                    wrapped.append(f"{line_indent_str}byte[] _cf_serializedResult{iter_id}_{call_counter} = null;")
                    # Start marker
                    wrapped.append(
                        f'{line_indent_str}System.out.println("!$######" + _cf_mod{iter_id} + ":" + _cf_cls{iter_id} + "." + _cf_test{iter_id} + ":" + _cf_fn{iter_id} + ":" + _cf_loop{iter_id} + ":{call_counter}" + "######$!");'
                    )
                    # Try block (use assignment, not declaration, since variable is declared above)
                    wrapped.append(f"{line_indent_str}try {{")
                    wrapped.append(f"{line_indent_str}    {start_stmt}")
                    wrapped.append(f"{line_indent_str}    {capture_stmt_assign}")
                    wrapped.append(f"{line_indent_str}    {end_stmt}")
                    wrapped.append(f"{line_indent_str}    {serialize_stmt}")
                    # Finally block with SQLite write
                    finally_lines = _generate_sqlite_write_code(
                        iter_id, call_counter, line_indent_str, class_name, func_name, test_method_name
                    )
                    wrapped.extend(finally_lines)
                    if in_assertion:
                        # Drop the assertion line — behavior capture handles correctness
                        # via serialized return value comparison. This avoids Object boxing
                        # type mismatches (Long vs Integer) that break assertEquals.
                        new_line = ""
                    else:
                        # Non-assertion embedded expression: replace call with captured variable
                        call_start_byte = call["start_byte"] - line_byte_start
                        call_end_byte = call["end_byte"] - line_byte_start
                        call_start_char = len(line_bytes[:call_start_byte].decode("utf8"))
                        call_end_char = len(line_bytes[:call_end_byte].decode("utf8"))
                        adj_start = call_start_char + char_shift
                        adj_end = call_end_char + char_shift
                        new_line = new_line[:adj_start] + var_with_cast + new_line[adj_end:]
                        char_shift += len(var_with_cast) - (call_end_char - call_start_char)
                else:
                    capture_line = f"{line_indent_str}{capture_stmt_with_decl}"
                    wrapped.append(capture_line)
                    serialize_line = f"{line_indent_str}{serialize_stmt}"
                    wrapped.append(serialize_line)

                    call_start_byte = call["start_byte"] - line_byte_start
                    call_end_byte = call["end_byte"] - line_byte_start
                    call_start_char = len(line_bytes[:call_start_byte].decode("utf8"))
                    call_end_char = len(line_bytes[:call_end_byte].decode("utf8"))
                    adj_start = call_start_char + char_shift
                    adj_end = call_end_char + char_shift
                    new_line = new_line[:adj_start] + var_with_cast + new_line[adj_end:]
                    char_shift += len(var_with_cast) - (call_end_char - call_start_char)

        # Keep the modified line only if it has meaningful content left
        if new_line.strip():
            wrapped.append(new_line)

    return wrapped, call_counter


def _is_inside_assertion(node: Any, wrapper_bytes: bytes, analyzer: JavaAnalyzer) -> bool:
    """Check if a tree-sitter node is inside an assertion method call.

    Walks up the AST to find a parent method_invocation whose name starts with "assert"
    (e.g., assertEquals, assertNotNull, assertThrows). Stops at statement boundaries.
    """
    current = node.parent
    while current is not None:
        if current.type in {"method_declaration", "block", "expression_statement"}:
            return False
        if current.type == "method_invocation":
            name_child = current.child_by_field_name("name")
            if name_child:
                name_text = analyzer.get_node_text(name_child, wrapper_bytes)
                if name_text.startswith("assert"):
                    return True
        current = current.parent
    return False


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
                        "in_assertion": _is_inside_assertion(node, wrapper_bytes, analyzer),
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

    # Get the original class name from the file name
    if test_path:
        original_class_name = test_path.stem  # e.g., "AlgorithmsTest"
    elif test_class_name is not None:
        original_class_name = test_class_name
    else:
        raise ValueError("test_path or test_class_name must be provided")

    if mode == "behavior":
        new_class_name = f"{original_class_name}__perfinstrumented"
    else:
        new_class_name = f"{original_class_name}__perfonlyinstrumented"

    # Rename class using tree-sitter-aware renaming that skips strings, comments, and imports.
    modified_source = _rename_class_treesitter(source, original_class_name, new_class_name)

    return_type = getattr(function_to_optimize, "return_type", None)

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
        modified_source = _add_behavior_instrumentation(modified_source, original_class_name, func_name, return_type=return_type)

    logger.debug("Java %s testing for %s: renamed class %s -> %s", mode, func_name, original_class_name, new_class_name)

    return True, modified_source


def _add_behavior_instrumentation(source: str, class_name: str, func_name: str, return_type: str | None = None) -> str:
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
    from codeflash.languages.java.parser import get_java_analyzer

    analyzer = get_java_analyzer()
    source_bytes = source.encode("utf8")
    tree = analyzer.parse(source_bytes)

    test_methods: list[tuple[Any, Any]] = []
    _collect_test_methods(tree.root_node, analyzer, source_bytes, test_methods)
    if not test_methods:
        return source

    replacements: list[tuple[int, int, bytes]] = []
    iteration_counter = 0

    for method_node, body_node in test_methods:
        iteration_counter += 1
        iter_id = iteration_counter

        body_start = body_node.start_byte + 1  # skip '{'
        body_end = body_node.end_byte - 1  # skip '}'
        raw_body = source_bytes[body_start:body_end].decode("utf8")

        # Split body into lines, removing leading empty line (after '{') and trailing
        # whitespace-only line (indent before '}') to match the format expected by
        # wrap_target_calls_with_treesitter.
        all_lines = raw_body.split("\n")
        body_lines = all_lines[1:]  # Remove leading empty from '\n' after '{'
        if body_lines and not body_lines[-1].strip():
            body_lines = body_lines[:-1]

        # Get test method name from AST
        name_node = method_node.child_by_field_name("name")
        test_method_name = analyzer.get_node_text(name_node, source_bytes) if name_node else "unknown"

        base_indent = method_node.start_point[1]
        indent = " " * (base_indent + 4)

        wrapped_body_lines, _call_counter = wrap_target_calls_with_treesitter(
            body_lines=body_lines,
            func_name=func_name,
            iter_id=iter_id,
            precise_call_timing=True,
            class_name=class_name,
            test_method_name=test_method_name,
            return_type=return_type,
        )

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

        body_content_lines = behavior_start_code + wrapped_body_lines
        new_body = "\n" + "\n".join(body_content_lines) + "\n" + " " * base_indent

        replacements.append((body_start, body_end, new_body.encode("utf8")))

    updated = source_bytes
    for start, end, new_bytes in sorted(replacements, key=lambda item: item[0], reverse=True):
        updated = updated[:start] + new_bytes + updated[end:]
    return updated.decode("utf8")


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

    def split_var_declaration(
        stmt_node: Any, source_bytes_ref: bytes
    ) -> tuple[str, str] | None:
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

        name_text = analyzer.get_node_text(name_node, source_bytes_ref)

        type_text = analyzer.get_node_text(type_node, source_bytes_ref)
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
        body_text: str, next_wrapper_id: int, base_indent: str, test_method_name: str = "unknown"
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

        statement_ranges: list[tuple[int, int, Any]] = []  # (char_start, char_end, ast_node)
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
            statement_ranges.append((stmt_start, stmt_end, stmt_node))

        # Deduplicate repeated calls within the same top-level statement.
        unique_ranges: list[tuple[int, int, Any]] = []
        seen_offsets: set[tuple[int, int]] = set()
        for stmt_start, stmt_end, stmt_node in statement_ranges:
            key = (stmt_start, stmt_end)
            if key in seen_offsets:
                continue
            seen_offsets.add(key)
            unique_ranges.append((stmt_start, stmt_end, stmt_node))
        if not unique_ranges:
            return body_text, next_wrapper_id

        if len(unique_ranges) == 1:
            stmt_start, stmt_end, stmt_ast_node = unique_ranges[0]
            prefix = body_text[:stmt_start]
            target_stmt = body_text[stmt_start:stmt_end]
            suffix = body_text[stmt_end:]

            current_id = next_wrapper_id + 1
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
                f'{inner_indent}System.out.println("!$######" + _cf_mod{current_id} + ":" + _cf_cls{current_id} + "." + _cf_test{current_id} + ":" + _cf_fn{current_id} + ":" + _cf_loopId{current_id} + ":" + _cf_i{current_id} + "######$!");',
                f"{inner_indent}long _cf_end{current_id} = -1;",
                f"{inner_indent}long _cf_start{current_id} = 0;",
                f"{inner_indent}try {{",
                f"{inner_body_indent}_cf_start{current_id} = System.nanoTime();",
                stmt_in_try,
                f"{inner_body_indent}_cf_end{current_id} = System.nanoTime();",
                f"{inner_indent}}} finally {{",
                f"{inner_body_indent}long _cf_end{current_id}_finally = System.nanoTime();",
                f"{inner_body_indent}long _cf_dur{current_id} = (_cf_end{current_id} != -1 ? _cf_end{current_id} : _cf_end{current_id}_finally) - _cf_start{current_id};",
                f'{inner_body_indent}System.out.println("!######" + _cf_mod{current_id} + ":" + _cf_cls{current_id} + "." + _cf_test{current_id} + ":" + _cf_fn{current_id} + ":" + _cf_loopId{current_id} + ":" + _cf_i{current_id} + ":" + _cf_dur{current_id} + "######!");',
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

        for stmt_start, stmt_end, stmt_ast_node in unique_ranges:
            prefix = body_text[cursor:stmt_start]
            target_stmt = body_text[stmt_start:stmt_end]
            multi_result_parts.append(prefix.rstrip(" \t"))

            wrapper_id += 1
            current_id = wrapper_id

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
                f'{inner_indent}System.out.println("!$######" + _cf_mod{current_id} + ":" + _cf_cls{current_id} + "." + _cf_test{current_id} + ":" + _cf_fn{current_id} + ":" + _cf_loopId{current_id} + ":{current_id}_" + _cf_i{current_id} + "######$!");',
                f"{inner_indent}long _cf_end{current_id} = -1;",
                f"{inner_indent}long _cf_start{current_id} = 0;",
                f"{inner_indent}try {{",
                f"{inner_body_indent}_cf_start{current_id} = System.nanoTime();",
                stmt_in_try,
                f"{inner_body_indent}_cf_end{current_id} = System.nanoTime();",
                f"{inner_indent}}} finally {{",
                f"{inner_body_indent}long _cf_end{current_id}_finally = System.nanoTime();",
                f"{inner_body_indent}long _cf_dur{current_id} = (_cf_end{current_id} != -1 ? _cf_end{current_id} : _cf_end{current_id}_finally) - _cf_start{current_id};",
                f'{inner_body_indent}System.out.println("!######" + _cf_mod{current_id} + ":" + _cf_cls{current_id} + "." + _cf_test{current_id} + ":" + _cf_fn{current_id} + ":" + _cf_loopId{current_id} + ":{current_id}_" + _cf_i{current_id} + ":" + _cf_dur{current_id} + "######!");',
                f"{inner_indent}}}",
                f"{indent}}}",
            ]

            multi_result_parts.append("\n" + "\n".join(setup_lines))
            multi_result_parts.append("\n".join(timing_lines))
            cursor = stmt_end

        multi_result_parts.append(body_text[cursor:])
        return "".join(multi_result_parts), wrapper_id

    test_methods: list[tuple[Any, Any]] = []
    _collect_test_methods(tree.root_node, analyzer, source_bytes, test_methods)
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
        new_body, new_wrapper_id = build_instrumented_body(body_text, next_wrapper_id, base_indent, test_method_name)
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

    from codeflash.languages.java.remove_asserts import transform_java_assertions

    test_code = transform_java_assertions(test_code, function_name, qualified_name)

    # Extract class name from the test code
    # Use pattern that starts at beginning of line to avoid matching words in comments
    class_match = re.search(r"^(?:public\s+)?class\s+(\w+)", test_code, re.MULTILINE)
    if not class_match:
        logger.warning("Could not find class name in generated test")
        return test_code

    original_class_name = class_match.group(1)

    if mode == "performance":
        new_class_name = f"{original_class_name}__perfonlyinstrumented"

        # Rename class using tree-sitter-aware renaming that skips strings, comments, and imports.
        modified_code = _rename_class_treesitter(test_code, original_class_name, new_class_name)

        modified_code = _add_timing_instrumentation(
            modified_code,
            original_class_name,  # Use original name in markers, not the renamed class
            function_name,
        )
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
