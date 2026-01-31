"""Java code replacement.

This module provides functionality to replace function implementations
in Java source code while preserving formatting and structure.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.languages.base import FunctionInfo
from codeflash.languages.java.parser import JavaAnalyzer, JavaMethodNode, get_java_analyzer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def replace_function(
    source: str,
    function: FunctionInfo,
    new_source: str,
    analyzer: JavaAnalyzer | None = None,
) -> str:
    """Replace a function in source code with new implementation.

    Preserves:
    - Surrounding whitespace and formatting
    - Javadoc comments (if they should be preserved)
    - Annotations

    Args:
        source: Original source code.
        function: FunctionInfo identifying the function to replace.
        new_source: New function source code.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Modified source code with function replaced.

    """
    analyzer = analyzer or get_java_analyzer()

    # Find the method in the source
    methods = analyzer.find_methods(source)
    target_method = None

    for method in methods:
        if method.name == function.name:
            if function.class_name is None or method.class_name == function.class_name:
                target_method = method
                break

    if not target_method:
        logger.error("Could not find method %s in source", function.name)
        return source

    # Determine replacement range
    # Include Javadoc if present
    start_line = target_method.javadoc_start_line or target_method.start_line
    end_line = target_method.end_line

    # Split source into lines
    lines = source.splitlines(keepends=True)

    # Get indentation from the original method
    original_first_line = lines[start_line - 1] if start_line <= len(lines) else ""
    indent = _get_indentation(original_first_line)

    # Ensure new source has correct indentation
    new_source_lines = new_source.splitlines(keepends=True)
    indented_new_source = _apply_indentation(new_source_lines, indent)

    # Ensure the new source ends with a newline to avoid concatenation issues
    if indented_new_source and not indented_new_source.endswith("\n"):
        indented_new_source += "\n"

    # Build the result
    before = lines[: start_line - 1]  # Lines before the method
    after = lines[end_line:]  # Lines after the method

    result = "".join(before) + indented_new_source + "".join(after)

    return result


def _get_indentation(line: str) -> str:
    """Extract the indentation from a line.

    Args:
        line: The line to analyze.

    Returns:
        The indentation string (spaces/tabs).

    """
    match = re.match(r"^(\s*)", line)
    return match.group(1) if match else ""


def _apply_indentation(lines: list[str], base_indent: str) -> str:
    """Apply indentation to all lines.

    Args:
        lines: Lines to indent.
        base_indent: Base indentation to apply.

    Returns:
        Indented source code.

    """
    if not lines:
        return ""

    # Detect the existing indentation from the first non-empty line
    # This includes Javadoc/comment lines to handle them correctly
    existing_indent = ""
    for line in lines:
        if line.strip():  # First non-empty line
            existing_indent = _get_indentation(line)
            break

    result_lines = []
    for line in lines:
        if not line.strip():
            result_lines.append(line)
        else:
            # Remove existing indentation and apply new base indentation
            stripped_line = line.lstrip()
            # Calculate relative indentation
            line_indent = _get_indentation(line)
            # When existing_indent is empty (first line has no indent), the relative
            # indent is the full line indent. Otherwise, calculate the difference.
            if line_indent.startswith(existing_indent):
                relative_indent = line_indent[len(existing_indent) :]
            else:
                relative_indent = ""
            result_lines.append(base_indent + relative_indent + stripped_line)

    return "".join(result_lines)


def replace_method_body(
    source: str,
    function: FunctionInfo,
    new_body: str,
    analyzer: JavaAnalyzer | None = None,
) -> str:
    """Replace just the body of a method, preserving signature.

    Args:
        source: Original source code.
        function: FunctionInfo identifying the function.
        new_body: New method body (code between braces).
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Modified source code.

    """
    analyzer = analyzer or get_java_analyzer()
    source_bytes = source.encode("utf8")

    # Find the method
    methods = analyzer.find_methods(source)
    target_method = None

    for method in methods:
        if method.name == function.name:
            if function.class_name is None or method.class_name == function.class_name:
                target_method = method
                break

    if not target_method:
        logger.error("Could not find method %s", function.name)
        return source

    # Find the body node
    body_node = target_method.node.child_by_field_name("body")
    if not body_node:
        logger.error("Method %s has no body (abstract?)", function.name)
        return source

    # Get the body's byte positions
    body_start = body_node.start_byte
    body_end = body_node.end_byte

    # Get indentation
    body_start_line = body_node.start_point[0]
    lines = source.splitlines(keepends=True)
    base_indent = _get_indentation(lines[body_start_line]) if body_start_line < len(lines) else "    "

    # Format the new body
    new_body = new_body.strip()
    if not new_body.startswith("{"):
        new_body = "{\n" + base_indent + "    " + new_body
    if not new_body.endswith("}"):
        new_body = new_body + "\n" + base_indent + "}"

    # Replace the body
    before = source_bytes[:body_start]
    after = source_bytes[body_end:]

    return (before + new_body.encode("utf8") + after).decode("utf8")


def insert_method(
    source: str,
    class_name: str,
    method_source: str,
    position: str = "end",  # "end" or "start"
    analyzer: JavaAnalyzer | None = None,
) -> str:
    """Insert a new method into a class.

    Args:
        source: The source code.
        class_name: Name of the class to insert into.
        method_source: Source code of the method to insert.
        position: Where to insert ("end" or "start" of class body).
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Source code with method inserted.

    """
    analyzer = analyzer or get_java_analyzer()

    # Find the class
    classes = analyzer.find_classes(source)
    target_class = None

    for cls in classes:
        if cls.name == class_name:
            target_class = cls
            break

    if not target_class:
        logger.error("Could not find class %s", class_name)
        return source

    # Find the class body
    body_node = target_class.node.child_by_field_name("body")
    if not body_node:
        logger.error("Class %s has no body", class_name)
        return source

    # Get insertion point
    source_bytes = source.encode("utf8")

    if position == "end":
        # Insert before the closing brace
        insert_point = body_node.end_byte - 1
    else:
        # Insert after the opening brace
        insert_point = body_node.start_byte + 1

    # Get indentation (typically 4 spaces inside a class)
    lines = source.splitlines(keepends=True)
    class_line = target_class.start_line - 1
    class_indent = _get_indentation(lines[class_line]) if class_line < len(lines) else ""
    method_indent = class_indent + "    "

    # Format the method
    method_lines = method_source.strip().splitlines(keepends=True)
    indented_method = _apply_indentation(method_lines, method_indent)

    # Ensure the indented method ends with a newline
    if indented_method and not indented_method.endswith("\n"):
        indented_method += "\n"

    # Insert the method
    before = source_bytes[:insert_point]
    after = source_bytes[insert_point:]

    # Use single newline as separator; for start position we need newline after opening brace
    separator = "\n" if position == "end" else "\n"

    return (before + separator.encode("utf8") + indented_method.encode("utf8") + after).decode("utf8")


def remove_method(
    source: str,
    function: FunctionInfo,
    analyzer: JavaAnalyzer | None = None,
) -> str:
    """Remove a method from source code.

    Args:
        source: The source code.
        function: FunctionInfo identifying the method to remove.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Source code with method removed.

    """
    analyzer = analyzer or get_java_analyzer()

    # Find the method
    methods = analyzer.find_methods(source)
    target_method = None

    for method in methods:
        if method.name == function.name:
            if function.class_name is None or method.class_name == function.class_name:
                target_method = method
                break

    if not target_method:
        logger.error("Could not find method %s", function.name)
        return source

    # Determine removal range (include Javadoc)
    start_line = target_method.javadoc_start_line or target_method.start_line
    end_line = target_method.end_line

    lines = source.splitlines(keepends=True)

    # Remove the method lines
    before = lines[: start_line - 1]
    after = lines[end_line:]

    return "".join(before) + "".join(after)


def remove_test_functions(
    test_source: str,
    functions_to_remove: list[str],
    analyzer: JavaAnalyzer | None = None,
) -> str:
    """Remove specific test functions from test source code.

    Args:
        test_source: Test source code.
        functions_to_remove: List of function names to remove.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Test source code with specified functions removed.

    """
    analyzer = analyzer or get_java_analyzer()

    # Find all methods
    methods = analyzer.find_methods(test_source)

    # Sort by start line in reverse order (remove from end first)
    methods_to_remove = [
        m for m in methods if m.name in functions_to_remove
    ]
    methods_to_remove.sort(key=lambda m: m.start_line, reverse=True)

    result = test_source

    for method in methods_to_remove:
        # Create a FunctionInfo for removal
        func_info = FunctionInfo(
            name=method.name,
            file_path=Path("temp.java"),
            start_line=method.start_line,
            end_line=method.end_line,
            parents=(),
            is_method=True,
        )
        result = remove_method(result, func_info, analyzer)

    return result


def add_runtime_comments(
    test_source: str,
    original_runtimes: dict[str, int],
    optimized_runtimes: dict[str, int],
    analyzer: JavaAnalyzer | None = None,
) -> str:
    """Add runtime performance comments to test source code.

    Adds comments showing the original vs optimized runtime for each
    function call (e.g., "// 1.5ms -> 0.3ms (80% faster)").

    Args:
        test_source: Test source code to annotate.
        original_runtimes: Map of invocation IDs to original runtimes (ns).
        optimized_runtimes: Map of invocation IDs to optimized runtimes (ns).
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Test source code with runtime comments added.

    """
    if not original_runtimes or not optimized_runtimes:
        return test_source

    # For now, add a summary comment at the top
    summary_lines = ["// Performance comparison:"]

    for inv_id in original_runtimes:
        original_ns = original_runtimes[inv_id]
        optimized_ns = optimized_runtimes.get(inv_id, original_ns)

        original_ms = original_ns / 1_000_000
        optimized_ms = optimized_ns / 1_000_000

        if original_ns > 0:
            speedup = ((original_ns - optimized_ns) / original_ns) * 100
            summary_lines.append(
                f"// {inv_id}: {original_ms:.3f}ms -> {optimized_ms:.3f}ms ({speedup:.1f}% faster)"
            )

    # Insert after imports
    lines = test_source.splitlines(keepends=True)
    insert_idx = 0

    for i, line in enumerate(lines):
        if line.strip().startswith("import "):
            insert_idx = i + 1
        elif line.strip() and not line.strip().startswith("//") and not line.strip().startswith("package"):
            if insert_idx == 0:
                insert_idx = i
            break

    # Insert summary
    summary = "\n".join(summary_lines) + "\n\n"
    lines.insert(insert_idx, summary)

    return "".join(lines)
