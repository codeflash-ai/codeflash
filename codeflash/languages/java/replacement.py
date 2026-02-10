"""Java code replacement.

This module provides functionality to replace function implementations
in Java source code while preserving formatting and structure.

Supports optimizations that add:
- New static fields
- New helper methods
- Additional class-level members
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.java.parser import get_java_analyzer

if TYPE_CHECKING:
    from codeflash.languages.java.parser import JavaAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ParsedOptimization:
    """Parsed optimization containing method and additional class members."""

    target_method_source: str
    new_fields: list[str]  # Source text of new fields to add
    new_helper_methods: list[str]  # Source text of new helper methods to add


def _parse_optimization_source(new_source: str, target_method_name: str, analyzer: JavaAnalyzer) -> ParsedOptimization:
    """Parse optimization source to extract method and additional class members.

    The new_source may contain:
    - Just a method definition
    - A class with the method and additional static fields/helper methods

    Args:
        new_source: The optimization source code.
        target_method_name: Name of the method being optimized.
        analyzer: JavaAnalyzer instance.

    Returns:
        ParsedOptimization with the method and any additional members.

    """
    new_fields: list[str] = []
    new_helper_methods: list[str] = []
    target_method_source = new_source  # Default to the whole source

    # Check if this is a full class or just a method
    classes = analyzer.find_classes(new_source)

    if classes:
        # It's a class - extract components
        methods = analyzer.find_methods(new_source)
        fields = analyzer.find_fields(new_source)

        # Find the target method
        target_method = None
        for method in methods:
            if method.name == target_method_name:
                target_method = method
                break

        if target_method:
            # Extract target method source (including Javadoc if present)
            lines = new_source.splitlines(keepends=True)
            start = (target_method.javadoc_start_line or target_method.start_line) - 1
            end = target_method.end_line
            target_method_source = "".join(lines[start:end])

        # Extract helper methods (methods other than the target)
        for method in methods:
            if method.name != target_method_name:
                lines = new_source.splitlines(keepends=True)
                start = (method.javadoc_start_line or method.start_line) - 1
                end = method.end_line
                helper_source = "".join(lines[start:end])
                new_helper_methods.append(helper_source)

        # Extract fields
        for field in fields:
            if field.source_text:
                new_fields.append(field.source_text)

    return ParsedOptimization(
        target_method_source=target_method_source, new_fields=new_fields, new_helper_methods=new_helper_methods
    )


def _insert_class_members(
    source: str, class_name: str, fields: list[str], methods: list[str], analyzer: JavaAnalyzer
) -> str:
    """Insert new class members (fields and methods) into a class.

    Fields are inserted at the beginning of the class body (after opening brace).
    Methods are inserted at the end of the class body (before closing brace).

    Args:
        source: The source code.
        class_name: Name of the class to modify.
        fields: List of field source texts to insert.
        methods: List of method source texts to insert.
        analyzer: JavaAnalyzer instance.

    Returns:
        Modified source code.

    """
    if not fields and not methods:
        return source

    classes = analyzer.find_classes(source)
    target_class = None

    for cls in classes:
        if cls.name == class_name:
            target_class = cls
            break

    if not target_class:
        logger.warning("Could not find class %s to insert members", class_name)
        return source

    # Get class body
    body_node = target_class.node.child_by_field_name("body")
    if not body_node:
        logger.warning("Class %s has no body", class_name)
        return source

    source_bytes = source.encode("utf8")
    lines = source.splitlines(keepends=True)

    # Get class indentation
    class_line = target_class.start_line - 1
    class_indent = _get_indentation(lines[class_line]) if class_line < len(lines) else ""
    member_indent = class_indent + "    "

    result = source

    # Insert fields at the beginning of the class body (after opening brace)
    if fields:
        # Re-parse to get current positions
        classes = analyzer.find_classes(result)
        for cls in classes:
            if cls.name == class_name:
                body_node = cls.node.child_by_field_name("body")
                break

        if body_node:
            result_bytes = result.encode("utf8")
            insert_point = body_node.start_byte + 1  # After opening brace

            # Format fields
            field_text = "\n"
            for field in fields:
                field_lines = field.strip().splitlines(keepends=True)
                indented_field = _apply_indentation(field_lines, member_indent)
                field_text += indented_field
                if not indented_field.endswith("\n"):
                    field_text += "\n"

            before = result_bytes[:insert_point]
            after = result_bytes[insert_point:]
            result = (before + field_text.encode("utf8") + after).decode("utf8")

    # Insert methods at the end of the class body (before closing brace)
    if methods:
        # Re-parse to get current positions
        classes = analyzer.find_classes(result)
        for cls in classes:
            if cls.name == class_name:
                body_node = cls.node.child_by_field_name("body")
                break

        if body_node:
            result_bytes = result.encode("utf8")
            insert_point = body_node.end_byte - 1  # Before closing brace

            # Format methods
            method_text = "\n"
            for method in methods:
                method_lines = method.strip().splitlines(keepends=True)
                indented_method = _apply_indentation(method_lines, member_indent)
                method_text += indented_method
                if not indented_method.endswith("\n"):
                    method_text += "\n"

            before = result_bytes[:insert_point]
            after = result_bytes[insert_point:]
            result = (before + method_text.encode("utf8") + after).decode("utf8")

    return result


def replace_function(
    source: str, function: FunctionToOptimize, new_source: str, analyzer: JavaAnalyzer | None = None
) -> str:
    """Replace a function in source code with new implementation.

    Supports optimizations that include:
    - Just the method being optimized
    - A class with the method plus additional static fields and helper methods

    When the new_source contains a full class with additional members,
    those members are also added to the original source.

    Preserves:
    - Surrounding whitespace and formatting
    - Javadoc comments (if they should be preserved)
    - Annotations

    Args:
        source: Original source code.
        function: FunctionToOptimize identifying the function to replace.
        new_source: New function source code (may include class with helpers).
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Modified source code with function replaced and any new members added.

    """
    analyzer = analyzer or get_java_analyzer()

    func_name = function.function_name
    func_start_line = function.starting_line
    func_end_line = function.ending_line

    # Parse the optimization to extract components
    parsed = _parse_optimization_source(new_source, func_name, analyzer)

    # Find the method in the original source
    methods = analyzer.find_methods(source)
    target_method = None
    target_overload_index = 0  # Track which overload we're targeting

    # Find all methods matching the name (there may be overloads)
    matching_methods = [
        m
        for m in methods
        if m.name == func_name and (function.class_name is None or m.class_name == function.class_name)
    ]

    if len(matching_methods) == 1:
        # Only one method with this name - use it
        target_method = matching_methods[0]
        target_overload_index = 0
    elif len(matching_methods) > 1:
        # Multiple overloads - use line numbers to find the exact one
        logger.debug(
            "Found %d overloads of %s. Function start_line=%s, end_line=%s",
            len(matching_methods),
            func_name,
            func_start_line,
            func_end_line,
        )
        for i, m in enumerate(matching_methods):
            logger.debug("  Overload %d: lines %d-%d", i, m.start_line, m.end_line)
        if func_start_line and func_end_line:
            for i, method in enumerate(matching_methods):
                # Check if the line numbers are close (account for minor differences
                # that can occur due to different parsing or file transformations)
                # Use a tolerance of 5 lines to handle edge cases
                if abs(method.start_line - func_start_line) <= 5:
                    target_method = method
                    target_overload_index = i
                    logger.debug(
                        "Matched overload %d at lines %d-%d (target: %d-%d)",
                        i,
                        method.start_line,
                        method.end_line,
                        func_start_line,
                        func_end_line,
                    )
                    break
        if not target_method:
            # Fallback: use the first match
            logger.warning("Multiple overloads of %s found but no line match, using first match", func_name)
            target_method = matching_methods[0]
            target_overload_index = 0

    if not target_method:
        logger.error("Could not find method %s in source", func_name)
        return source

    # Get the class name for inserting new members
    class_name = target_method.class_name or function.class_name

    # First, add any new fields and helper methods to the class
    if class_name and (parsed.new_fields or parsed.new_helper_methods):
        # Filter out fields/methods that already exist
        existing_methods = {m.name for m in methods}
        existing_fields = {f.name for f in analyzer.find_fields(source)}

        # Filter helper methods
        new_helpers_to_add = []
        for helper_src in parsed.new_helper_methods:
            helper_methods = analyzer.find_methods(helper_src)
            if helper_methods and helper_methods[0].name not in existing_methods:
                new_helpers_to_add.append(helper_src)

        # Filter fields
        new_fields_to_add = []
        for field_src in parsed.new_fields:
            # Parse field to get its name by wrapping in a dummy class
            # (find_fields requires class context to parse field declarations)
            dummy_class = f"class __DummyClass__ {{\n{field_src}\n}}"
            field_infos = analyzer.find_fields(dummy_class)
            for field_info in field_infos:
                if field_info.name not in existing_fields:
                    new_fields_to_add.append(field_src)
                    break  # Only add once per field declaration

        if new_fields_to_add or new_helpers_to_add:
            logger.debug(
                "Adding %d new fields and %d helper methods to class %s",
                len(new_fields_to_add),
                len(new_helpers_to_add),
                class_name,
            )
            source = _insert_class_members(source, class_name, new_fields_to_add, new_helpers_to_add, analyzer)

            # Re-find the target method after modifications
            # Line numbers have shifted, but the relative order of overloads is preserved
            # Use the target_overload_index we saved earlier
            methods = analyzer.find_methods(source)
            matching_methods = [
                m
                for m in methods
                if m.name == func_name and (function.class_name is None or m.class_name == function.class_name)
            ]

            if matching_methods and target_overload_index < len(matching_methods):
                target_method = matching_methods[target_overload_index]
                logger.debug(
                    "Re-found target method at overload index %d (lines %d-%d after shift)",
                    target_overload_index,
                    target_method.start_line,
                    target_method.end_line,
                )
            else:
                logger.error(
                    "Lost target method %s after adding members (had index %d, found %d overloads)",
                    func_name,
                    target_overload_index,
                    len(matching_methods),
                )
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
    method_source = parsed.target_method_source
    new_source_lines = method_source.splitlines(keepends=True)
    indented_new_source = _apply_indentation(new_source_lines, indent)

    # Ensure the new source ends with a newline to avoid concatenation issues
    if indented_new_source and not indented_new_source.endswith("\n"):
        indented_new_source += "\n"

    # Build the result
    before = lines[: start_line - 1]  # Lines before the method
    after = lines[end_line:]  # Lines after the method

    return "".join(before) + indented_new_source + "".join(after)


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
    source: str, function: FunctionToOptimize, new_body: str, analyzer: JavaAnalyzer | None = None
) -> str:
    """Replace just the body of a method, preserving signature.

    Args:
        source: Original source code.
        function: FunctionToOptimize identifying the function.
        new_body: New method body (code between braces).
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Modified source code.

    """
    analyzer = analyzer or get_java_analyzer()
    source_bytes = source.encode("utf8")

    func_name = function.function_name

    # Find the method
    methods = analyzer.find_methods(source)
    target_method = None

    for method in methods:
        if method.name == func_name:
            if function.class_name is None or method.class_name == function.class_name:
                target_method = method
                break

    if not target_method:
        logger.error("Could not find method %s", func_name)
        return source

    # Find the body node
    body_node = target_method.node.child_by_field_name("body")
    if not body_node:
        logger.error("Method %s has no body (abstract?)", func_name)
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

    # Use single newline as separator
    separator = "\n"

    return (before + separator.encode("utf8") + indented_method.encode("utf8") + after).decode("utf8")


def remove_method(source: str, function: FunctionToOptimize, analyzer: JavaAnalyzer | None = None) -> str:
    """Remove a method from source code.

    Args:
        source: The source code.
        function: FunctionToOptimize identifying the method to remove.
        analyzer: Optional JavaAnalyzer instance.

    Returns:
        Source code with method removed.

    """
    analyzer = analyzer or get_java_analyzer()

    func_name = function.function_name

    # Find the method
    methods = analyzer.find_methods(source)
    target_method = None

    for method in methods:
        if method.name == func_name:
            if function.class_name is None or method.class_name == function.class_name:
                target_method = method
                break

    if not target_method:
        logger.error("Could not find method %s", func_name)
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
    test_source: str, functions_to_remove: list[str], analyzer: JavaAnalyzer | None = None
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
    methods_to_remove = [m for m in methods if m.name in functions_to_remove]
    methods_to_remove.sort(key=lambda m: m.start_line, reverse=True)

    result = test_source

    for method in methods_to_remove:
        # Create a FunctionToOptimize for removal
        func_info = FunctionToOptimize(
            function_name=method.name,
            file_path=Path("temp.java"),
            starting_line=method.start_line,
            ending_line=method.end_line,
            parents=[],
            is_method=True,
            language="java",
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
            summary_lines.append(f"// {inv_id}: {original_ms:.3f}ms -> {optimized_ms:.3f}ms ({speedup:.1f}% faster)")

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
