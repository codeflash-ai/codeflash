"""JavaScript test editing utilities.

This module provides functionality for editing JavaScript/TypeScript test files,
including adding runtime comments and removing test functions.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger

if TYPE_CHECKING:
    from pathlib import Path
from codeflash.code_utils.time_utils import format_perf, format_time
from codeflash.result.critic import performance_gain


def format_runtime_comment(original_time: int, optimized_time: int) -> str:
    """Format a runtime comparison comment for JavaScript.

    Args:
        original_time: Original runtime in nanoseconds.
        optimized_time: Optimized runtime in nanoseconds.

    Returns:
        Formatted comment string with // prefix.

    """
    perf_gain = format_perf(
        abs(performance_gain(original_runtime_ns=original_time, optimized_runtime_ns=optimized_time) * 100)
    )
    status = "slower" if optimized_time > original_time else "faster"
    return f"// {format_time(original_time)} -> {format_time(optimized_time)} ({perf_gain}% {status})"


def _build_timing_lookup(
    original_runtimes: dict[str, int],
    optimized_runtimes: dict[str, int],
    test_file_path: Path | None = None,
) -> dict[str, tuple[int, int]]:
    """Build a lookup table mapping test names to timing data.

    Supports matching by:
    1. Full test name (including describe blocks)
    2. Iteration ID based matching (like Python implementation)

    Args:
        original_runtimes: Map of invocation keys to original runtimes (ns).
        optimized_runtimes: Map of invocation keys to optimized runtimes (ns).
        test_file_path: Optional path to the test file for path-based matching.

    Returns:
        Dict mapping test names to (original_time, optimized_time) tuples.

    """
    timing_by_test: dict[str, tuple[int, int]] = {}
    timing_by_iteration: dict[str, tuple[int, int]] = {}

    for key in original_runtimes:
        if key not in optimized_runtimes:
            continue

        # Keys look like: "test_name#/path/to/test#iteration_id"
        parts = key.split("#")
        if len(parts) >= 1:
            test_name = parts[0]
            iteration_id = parts[-1] if len(parts) >= 3 else None

            orig_time = original_runtimes[key]
            opt_time = optimized_runtimes[key]

            # Store by full test name
            if test_name not in timing_by_test:
                timing_by_test[test_name] = (orig_time, opt_time)
            else:
                # Sum up timings for same test (multiple invocations)
                old_orig, old_opt = timing_by_test[test_name]
                timing_by_test[test_name] = (old_orig + orig_time, old_opt + opt_time)

            # Store by iteration ID for line-based matching
            if iteration_id:
                iteration_key = f"{test_name}#{iteration_id}"
                if iteration_key not in timing_by_iteration:
                    timing_by_iteration[iteration_key] = (orig_time, opt_time)
                else:
                    old_orig, old_opt = timing_by_iteration[iteration_key]
                    timing_by_iteration[iteration_key] = (old_orig + orig_time, old_opt + opt_time)

    logger.debug(f"[js-annotations] Built timing_by_test with {len(timing_by_test)} entries")
    logger.debug(f"[js-annotations] Built timing_by_iteration with {len(timing_by_iteration)} entries")

    return timing_by_test


def _find_matching_test(test_description: str, timing_by_test: dict[str, tuple[int, int]]) -> str | None:
    """Find a timing key that matches the given test description.

    Supports multiple matching strategies:
    1. Exact match
    2. Suffix match (for Jest tests with describe block prefixes)

    Args:
        test_description: The test name from the source code.
        timing_by_test: Dict of test names to timing data.

    Returns:
        The matched test name key, or None if no match found.

    """
    # Try exact match first
    if test_description in timing_by_test:
        return test_description

    # Try suffix match (timing keys may include describe block names)
    # e.g., timing key: "fibonacci Edge cases should return 0"
    # test description: "should return 0"
    for full_name in timing_by_test:
        if full_name.lower().endswith(test_description.lower()):
            logger.debug(f"[js-annotations] Suffix match: '{test_description}' matches '{full_name}'")
            return full_name

    # Try substring match as last resort
    test_desc_lower = test_description.lower()
    for full_name in timing_by_test:
        if test_desc_lower in full_name.lower():
            logger.debug(f"[js-annotations] Substring match: '{test_description}' in '{full_name}'")
            return full_name

    return None


def add_runtime_comments(source: str, original_runtimes: dict[str, int], optimized_runtimes: dict[str, int]) -> str:
    """Add runtime comments to JavaScript test source code.

    For JavaScript, we match timing data by test function name and add comments
    to expect() or function call lines.

    Args:
        source: JavaScript test source code.
        original_runtimes: Map of invocation keys to original runtimes (ns).
        optimized_runtimes: Map of invocation keys to optimized runtimes (ns).

    Returns:
        Source code with runtime comments added.

    """
    logger.debug(f"[js-annotations] original_runtimes has {len(original_runtimes)} entries")
    logger.debug(f"[js-annotations] optimized_runtimes has {len(optimized_runtimes)} entries")

    if not original_runtimes or not optimized_runtimes:
        logger.debug("[js-annotations] No runtimes available, returning unchanged source")
        return source

    # Build timing lookup
    timing_by_test = _build_timing_lookup(original_runtimes, optimized_runtimes)

    if not timing_by_test:
        logger.debug("[js-annotations] No matching timing data found")
        return source

    lines = source.split("\n")
    modified_lines = []

    # Track current test context
    current_test_name: str | None = None
    current_matched_full_name: str | None = None

    # Patterns for test identification
    test_pattern = re.compile(r"(?:test|it)\s*\(\s*['\"]([^'\"]+)['\"]")

    # Pattern for function calls - matches:
    # - expect(someFunc(...))
    # - const result = someFunc(...)
    # - someFunc(...)
    # But excludes common non-function patterns like describe(), it(), etc.
    func_call_pattern = re.compile(r"(?:expect\s*\(\s*)?(\w+)\s*\([^)]*\)")

    # Track lines with function calls inside tests
    for i, line in enumerate(lines):
        # Check if this line starts a new test
        test_match = test_pattern.search(line)
        if test_match:
            current_test_name = test_match.group(1)
            logger.debug(f"[js-annotations] Found test at line {i + 1}: '{current_test_name}'")
            # Find the matching full name from timing data
            current_matched_full_name = _find_matching_test(current_test_name, timing_by_test)
            if current_matched_full_name:
                logger.debug(f"[js-annotations] Test '{current_test_name}' matched to '{current_matched_full_name}'")

        # Check if this line has a function call and we have timing for current test
        if current_matched_full_name and current_matched_full_name in timing_by_test:
            # Check if line has a function call and doesn't already have a comment
            if func_call_pattern.search(line) and "//" not in line:
                # Verify it's not just a test framework call
                line_stripped = line.strip()
                if (
                    "expect(" in line
                    or ("=" in line and "(" in line)
                    or (
                        line_stripped.startswith(tuple("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
                        and "(" in line
                    )
                ) and not any(
                    kw in line_stripped for kw in ["describe(", "it(", "test(", "beforeEach(", "afterEach(", "import "]
                ):
                    orig_time, opt_time = timing_by_test[current_matched_full_name]
                    comment = format_runtime_comment(orig_time, opt_time)
                    logger.debug(f"[js-annotations] Adding comment at line {i + 1}: {comment}")
                    # Add comment at end of line
                    line = f"{line.rstrip()}  {comment}"
                    # Clear timing so we only annotate first call in each test
                    del timing_by_test[current_matched_full_name]
                    current_matched_full_name = None

        modified_lines.append(line)

    return "\n".join(modified_lines)


def remove_test_functions(source: str, functions_to_remove: list[str]) -> str:
    """Remove specific test functions from JavaScript test source code.

    Handles Jest test patterns: test(), it(), and describe() blocks.

    Args:
        source: JavaScript test source code.
        functions_to_remove: List of test function/describe names to remove.

    Returns:
        Source code with specified functions removed.

    """
    if not functions_to_remove:
        return source

    for func_name in functions_to_remove:
        # Pattern to match test('name', ...) or it('name', ...) blocks
        # This handles nested callbacks and multi-line test bodies
        test_pattern = re.compile(
            r"(?:test|it)\s*\(\s*['\"]" + re.escape(func_name) + r"['\"].*?\)\s*;?\s*\n?", re.DOTALL
        )

        # Try to find and remove matching test blocks
        # For more complex removal, we'd need to track brace matching
        match = test_pattern.search(source)
        if match:
            # Find the full test block by tracking braces
            start = match.start()
            end = _find_block_end(source, match.end() - 1)
            if end > start:
                source = source[:start] + source[end:]

    return source


def _find_block_end(source: str, start: int) -> int:
    """Find the end of a JavaScript block starting from a position.

    Tracks brace matching to find where a function/block ends.

    Args:
        source: Source code.
        start: Starting position (should be at or before opening brace).

    Returns:
        Position after the closing brace, or start if not found.

    """
    # Find the opening brace
    brace_pos = source.find("{", start)
    if brace_pos == -1:
        # No block found, try to find end of arrow function or simple statement
        semicolon_pos = source.find(";", start)
        newline_pos = source.find("\n", start)
        if semicolon_pos != -1:
            return semicolon_pos + 1
        if newline_pos != -1:
            return newline_pos + 1
        return start

    # Track brace depth
    depth = 0
    in_string = False
    string_char = None
    i = brace_pos

    while i < len(source):
        char = source[i]

        # Handle string literals
        if char in ('"', "'", "`") and (i == 0 or source[i - 1] != "\\"):
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char:
                in_string = False
                string_char = None
        elif not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    # Found the matching closing brace
                    # Skip any trailing semicolon or newline
                    end = i + 1
                    while end < len(source) and source[end] in " \t":
                        end += 1
                    if end < len(source) and source[end] == ";":
                        end += 1
                    while end < len(source) and source[end] in " \t\n":
                        end += 1
                    return end

        i += 1

    return start
