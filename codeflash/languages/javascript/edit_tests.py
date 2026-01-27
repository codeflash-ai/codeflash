"""JavaScript test editing utilities.

This module provides functionality for editing JavaScript/TypeScript test files,
including adding runtime comments and removing test functions.
"""

from __future__ import annotations

import re

from codeflash.cli_cmds.console import logger
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

    lines = source.split("\n")
    modified_lines = []

    # Build a lookup by FULL test name (including describe blocks) for suffix matching
    # The keys in original_runtimes look like: "full_test_name#/path/to/test#invocation_id"
    # where full_test_name includes describe blocks: "fibonacci Edge cases should return 0"
    timing_by_full_name: dict[str, tuple[int, int]] = {}
    for key in original_runtimes:
        if key in optimized_runtimes:
            # Extract test function name from the key (first part before #)
            parts = key.split("#")
            if parts:
                full_test_name = parts[0]
                logger.debug(f"[js-annotations] Found timing for full test name: '{full_test_name}'")
                if full_test_name not in timing_by_full_name:
                    timing_by_full_name[full_test_name] = (original_runtimes[key], optimized_runtimes[key])
                else:
                    # Sum up timings for same test
                    old_orig, old_opt = timing_by_full_name[full_test_name]
                    timing_by_full_name[full_test_name] = (
                        old_orig + original_runtimes[key],
                        old_opt + optimized_runtimes[key],
                    )

    logger.debug(f"[js-annotations] Built timing_by_full_name with {len(timing_by_full_name)} entries")

    def find_matching_test(test_description: str) -> str | None:
        """Find a timing key that ends with the given test description (suffix match).

        Timing keys are like: "fibonacci Edge cases should return 0"
        Source test names are like: "should return 0"
        We need to match by suffix because timing includes all describe block names.
        """
        # Try to match by finding a key that ends with the test description
        for full_name in timing_by_full_name:
            # Check if the full name ends with the test description (case-insensitive)
            if full_name.lower().endswith(test_description.lower()):
                logger.debug(f"[js-annotations] Suffix match: '{test_description}' matches '{full_name}'")
                return full_name
        return None

    # Track current test context
    current_test_name = None
    current_matched_full_name = None
    test_pattern = re.compile(r"(?:test|it)\s*\(\s*['\"]([^'\"]+)['\"]")
    # Match function calls that look like: funcName(args) or expect(funcName(args))
    func_call_pattern = re.compile(r"(?:expect\s*\(\s*)?(\w+)\s*\([^)]*\)")

    for line in lines:
        # Check if this line starts a new test
        test_match = test_pattern.search(line)
        if test_match:
            current_test_name = test_match.group(1)
            logger.debug(f"[js-annotations] Found test: '{current_test_name}'")
            # Find the matching full name from timing data using suffix match
            current_matched_full_name = find_matching_test(current_test_name)
            if current_matched_full_name:
                logger.debug(f"[js-annotations] Test '{current_test_name}' matched to '{current_matched_full_name}'")

        # Check if this line has a function call and we have timing for current test
        if current_matched_full_name and current_matched_full_name in timing_by_full_name:
            # Only add comment if line has a function call and doesn't already have a comment
            if func_call_pattern.search(line) and "//" not in line and "expect(" in line:
                orig_time, opt_time = timing_by_full_name[current_matched_full_name]
                comment = format_runtime_comment(orig_time, opt_time)
                logger.debug(f"[js-annotations] Adding comment to test '{current_test_name}': {comment}")
                # Add comment at end of line
                line = f"{line.rstrip()}  {comment}"
                # Clear timing so we only annotate first call in each test
                del timing_by_full_name[current_matched_full_name]
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
