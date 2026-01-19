"""JavaScript test instrumentation for existing tests.

This module provides functionality to inject profiling code into existing JavaScript
test files, similar to Python's inject_profiling_into_existing_test.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger

if TYPE_CHECKING:
    from codeflash.code_utils.code_position import CodePosition
    from codeflash.discovery.functions_to_optimize import FunctionToOptimize


class TestingMode:
    """Testing mode constants."""

    BEHAVIOR = "behavior"
    PERFORMANCE = "performance"


def inject_profiling_into_existing_js_test(
    test_path: Path,
    call_positions: list[CodePosition],
    function_to_optimize: FunctionToOptimize,
    tests_project_root: Path,
    mode: str = TestingMode.BEHAVIOR,
) -> tuple[bool, str | None]:
    """Inject profiling code into an existing JavaScript test file.

    This function wraps function calls with codeflash.capture() or codeflash.capturePerfLooped()
    to enable behavioral verification and performance benchmarking.

    Args:
        test_path: Path to the test file.
        call_positions: List of code positions where the function is called.
        function_to_optimize: The function being optimized.
        tests_project_root: Root directory of tests.
        mode: Testing mode - "behavior" or "performance".

    Returns:
        Tuple of (success, instrumented_code).
    """
    try:
        with test_path.open(encoding="utf8") as f:
            test_code = f.read()
    except Exception as e:
        logger.error(f"Failed to read test file {test_path}: {e}")
        return False, None

    func_name = function_to_optimize.function_name

    # Get the relative path for test identification
    try:
        rel_path = test_path.relative_to(tests_project_root)
    except ValueError:
        rel_path = test_path

    # Check if the function is imported/required in this test file
    if not _is_function_used_in_test(test_code, func_name):
        logger.debug(f"Function '{func_name}' not found in test file {test_path}")
        return False, None

    # Instrument the test code
    instrumented_code = _instrument_js_test_code(
        test_code, func_name, str(rel_path), mode, function_to_optimize.qualified_name
    )

    if instrumented_code == test_code:
        logger.debug(f"No changes made to test file {test_path}")
        return False, None

    return True, instrumented_code


def _is_function_used_in_test(code: str, func_name: str) -> bool:
    """Check if a function is imported or used in the test code."""
    # Check for CommonJS require
    require_pattern = rf"(?:const|let|var)\s+\{{\s*[^}}]*\b{re.escape(func_name)}\b[^}}]*\}}\s*=\s*require\s*\("
    if re.search(require_pattern, code):
        return True

    # Check for ES6 import
    import_pattern = rf"import\s+\{{\s*[^}}]*\b{re.escape(func_name)}\b[^}}]*\}}\s+from"
    if re.search(import_pattern, code):
        return True

    # Check for default import (import func from or const func = require())
    default_require = rf"(?:const|let|var)\s+{re.escape(func_name)}\s*=\s*require\s*\("
    if re.search(default_require, code):
        return True

    default_import = rf"import\s+{re.escape(func_name)}\s+from"
    if re.search(default_import, code):
        return True

    return False


def _instrument_js_test_code(
    code: str,
    func_name: str,
    test_file_path: str,
    mode: str,
    qualified_name: str,
) -> str:
    """Instrument JavaScript test code with profiling capture calls.

    Args:
        code: Original test code.
        func_name: Name of the function to instrument.
        test_file_path: Relative path to test file.
        mode: Testing mode (behavior or performance).
        qualified_name: Fully qualified function name.

    Returns:
        Instrumented code.
    """
    # Add codeflash helper require if not already present
    if "codeflash-jest-helper" not in code:
        # Find the first require/import statement to add after
        import_match = re.search(
            r"^((?:const|let|var|import)\s+.+?(?:require\([^)]+\)|from\s+['\"][^'\"]+['\"]).*;?\s*\n)",
            code,
            re.MULTILINE,
        )
        if import_match:
            insert_pos = import_match.end()
            helper_require = "const codeflash = require('./codeflash-jest-helper');\n"
            code = code[:insert_pos] + helper_require + code[insert_pos:]
        else:
            # Add at the beginning if no imports found
            code = "const codeflash = require('./codeflash-jest-helper');\n\n" + code

    # Choose capture function based on mode
    capture_func = "capturePerfLooped" if mode == TestingMode.PERFORMANCE else "capture"

    # Track invocations for unique IDs
    invocation_counter = [0]

    def get_test_context(code_before: str) -> str:
        """Extract the current test name from preceding code."""
        # Look for ALL test('name', ...) or it('name', ...) patterns and get the LAST one
        # This ensures we get the most recent test context
        test_matches = list(re.finditer(r"(?:test|it)\s*\(\s*['\"]([^'\"]+)['\"]", code_before))
        if test_matches:
            last_match = test_matches[-1]
            return last_match.group(1).replace(" ", "_").replace("'", "")[:50]
        return "test"

    # Use a function-based approach to handle nested parentheses properly
    # The simple regex [^)]* fails for nested parens like func(getN(5))

    def find_and_replace_expect_calls(code: str) -> str:
        """Find expect(func(...)) patterns and replace them, handling nested parens."""
        result = []
        i = 0
        pattern = re.compile(rf"(\s*)expect\s*\(\s*{re.escape(func_name)}\s*\(")

        while i < len(code):
            match = pattern.search(code, i)
            if not match:
                result.append(code[i:])
                break

            # Add everything before the match
            result.append(code[i:match.start()])

            leading_ws = match.group(1)

            # Find the matching closing paren of func_name(
            func_call_start = match.end() - 1  # Position of ( after func_name
            args_start = match.end()

            # Count parentheses to find matching close
            paren_depth = 1
            pos = args_start
            while pos < len(code) and paren_depth > 0:
                if code[pos] == '(':
                    paren_depth += 1
                elif code[pos] == ')':
                    paren_depth -= 1
                pos += 1

            if paren_depth != 0:
                # Unmatched parens, skip this match
                result.append(code[match.start():match.end()])
                i = match.end()
                continue

            # pos is now right after the closing ) of func_name(args)
            args = code[args_start:pos - 1]

            # Skip whitespace and find the closing ) of expect(
            expect_close_pos = pos
            while expect_close_pos < len(code) and code[expect_close_pos].isspace():
                expect_close_pos += 1

            if expect_close_pos >= len(code) or code[expect_close_pos] != ')':
                # No closing ) for expect, skip
                result.append(code[match.start():match.end()])
                i = match.end()
                continue

            expect_close_pos += 1  # Move past )

            # Now look for .toXXX(value)
            assertion_match = re.match(r'(\.to\w+\([^)]*\))', code[expect_close_pos:])
            if not assertion_match:
                # No assertion found, skip
                result.append(code[match.start():match.end()])
                i = match.end()
                continue

            assertion = assertion_match.group(1)
            end_pos = expect_close_pos + assertion_match.end()

            # Generate the wrapped call
            # The capture function signature is: capture(funcName, lineId, fn, ...args)
            invocation_counter[0] += 1
            line_id = str(invocation_counter[0])

            # Build args list - need to handle empty args case
            args_str = args.strip()
            if args_str:
                wrapped = (
                    f"{leading_ws}expect(codeflash.{capture_func}('{qualified_name}', "
                    f"'{line_id}', {func_name}, {args_str})){assertion}"
                )
            else:
                wrapped = (
                    f"{leading_ws}expect(codeflash.{capture_func}('{qualified_name}', "
                    f"'{line_id}', {func_name})){assertion}"
                )

            result.append(wrapped)
            i = end_pos

        return ''.join(result)

    code = find_and_replace_expect_calls(code)

    return code


def get_instrumented_test_path(original_path: Path, mode: str) -> Path:
    """Generate path for instrumented test file.

    Args:
        original_path: Original test file path.
        mode: Testing mode (behavior or performance).

    Returns:
        Path for instrumented file.
    """
    suffix = "_codeflash_behavior" if mode == TestingMode.BEHAVIOR else "_codeflash_perf"
    stem = original_path.stem
    # Handle .test.js -> .test_codeflash_behavior.js
    if ".test" in stem:
        parts = stem.rsplit(".test", 1)
        new_stem = f"{parts[0]}{suffix}.test"
    elif ".spec" in stem:
        parts = stem.rsplit(".spec", 1)
        new_stem = f"{parts[0]}{suffix}.spec"
    else:
        new_stem = f"{stem}{suffix}"

    return original_path.parent / f"{new_stem}{original_path.suffix}"