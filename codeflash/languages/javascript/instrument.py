"""JavaScript test instrumentation for existing tests.

This module provides functionality to inject profiling code into existing JavaScript
test files, similar to Python's inject_profiling_into_existing_test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
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


@dataclass
class ExpectCallMatch:
    """Represents a matched expect(func(...)).toXXX() call."""

    start_pos: int
    end_pos: int
    leading_whitespace: str
    func_args: str
    assertion_chain: str
    has_trailing_semicolon: bool
    object_prefix: str = ""  # Object prefix like "calc." or "this." or ""


@dataclass
class StandaloneCallMatch:
    """Represents a matched standalone func(...) call."""

    start_pos: int
    end_pos: int
    leading_whitespace: str
    func_args: str
    prefix: str  # "await " or ""
    object_prefix: str  # Object prefix like "calc." or "this." or ""
    has_trailing_semicolon: bool


class StandaloneCallTransformer:
    """Transforms standalone func(...) calls in JavaScript test code.

    This class handles the transformation of standalone function calls that are NOT
    inside expect() wrappers. These calls need to be wrapped with codeflash.capture()
    or codeflash.capturePerf() for instrumentation.

    Examples:
    - await func(args) -> await codeflash.capturePerf('name', 'id', func, args)
    - func(args) -> codeflash.capturePerf('name', 'id', func, args)
    - const result = func(args) -> const result = codeflash.capturePerf(...)
    - arr.map(() => func(args)) -> arr.map(() => codeflash.capturePerf(..., func, args))
    - calc.fibonacci(n) -> codeflash.capturePerf('...', 'id', calc.fibonacci.bind(calc), n)

    """

    def __init__(self, func_name: str, qualified_name: str, capture_func: str) -> None:
        self.func_name = func_name
        self.qualified_name = qualified_name
        self.capture_func = capture_func
        self.invocation_counter = 0
        # Pattern to match func_name( with optional leading await and optional object prefix
        # Captures: (whitespace)(await )?(object.)*func_name(
        # We'll filter out expect() and codeflash. cases in the transform loop
        self._call_pattern = re.compile(rf"(\s*)(await\s+)?((?:\w+\.)*){re.escape(func_name)}\s*\(")

    def transform(self, code: str) -> str:
        """Transform all standalone calls in the code."""
        result: list[str] = []
        pos = 0

        while pos < len(code):
            match = self._call_pattern.search(code, pos)
            if not match:
                result.append(code[pos:])
                break

            match_start = match.start()

            # Check if this call is inside an expect() or already transformed
            if self._should_skip_match(code, match_start, match):
                result.append(code[pos : match.end()])
                pos = match.end()
                continue

            # Add everything before the match
            result.append(code[pos:match_start])

            # Try to parse the full standalone call
            standalone_match = self._parse_standalone_call(code, match)
            if standalone_match is None:
                # Couldn't parse, skip this match
                result.append(code[match_start : match.end()])
                pos = match.end()
                continue

            # Generate the transformed code
            self.invocation_counter += 1
            transformed = self._generate_transformed_call(standalone_match)
            result.append(transformed)
            pos = standalone_match.end_pos

        return "".join(result)

    def _should_skip_match(self, code: str, start: int, match: re.Match) -> bool:
        """Check if the match should be skipped (inside expect, already transformed, etc.)."""
        # Look backwards to check context
        lookback_start = max(0, start - 200)
        lookback = code[lookback_start:start]

        # Skip if already transformed with codeflash.capture
        if f"codeflash.{self.capture_func}(" in lookback[-60:]:
            return True

        # Skip if this is a function/method definition, not a call
        # Patterns to skip:
        # - ClassName.prototype.funcName = function(
        # - funcName = function(
        # - funcName: function(
        # - function funcName(
        # - funcName() { (method definition in class)
        near_context = lookback[-80:] if len(lookback) >= 80 else lookback

        # Skip prototype assignment: ClassName.prototype.funcName = function(
        if re.search(r"\.prototype\.\w+\s*=\s*function\s*$", near_context):
            return True

        # Skip function assignment: funcName = function(
        if re.search(rf"{re.escape(self.func_name)}\s*=\s*function\s*$", near_context):
            return True

        # Skip function declaration: function funcName(
        if re.search(rf"function\s+{re.escape(self.func_name)}\s*$", near_context):
            return True

        # Skip method definition in class body: funcName(params) { or async funcName(params) {
        # Check by looking at what comes after the closing paren
        # The match ends at the opening paren, so find the closing paren and check what follows
        close_paren_pos = self._find_matching_paren(code, match.end() - 1)
        if close_paren_pos != -1:
            # Check if followed by { (method definition) after optional whitespace
            after_close = code[close_paren_pos : close_paren_pos + 20].lstrip()
            if after_close.startswith("{"):
                # This is a method definition like "fibonacci(n) {"
                # But we still want to capture certain patterns like arrow functions
                # Check if there's no => before the {
                between = code[close_paren_pos : close_paren_pos + 20].strip()
                if not between.startswith("=>"):
                    return True

        # Skip if inside expect() - look for 'expect(' with unmatched parens
        # Find the last 'expect(' and check if it's still open
        expect_search_start = max(0, start - 100)
        expect_lookback = code[expect_search_start:start]

        # Find all expect( positions
        expect_pos = expect_lookback.rfind("expect(")
        if expect_pos != -1:
            # Count parens from expect( to our match position
            between = expect_lookback[expect_pos:]
            open_parens = between.count("(") - between.count(")")
            if open_parens > 0:
                # We're inside an unclosed expect()
                return True

        return False

    def _find_matching_paren(self, code: str, open_paren_pos: int) -> int:
        """Find the position of the closing paren for the given opening paren."""
        if open_paren_pos >= len(code) or code[open_paren_pos] != "(":
            return -1

        depth = 1
        pos = open_paren_pos + 1

        while pos < len(code) and depth > 0:
            if code[pos] == "(":
                depth += 1
            elif code[pos] == ")":
                depth -= 1
            pos += 1

        return pos if depth == 0 else -1

    def _parse_standalone_call(self, code: str, match: re.Match) -> StandaloneCallMatch | None:
        """Parse a complete standalone func(...) call."""
        leading_ws = match.group(1)
        prefix = match.group(2) or ""  # "await " or ""
        object_prefix = match.group(3) or ""  # Object prefix like "calc." or ""

        # If qualified_name is a standalone function (no dot), don't match method calls
        # e.g., if qualified_name="func", don't match "obj.func()" - only match "func()"
        if "." not in self.qualified_name and object_prefix:
            return None

        # Find the opening paren position
        match_text = match.group(0)
        paren_offset = match_text.rfind("(")
        open_paren_pos = match.start() + paren_offset

        # Find the arguments (content inside parens)
        func_args, close_pos = self._find_balanced_parens(code, open_paren_pos)
        if func_args is None:
            return None

        # Check for trailing semicolon
        end_pos = close_pos
        # Skip whitespace
        while end_pos < len(code) and code[end_pos] in " \t":
            end_pos += 1

        has_trailing_semicolon = end_pos < len(code) and code[end_pos] == ";"
        if has_trailing_semicolon:
            end_pos += 1

        return StandaloneCallMatch(
            start_pos=match.start(),
            end_pos=end_pos,
            leading_whitespace=leading_ws,
            func_args=func_args,
            prefix=prefix,
            object_prefix=object_prefix,
            has_trailing_semicolon=has_trailing_semicolon,
        )

    def _find_balanced_parens(self, code: str, open_paren_pos: int) -> tuple[str | None, int]:
        """Find content within balanced parentheses."""
        if open_paren_pos >= len(code) or code[open_paren_pos] != "(":
            return None, -1

        depth = 1
        pos = open_paren_pos + 1
        in_string = False
        string_char = None

        while pos < len(code) and depth > 0:
            char = code[pos]

            # Handle string literals
            if char in "\"'`" and (pos == 0 or code[pos - 1] != "\\"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
            elif not in_string:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1

            pos += 1

        if depth != 0:
            return None, -1

        return code[open_paren_pos + 1 : pos - 1], pos

    def _generate_transformed_call(self, match: StandaloneCallMatch) -> str:
        """Generate the transformed code for a standalone call."""
        line_id = str(self.invocation_counter)
        args_str = match.func_args.strip()
        semicolon = ";" if match.has_trailing_semicolon else ""

        # Handle method calls on objects (e.g., calc.fibonacci, this.method)
        if match.object_prefix:
            # Remove trailing dot from object prefix for the bind call
            obj = match.object_prefix.rstrip(".")
            full_method = f"{obj}.{self.func_name}"

            if args_str:
                return (
                    f"{match.leading_whitespace}{match.prefix}codeflash.{self.capture_func}('{self.qualified_name}', "
                    f"'{line_id}', {full_method}.bind({obj}), {args_str}){semicolon}"
                )
            return (
                f"{match.leading_whitespace}{match.prefix}codeflash.{self.capture_func}('{self.qualified_name}', "
                f"'{line_id}', {full_method}.bind({obj})){semicolon}"
            )

        # Handle standalone function calls
        if args_str:
            return (
                f"{match.leading_whitespace}{match.prefix}codeflash.{self.capture_func}('{self.qualified_name}', "
                f"'{line_id}', {self.func_name}, {args_str}){semicolon}"
            )
        return (
            f"{match.leading_whitespace}{match.prefix}codeflash.{self.capture_func}('{self.qualified_name}', "
            f"'{line_id}', {self.func_name}){semicolon}"
        )


def transform_standalone_calls(
    code: str, func_name: str, qualified_name: str, capture_func: str, start_counter: int = 0
) -> tuple[str, int]:
    """Transform standalone func(...) calls in JavaScript test code.

    This transforms function calls that are NOT inside expect() wrappers.

    Args:
        code: The test code to transform.
        func_name: Name of the function being tested.
        qualified_name: Fully qualified function name.
        capture_func: The capture function to use ('capture' or 'capturePerf').
        start_counter: Starting value for the invocation counter.

    Returns:
        Tuple of (transformed code, final counter value).

    """
    transformer = StandaloneCallTransformer(
        func_name=func_name, qualified_name=qualified_name, capture_func=capture_func
    )
    transformer.invocation_counter = start_counter
    result = transformer.transform(code)
    return result, transformer.invocation_counter


class ExpectCallTransformer:
    """Transforms expect(func(...)).assertion() calls in JavaScript test code.

    This class handles the parsing and transformation of Jest/Vitest expect calls,
    supporting various assertion patterns including:
    - Basic: expect(func(args)).toBe(value)
    - Negated: expect(func(args)).not.toBe(value)
    - Async: expect(func(args)).resolves.toBe(value)
    - Chained: expect(func(args)).not.resolves.toBe(value)
    - No-arg assertions: expect(func(args)).toBeTruthy()
    - Multi-arg assertions: expect(func(args)).toBeCloseTo(0.5, 2)
    """

    def __init__(self, func_name: str, qualified_name: str, capture_func: str, remove_assertions: bool = False) -> None:
        self.func_name = func_name
        self.qualified_name = qualified_name
        self.capture_func = capture_func
        self.remove_assertions = remove_assertions
        self.invocation_counter = 0
        # Pattern to match start of expect((object.)*func_name(
        # Captures: (whitespace), (object prefix like calc. or this.)
        self._expect_pattern = re.compile(rf"(\s*)expect\s*\(\s*((?:\w+\.)*){re.escape(func_name)}\s*\(")

    def transform(self, code: str) -> str:
        """Transform all expect calls in the code."""
        result: list[str] = []
        pos = 0

        while pos < len(code):
            match = self._expect_pattern.search(code, pos)
            if not match:
                result.append(code[pos:])
                break

            # Add everything before the match
            result.append(code[pos : match.start()])

            # Try to parse the full expect call
            expect_match = self._parse_expect_call(code, match)
            if expect_match is None:
                # Couldn't parse, skip this match
                result.append(code[match.start() : match.end()])
                pos = match.end()
                continue

            # Generate the transformed code
            self.invocation_counter += 1
            transformed = self._generate_transformed_call(expect_match)
            result.append(transformed)
            pos = expect_match.end_pos

        return "".join(result)

    def _parse_expect_call(self, code: str, match: re.Match) -> ExpectCallMatch | None:
        """Parse a complete expect(func(...)).assertion() call.

        Returns None if the pattern doesn't match expected structure.
        """
        leading_ws = match.group(1)
        object_prefix = match.group(2) or ""  # Object prefix like "calc." or ""

        # If qualified_name is a standalone function (no dot), don't match method calls
        # e.g., if qualified_name="func", don't match "obj.func()" - only match "func()"
        if "." not in self.qualified_name and object_prefix:
            return None

        # Find the arguments of the function call (handling nested parens)
        args_start = match.end()
        func_args, func_close_pos = self._find_balanced_parens(code, args_start - 1)
        if func_args is None:
            return None

        # Skip whitespace and find closing ) of expect(
        expect_close_pos = func_close_pos
        while expect_close_pos < len(code) and code[expect_close_pos].isspace():
            expect_close_pos += 1

        if expect_close_pos >= len(code) or code[expect_close_pos] != ")":
            return None

        expect_close_pos += 1  # Move past )

        # Parse the assertion chain (e.g., .not.resolves.toBe(value))
        assertion_chain, chain_end_pos = self._parse_assertion_chain(code, expect_close_pos)
        if assertion_chain is None:
            return None

        # Check for trailing semicolon
        has_trailing_semicolon = chain_end_pos < len(code) and code[chain_end_pos] == ";"
        if has_trailing_semicolon:
            chain_end_pos += 1

        return ExpectCallMatch(
            start_pos=match.start(),
            end_pos=chain_end_pos,
            leading_whitespace=leading_ws,
            func_args=func_args,
            assertion_chain=assertion_chain,
            has_trailing_semicolon=has_trailing_semicolon,
            object_prefix=object_prefix,
        )

    def _find_balanced_parens(self, code: str, open_paren_pos: int) -> tuple[str | None, int]:
        """Find content within balanced parentheses.

        Args:
            code: The source code
            open_paren_pos: Position of the opening parenthesis

        Returns:
            Tuple of (content inside parens, position after closing paren) or (None, -1)

        """
        if open_paren_pos >= len(code) or code[open_paren_pos] != "(":
            return None, -1

        depth = 1
        pos = open_paren_pos + 1
        in_string = False
        string_char = None

        while pos < len(code) and depth > 0:
            char = code[pos]

            # Handle string literals
            if char in "\"'`" and (pos == 0 or code[pos - 1] != "\\"):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
            elif not in_string:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1

            pos += 1

        if depth != 0:
            return None, -1

        # Return content (excluding parens) and position after closing paren
        return code[open_paren_pos + 1 : pos - 1], pos

    def _parse_assertion_chain(self, code: str, start_pos: int) -> tuple[str | None, int]:
        """Parse assertion chain like .not.resolves.toBe(value).

        Handles:
        - .toBe(value)
        - .not.toBe(value)
        - .resolves.toBe(value)
        - .rejects.toThrow()
        - .not.resolves.toBe(value)
        - .toBeTruthy() (no args)
        - .toBeCloseTo(0.5, 2) (multiple args with nested parens)

        Returns:
            Tuple of (assertion chain string, end position) or (None, -1)

        """
        pos = start_pos
        chain_parts: list[str] = []

        # Skip any leading whitespace (for multi-line)
        while pos < len(code) and code[pos] in " \t\n\r":
            pos += 1

        # Must start with a dot
        if pos >= len(code) or code[pos] != ".":
            return None, -1

        while pos < len(code):
            # Skip whitespace between chain elements
            while pos < len(code) and code[pos] in " \t\n\r":
                pos += 1

            if pos >= len(code) or code[pos] != ".":
                break

            pos += 1  # Skip the dot

            # Skip whitespace after dot
            while pos < len(code) and code[pos] in " \t\n\r":
                pos += 1

            # Parse the method name
            method_start = pos
            while pos < len(code) and (code[pos].isalnum() or code[pos] == "_"):
                pos += 1

            if pos == method_start:
                return None, -1

            method_name = code[method_start:pos]

            # Skip whitespace before potential parens
            while pos < len(code) and code[pos] in " \t\n\r":
                pos += 1

            # Check for parentheses (method call)
            if pos < len(code) and code[pos] == "(":
                args_content, after_paren = self._find_balanced_parens(code, pos)
                if args_content is None:
                    return None, -1
                chain_parts.append(f".{method_name}({args_content})")
                pos = after_paren
            else:
                # Method without parens (like .not, .resolves, .rejects)
                # Or assertion without args like .toBeTruthy
                chain_parts.append(f".{method_name}")

            # If this is a terminal assertion (starts with 'to'), we're done
            if method_name.startswith("to"):
                break

        if not chain_parts:
            return None, -1

        # Verify we have a terminal assertion (should end with .toXXX)
        last_part = chain_parts[-1]
        if not last_part.startswith(".to"):
            return None, -1

        return "".join(chain_parts), pos

    def _generate_transformed_call(self, match: ExpectCallMatch) -> str:
        """Generate the transformed code for an expect call."""
        line_id = str(self.invocation_counter)
        args_str = match.func_args.strip()

        # Determine the function reference to use
        if match.object_prefix:
            # Method call on object: calc.fibonacci -> calc.fibonacci.bind(calc)
            obj = match.object_prefix.rstrip(".")
            func_ref = f"{obj}.{self.func_name}.bind({obj})"
        else:
            func_ref = self.func_name

        if self.remove_assertions:
            # For generated/regression tests: remove expect wrapper and assertion
            if args_str:
                return (
                    f"{match.leading_whitespace}codeflash.{self.capture_func}('{self.qualified_name}', "
                    f"'{line_id}', {func_ref}, {args_str});"
                )
            return (
                f"{match.leading_whitespace}codeflash.{self.capture_func}('{self.qualified_name}', "
                f"'{line_id}', {func_ref});"
            )

        # For existing tests: keep the expect wrapper
        semicolon = ";" if match.has_trailing_semicolon else ""
        if args_str:
            return (
                f"{match.leading_whitespace}expect(codeflash.{self.capture_func}('{self.qualified_name}', "
                f"'{line_id}', {func_ref}, {args_str})){match.assertion_chain}{semicolon}"
            )
        return (
            f"{match.leading_whitespace}expect(codeflash.{self.capture_func}('{self.qualified_name}', "
            f"'{line_id}', {func_ref})){match.assertion_chain}{semicolon}"
        )


def transform_expect_calls(
    code: str, func_name: str, qualified_name: str, capture_func: str, remove_assertions: bool = False
) -> tuple[str, int]:
    """Transform expect(func(...)).assertion() calls in JavaScript test code.

    This is the main entry point for expect call transformation.

    Args:
        code: The test code to transform.
        func_name: Name of the function being tested.
        qualified_name: Fully qualified function name.
        capture_func: The capture function to use ('capture' or 'capturePerf').
        remove_assertions: If True, remove assertions entirely (for generated tests).

    Returns:
        Tuple of (transformed code, final invocation counter value).

    """
    transformer = ExpectCallTransformer(
        func_name=func_name,
        qualified_name=qualified_name,
        capture_func=capture_func,
        remove_assertions=remove_assertions,
    )
    result = transformer.transform(code)
    return result, transformer.invocation_counter


def inject_profiling_into_existing_js_test(
    test_path: Path,
    call_positions: list[CodePosition],
    function_to_optimize: FunctionToOptimize,
    tests_project_root: Path,
    mode: str = TestingMode.BEHAVIOR,
) -> tuple[bool, str | None]:
    """Inject profiling code into an existing JavaScript test file.

    This function wraps function calls with codeflash.capture() or codeflash.capturePerf()
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
    """Check if a function is imported or used in the test code.

    This function handles both standalone functions and class methods.
    For class methods, it checks if the method is called on any object
    (e.g., calc.fibonacci, this.fibonacci).
    """
    # Check for CommonJS require with named export
    require_pattern = rf"(?:const|let|var)\s+\{{\s*[^}}]*\b{re.escape(func_name)}\b[^}}]*\}}\s*=\s*require\s*\("
    if re.search(require_pattern, code):
        return True

    # Check for ES6 import with named export
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

    # Check for method calls: obj.funcName( or this.funcName(
    # This handles class methods called on instances
    method_call_pattern = rf"\w+\.{re.escape(func_name)}\s*\("
    return bool(re.search(method_call_pattern, code))


def _instrument_js_test_code(
    code: str, func_name: str, test_file_path: str, mode: str, qualified_name: str, remove_assertions: bool = False
) -> str:
    """Instrument JavaScript test code with profiling capture calls.

    Args:
        code: Original test code.
        func_name: Name of the function to instrument.
        test_file_path: Relative path to test file.
        mode: Testing mode (behavior or performance).
        qualified_name: Fully qualified function name.
        remove_assertions: If True, remove expect assertions entirely (for generated/regression tests).
                          If False, keep the expect wrapper (for existing user-written tests).

    Returns:
        Instrumented code.

    """
    # Add codeflash helper import if not already present
    # Support both npm package (codeflash) and legacy local file (codeflash-jest-helper)
    has_codeflash_import = "codeflash" in code
    if not has_codeflash_import:
        # Detect module system: ESM uses "import ... from", CommonJS uses "require()"
        is_esm = bool(re.search(r"^\s*import\s+.+\s+from\s+['\"]", code, re.MULTILINE))

        if is_esm:
            # ESM: Use import statement at the top of the file (after any other imports)
            helper_import = "import codeflash from 'codeflash';\n"
            # Find the last import statement to add after
            import_matches = list(re.finditer(r"^import\s+.+\s+from\s+['\"][^'\"]+['\"]\s*;?\s*\n", code, re.MULTILINE))
            if import_matches:
                # Add after the last import
                last_import = import_matches[-1]
                insert_pos = last_import.end()
                code = code[:insert_pos] + helper_import + code[insert_pos:]
            else:
                # No imports found, add at beginning
                code = helper_import + "\n" + code
        else:
            # CommonJS: Use require statement
            helper_require = "const codeflash = require('codeflash');\n"
            # Find the first require statement to add after
            import_match = re.search(r"^((?:const|let|var)\s+.+?require\([^)]+\).*;?\s*\n)", code, re.MULTILINE)
            if import_match:
                insert_pos = import_match.end()
                code = code[:insert_pos] + helper_require + code[insert_pos:]
            else:
                # Add at the beginning if no requires found
                code = helper_require + "\n" + code

    # Choose capture function based on mode
    capture_func = "capturePerf" if mode == TestingMode.PERFORMANCE else "capture"

    # Transform expect calls using the refactored transformer
    code, expect_counter = transform_expect_calls(
        code=code,
        func_name=func_name,
        qualified_name=qualified_name,
        capture_func=capture_func,
        remove_assertions=remove_assertions,
    )

    # Transform standalone calls (not inside expect wrappers)
    # Continue counter from expect transformer to ensure unique IDs
    code, _final_counter = transform_standalone_calls(
        code=code,
        func_name=func_name,
        qualified_name=qualified_name,
        capture_func=capture_func,
        start_counter=expect_counter,
    )

    return code


def validate_and_fix_import_style(test_code: str, source_file_path: Path, function_name: str) -> str:
    """Validate and fix import style in generated test code to match source export.

    The AI may generate tests with incorrect import styles (e.g., using named import
    for a default export). This function detects such mismatches and fixes them.

    Args:
        test_code: The generated test code.
        source_file_path: Path to the source file being tested.
        function_name: Name of the function being tested.

    Returns:
        Fixed test code with correct import style.

    """
    from codeflash.languages.treesitter_utils import get_analyzer_for_file

    # Read source file to determine export style
    try:
        source_code = source_file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Could not read source file {source_file_path}: {e}")
        return test_code

    # Get analyzer for the source file
    try:
        analyzer = get_analyzer_for_file(source_file_path)
        exports = analyzer.find_exports(source_code)
    except Exception as e:
        logger.warning(f"Could not analyze exports in {source_file_path}: {e}")
        return test_code

    if not exports:
        return test_code

    # Determine how the function is exported
    is_default_export = False
    is_named_export = False

    for export in exports:
        if export.default_export == function_name:
            is_default_export = True
            break
        for name, _alias in export.exported_names:
            if name == function_name:
                is_named_export = True
                break
        if is_named_export:
            break

    # If we can't determine export style, don't modify
    if not is_default_export and not is_named_export:
        # Check if it might be a default export without name
        for export in exports:
            if export.default_export == "default":
                is_default_export = True
                break

    if not is_default_export and not is_named_export:
        return test_code

    # Find import statements in test code that import from the source file
    # Normalize path for matching
    source_name = source_file_path.stem
    source_patterns = [source_name, f"./{source_name}", f"../{source_name}", source_file_path.as_posix()]

    # Pattern for named import: const { funcName } = require(...) or import { funcName } from ...
    named_require_pattern = re.compile(
        rf"(const|let|var)\s+\{{\s*{re.escape(function_name)}\s*\}}\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
    )
    named_import_pattern = re.compile(rf"import\s+\{{\s*{re.escape(function_name)}\s*\}}\s+from\s+['\"]([^'\"]+)['\"]")

    # Pattern for default import: const funcName = require(...) or import funcName from ...
    default_require_pattern = re.compile(
        rf"(const|let|var)\s+{re.escape(function_name)}\s*=\s*require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"
    )
    default_import_pattern = re.compile(rf"import\s+{re.escape(function_name)}\s+from\s+['\"]([^'\"]+)['\"]")

    def is_relevant_import(module_path: str) -> bool:
        """Check if the module path refers to our source file."""
        # Normalize and compare
        module_name = Path(module_path).stem
        return any(p in module_path or module_name == source_name for p in source_patterns)

    # Check for mismatch and fix
    if is_default_export:
        # Function is default exported, but test uses named import - need to fix
        for match in named_require_pattern.finditer(test_code):
            module_path = match.group(2)
            if is_relevant_import(module_path):
                logger.debug(f"Fixing named require to default for {function_name} from {module_path}")
                old_import = match.group(0)
                new_import = f"{match.group(1)} {function_name} = require('{module_path}')"
                test_code = test_code.replace(old_import, new_import)

        for match in named_import_pattern.finditer(test_code):
            module_path = match.group(1)
            if is_relevant_import(module_path):
                logger.debug(f"Fixing named import to default for {function_name} from {module_path}")
                old_import = match.group(0)
                new_import = f"import {function_name} from '{module_path}'"
                test_code = test_code.replace(old_import, new_import)

    elif is_named_export:
        # Function is named exported, but test uses default import - need to fix
        for match in default_require_pattern.finditer(test_code):
            module_path = match.group(2)
            if is_relevant_import(module_path):
                logger.debug(f"Fixing default require to named for {function_name} from {module_path}")
                old_import = match.group(0)
                new_import = f"{match.group(1)} {{ {function_name} }} = require('{module_path}')"
                test_code = test_code.replace(old_import, new_import)

        for match in default_import_pattern.finditer(test_code):
            module_path = match.group(1)
            if is_relevant_import(module_path):
                logger.debug(f"Fixing default import to named for {function_name} from {module_path}")
                old_import = match.group(0)
                new_import = f"import {{ {function_name} }} from '{module_path}'"
                test_code = test_code.replace(old_import, new_import)

    return test_code


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


def instrument_generated_js_test(
    test_code: str, function_name: str, qualified_name: str, mode: str = TestingMode.BEHAVIOR
) -> str:
    """Instrument generated JavaScript/TypeScript test code.

    This function is used to instrument tests generated by the aiservice.
    Unlike inject_profiling_into_existing_js_test, this takes the test code
    as a string rather than reading from a file.

    For generated tests, we remove the expect() assertions entirely because:
    1. LLM-generated expected values may be incorrect
    2. These are treated as regression tests where correctness is verified
       by comparing outputs between original and optimized code

    Args:
        test_code: The generated test code to instrument.
        function_name: Name of the function being tested.
        qualified_name: Fully qualified function name (e.g., 'module.funcName').
        mode: Testing mode - "behavior" or "performance".

    Returns:
        Instrumented test code with assertions removed.

    """
    if not test_code or not test_code.strip():
        return test_code

    # Use the internal instrumentation function with assertion removal enabled
    # Generated tests are treated as regression tests, so we remove LLM-generated assertions
    return _instrument_js_test_code(
        code=test_code,
        func_name=function_name,
        test_file_path="generated_test",
        mode=mode,
        qualified_name=qualified_name,
        remove_assertions=True,
    )

