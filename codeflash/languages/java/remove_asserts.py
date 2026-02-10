"""Java assertion removal transformer for converting tests to regression tests.

This module removes assertion statements from Java test code while preserving
function calls, enabling behavioral verification by comparing outputs between
original and optimized code.

Supported frameworks:
- JUnit 5 (Jupiter): assertEquals, assertTrue, assertThrows, etc.
- JUnit 4: org.junit.Assert.*
- AssertJ: assertThat(...).isEqualTo(...)
- TestNG: org.testng.Assert.*
- Hamcrest: assertThat(actual, is(expected))
- Truth: assertThat(actual).isEqualTo(expected)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeflash.languages.java.parser import get_java_analyzer

if TYPE_CHECKING:
    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.languages.java.parser import JavaAnalyzer

logger = logging.getLogger(__name__)


# JUnit 5 assertion methods that take (expected, actual, ...) or (actual, ...)
JUNIT5_VALUE_ASSERTIONS = frozenset(
    {
        "assertEquals",
        "assertNotEquals",
        "assertSame",
        "assertNotSame",
        "assertArrayEquals",
        "assertIterableEquals",
        "assertLinesMatch",
    }
)

# JUnit 5 assertions that take a single boolean/object argument
JUNIT5_CONDITION_ASSERTIONS = frozenset({"assertTrue", "assertFalse", "assertNull", "assertNotNull"})

# JUnit 5 assertions that handle exceptions (need special treatment)
JUNIT5_EXCEPTION_ASSERTIONS = frozenset({"assertThrows", "assertDoesNotThrow"})

# JUnit 5 timeout assertions
JUNIT5_TIMEOUT_ASSERTIONS = frozenset({"assertTimeout", "assertTimeoutPreemptively"})

# JUnit 5 grouping assertion
JUNIT5_GROUP_ASSERTIONS = frozenset({"assertAll"})

# All JUnit 5 assertions
JUNIT5_ALL_ASSERTIONS = (
    JUNIT5_VALUE_ASSERTIONS
    | JUNIT5_CONDITION_ASSERTIONS
    | JUNIT5_EXCEPTION_ASSERTIONS
    | JUNIT5_TIMEOUT_ASSERTIONS
    | JUNIT5_GROUP_ASSERTIONS
)

# AssertJ terminal assertions (methods that end the chain)
ASSERTJ_TERMINAL_METHODS = frozenset(
    {
        "isEqualTo",
        "isNotEqualTo",
        "isSameAs",
        "isNotSameAs",
        "isNull",
        "isNotNull",
        "isTrue",
        "isFalse",
        "isEmpty",
        "isNotEmpty",
        "isBlank",
        "isNotBlank",
        "contains",
        "containsOnly",
        "containsExactly",
        "containsExactlyInAnyOrder",
        "doesNotContain",
        "startsWith",
        "endsWith",
        "matches",
        "hasSize",
        "hasSizeBetween",
        "hasSizeGreaterThan",
        "hasSizeLessThan",
        "isGreaterThan",
        "isGreaterThanOrEqualTo",
        "isLessThan",
        "isLessThanOrEqualTo",
        "isBetween",
        "isCloseTo",
        "isPositive",
        "isNegative",
        "isZero",
        "isNotZero",
        "isInstanceOf",
        "isNotInstanceOf",
        "isIn",
        "isNotIn",
        "containsKey",
        "containsKeys",
        "containsValue",
        "containsValues",
        "containsEntry",
        "hasFieldOrPropertyWithValue",
        "extracting",
        "satisfies",
        "doesNotThrow",
    }
)

# Hamcrest matcher methods
HAMCREST_MATCHERS = frozenset(
    {
        "is",
        "equalTo",
        "not",
        "nullValue",
        "notNullValue",
        "hasItem",
        "hasItems",
        "hasSize",
        "containsString",
        "startsWith",
        "endsWith",
        "greaterThan",
        "lessThan",
        "closeTo",
        "instanceOf",
        "anything",
        "allOf",
        "anyOf",
    }
)


@dataclass
class TargetCall:
    """Represents a method call that should be captured."""

    receiver: str | None  # 'calc', 'algorithms' (None for static)
    method_name: str
    arguments: str
    full_call: str  # 'calc.fibonacci(10)'
    start_pos: int
    end_pos: int


@dataclass
class AssertionMatch:
    """Represents a matched assertion statement."""

    start_pos: int
    end_pos: int
    statement_type: str  # 'junit5', 'assertj', 'junit4', 'testng', 'hamcrest'
    assertion_method: str
    target_calls: list[TargetCall] = field(default_factory=list)
    leading_whitespace: str = ""
    original_text: str = ""
    is_exception_assertion: bool = False
    lambda_body: str | None = None  # For assertThrows lambda content
    assigned_var_type: str | None = None  # For Type var = assertThrows(...)
    assigned_var_name: str | None = None


class JavaAssertTransformer:
    """Transforms Java test code by removing assertions and preserving function calls.

    This class uses tree-sitter for AST-based analysis and regex for text manipulation.
    It handles various Java testing frameworks including JUnit 5, JUnit 4, AssertJ,
    TestNG, Hamcrest, and Truth.
    """

    def __init__(
        self, function_name: str, qualified_name: str | None = None, analyzer: JavaAnalyzer | None = None
    ) -> None:
        self.analyzer = analyzer or get_java_analyzer()
        self.func_name = function_name
        self.qualified_name = qualified_name or function_name
        self.invocation_counter = 0
        self._detected_framework: str | None = None

    def transform(self, source: str) -> str:
        """Remove assertions from source code, preserving target function calls.

        Args:
            source: Java source code containing test assertions.

        Returns:
            Transformed source with assertions replaced by captured function calls.

        """
        if not source or not source.strip():
            return source

        # Detect framework from imports
        self._detected_framework = self._detect_framework(source)

        # Find all assertion statements
        assertions = self._find_assertions(source)

        if not assertions:
            return source

        # Filter to only assertions that contain target calls
        assertions_with_targets = [a for a in assertions if a.target_calls or a.is_exception_assertion]

        if not assertions_with_targets:
            return source

        # Sort by position (forward order) to assign counter numbers in source order
        assertions_with_targets.sort(key=lambda a: a.start_pos)

        # Filter out nested assertions (e.g., assertEquals inside assertAll)
        # An assertion is nested if it's completely contained within another assertion
        non_nested: list[AssertionMatch] = []
        for i, assertion in enumerate(assertions_with_targets):
            is_nested = False
            for j, other in enumerate(assertions_with_targets):
                if i != j:
                    # Check if 'assertion' is nested inside 'other'
                    if other.start_pos <= assertion.start_pos and assertion.end_pos <= other.end_pos:
                        is_nested = True
                        break
            if not is_nested:
                non_nested.append(assertion)

        assertions_with_targets = non_nested

        # Pre-compute all replacements with correct counter values
        replacements: list[tuple[int, int, str]] = []
        for assertion in assertions_with_targets:
            replacement = self._generate_replacement(assertion)
            replacements.append((assertion.start_pos, assertion.end_pos, replacement))

        # Apply replacements in reverse order to preserve positions
        result = source
        for start_pos, end_pos, replacement in reversed(replacements):
            result = result[:start_pos] + replacement + result[end_pos:]

        return result

    def _detect_framework(self, source: str) -> str:
        """Detect which testing framework is being used from imports.

        Checks more specific frameworks first (AssertJ, Hamcrest) before
        falling back to generic JUnit.
        """
        imports = self.analyzer.find_imports(source)

        # First pass: check for specific assertion libraries
        for imp in imports:
            path = imp.import_path.lower()
            if "org.assertj" in path:
                return "assertj"
            if "org.hamcrest" in path:
                return "hamcrest"
            if "com.google.common.truth" in path:
                return "truth"
            if "org.testng" in path:
                return "testng"

        # Second pass: check for JUnit versions
        for imp in imports:
            path = imp.import_path.lower()
            if "org.junit.jupiter" in path or "junit.jupiter" in path:
                return "junit5"
            if "org.junit" in path:
                return "junit4"

        # Default to JUnit 5 if no specific imports found
        return "junit5"

    def _find_assertions(self, source: str) -> list[AssertionMatch]:
        """Find all assertion statements in the source code."""
        assertions: list[AssertionMatch] = []

        # Find JUnit-style assertions
        assertions.extend(self._find_junit_assertions(source))

        # Find AssertJ/Truth-style fluent assertions
        assertions.extend(self._find_fluent_assertions(source))

        # Find Hamcrest assertions
        assertions.extend(self._find_hamcrest_assertions(source))

        return assertions

    def _find_junit_assertions(self, source: str) -> list[AssertionMatch]:
        """Find JUnit 4/5 and TestNG style assertions."""
        assertions: list[AssertionMatch] = []

        # Pattern for JUnit assertions: (Assert.|Assertions.)?assertXxx(...)
        # This handles both static imports and qualified calls:
        # - assertEquals (static import)
        # - Assert.assertEquals (JUnit 4)
        # - Assertions.assertEquals (JUnit 5)
        # - org.junit.jupiter.api.Assertions.assertEquals (fully qualified)
        all_assertions = "|".join(JUNIT5_ALL_ASSERTIONS)
        pattern = re.compile(
            rf"(\s*)((?:(?:\w+\.)*Assert(?:ions)?\.)?({all_assertions}))\s*\(", re.MULTILINE
        )

        for match in pattern.finditer(source):
            leading_ws = match.group(1)
            full_method = match.group(2)
            assertion_method = match.group(3)

            # Find the complete assertion statement (balanced parens)
            start_pos = match.start()
            paren_start = match.end() - 1  # Position of opening paren

            args_content, end_pos = self._find_balanced_parens(source, paren_start)
            if args_content is None:
                continue

            # Check for semicolon after closing paren
            while end_pos < len(source) and source[end_pos] in " \t\n\r":
                end_pos += 1
            if end_pos < len(source) and source[end_pos] == ";":
                end_pos += 1

            # Extract target calls from the arguments
            target_calls = self._extract_target_calls(args_content, match.end())
            is_exception = assertion_method in JUNIT5_EXCEPTION_ASSERTIONS

            # For exception assertions, extract the lambda body
            lambda_body = None
            if is_exception:
                lambda_body = self._extract_lambda_body(args_content)

            original_text = source[start_pos:end_pos]

            # Detect variable assignment: Type var = assertXxx(...)
            # This applies to all assertions (assertThrows, assertTimeout, etc.)
            assigned_var_type = None
            assigned_var_name = None

            before = source[:start_pos]
            last_nl_idx = before.rfind("\n")
            if last_nl_idx >= 0:
                line_prefix = source[last_nl_idx + 1 : start_pos]
            else:
                line_prefix = source[:start_pos]

            var_match = re.match(r"([ \t]*)(?:final\s+)?([\w.<>\[\]]+)\s+(\w+)\s*=\s*$", line_prefix)
            if var_match:
                if last_nl_idx >= 0:
                    start_pos = last_nl_idx
                    leading_ws = "\n" + var_match.group(1)
                else:
                    start_pos = 0
                    leading_ws = var_match.group(1)

                assigned_var_type = var_match.group(2)
                assigned_var_name = var_match.group(3)
                original_text = source[start_pos:end_pos]

            # Determine statement type based on detected framework
            detected = self._detected_framework or "junit5"
            if "jupiter" in detected or detected == "junit5":
                stmt_type = "junit5"
            else:
                stmt_type = detected

            assertions.append(
                AssertionMatch(
                    start_pos=start_pos,
                    end_pos=end_pos,
                    statement_type=stmt_type,
                    assertion_method=assertion_method,
                    target_calls=target_calls,
                    leading_whitespace=leading_ws,
                    original_text=original_text,
                    is_exception_assertion=is_exception,
                    lambda_body=lambda_body,
                    assigned_var_type=assigned_var_type,
                    assigned_var_name=assigned_var_name,
                )
            )

        return assertions

    def _find_fluent_assertions(self, source: str) -> list[AssertionMatch]:
        """Find AssertJ and Truth style fluent assertions (assertThat chains)."""
        assertions: list[AssertionMatch] = []

        # Pattern for fluent assertions: assertThat(...).<chain>
        # Handles both org.assertj and com.google.common.truth
        pattern = re.compile(r"(\s*)((?:Assertions?\.)?assertThat)\s*\(", re.MULTILINE)

        for match in pattern.finditer(source):
            leading_ws = match.group(1)
            start_pos = match.start()
            paren_start = match.end() - 1

            # Find assertThat(...) content
            args_content, after_paren = self._find_balanced_parens(source, paren_start)
            if args_content is None:
                continue

            # Find the assertion chain (e.g., .isEqualTo(5).hasSize(3))
            chain_end = self._find_fluent_chain_end(source, after_paren)
            if chain_end == after_paren:
                # No chain found, skip
                continue

            # Check for semicolon
            end_pos = chain_end
            while end_pos < len(source) and source[end_pos] in " \t\n\r":
                end_pos += 1
            if end_pos < len(source) and source[end_pos] == ";":
                end_pos += 1

            # Extract target calls from assertThat argument
            target_calls = self._extract_target_calls(args_content, match.end())
            original_text = source[start_pos:end_pos]

            # Determine statement type based on detected framework
            detected = self._detected_framework or "assertj"
            stmt_type = "assertj" if "assertj" in detected else "truth"

            assertions.append(
                AssertionMatch(
                    start_pos=start_pos,
                    end_pos=end_pos,
                    statement_type=stmt_type,
                    assertion_method="assertThat",
                    target_calls=target_calls,
                    leading_whitespace=leading_ws,
                    original_text=original_text,
                )
            )

        return assertions

    def _find_hamcrest_assertions(self, source: str) -> list[AssertionMatch]:
        """Find Hamcrest style assertions: assertThat(actual, matcher)."""
        assertions: list[AssertionMatch] = []

        if self._detected_framework != "hamcrest":
            return assertions

        # Pattern for Hamcrest: assertThat(actual, is(...)) or assertThat(reason, actual, matcher)
        pattern = re.compile(r"(\s*)((?:MatcherAssert\.)?assertThat)\s*\(", re.MULTILINE)

        for match in pattern.finditer(source):
            leading_ws = match.group(1)
            start_pos = match.start()
            paren_start = match.end() - 1

            args_content, end_pos = self._find_balanced_parens(source, paren_start)
            if args_content is None:
                continue

            # Check for semicolon
            while end_pos < len(source) and source[end_pos] in " \t\n\r":
                end_pos += 1
            if end_pos < len(source) and source[end_pos] == ";":
                end_pos += 1

            # For Hamcrest, the first arg (or second if reason given) is the actual value
            target_calls = self._extract_target_calls(args_content, match.end())
            original_text = source[start_pos:end_pos]

            assertions.append(
                AssertionMatch(
                    start_pos=start_pos,
                    end_pos=end_pos,
                    statement_type="hamcrest",
                    assertion_method="assertThat",
                    target_calls=target_calls,
                    leading_whitespace=leading_ws,
                    original_text=original_text,
                )
            )

        return assertions

    def _find_fluent_chain_end(self, source: str, start_pos: int) -> int:
        """Find the end of a fluent assertion chain."""
        pos = start_pos

        while pos < len(source):
            # Skip whitespace
            while pos < len(source) and source[pos] in " \t\n\r":
                pos += 1

            if pos >= len(source) or source[pos] != ".":
                break

            pos += 1  # Skip dot

            # Skip whitespace after dot
            while pos < len(source) and source[pos] in " \t\n\r":
                pos += 1

            # Read method name
            method_start = pos
            while pos < len(source) and (source[pos].isalnum() or source[pos] == "_"):
                pos += 1

            if pos == method_start:
                break

            method_name = source[method_start:pos]

            # Skip whitespace before potential parens
            while pos < len(source) and source[pos] in " \t\n\r":
                pos += 1

            # Check for parentheses
            if pos < len(source) and source[pos] == "(":
                _, new_pos = self._find_balanced_parens(source, pos)
                if new_pos == -1:
                    break
                pos = new_pos

            # Check if this is a terminal assertion method
            if method_name in ASSERTJ_TERMINAL_METHODS:
                # Continue looking for chained assertions
                continue

        return pos

    def _extract_target_calls(self, content: str, base_offset: int) -> list[TargetCall]:
        """Extract calls to the target function from assertion arguments."""
        target_calls: list[TargetCall] = []

        # Pattern to match method calls with various receiver styles:
        # - obj.method(args)
        # - ClassName.staticMethod(args)
        # - new ClassName().method(args)
        # - new ClassName(args).method(args)
        # - method(args) (no receiver)
        #
        # Strategy: Find the function name, then look backwards for the receiver
        pattern = re.compile(rf"({re.escape(self.func_name)})\s*\(", re.MULTILINE)

        for match in pattern.finditer(content):
            method_name = match.group(1)
            method_start = match.start()

            # Find the arguments
            paren_pos = match.end() - 1
            args_content, end_pos = self._find_balanced_parens(content, paren_pos)
            if args_content is None:
                continue

            # Look backwards from the method name to find the receiver
            receiver_start = method_start

            # Check if there's a dot before the method name (indicating a receiver)
            before_method = content[:method_start]
            stripped_before = before_method.rstrip()
            if stripped_before.endswith("."):
                dot_pos = len(stripped_before) - 1
                before_dot = content[:dot_pos]

                # Check for new ClassName() or new ClassName(args)
                stripped_before_dot = before_dot.rstrip()
                if stripped_before_dot.endswith(")"):
                    # Find matching opening paren for constructor args
                    close_paren_pos = len(stripped_before_dot) - 1
                    paren_depth = 1
                    i = close_paren_pos - 1
                    while i >= 0 and paren_depth > 0:
                        if stripped_before_dot[i] == ")":
                            paren_depth += 1
                        elif stripped_before_dot[i] == "(":
                            paren_depth -= 1
                        i -= 1
                    if paren_depth == 0:
                        open_paren_pos = i + 1
                        # Look for "new ClassName" before the opening paren
                        before_paren = stripped_before_dot[:open_paren_pos].rstrip()
                        new_match = re.search(r"new\s+[a-zA-Z_]\w*\s*$", before_paren)
                        if new_match:
                            receiver_start = new_match.start()
                        else:
                            # Could be chained call like something().method()
                            # For now, just use the part from open paren
                            receiver_start = open_paren_pos
                else:
                    # Simple identifier: obj.method() or Class.method() or pkg.Class.method()
                    ident_match = re.search(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*\s*$", stripped_before_dot)
                    if ident_match:
                        receiver_start = ident_match.start()

            full_call = content[receiver_start:end_pos]
            receiver = (
                content[receiver_start:method_start].rstrip(".").strip() if receiver_start < method_start else None
            )

            target_calls.append(
                TargetCall(
                    receiver=receiver,
                    method_name=method_name,
                    arguments=args_content,
                    full_call=full_call,
                    start_pos=base_offset + receiver_start,
                    end_pos=base_offset + end_pos,
                )
            )

        return target_calls

    def _extract_lambda_body(self, content: str) -> str | None:
        """Extract the body of a lambda expression from assertThrows arguments.

        For assertThrows(Exception.class, () -> code()), we want to extract 'code()'.
        For assertThrows(Exception.class, () -> { code(); }), we want 'code();'.
        """
        # Look for lambda: () -> expr or () -> { block }
        lambda_match = re.search(r"\(\s*\)\s*->\s*", content)
        if not lambda_match:
            return None

        body_start = lambda_match.end()
        remaining = content[body_start:].strip()

        if remaining.startswith("{"):
            # Block lambda: () -> { code }
            _, block_end = self._find_balanced_braces(content, body_start + content[body_start:].index("{"))
            if block_end != -1:
                # Extract content inside braces
                brace_content = content[body_start + content[body_start:].index("{") + 1 : block_end - 1]
                return brace_content.strip()
        else:
            # Expression lambda: () -> expr
            # Find the end (before the closing paren of assertThrows, or comma at depth 0)
            depth = 0
            end = len(content)
            for i, ch in enumerate(content[body_start:]):
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    if depth == 0:
                        end = body_start + i
                        break
                    depth -= 1
                elif ch == "," and depth == 0:
                    end = body_start + i
                    break
            return content[body_start:end].strip()

        return None

    def _find_balanced_parens(self, code: str, open_paren_pos: int) -> tuple[str | None, int]:
        """Find content within balanced parentheses.

        Args:
            code: The source code.
            open_paren_pos: Position of the opening parenthesis.

        Returns:
            Tuple of (content inside parens, position after closing paren) or (None, -1).

        """
        if open_paren_pos >= len(code) or code[open_paren_pos] != "(":
            return None, -1

        depth = 1
        pos = open_paren_pos + 1
        in_string = False
        string_char = None
        in_char = False

        while pos < len(code) and depth > 0:
            char = code[pos]
            prev_char = code[pos - 1] if pos > 0 else ""

            # Handle character literals
            if char == "'" and not in_string and prev_char != "\\":
                in_char = not in_char
            # Handle string literals (double quotes)
            elif char == '"' and not in_char and prev_char != "\\":
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
            elif not in_string and not in_char:
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1

            pos += 1

        if depth != 0:
            return None, -1

        return code[open_paren_pos + 1 : pos - 1], pos

    def _find_balanced_braces(self, code: str, open_brace_pos: int) -> tuple[str | None, int]:
        """Find content within balanced braces."""
        if open_brace_pos >= len(code) or code[open_brace_pos] != "{":
            return None, -1

        depth = 1
        pos = open_brace_pos + 1
        in_string = False
        string_char = None
        in_char = False

        while pos < len(code) and depth > 0:
            char = code[pos]
            prev_char = code[pos - 1] if pos > 0 else ""

            if char == "'" and not in_string and prev_char != "\\":
                in_char = not in_char
            elif char == '"' and not in_char and prev_char != "\\":
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
            elif not in_string and not in_char:
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1

            pos += 1

        if depth != 0:
            return None, -1

        return code[open_brace_pos + 1 : pos - 1], pos

    def _generate_replacement(self, assertion: AssertionMatch) -> str:
        """Generate replacement code for an assertion.

        The replacement captures target function return values and removes assertions.

        Args:
            assertion: The assertion to replace.

        Returns:
            Replacement code string.

        """
        if assertion.is_exception_assertion:
            return self._generate_exception_replacement(assertion)

        if not assertion.target_calls:
            # No target calls found, just comment out the assertion
            return f"{assertion.leading_whitespace}// Removed assertion: no target calls found"

        # Generate capture statements for each target call
        replacements = []
        # For the first replacement, use the full leading whitespace
        # For subsequent ones, strip leading newlines to avoid extra blank lines
        base_indent = assertion.leading_whitespace.lstrip("\n\r")
        for i, call in enumerate(assertion.target_calls):
            self.invocation_counter += 1
            var_name = f"_cf_result{self.invocation_counter}"
            if i == 0:
                replacements.append(f"{assertion.leading_whitespace}Object {var_name} = {call.full_call};")
            else:
                replacements.append(f"{base_indent}Object {var_name} = {call.full_call};")

        return "\n".join(replacements)

    def _generate_exception_replacement(self, assertion: AssertionMatch) -> str:
        """Generate replacement for assertThrows/assertDoesNotThrow.

        Transforms:
            assertThrows(Exception.class, () -> calculator.divide(1, 0));
        To:
            try { calculator.divide(1, 0); } catch (Exception _cf_ignored1) {}

        When assigned to a variable:
            IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> calc.divide(1, 0));
        To:
            IllegalArgumentException ex = null;
            try { calc.divide(1, 0); } catch (IllegalArgumentException _cf_caught1) { ex = _cf_caught1; }

        """
        self.invocation_counter += 1
        counter = self.invocation_counter
        ws = assertion.leading_whitespace
        base_indent = ws.lstrip("\n\r")

        if assertion.lambda_body:
            code_to_run = assertion.lambda_body
            if not code_to_run.endswith(";"):
                code_to_run += ";"

            # Handle variable assignment: Type var = assertThrows(...)
            if assertion.assigned_var_name and assertion.assigned_var_type:
                var_type = assertion.assigned_var_type
                var_name = assertion.assigned_var_name
                if assertion.assertion_method == "assertDoesNotThrow":
                    if ";" not in assertion.lambda_body.strip():
                        return f"{ws}{var_type} {var_name} = {assertion.lambda_body.strip()};"
                    return f"{ws}{code_to_run}"
                return (
                    f"{ws}{var_type} {var_name} = null;\n"
                    f"{base_indent}try {{ {code_to_run} }} "
                    f"catch ({var_type} _cf_caught{counter}) {{ {var_name} = _cf_caught{counter}; }}"
                )

            return (
                f"{ws}try {{ {code_to_run} }} "
                f"catch (Exception _cf_ignored{counter}) {{}}"
            )

        # If no lambda body found, try to extract from target calls
        if assertion.target_calls:
            call = assertion.target_calls[0]
            return (
                f"{ws}try {{ {call.full_call}; }} "
                f"catch (Exception _cf_ignored{counter}) {{}}"
            )

        # Fallback: comment out the assertion
        return f"{ws}// Removed assertThrows: could not extract callable"


def transform_java_assertions(source: str, function_name: str, qualified_name: str | None = None) -> str:
    """Transform Java test code by removing assertions and capturing function calls.

    This is the main entry point for Java assertion transformation.

    Args:
        source: The Java test source code.
        function_name: Name of the function being tested.
        qualified_name: Optional fully qualified name of the function.

    Returns:
        Transformed source code with assertions replaced by capture statements.

    """
    transformer = JavaAssertTransformer(function_name=function_name, qualified_name=qualified_name)
    return transformer.transform(source)


def remove_assertions_from_test(source: str, target_function: FunctionToOptimize) -> str:
    """Remove assertions from test code for the given target function.

    This is a convenience wrapper around transform_java_assertions that
    takes a FunctionToOptimize object.

    Args:
        source: The Java test source code.
        target_function: The function being optimized.

    Returns:
        Transformed source code.

    """
    return transform_java_assertions(
        source=source, function_name=target_function.function_name, qualified_name=target_function.qualified_name
    )
