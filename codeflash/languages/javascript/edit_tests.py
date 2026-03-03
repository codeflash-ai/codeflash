"""JavaScript test editing utilities.

This module provides functionality for editing JavaScript/TypeScript test files,
including adding runtime comments and removing test functions.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.time_utils import format_runtime_comment
from codeflash.models.models import GeneratedTests, GeneratedTestsList


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
                comment = format_runtime_comment(orig_time, opt_time, comment_prefix="//")
                logger.debug(f"[js-annotations] Adding comment to test '{current_test_name}': {comment}")
                # Add comment at end of line
                line = f"{line.rstrip()}  {comment}"
                # Clear timing so we only annotate first call in each test
                del timing_by_full_name[current_matched_full_name]
                current_matched_full_name = None

        modified_lines.append(line)

    return "\n".join(modified_lines)


JS_TEST_EXTENSIONS = (
    ".test.ts",
    ".test.js",
    ".test.tsx",
    ".test.jsx",
    ".spec.ts",
    ".spec.js",
    ".spec.tsx",
    ".spec.jsx",
    ".ts",
    ".js",
    ".tsx",
    ".jsx",
    ".mjs",
    ".mts",
)


# TODO:{self} Needs cleanup for jest logic in else block
# Author: Sarthak Agarwal <sarthak.saga@gmail.com>
def is_js_test_module_path(test_module_path: str) -> bool:
    """Return True when the module path looks like a JS/TS test path."""
    return any(test_module_path.endswith(ext) for ext in JS_TEST_EXTENSIONS)


# Author: Sarthak Agarwal <sarthak.saga@gmail.com>
def resolve_js_test_module_path(test_module_path: str, tests_project_rootdir: Path) -> Path:
    """Resolve a JS/TS test module path to a concrete file path."""
    if "/" in test_module_path or "\\" in test_module_path:
        return tests_project_rootdir / Path(test_module_path)

    matched_ext = None
    for ext in JS_TEST_EXTENSIONS:
        if test_module_path.endswith(ext):
            matched_ext = ext
            break

    if matched_ext:
        base_path = test_module_path[: -len(matched_ext)]
        file_path = base_path.replace(".", os.sep) + matched_ext
        tests_dir_name = tests_project_rootdir.name
        if file_path.startswith((tests_dir_name + os.sep, tests_dir_name + "/")):
            return tests_project_rootdir.parent / Path(file_path)
        return tests_project_rootdir / Path(file_path)

    return tests_project_rootdir / Path(test_module_path)


# Patterns for normalizing codeflash imports (legacy -> npm package)
# Author: Sarthak Agarwal <sarthak.saga@gmail.com>
_CODEFLASH_REQUIRE_PATTERN = re.compile(
    r"(const|let|var)\s+(\w+)\s*=\s*require\s*\(\s*['\"]\.?/?codeflash-jest-helper['\"]\s*\)"
)
_CODEFLASH_IMPORT_PATTERN = re.compile(r"import\s+(?:\*\s+as\s+)?(\w+)\s+from\s+['\"]\.?/?codeflash-jest-helper['\"]")


# Author: Sarthak Agarwal <sarthak.saga@gmail.com>
def normalize_codeflash_imports(source: str) -> str:
    """Normalize codeflash imports to use the npm package.

    Replaces legacy local file imports:
        const codeflash = require('./codeflash-jest-helper')
        import codeflash from './codeflash-jest-helper'

    With npm package imports:
        const codeflash = require('codeflash')

    Args:
        source: JavaScript/TypeScript source code.

    Returns:
        Source code with normalized imports.

    """
    # Replace CommonJS require
    source = _CODEFLASH_REQUIRE_PATTERN.sub(r"\1 \2 = require('codeflash')", source)
    # Replace ES module import
    return _CODEFLASH_IMPORT_PATTERN.sub(r"import \1 from 'codeflash'", source)


# Author: ali <mohammed18200118@gmail.com>
def inject_test_globals(
    generated_tests: GeneratedTestsList, test_framework: str = "jest", module_system: str = "esm"
) -> GeneratedTestsList:
    # TODO: inside the prompt tell the llm if it should import jest functions or it's already injected in the global window
    """Inject test globals into all generated tests.

    Args:
        generated_tests: List of generated tests.
        test_framework: The test framework being used ("jest", "vitest", or "mocha").
        module_system: The module system ("esm" or "commonjs").

    Returns:
        Generated tests with test globals injected.

    """
    is_cjs = module_system == "commonjs"
    # Use vitest imports for vitest projects, jest imports for jest projects
    if test_framework == "vitest":
        global_import = "import { vi, describe, it, expect, beforeEach, afterEach, beforeAll, test } from 'vitest'\n"
    elif test_framework == "mocha":
        if is_cjs:
            global_import = "const assert = require('node:assert/strict');\n"
        else:
            global_import = "import assert from 'node:assert/strict';\n"
    else:
        # Default to jest imports for jest and other frameworks
        global_import = (
            "import { jest, describe, it, expect, beforeEach, afterEach, beforeAll, test } from '@jest/globals'\n"
        )

    for test in generated_tests.generated_tests:
        # Skip injection if the source already has the import (LLM may have included it)
        if global_import.strip() not in test.generated_original_test_source:
            test.generated_original_test_source = global_import + test.generated_original_test_source
        if global_import.strip() not in test.instrumented_behavior_test_source:
            test.instrumented_behavior_test_source = global_import + test.instrumented_behavior_test_source
        if global_import.strip() not in test.instrumented_perf_test_source:
            test.instrumented_perf_test_source = global_import + test.instrumented_perf_test_source
    return generated_tests


_VITEST_IMPORT_RE = re.compile(r"^.*import\s+\{[^}]*\}\s+from\s+['\"]vitest['\"].*\n?", re.MULTILINE)
_VITEST_REQUIRE_RE = re.compile(
    r"^.*(?:const|let|var)\s+\{[^}]*\}\s*=\s*require\s*\(\s*['\"]vitest['\"]\s*\).*\n?", re.MULTILINE
)
_JEST_GLOBALS_IMPORT_RE = re.compile(r"^.*import\s+\{[^}]*\}\s+from\s+['\"]@jest/globals['\"].*\n?", re.MULTILINE)
_JEST_GLOBALS_REQUIRE_RE = re.compile(
    r"^.*(?:const|let|var)\s+\{[^}]*\}\s*=\s*require\s*\(\s*['\"]@jest/globals['\"]\s*\).*\n?", re.MULTILINE
)
_MOCHA_REQUIRE_RE = re.compile(
    r"^.*(?:const|let|var)\s+\{[^}]*\}\s*=\s*require\s*\(\s*['\"]mocha['\"]\s*\).*\n?", re.MULTILINE
)
_VITEST_COMMENT_RE = re.compile(r"^.*//.*vitest imports.*\n?", re.MULTILINE | re.IGNORECASE)

# Chai import patterns — LLMs sometimes associate Mocha with Chai
_CHAI_IMPORT_RE = re.compile(r"^.*import\s+.*\s+from\s+['\"]chai['\"].*\n?", re.MULTILINE)
_CHAI_REQUIRE_RE = re.compile(r"^.*(?:const|let|var)\s+.*\s*=\s*require\s*\(\s*['\"]chai['\"]\s*\).*\n?", re.MULTILINE)

# Pattern to convert test() → it() — Mocha uses it(), not test()
_TEST_CALL_RE = re.compile(r"(\s*)test\s*\(")


def sanitize_mocha_imports(source: str) -> str:
    """Remove vitest/jest/mocha-require/chai imports from Mocha test source.

    The AI service sometimes generates vitest or jest-style imports when the
    framework is mocha. Mocha provides describe/it/before*/after* as globals,
    so these imports must be removed. Also removes ``require('mocha')``
    destructures since Mocha doesn't export those.

    Additionally converts ``test()`` calls to ``it()`` since Mocha only
    supports ``it()`` as its test function.

    Args:
        source: Generated test source code.

    Returns:
        Source with incorrect framework imports stripped and test() converted to it().

    """
    source = _VITEST_IMPORT_RE.sub("", source)
    source = _VITEST_REQUIRE_RE.sub("", source)
    source = _JEST_GLOBALS_IMPORT_RE.sub("", source)
    source = _JEST_GLOBALS_REQUIRE_RE.sub("", source)
    source = _MOCHA_REQUIRE_RE.sub("", source)
    source = _VITEST_COMMENT_RE.sub("", source)
    source = _CHAI_IMPORT_RE.sub("", source)
    source = _CHAI_REQUIRE_RE.sub("", source)
    source = _TEST_CALL_RE.sub(r"\1it(", source)
    return convert_expect_to_assert(source)


def _find_matching_paren(source: str, open_pos: int) -> int:
    """Find the position of the closing parenthesis matching the one at open_pos."""
    depth = 0
    in_string = False
    string_char = None
    i = open_pos
    while i < len(source):
        char = source[i]
        if char in ('"', "'", "`") and (i == 0 or source[i - 1] != "\\"):
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
                if depth == 0:
                    return i
        i += 1
    return -1


def convert_expect_to_assert(source: str) -> str:
    """Convert expect()-style assertions to node:assert/strict equivalents.

    LLMs frequently generate Chai-style (``expect(x).to.equal(y)``) or
    Jest-style (``expect(x).toBe(y)``) assertions for Mocha tests despite
    being instructed to use ``assert``.  This function converts the common
    patterns to their ``node:assert/strict`` equivalents so that
    instrumentation and Mocha execution work correctly.

    Any ``expect()`` calls that cannot be converted are commented out with
    ``// SKIPPED`` to prevent ``ReferenceError: expect is not defined``.

    Args:
        source: Test source code that may contain expect() calls.

    Returns:
        Source with expect() calls converted to assert equivalents.

    """
    if "expect(" not in source:
        return source

    lines = source.split("\n")
    converted: list[str] = []

    for line in lines:
        converted_line = _convert_expect_line(line)
        converted.append(converted_line)

    return "\n".join(converted)


# Patterns mapping (chain_suffix → conversion_type)
# "simple" = assert.func(actual, value), "ok_cmp" = assert.ok(actual OP value)
# "ok_method" = assert.ok(actual.method(value)), "type" = assert.ok(typeof actual === ...)
# "truthy" = assert.ok(actual) / assert.strictEqual(actual, bool)
# "throws" = assert.throws, "noop" = assert.ok(actual !== undefined)
_EXPECT_CHAIN_MAP: list[tuple[str, str, str | None]] = [
    # Jest patterns (most common)
    (".toBe(", "simple_strictEqual", None),
    (".toEqual(", "simple_deepStrictEqual", None),
    (".toStrictEqual(", "simple_deepStrictEqual", None),
    (".toBeGreaterThan(", "ok_gt", None),
    (".toBeGreaterThanOrEqual(", "ok_gte", None),
    (".toBeLessThan(", "ok_lt", None),
    (".toBeLessThanOrEqual(", "ok_lte", None),
    (".toContain(", "ok_includes", None),
    (".toHaveLength(", "ok_length", None),
    (".toBeNull(", "null_check", None),
    (".toBeUndefined(", "undef_check", None),
    (".toBeTruthy(", "truthy", None),
    (".toBeFalsy(", "falsy", None),
    (".toThrow(", "throws", None),
    (".toMatch(", "ok_match", None),
    # Chai .to. patterns
    (".to.equal(", "simple_strictEqual", None),
    (".to.eql(", "simple_deepStrictEqual", None),
    (".to.deep.equal(", "simple_deepStrictEqual", None),
    (".to.be.greaterThan(", "ok_gt", None),
    (".to.be.lessThan(", "ok_lt", None),
    (".to.be.above(", "ok_gt", None),
    (".to.be.below(", "ok_lt", None),
    (".to.be.at.least(", "ok_gte", None),
    (".to.be.at.most(", "ok_lte", None),
    (".to.include(", "ok_includes", None),
    (".to.contain(", "ok_includes", None),
    (".to.not.include(", "ok_not_includes", None),
    (".to.not.contain(", "ok_not_includes", None),
    (".to.have.length(", "ok_length", None),
    (".to.have.lengthOf(", "ok_length", None),
    (".to.throw(", "throws", None),
    (".to.match(", "ok_match", None),
    (".to.be.a(", "noop", None),
    (".to.be.an(", "noop", None),
    (".to.be.instanceOf(", "noop", None),
    (".to.be.instanceof(", "noop", None),
    (".to.exist", "truthy_no_arg", None),
    (".to.not.exist", "falsy_no_arg", None),
    (".to.be.true", "true_no_arg", None),
    (".to.be.false", "false_no_arg", None),
    (".to.be.null", "null_no_arg", None),
    (".to.be.undefined", "undef_no_arg", None),
    (".to.be.ok", "truthy_no_arg", None),
    (".to.not.be.ok", "falsy_no_arg", None),
]


def _convert_expect_line(line: str) -> str:
    """Convert a single line containing expect() to an assert equivalent."""
    stripped = line.lstrip()
    if "expect(" not in stripped:
        return line

    indent = line[: len(line) - len(stripped)]

    expect_idx = line.find("expect(")
    if expect_idx == -1:
        return line

    open_paren = expect_idx + len("expect")
    close_paren = _find_matching_paren(line, open_paren)
    if close_paren == -1:
        # Multi-line expect or malformed — comment out to prevent ReferenceError
        return f"{indent}// SKIPPED (unconvertible expect): {stripped}"

    actual_expr = line[open_paren + 1 : close_paren]
    rest = line[close_paren + 1 :].strip()
    trailing_semi = ";" if rest.endswith(";") else ""

    # Try each chain pattern
    for chain_prefix, conversion_type, _ in _EXPECT_CHAIN_MAP:
        if not rest.startswith(chain_prefix):
            continue

        # No-argument chains (e.g. .to.be.true, .to.exist)
        if conversion_type.endswith("_no_arg"):
            return _convert_no_arg(indent, actual_expr, conversion_type, trailing_semi)

        # Extract the argument inside the chain's parentheses
        chain_open = rest.find("(")
        if chain_open == -1:
            break
        chain_close = _find_matching_paren(rest, chain_open)
        if chain_close == -1:
            break
        value_expr = rest[chain_open + 1 : chain_close]

        return _convert_with_arg(indent, actual_expr, value_expr, conversion_type, trailing_semi)

    # Fallback: comment out unconvertible expect() to prevent ReferenceError
    return f"{indent}// SKIPPED (unconvertible expect): {stripped}"


def _convert_no_arg(indent: str, actual: str, conversion_type: str, semi: str) -> str:
    """Convert expect patterns that take no argument (e.g., .to.be.true)."""
    if conversion_type == "true_no_arg":
        return f"{indent}assert.strictEqual({actual}, true){semi}"
    if conversion_type == "false_no_arg":
        return f"{indent}assert.strictEqual({actual}, false){semi}"
    if conversion_type == "null_no_arg":
        return f"{indent}assert.strictEqual({actual}, null){semi}"
    if conversion_type == "undef_no_arg":
        return f"{indent}assert.strictEqual({actual}, undefined){semi}"
    if conversion_type == "truthy_no_arg":
        return f"{indent}assert.ok({actual}){semi}"
    if conversion_type == "falsy_no_arg":
        return f"{indent}assert.ok(!({actual})){semi}"
    return f"{indent}assert.ok({actual} !== undefined){semi}"


def _convert_with_arg(indent: str, actual: str, value: str, conversion_type: str, semi: str) -> str:
    """Convert expect patterns that take an argument."""
    if conversion_type == "simple_strictEqual":
        return f"{indent}assert.strictEqual({actual}, {value}){semi}"
    if conversion_type == "simple_deepStrictEqual":
        return f"{indent}assert.deepStrictEqual({actual}, {value}){semi}"
    if conversion_type == "ok_gt":
        return f"{indent}assert.ok(({actual}) > ({value})){semi}"
    if conversion_type == "ok_gte":
        return f"{indent}assert.ok(({actual}) >= ({value})){semi}"
    if conversion_type == "ok_lt":
        return f"{indent}assert.ok(({actual}) < ({value})){semi}"
    if conversion_type == "ok_lte":
        return f"{indent}assert.ok(({actual}) <= ({value})){semi}"
    if conversion_type == "ok_includes":
        return f"{indent}assert.ok(String({actual}).includes({value})){semi}"
    if conversion_type == "ok_not_includes":
        return f"{indent}assert.ok(!String({actual}).includes({value})){semi}"
    if conversion_type == "ok_length":
        return f"{indent}assert.strictEqual(({actual}).length, {value}){semi}"
    if conversion_type == "ok_match":
        return f"{indent}assert.match(String({actual}), {value}){semi}"
    if conversion_type == "null_check":
        return f"{indent}assert.strictEqual({actual}, null){semi}"
    if conversion_type == "undef_check":
        return f"{indent}assert.strictEqual({actual}, undefined){semi}"
    if conversion_type == "truthy":
        return f"{indent}assert.ok({actual}){semi}"
    if conversion_type == "falsy":
        return f"{indent}assert.ok(!({actual})){semi}"
    if conversion_type == "throws":
        if value:
            return f"{indent}assert.throws(() => {{ {actual}; }}, {value}){semi}"
        return f"{indent}assert.throws(() => {{ {actual}; }}){semi}"
    # noop: type checks like .to.be.a('string') — just verify defined
    if conversion_type == "noop":
        return f"{indent}assert.ok({actual} !== undefined){semi}"
    return f"{indent}assert.ok({actual} !== undefined){semi}"


# Author: ali <mohammed18200118@gmail.com>
def disable_ts_check(generated_tests: GeneratedTestsList) -> GeneratedTestsList:
    """Disable TypeScript type checking in all generated tests.

    Args:
        generated_tests: List of generated tests.

    Returns:
        Generated tests with TypeScript type checking disabled.

    """
    # we only inject test globals for esm modules
    ts_nocheck = "// @ts-nocheck\n"

    for test in generated_tests.generated_tests:
        test.generated_original_test_source = ts_nocheck + test.generated_original_test_source
        test.instrumented_behavior_test_source = ts_nocheck + test.instrumented_behavior_test_source
        test.instrumented_perf_test_source = ts_nocheck + test.instrumented_perf_test_source
    return generated_tests


# Author: Sarthak Agarwal <sarthak.saga@gmail.com>
def normalize_generated_tests_imports(generated_tests: GeneratedTestsList) -> GeneratedTestsList:
    """Normalize codeflash imports in all generated tests.

    Args:
        generated_tests: List of generated tests.

    Returns:
        Generated tests with normalized imports.

    """
    normalized_tests = []
    for test in generated_tests.generated_tests:
        # Only normalize JS/TS files
        if test.behavior_file_path.suffix in (".js", ".ts", ".jsx", ".tsx", ".mjs", ".mts"):
            normalized_test = GeneratedTests(
                generated_original_test_source=normalize_codeflash_imports(test.generated_original_test_source),
                instrumented_behavior_test_source=normalize_codeflash_imports(test.instrumented_behavior_test_source),
                instrumented_perf_test_source=normalize_codeflash_imports(test.instrumented_perf_test_source),
                behavior_file_path=test.behavior_file_path,
                perf_file_path=test.perf_file_path,
            )
            normalized_tests.append(normalized_test)
        else:
            normalized_tests.append(test)
    return GeneratedTestsList(generated_tests=normalized_tests)


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
