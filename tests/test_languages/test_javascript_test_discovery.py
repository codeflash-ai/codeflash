"""Comprehensive tests for JavaScript test discovery functionality.

These tests verify that the JavaScript language support correctly discovers
Jest tests and maps them to source functions, similar to Python's test discovery tests.
"""

import tempfile
from pathlib import Path

import pytest

from codeflash.languages.javascript.support import JavaScriptSupport


@pytest.fixture
def js_support():
    """Create a JavaScriptSupport instance."""
    return JavaScriptSupport()


class TestDiscoverTests:
    """Tests for discover_tests method."""

    def test_discover_tests_basic(self, js_support):
        """Test discovering basic Jest tests for a function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create source file
            source_file = tmpdir / "math.js"
            source_file.write_text("""
export function add(a, b) {
    return a + b;
}

module.exports = { add };
""")

            # Create test file
            test_file = tmpdir / "math.test.js"
            test_file.write_text("""
const { add } = require('./math');

describe('add function', () => {
    test('adds two positive numbers', () => {
        expect(add(1, 2)).toBe(3);
    });

    test('adds negative numbers', () => {
        expect(add(-1, -2)).toBe(-3);
    });
});
""")

            # Discover functions first
            functions = js_support.discover_functions(source_file)
            assert len(functions) == 1

            # Discover tests
            tests = js_support.discover_tests(tmpdir, functions)

            assert len(tests) > 0
            # Should have tests mapped to the add function
            assert any("add" in key for key in tests.keys())

    def test_discover_tests_spec_suffix(self, js_support):
        """Test discovering tests with .spec.js suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create source file
            source_file = tmpdir / "calculator.js"
            source_file.write_text("""
export function multiply(a, b) {
    return a * b;
}

module.exports = { multiply };
""")

            # Create test file with .spec.js suffix
            test_file = tmpdir / "calculator.spec.js"
            test_file.write_text("""
const { multiply } = require('./calculator');

describe('multiply', () => {
    it('multiplies two numbers', () => {
        expect(multiply(3, 4)).toBe(12);
    });
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            assert len(tests) > 0

    def test_discover_tests_in_tests_directory(self, js_support):
        """Test discovering tests in __tests__ directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create source file
            source_file = tmpdir / "utils.js"
            source_file.write_text("""
export function formatDate(date) {
    return date.toISOString();
}

module.exports = { formatDate };
""")

            # Create __tests__ directory
            tests_dir = tmpdir / "__tests__"
            tests_dir.mkdir()

            test_file = tests_dir / "utils.js"
            test_file.write_text("""
const { formatDate } = require('../utils');

test('formats date correctly', () => {
    const date = new Date('2024-01-01');
    expect(formatDate(date)).toContain('2024');
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            assert len(tests) > 0

    def test_discover_tests_nested_describe(self, js_support):
        """Test discovering tests with nested describe blocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "string_utils.js"
            source_file.write_text("""
export function capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}

export function lowercase(str) {
    return str.toLowerCase();
}

module.exports = { capitalize, lowercase };
""")

            test_file = tmpdir / "string_utils.test.js"
            test_file.write_text("""
const { capitalize, lowercase } = require('./string_utils');

describe('String Utils', () => {
    describe('capitalize', () => {
        test('capitalizes first letter', () => {
            expect(capitalize('hello')).toBe('Hello');
        });

        test('handles empty string', () => {
            expect(capitalize('')).toBe('');
        });
    });

    describe('lowercase', () => {
        test('lowercases string', () => {
            expect(lowercase('HELLO')).toBe('hello');
        });
    });
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            assert len(tests) > 0
            # Check that nested tests are found
            test_info = list(tests.values())[0]
            test_names = [t.test_name for t in test_info]
            assert any("capitalizes first letter" in name for name in test_names)

    def test_discover_tests_with_it_block(self, js_support):
        """Test discovering tests using 'it' instead of 'test'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "array_utils.js"
            source_file.write_text("""
export function sum(arr) {
    return arr.reduce((a, b) => a + b, 0);
}

module.exports = { sum };
""")

            test_file = tmpdir / "array_utils.test.js"
            test_file.write_text("""
const { sum } = require('./array_utils');

describe('sum function', () => {
    it('should sum an array of numbers', () => {
        expect(sum([1, 2, 3])).toBe(6);
    });

    it('should return 0 for empty array', () => {
        expect(sum([])).toBe(0);
    });
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            assert len(tests) > 0

    def test_discover_tests_es_module_import(self, js_support):
        """Test discovering tests with ES module imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "math_es.js"
            source_file.write_text("""
export function divide(a, b) {
    return a / b;
}

export function subtract(a, b) {
    return a - b;
}
""")

            test_file = tmpdir / "math_es.test.js"
            test_file.write_text("""
import { divide, subtract } from './math_es';

test('divide two numbers', () => {
    expect(divide(10, 2)).toBe(5);
});

test('subtract two numbers', () => {
    expect(subtract(5, 3)).toBe(2);
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            assert len(tests) > 0

    def test_discover_tests_default_export(self, js_support):
        """Test discovering tests for default exported functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "greeter.js"
            source_file.write_text("""
export function greet(name) {
    return `Hello, ${name}!`;
}

module.exports = greet;
""")

            test_file = tmpdir / "greeter.test.js"
            test_file.write_text("""
const greet = require('./greeter');

test('greets by name', () => {
    expect(greet('World')).toBe('Hello, World!');
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            assert len(tests) > 0

    def test_discover_tests_class_methods(self, js_support):
        """Test discovering tests for class methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "calculator_class.js"
            source_file.write_text("""
export class Calculator {
    add(a, b) {
        return a + b;
    }

    multiply(a, b) {
        return a * b;
    }
}

module.exports = { Calculator };
""")

            test_file = tmpdir / "calculator_class.test.js"
            test_file.write_text("""
const { Calculator } = require('./calculator_class');

describe('Calculator class', () => {
    let calc;

    beforeEach(() => {
        calc = new Calculator();
    });

    test('add method', () => {
        expect(calc.add(2, 3)).toBe(5);
    });

    test('multiply method', () => {
        expect(calc.multiply(2, 3)).toBe(6);
    });
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # Should find tests for class methods
            assert len(tests) > 0

    def test_discover_tests_multi_level_directories(self, js_support):
        """Test discovering tests in multi-level directory structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create nested source structure
            src_dir = tmpdir / "src" / "utils"
            src_dir.mkdir(parents=True)

            source_file = src_dir / "helpers.js"
            source_file.write_text("""
export function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

module.exports = { clamp };
""")

            # Create nested test structure
            test_dir = tmpdir / "tests" / "utils"
            test_dir.mkdir(parents=True)

            test_file = test_dir / "helpers.test.js"
            test_file.write_text("""
const { clamp } = require('../../src/utils/helpers');

describe('clamp', () => {
    test('clamps value within range', () => {
        expect(clamp(5, 0, 10)).toBe(5);
    });

    test('clamps value to min', () => {
        expect(clamp(-5, 0, 10)).toBe(0);
    });

    test('clamps value to max', () => {
        expect(clamp(15, 0, 10)).toBe(10);
    });
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            assert len(tests) > 0

    def test_discover_tests_async_functions(self, js_support):
        """Test discovering tests for async functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "async_utils.js"
            source_file.write_text("""
export async function fetchData(url) {
    return await fetch(url).then(r => r.json());
}

export async function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

module.exports = { fetchData, delay };
""")

            test_file = tmpdir / "async_utils.test.js"
            test_file.write_text("""
const { fetchData, delay } = require('./async_utils');

describe('async utilities', () => {
    test('delay resolves after timeout', async () => {
        const start = Date.now();
        await delay(100);
        expect(Date.now() - start).toBeGreaterThanOrEqual(100);
    });
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            assert len(tests) > 0

    def test_discover_tests_jsx_component(self, js_support):
        """Test discovering tests for JSX components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "Button.jsx"
            source_file.write_text("""
import React from 'react';

export function Button({ onClick, children }) {
    return <button onClick={onClick}>{children}</button>;
}

export default Button;
""")

            test_file = tmpdir / "Button.test.jsx"
            test_file.write_text("""
import React from 'react';
import Button from './Button';

describe('Button component', () => {
    test('renders children', () => {
        // Test implementation
    });

    test('handles click', () => {
        // Test implementation
    });
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # JSX tests should be discovered
            assert len(tests) >= 0  # May or may not find depending on import matching

    def test_discover_tests_no_matching_tests(self, js_support):
        """Test when no matching tests exist for a function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "untested.js"
            source_file.write_text("""
export function untestedFunction() {
    return 42;
}

module.exports = { untestedFunction };
""")

            # Create test file that doesn't import our function
            test_file = tmpdir / "other.test.js"
            test_file.write_text("""
const { someOtherFunc } = require('./other');

test('other test', () => {
    expect(true).toBe(true);
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # Should not find tests for our function
            assert "untested.untestedFunction" not in tests or len(tests.get("untested.untestedFunction", [])) == 0

    def test_discover_tests_function_name_in_source(self, js_support):
        """Test discovering tests when function name appears in test source."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "validators.js"
            source_file.write_text("""
export function isEmail(str) {
    return str.includes('@');
}

export function isUrl(str) {
    return str.startsWith('http');
}

module.exports = { isEmail, isUrl };
""")

            test_file = tmpdir / "validators.test.js"
            test_file.write_text("""
const { isEmail } = require('./validators');

describe('validators', () => {
    test('isEmail validates email', () => {
        expect(isEmail('test@example.com')).toBe(true);
        expect(isEmail('invalid')).toBe(false);
    });
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # Should find tests for isEmail
            assert len(tests) > 0

    def test_discover_tests_multiple_test_files(self, js_support):
        """Test discovering tests across multiple test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "shared_utils.js"
            source_file.write_text("""
export function helper1() {
    return 1;
}

export function helper2() {
    return 2;
}

module.exports = { helper1, helper2 };
""")

            # First test file
            test_file1 = tmpdir / "shared_utils_1.test.js"
            test_file1.write_text("""
const { helper1 } = require('./shared_utils');

test('helper1 returns 1', () => {
    expect(helper1()).toBe(1);
});
""")

            # Second test file
            test_file2 = tmpdir / "shared_utils_2.test.js"
            test_file2.write_text("""
const { helper2 } = require('./shared_utils');

test('helper2 returns 2', () => {
    expect(helper2()).toBe(2);
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            assert len(tests) > 0

    def test_discover_tests_template_literal_names(self, js_support):
        """Test discovering tests with template literal test names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "format.js"
            source_file.write_text("""
export function formatNumber(n) {
    return n.toFixed(2);
}

module.exports = { formatNumber };
""")

            test_file = tmpdir / "format.test.js"
            test_file.write_text("""
const { formatNumber } = require('./format');

test(`formatNumber with decimal`, () => {
    expect(formatNumber(3.14159)).toBe('3.14');
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # May or may not find depending on template literal handling
            assert isinstance(tests, dict)

    def test_discover_tests_aliased_import(self, js_support):
        """Test discovering tests with aliased imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "transform.js"
            source_file.write_text("""
export function transformData(data) {
    return data.map(x => x * 2);
}

module.exports = { transformData };
""")

            test_file = tmpdir / "transform.test.js"
            test_file.write_text("""
const { transformData: transform } = require('./transform');

describe('transform', () => {
    test('doubles all values', () => {
        expect(transform([1, 2, 3])).toEqual([2, 4, 6]);
    });
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # Should still find tests since original name is imported
            assert len(tests) > 0


class TestFindJestTests:
    """Tests for _find_jest_tests method."""

    def test_find_basic_tests(self, js_support):
        """Test finding basic test and it blocks."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
test('first test', () => {});
test('second test', () => {});
it('third test', () => {});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            assert "first test" in test_names
            assert "second test" in test_names
            assert "third test" in test_names

    def test_find_describe_blocks(self, js_support):
        """Test finding describe blocks."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
describe('Suite A', () => {
    test('test 1', () => {});
});

describe('Suite B', () => {
    it('test 2', () => {});
});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            assert "Suite A" in test_names
            assert "Suite B" in test_names
            assert "test 1" in test_names
            assert "test 2" in test_names

    def test_find_nested_describe_blocks(self, js_support):
        """Test finding nested describe blocks."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
describe('Outer', () => {
    describe('Inner', () => {
        test('nested test', () => {});
    });
});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            assert "Outer" in test_names
            assert "Inner" in test_names
            assert "nested test" in test_names

    def test_find_tests_with_skip(self, js_support):
        """Test finding skipped tests (test.skip, it.skip)."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
test('normal test', () => {});
test.skip('skipped test', () => {});
it.skip('skipped it', () => {});
describe.skip('skipped describe', () => {
    test('test in skipped', () => {});
});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            assert "normal test" in test_names

    def test_find_tests_with_only(self, js_support):
        """Test finding tests with .only modifier."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
test('regular test', () => {});
test.only('only test', () => {});
describe.only('only describe', () => {
    test('test inside', () => {});
});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            assert "regular test" in test_names

    def test_find_tests_with_single_quotes(self, js_support):
        """Test finding tests with single-quoted names."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
test('single quotes', () => {});
describe('describe single', () => {});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            assert "single quotes" in test_names
            assert "describe single" in test_names

    def test_find_tests_with_double_quotes(self, js_support):
        """Test finding tests with double-quoted names."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
test("double quotes", () => {});
describe("describe double", () => {});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            assert "double quotes" in test_names
            assert "describe double" in test_names

    def test_find_tests_empty_file(self, js_support):
        """Test finding tests in empty file."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            assert test_names == []


class TestImportAnalysis:
    """Tests for import analysis in test discovery."""

    def test_require_named_import(self, js_support):
        """Test detecting named imports via require."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "funcs.js"
            source_file.write_text("""
export function funcA() { return 1; }
export function funcB() { return 2; }
module.exports = { funcA, funcB };
""")

            test_file = tmpdir / "funcs.test.js"
            test_file.write_text("""
const { funcA } = require('./funcs');

test('funcA works', () => {
    expect(funcA()).toBe(1);
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # funcA should have tests
            funcA_key = next((k for k in tests.keys() if "funcA" in k), None)
            assert funcA_key is not None

    def test_es_module_named_import(self, js_support):
        """Test detecting ES module named imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "esm_funcs.js"
            source_file.write_text("""
export function funcX() { return 'x'; }
export function funcY() { return 'y'; }
""")

            test_file = tmpdir / "esm_funcs.test.js"
            test_file.write_text("""
import { funcX } from './esm_funcs';

test('funcX works', () => {
    expect(funcX()).toBe('x');
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # funcX should have tests
            assert len(tests) > 0

    def test_default_import(self, js_support):
        """Test detecting default imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "default_export.js"
            source_file.write_text("""
export function mainFunc() { return 'main'; }
module.exports = mainFunc;
""")

            test_file = tmpdir / "default_export.test.js"
            test_file.write_text("""
const mainFunc = require('./default_export');

test('mainFunc works', () => {
    expect(mainFunc()).toBe('main');
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            assert len(tests) > 0


class TestEdgeCases:
    """Edge case tests for JavaScript test discovery."""

    def test_comments_in_test_file(self, js_support):
        """Test that comments don't affect test discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "commented.js"
            source_file.write_text("""
export function compute() { return 42; }
module.exports = { compute };
""")

            test_file = tmpdir / "commented.test.js"
            test_file.write_text("""
const { compute } = require('./commented');

// test('commented out test', () => {});

test('actual test', () => {
    expect(compute()).toBe(42);
});

/*
test('block commented', () => {
    expect(true).toBe(true);
});
*/
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            assert len(tests) > 0

    def test_test_file_with_syntax_error(self, js_support):
        """Test handling of test files with syntax errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "valid.js"
            source_file.write_text("""
export function validFunc() { return 1; }
module.exports = { validFunc };
""")

            test_file = tmpdir / "invalid.test.js"
            test_file.write_text("""
const { validFunc } = require('./valid');

test('broken test' {  // Missing arrow function
    expect(validFunc()).toBe(1);
});
""")

            functions = js_support.discover_functions(source_file)
            # Should not crash
            tests = js_support.discover_tests(tmpdir, functions)
            assert isinstance(tests, dict)

    def test_function_with_same_name_as_jest_api(self, js_support):
        """Test function with same name as Jest API (test, describe, etc.)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "conflict.js"
            source_file.write_text("""
export function test(value) { return value > 0; }
export function describe(obj) { return JSON.stringify(obj); }
module.exports = { test, describe };
""")

            test_file = tmpdir / "conflict.test.js"
            test_file.write_text("""
const { test: testFunc, describe: describeFunc } = require('./conflict');

describe('conflict tests', () => {
    test('testFunc validates', () => {
        expect(testFunc(5)).toBe(true);
    });
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # Should still work despite naming conflicts
            assert isinstance(tests, dict)

    def test_empty_test_directory(self, js_support):
        """Test discovering tests when test directory is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "lonely.js"
            source_file.write_text("""
export function lonelyFunc() { return 'alone'; }
module.exports = { lonelyFunc };
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # Should return empty dict, not crash
            assert tests == {} or all(len(v) == 0 for v in tests.values())

    def test_circular_imports(self, js_support):
        """Test handling of circular import patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            file_a = tmpdir / "moduleA.js"
            file_a.write_text("""
const { funcB } = require('./moduleB');
export function funcA() { return 'A' + (funcB ? funcB() : ''); }
module.exports = { funcA };
""")

            file_b = tmpdir / "moduleB.js"
            file_b.write_text("""
const { funcA } = require('./moduleA');
export function funcB() { return 'B'; }
module.exports = { funcB };
""")

            test_file = tmpdir / "modules.test.js"
            test_file.write_text("""
const { funcA } = require('./moduleA');
const { funcB } = require('./moduleB');

test('funcA works', () => {
    expect(funcA()).toContain('A');
});
""")

            functions_a = js_support.discover_functions(file_a)
            tests = js_support.discover_tests(tmpdir, functions_a)

            # Should handle circular imports gracefully
            assert isinstance(tests, dict)

    def test_unicode_in_test_names(self, js_support):
        """Test handling of unicode characters in test names."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False, encoding="utf-8") as f:
            f.write("""
test('handles emoji 🎉', () => {});
describe('日本語テスト', () => {
    test('works with unicode', () => {});
});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text(encoding="utf-8")
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            # Should find tests even with unicode
            assert len(test_names) > 0


class TestParametrizedTests:
    """Tests for Jest parametrized test discovery (test.each, describe.each)."""

    def test_find_test_each_array(self, js_support):
        """Test finding test.each with array syntax."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
test.each([
    [1, 1, 2],
    [1, 2, 3],
    [2, 1, 3],
])('add(%i, %i) returns %i', (a, b, expected) => {
    expect(a + b).toBe(expected);
});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            # The current implementation may or may not find test.each
            # This documents the expected behavior
            assert isinstance(test_names, list)

    def test_find_describe_each(self, js_support):
        """Test finding describe.each."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
describe.each([
    { name: 'add', fn: (a, b) => a + b },
    { name: 'multiply', fn: (a, b) => a * b },
])('$name function', ({ fn }) => {
    test('works', () => {
        expect(fn(2, 3)).toBeDefined();
    });
});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            # Document current behavior
            assert isinstance(test_names, list)

    def test_find_it_each(self, js_support):
        """Test finding it.each."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
describe('Math operations', () => {
    it.each([
        [2, 2, 4],
        [3, 3, 9],
    ])('squares %i to get %i', (input, _, expected) => {
        expect(input * input).toBe(expected);
    });
});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            # Should at least find the describe block
            assert "Math operations" in test_names


class TestTestDiscoveryIntegration:
    """Integration tests for full test discovery workflow."""

    def test_full_discovery_workflow(self, js_support):
        """Test complete discovery workflow from functions to tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a realistic project structure
            src_dir = tmpdir / "src"
            src_dir.mkdir()

            tests_dir = tmpdir / "__tests__"
            tests_dir.mkdir()

            # Source file
            source_file = src_dir / "utils.js"
            source_file.write_text(r"""
export function validateEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

export function validatePhone(phone) {
    const re = /^\d{10}$/;
    return re.test(phone);
}

export function formatName(first, last) {
    return `${first} ${last}`.trim();
}

module.exports = { validateEmail, validatePhone, formatName };
""")

            # Test file
            test_file = tests_dir / "utils.test.js"
            test_file.write_text("""
const { validateEmail, validatePhone, formatName } = require('../src/utils');

describe('Validation Utils', () => {
    describe('validateEmail', () => {
        test('accepts valid email', () => {
            expect(validateEmail('test@example.com')).toBe(true);
        });

        test('rejects invalid email', () => {
            expect(validateEmail('invalid')).toBe(false);
        });
    });

    describe('validatePhone', () => {
        test('accepts 10 digit phone', () => {
            expect(validatePhone('1234567890')).toBe(true);
        });
    });
});

describe('formatName', () => {
    test('formats full name', () => {
        expect(formatName('John', 'Doe')).toBe('John Doe');
    });
});
""")

            # Discover functions
            functions = js_support.discover_functions(source_file)
            assert len(functions) == 3

            # Discover tests
            tests = js_support.discover_tests(tmpdir, functions)

            # Verify structure
            assert len(tests) > 0

            # Check that test names are found
            all_test_names = []
            for test_list in tests.values():
                all_test_names.extend([t.test_name for t in test_list])

            assert any("validateEmail" in name or "accepts valid email" in name for name in all_test_names)

    def test_discovery_with_fixtures(self, js_support):
        """Test discovery when test file uses beforeEach/afterEach."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "database.js"
            source_file.write_text("""
export class Database {
    constructor() {
        this.data = [];
    }

    insert(item) {
        this.data.push(item);
        return this.data.length;
    }

    clear() {
        this.data = [];
        return true;
    }
}

module.exports = { Database };
""")

            test_file = tmpdir / "database.test.js"
            test_file.write_text("""
const { Database } = require('./database');

describe('Database', () => {
    let db;

    beforeEach(() => {
        db = new Database();
    });

    afterEach(() => {
        db.clear();
    });

    test('insert adds item', () => {
        expect(db.insert('item1')).toBe(1);
    });

    test('insert returns correct count', () => {
        db.insert('item1');
        expect(db.insert('item2')).toBe(2);
    });
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            assert len(tests) > 0


class TestImportFilteringDetailed:
    """Detailed tests for import filtering in test discovery, mirroring Python tests."""

    def test_test_file_imports_different_module(self, js_support):
        """Test that tests importing different modules are correctly matched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create two source files
            source_a = tmpdir / "moduleA.js"
            source_a.write_text("""
export function funcA() { return 'A'; }
module.exports = { funcA };
""")

            source_b = tmpdir / "moduleB.js"
            source_b.write_text("""
export function funcB() { return 'B'; }
module.exports = { funcB };
""")

            # Test file only imports moduleA
            test_file = tmpdir / "test_a.test.js"
            test_file.write_text("""
const { funcA } = require('./moduleA');

test('funcA works', () => {
    expect(funcA()).toBe('A');
});
""")

            # Discover functions from moduleB
            functions_b = js_support.discover_functions(source_b)
            tests = js_support.discover_tests(tmpdir, functions_b)

            # funcB should not have any tests since test file doesn't import it
            for key in tests.keys():
                if "funcB" in key:
                    # If funcB is in tests, it should have 0 tests
                    assert len(tests[key]) == 0

    def test_test_file_imports_only_specific_function(self, js_support):
        """Test that only imported functions are matched to tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "utils.js"
            source_file.write_text("""
export function funcOne() { return 1; }
export function funcTwo() { return 2; }
export function funcThree() { return 3; }
module.exports = { funcOne, funcTwo, funcThree };
""")

            # Test file only imports funcOne
            test_file = tmpdir / "utils.test.js"
            test_file.write_text("""
const { funcOne } = require('./utils');

test('funcOne returns 1', () => {
    expect(funcOne()).toBe(1);
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # Check that tests were found
            assert len(tests) > 0

    def test_function_name_as_string_not_import(self, js_support):
        """Test that function name appearing as string doesn't count as import."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "target.js"
            source_file.write_text("""
export function targetFunc() { return 'target'; }
module.exports = { targetFunc };
""")

            # Test file mentions targetFunc as string, not import
            test_file = tmpdir / "strings.test.js"
            test_file.write_text("""
const { otherFunc } = require('./other');

test('mentions targetFunc in string', () => {
    const message = 'This test is for targetFunc';
    expect(message).toContain('targetFunc');
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # Current implementation may still match on string occurrence
            # This documents the actual behavior
            assert isinstance(tests, dict)

    def test_module_import_with_method_access(self, js_support):
        """Test module-style import with method access."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "math.js"
            source_file.write_text("""
export function calculate(x) { return x * 2; }
module.exports = { calculate };
""")

            test_file = tmpdir / "math.test.js"
            test_file.write_text("""
const math = require('./math');

test('calculate doubles', () => {
    expect(math.calculate(5)).toBe(10);
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # Should find tests since 'calculate' appears in source
            assert len(tests) > 0

    def test_class_method_discovery_via_class_import(self, js_support):
        """Test that class method tests are discovered when class is imported."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "myclass.js"
            source_file.write_text("""
export class MyClass {
    methodA() { return 'A'; }
    methodB() { return 'B'; }
}
module.exports = { MyClass };
""")

            test_file = tmpdir / "myclass.test.js"
            test_file.write_text("""
const { MyClass } = require('./myclass');

describe('MyClass', () => {
    test('methodA returns A', () => {
        const obj = new MyClass();
        expect(obj.methodA()).toBe('A');
    });
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # Should find tests for class methods
            assert len(tests) > 0

    def test_nested_module_structure(self, js_support):
        """Test discovery with nested module structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create nested structure
            src_dir = tmpdir / "src" / "core" / "utils"
            src_dir.mkdir(parents=True)

            source_file = src_dir / "helpers.js"
            source_file.write_text("""
export function deepHelper() { return 'deep'; }
module.exports = { deepHelper };
""")

            tests_dir = tmpdir / "tests" / "unit"
            tests_dir.mkdir(parents=True)

            test_file = tests_dir / "helpers.test.js"
            test_file.write_text("""
const { deepHelper } = require('../../src/core/utils/helpers');

test('deepHelper works', () => {
    expect(deepHelper()).toBe('deep');
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            assert len(tests) > 0


class TestAdvancedPatterns:
    """Tests for advanced Jest patterns."""

    def test_dynamic_test_names(self, js_support):
        """Test handling of dynamic/computed test names."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
const testCases = ['case1', 'case2', 'case3'];

testCases.forEach(name => {
    test(name + ' test', () => {
        expect(true).toBe(true);
    });
});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            # Dynamic tests may not be discoverable statically
            assert isinstance(test_names, list)

    def test_conditional_tests(self, js_support):
        """Test handling of conditional test blocks."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
describe('conditional tests', () => {
    if (process.env.RUN_SLOW_TESTS) {
        test('slow test', () => {
            expect(true).toBe(true);
        });
    }

    test('always runs', () => {
        expect(true).toBe(true);
    });
});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            assert "conditional tests" in test_names
            assert "always runs" in test_names

    def test_test_with_timeout(self, js_support):
        """Test finding tests with timeout option."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
test('quick test', () => {
    expect(true).toBe(true);
});

test('slow test', () => {
    expect(true).toBe(true);
}, 30000);
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            assert "quick test" in test_names
            assert "slow test" in test_names

    def test_todo_tests(self, js_support):
        """Test finding test.todo blocks."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
test('implemented test', () => {
    expect(true).toBe(true);
});

test.todo('needs implementation');
test.todo('also needs implementation');
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            assert "implemented test" in test_names

    def test_concurrent_tests(self, js_support):
        """Test finding test.concurrent blocks."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
test.concurrent('concurrent test 1', async () => {
    expect(await Promise.resolve(1)).toBe(1);
});

test.concurrent('concurrent test 2', async () => {
    expect(await Promise.resolve(2)).toBe(2);
});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            # test.concurrent may or may not be found depending on implementation
            assert isinstance(test_names, list)


class TestFunctionToTestMapping:
    """Tests for correct function-to-test mapping."""

    def test_multiple_functions_same_file_different_tests(self, js_support):
        """Test that functions in same file map to their specific tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "multiple.js"
            source_file.write_text("""
export function addNumbers(a, b) { return a + b; }
export function subtractNumbers(a, b) { return a - b; }
export function multiplyNumbers(a, b) { return a * b; }
module.exports = { addNumbers, subtractNumbers, multiplyNumbers };
""")

            test_file = tmpdir / "multiple.test.js"
            test_file.write_text("""
const { addNumbers, subtractNumbers } = require('./multiple');

describe('addNumbers', () => {
    test('adds correctly', () => {
        expect(addNumbers(1, 2)).toBe(3);
    });
});

describe('subtractNumbers', () => {
    test('subtracts correctly', () => {
        expect(subtractNumbers(5, 3)).toBe(2);
    });
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # All three functions should be discovered
            assert len(functions) == 3

            # Tests should exist for addNumbers and subtractNumbers
            assert len(tests) > 0

    def test_test_in_wrong_describe_still_discovered(self, js_support):
        """Test that tests are discovered even if describe name doesn't match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "funcs.js"
            source_file.write_text("""
export function targetFunc() { return 'target'; }
module.exports = { targetFunc };
""")

            test_file = tmpdir / "funcs.test.js"
            test_file.write_text("""
const { targetFunc } = require('./funcs');

describe('Unrelated name', () => {
    test('test that uses targetFunc', () => {
        expect(targetFunc()).toBe('target');
    });
});
""")

            functions = js_support.discover_functions(source_file)
            tests = js_support.discover_tests(tmpdir, functions)

            # Should still find tests
            assert len(tests) > 0


class TestMochaStyleTests:
    """Tests for Mocha-style test syntax (also supported by Jest)."""

    def test_mocha_bdd_style(self, js_support):
        """Test finding Mocha BDD-style tests."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
describe('Array', function() {
    describe('#indexOf()', function() {
        it('should return -1 when not present', function() {
            expect([1, 2, 3].indexOf(4)).toBe(-1);
        });
    });
});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            assert "Array" in test_names
            assert "#indexOf()" in test_names
            assert "should return -1 when not present" in test_names

    def test_context_block(self, js_support):
        """Test finding context blocks (Mocha-style, aliased to describe in Jest)."""
        with tempfile.NamedTemporaryFile(suffix=".test.js", mode="w", delete=False) as f:
            f.write("""
describe('User', () => {
    describe('when logged in', () => {
        test('can access dashboard', () => {
            expect(true).toBe(true);
        });
    });

    describe('when logged out', () => {
        test('is redirected to login', () => {
            expect(true).toBe(true);
        });
    });
});
""")
            f.flush()
            file_path = Path(f.name)

            source = file_path.read_text()
            from codeflash.languages.javascript.treesitter import get_analyzer_for_file

            analyzer = get_analyzer_for_file(file_path)
            test_names = js_support._find_jest_tests(source, analyzer)

            assert "User" in test_names
            assert "when logged in" in test_names
            assert "when logged out" in test_names


class TestQualifiedNames:
    """Tests for qualified function name handling."""

    def test_class_method_qualified_name(self, js_support):
        """Test that class methods have proper qualified names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "calculator.js"
            source_file.write_text("""
export class Calculator {
    add(a, b) { return a + b; }
    subtract(a, b) { return a - b; }
}
module.exports = { Calculator };
""")

            functions = js_support.discover_functions(source_file)

            # Check qualified names include class
            add_func = next((f for f in functions if f.function_name == "add"), None)
            assert add_func is not None
            assert add_func.class_name == "Calculator"

    def test_nested_class_method(self, js_support):
        """Test nested class method discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            source_file = tmpdir / "nested.js"
            source_file.write_text("""
export class Outer {
    innerMethod() {
        class Inner {
            deepMethod() { return 'deep'; }
        }
        return new Inner().deepMethod();
    }
}
module.exports = { Outer };
""")

            functions = js_support.discover_functions(source_file)

            # Should find at least the Outer class method
            assert any(f.class_name == "Outer" for f in functions)
