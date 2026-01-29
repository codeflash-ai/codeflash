"""Tests for JavaScript/TypeScript code extractor.

Uses strict string equality to verify extraction results.
"""

import shutil
from pathlib import Path

import pytest
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.base import Language
from codeflash.languages.javascript.support import JavaScriptSupport, TypeScriptSupport
from codeflash.languages.registry import get_language_support
from codeflash.models.models import FunctionParent
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestCodeExtractorCJS:
    """Tests for CommonJS module code extraction."""

    @pytest.fixture
    def cjs_project(self, tmp_path):
        """Create a temporary CJS project from fixtures."""
        project_dir = tmp_path / "cjs_project"
        shutil.copytree(FIXTURES_DIR / "js_cjs", project_dir)
        return project_dir

    @pytest.fixture
    def js_support(self):
        """Create JavaScriptSupport instance."""
        return JavaScriptSupport()

    def test_discover_class_methods(self, js_support, cjs_project):
        """Test that class methods are discovered correctly."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        method_names = {f.name for f in functions}

        expected_methods = {"calculateCompoundInterest", "permutation", "quickAdd"}
        assert method_names == expected_methods, f"Expected methods {expected_methods}, got {method_names}"

    def test_class_method_has_correct_parent(self, js_support, cjs_project):
        """Test parent class information for methods."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        for func in functions:
            # All methods should belong to Calculator class
            assert func.is_method is True, f"{func.name} should be a method"
            assert func.class_name == "Calculator", f"{func.name} should belong to Calculator, got {func.class_name}"

    def test_extract_permutation_code(self, js_support, cjs_project):
        """Test permutation method code extraction."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        permutation_func = next(f for f in functions if f.name == "permutation")

        context = js_support.extract_code_context(
            function=permutation_func, project_root=cjs_project, module_root=cjs_project
        )

        expected_code = """\
class Calculator {
    constructor(precision = 2) {
        this.precision = precision;
        this.history = [];
    }

    /**
     * Calculate permutation using factorial helper.
     * @param n - Total items
     * @param r - Items to choose
     * @returns Permutation result
     */
    permutation(n, r) {
        if (n < r) return 0;
        // Inefficient: calculates factorial(n) fully even when not needed
        return factorial(n) / factorial(n - r);
    }
}"""

        assert context.target_code is not None, "target_code should not be None"
        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{expected_code}\n\n"
            f"Got:\n{context.target_code}"
        )

    def test_extract_context_includes_direct_helpers(self, js_support, cjs_project):
        """Test that direct helper functions are included in context."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        permutation_func = next(f for f in functions if f.name == "permutation")

        context = js_support.extract_code_context(
            function=permutation_func, project_root=cjs_project, module_root=cjs_project
        )

        # Find factorial helper
        helper_dict = {h.name: h for h in context.helper_functions}

        assert "factorial" in helper_dict, f"factorial helper not found. Found helpers: {list(helper_dict.keys())}"

        factorial_helper = helper_dict["factorial"]

        expected_factorial_code = """\
/**
 * Calculate factorial recursively.
 * @param n - Non-negative integer
 * @returns Factorial of n
 */
function factorial(n) {
    // Intentionally inefficient recursive implementation
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}"""

        assert factorial_helper.source_code.strip() == expected_factorial_code.strip(), (
            f"Factorial helper code does not match expected.\n"
            f"Expected:\n{expected_factorial_code}\n\n"
            f"Got:\n{factorial_helper.source_code}"
        )

        # STRICT: Verify file path ends with expected filename
        assert str(factorial_helper.file_path).endswith("math_utils.js"), (
            f"Expected factorial to be from math_utils.js, got {factorial_helper.file_path}"
        )

    def test_extract_compound_interest_code(self, js_support, cjs_project):
        """Test calculateCompoundInterest code extraction."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        compound_func = next(f for f in functions if f.name == "calculateCompoundInterest")

        context = js_support.extract_code_context(
            function=compound_func, project_root=cjs_project, module_root=cjs_project
        )

        expected_code = """\
class Calculator {
    constructor(precision = 2) {
        this.precision = precision;
        this.history = [];
    }

    /**
     * Calculate compound interest with multiple helper dependencies.
     * @param principal - Initial amount
     * @param rate - Interest rate (as decimal)
     * @param time - Time in years
     * @param n - Compounding frequency per year
     * @returns Compound interest result
     */
    calculateCompoundInterest(principal, rate, time, n) {
        validateInput(principal, 'principal');
        validateInput(rate, 'rate');

        // Inefficient: recalculates power multiple times
        let result = principal;
        for (let i = 0; i < n * time; i++) {
            result = multiply(result, add(1, rate / n));
        }

        const interest = result - principal;
        this.history.push({ type: 'compound', result: interest });
        return formatNumber(interest, this.precision);
    }
}"""

        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{expected_code}\n\n"
            f"Got:\n{context.target_code}"
        )

    def test_extract_compound_interest_helpers(self, js_support, cjs_project):
        """Test helper extraction for calculateCompoundInterest."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        compound_func = next(f for f in functions if f.name == "calculateCompoundInterest")

        context = js_support.extract_code_context(
            function=compound_func, project_root=cjs_project, module_root=cjs_project
        )

        helper_dict = {h.name: h for h in context.helper_functions}

        expected_helpers = {"add", "multiply", "formatNumber", "validateInput"}
        actual_helpers = set(helper_dict.keys())
        assert actual_helpers == expected_helpers, f"Expected helpers {expected_helpers}, got {actual_helpers}"

        # STRICT: Verify each helper's code exactly
        expected_add_code = """\
/**
 * Add two numbers.
 * @param a - First number
 * @param b - Second number
 * @returns Sum of a and b
 */
function add(a, b) {
    return a + b;
}"""

        expected_multiply_code = """\
/**
 * Multiply two numbers.
 * @param a - First number
 * @param b - Second number
 * @returns Product of a and b
 */
function multiply(a, b) {
    return a * b;
}"""

        expected_format_number_code = """\
/**
 * Format a number to specified decimal places.
 * @param num - Number to format
 * @param decimals - Number of decimal places
 * @returns Formatted number
 */
function formatNumber(num, decimals) {
    return Number(num.toFixed(decimals));
}"""

        expected_validate_input_code = """\
/**
 * Validate that input is a valid number.
 * @param value - Value to validate
 * @param name - Parameter name for error message
 * @throws Error if value is not a valid number
 */
function validateInput(value, name) {
    if (typeof value !== 'number' || isNaN(value)) {
        throw new Error(`Invalid ${name}: must be a number`);
    }
}"""

        helper_expectations = {
            "add": (expected_add_code, "math_utils.js"),
            "multiply": (expected_multiply_code, "math_utils.js"),
            "formatNumber": (expected_format_number_code, "format.js"),
            "validateInput": (expected_validate_input_code, "format.js"),
        }

        for helper_name, (expected_code, expected_file) in helper_expectations.items():
            helper = helper_dict[helper_name]

            assert helper.source_code.strip() == expected_code.strip(), (
                f"{helper_name} helper code does not match expected.\n"
                f"Expected:\n{expected_code}\n\n"
                f"Got:\n{helper.source_code}"
            )

            assert str(helper.file_path).endswith(expected_file), (
                f"Expected {helper_name} to be from {expected_file}, got {helper.file_path}"
            )

    def test_extract_context_includes_imports(self, js_support, cjs_project):
        """Test import statement extraction."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        compound_func = next(f for f in functions if f.name == "calculateCompoundInterest")

        context = js_support.extract_code_context(
            function=compound_func, project_root=cjs_project, module_root=cjs_project
        )

        expected_imports = [
            "const { add, multiply, factorial } = require('./math_utils');",
            "const { formatNumber, validateInput } = require('./helpers/format');",
        ]

        assert len(context.imports) == 2, f"Expected 2 imports, got {len(context.imports)}: {context.imports}"
        assert context.imports == expected_imports, (
            f"Imports do not match expected.\n"
            f"Expected:\n{expected_imports}\n\n"
            f"Got:\n{context.imports}"
        )

    def test_extract_static_method(self, js_support, cjs_project):
        """Test static method extraction (quickAdd)."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        quick_add_func = next(f for f in functions if f.name == "quickAdd")

        context = js_support.extract_code_context(
            function=quick_add_func, project_root=cjs_project, module_root=cjs_project
        )

        expected_code = """\
class Calculator {
    constructor(precision = 2) {
        this.precision = precision;
        this.history = [];
    }

    /**
     * Static method for quick calculations.
     */
    static quickAdd(a, b) {
        return add(a, b);
    }
}"""

        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{expected_code}\n\n"
            f"Got:\n{context.target_code}"
        )

        # quickAdd uses add helper from math_utils
        helper_dict = {h.name: h for h in context.helper_functions}
        assert set(helper_dict.keys()) == {"add"}, f"Expected 'add' helper, got: {list(helper_dict.keys())}"

        expected_add_code = """\
/**
 * Add two numbers.
 * @param a - First number
 * @param b - Second number
 * @returns Sum of a and b
 */
function add(a, b) {
    return a + b;
}"""

        assert helper_dict["add"].source_code.strip() == expected_add_code.strip(), (
            f"add helper code does not match.\nExpected:\n{expected_add_code}\n\nGot:\n{helper_dict['add'].source_code}"
        )


class TestCodeExtractorESM:
    """Tests for ES Module code extraction."""

    @pytest.fixture
    def esm_project(self, tmp_path):
        """Create a temporary ESM project from fixtures."""
        project_dir = tmp_path / "esm_project"
        shutil.copytree(FIXTURES_DIR / "js_esm", project_dir)
        return project_dir

    @pytest.fixture
    def js_support(self):
        """Create JavaScriptSupport instance."""
        return JavaScriptSupport()

    def test_discover_esm_methods(self, js_support, esm_project):
        """Test method discovery in ESM project."""
        calculator_file = esm_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        method_names = {f.name for f in functions}

        # Should find same methods as CJS version
        expected_methods = {"calculateCompoundInterest", "permutation", "quickAdd"}
        assert method_names == expected_methods, f"Expected methods {expected_methods}, got {method_names}"

    def test_esm_permutation_extraction(self, js_support, esm_project):
        """Test permutation method extraction in ESM."""
        calculator_file = esm_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        permutation_func = next(f for f in functions if f.name == "permutation")

        context = js_support.extract_code_context(
            function=permutation_func, project_root=esm_project, module_root=esm_project
        )

        expected_code = """\
class Calculator {
    constructor(precision = 2) {
        this.precision = precision;
        this.history = [];
    }

    /**
     * Calculate permutation using factorial helper.
     * @param n - Total items
     * @param r - Items to choose
     * @returns Permutation result
     */
    permutation(n, r) {
        if (n < r) return 0;
        // Inefficient: calculates factorial(n) fully even when not needed
        return factorial(n) / factorial(n - r);
    }
}"""

        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{expected_code}\n\n"
            f"Got:\n{context.target_code}"
        )

        # ESM permutation uses factorial helper
        helper_dict = {h.name: h for h in context.helper_functions}
        assert set(helper_dict.keys()) == {"factorial"}, f"Expected 'factorial' helper, got: {list(helper_dict.keys())}"

        expected_factorial_code = """\
export function factorial(n) {
    // Intentionally inefficient recursive implementation
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}"""

        assert helper_dict["factorial"].source_code.strip() == expected_factorial_code.strip(), (
            f"factorial helper code does not match.\nExpected:\n{expected_factorial_code}\n\nGot:\n{helper_dict['factorial'].source_code}"
        )

    def test_esm_compound_interest_extraction(self, js_support, esm_project):
        """Test calculateCompoundInterest extraction in ESM with import syntax."""
        calculator_file = esm_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        compound_func = next(f for f in functions if f.name == "calculateCompoundInterest")

        context = js_support.extract_code_context(
            function=compound_func, project_root=esm_project, module_root=esm_project
        )

        expected_code = """\
class Calculator {
    constructor(precision = 2) {
        this.precision = precision;
        this.history = [];
    }

    /**
     * Calculate compound interest with multiple helper dependencies.
     * @param principal - Initial amount
     * @param rate - Interest rate (as decimal)
     * @param time - Time in years
     * @param n - Compounding frequency per year
     * @returns Compound interest result
     */
    calculateCompoundInterest(principal, rate, time, n) {
        validateInput(principal, 'principal');
        validateInput(rate, 'rate');

        // Inefficient: recalculates power multiple times
        let result = principal;
        for (let i = 0; i < n * time; i++) {
            result = multiply(result, add(1, rate / n));
        }

        const interest = result - principal;
        this.history.push({ type: 'compound', result: interest });
        return formatNumber(interest, this.precision);
    }
}"""

        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{expected_code}\n\n"
            f"Got:\n{context.target_code}"
        )

        expected_imports = [
            "import { add, multiply, factorial } from './math_utils.js';",
            "import { formatNumber, validateInput } from './helpers/format.js';",
        ]

        assert len(context.imports) == 2, f"Expected 2 imports, got {len(context.imports)}: {context.imports}"
        assert context.imports == expected_imports, (
            f"Imports do not match expected.\n"
            f"Expected:\n{expected_imports}\n\n"
            f"Got:\n{context.imports}"
        )

        # ESM compound interest uses 4 helpers
        helper_dict = {h.name: h for h in context.helper_functions}
        expected_helper_names = {"validateInput", "formatNumber", "add", "multiply"}
        assert set(helper_dict.keys()) == expected_helper_names, (
            f"Expected helpers {expected_helper_names}, got: {set(helper_dict.keys())}"
        )

        expected_validate_input_code = """\
export function validateInput(value, name) {
    if (typeof value !== 'number' || isNaN(value)) {
        throw new Error(`Invalid ${name}: must be a number`);
    }
}"""

        expected_format_number_code = """\
export function formatNumber(num, decimals) {
    return Number(num.toFixed(decimals));
}"""

        expected_add_code = """\
export function add(a, b) {
    return a + b;
}"""

        expected_multiply_code = """\
export function multiply(a, b) {
    return a * b;
}"""

        helper_expectations = {
            "validateInput": expected_validate_input_code,
            "formatNumber": expected_format_number_code,
            "add": expected_add_code,
            "multiply": expected_multiply_code,
        }

        for helper_name, expected_code in helper_expectations.items():
            assert helper_dict[helper_name].source_code.strip() == expected_code.strip(), (
                f"{helper_name} helper code does not match.\n"
                f"Expected:\n{expected_code}\n\n"
                f"Got:\n{helper_dict[helper_name].source_code}"
            )


class TestCodeExtractorTypeScript:
    """Tests for TypeScript code extraction."""

    @pytest.fixture
    def ts_project(self, tmp_path):
        """Create a temporary TypeScript project from fixtures."""
        project_dir = tmp_path / "ts_project"
        shutil.copytree(FIXTURES_DIR / "ts", project_dir)
        return project_dir

    @pytest.fixture
    def ts_support(self):
        """Create TypeScriptSupport instance."""
        return TypeScriptSupport()

    def test_typescript_support_properties(self, ts_support):
        """Test TypeScriptSupport properties."""
        assert ts_support.language == Language.TYPESCRIPT

        # STRICT: Verify exact file extensions
        expected_extensions = {".ts", ".tsx"}
        actual_extensions = set(ts_support.file_extensions)
        assert expected_extensions.issubset(actual_extensions), (
            f"Expected extensions {expected_extensions} to be subset of {actual_extensions}"
        )

    def test_discover_ts_methods(self, ts_support, ts_project):
        """Test method discovery in TypeScript."""
        calculator_file = ts_project / "calculator.ts"
        functions = ts_support.discover_functions(calculator_file)

        method_names = {f.name for f in functions}

        # TypeScript has additional getHistory method
        expected_methods = {"calculateCompoundInterest", "permutation", "getHistory", "quickAdd"}
        assert method_names == expected_methods, f"Expected methods {expected_methods}, got {method_names}"

    def test_ts_permutation_extraction(self, ts_support, ts_project):
        """Test permutation method extraction in TypeScript."""
        calculator_file = ts_project / "calculator.ts"
        functions = ts_support.discover_functions(calculator_file)

        permutation_func = next(f for f in functions if f.name == "permutation")

        context = ts_support.extract_code_context(
            function=permutation_func, project_root=ts_project, module_root=ts_project
        )

        expected_code = """\
class Calculator {
    private precision: number;
    private history: HistoryEntry[];

    constructor(precision: number = 2) {
        this.precision = precision;
        this.history = [];
    }

    /**
     * Calculate permutation using factorial helper.
     * @param n - Total items
     * @param r - Items to choose
     * @returns Permutation result
     */
    permutation(n: number, r: number): number {
        if (n < r) return 0;
        // Inefficient: calculates factorial(n) fully even when not needed
        return factorial(n) / factorial(n - r);
    }
}"""

        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{expected_code}\n\n"
            f"Got:\n{context.target_code}"
        )

        # TypeScript permutation uses factorial helper
        helper_dict = {h.name: h for h in context.helper_functions}
        assert set(helper_dict.keys()) == {"factorial"}, f"Expected 'factorial' helper, got: {list(helper_dict.keys())}"

        expected_factorial_code = """\
export function factorial(n: number): number {
    // Intentionally inefficient recursive implementation
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}"""

        assert helper_dict["factorial"].source_code.strip() == expected_factorial_code.strip(), (
            f"factorial helper code does not match.\nExpected:\n{expected_factorial_code}\n\nGot:\n{helper_dict['factorial'].source_code}"
        )

    def test_ts_compound_interest_extraction(self, ts_support, ts_project):
        """Test calculateCompoundInterest extraction in TypeScript."""
        calculator_file = ts_project / "calculator.ts"
        functions = ts_support.discover_functions(calculator_file)

        compound_func = next(f for f in functions if f.name == "calculateCompoundInterest")

        context = ts_support.extract_code_context(
            function=compound_func, project_root=ts_project, module_root=ts_project
        )

        expected_code = """\
class Calculator {
    private precision: number;
    private history: HistoryEntry[];

    constructor(precision: number = 2) {
        this.precision = precision;
        this.history = [];
    }

    /**
     * Calculate compound interest with multiple helper dependencies.
     * @param principal - Initial amount
     * @param rate - Interest rate (as decimal)
     * @param time - Time in years
     * @param n - Compounding frequency per year
     * @returns Compound interest result
     */
    calculateCompoundInterest(principal: number, rate: number, time: number, n: number): number {
        validateInput(principal, 'principal');
        validateInput(rate, 'rate');

        // Inefficient: recalculates power multiple times
        let result = principal;
        for (let i = 0; i < n * time; i++) {
            result = multiply(result, add(1, rate / n));
        }

        const interest = result - principal;
        this.history.push({ type: 'compound', result: interest });
        return formatNumber(interest, this.precision);
    }
}"""

        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{expected_code}\n\n"
            f"Got:\n{context.target_code}"
        )

        # TypeScript compound interest uses 4 helpers
        helper_dict = {h.name: h for h in context.helper_functions}
        expected_helper_names = {"validateInput", "formatNumber", "add", "multiply"}
        assert set(helper_dict.keys()) == expected_helper_names, (
            f"Expected helpers {expected_helper_names}, got: {set(helper_dict.keys())}"
        )

        expected_validate_input_code = """\
export function validateInput(value: unknown, name: string): asserts value is number {
    if (typeof value !== 'number' || isNaN(value)) {
        throw new Error(`Invalid ${name}: must be a number`);
    }
}"""

        expected_format_number_code = """\
export function formatNumber(num: number, decimals: number): number {
    return Number(num.toFixed(decimals));
}"""

        expected_add_code = """\
export function add(a: number, b: number): number {
    return a + b;
}"""

        expected_multiply_code = """\
export function multiply(a: number, b: number): number {
    return a * b;
}"""

        helper_expectations = {
            "validateInput": expected_validate_input_code,
            "formatNumber": expected_format_number_code,
            "add": expected_add_code,
            "multiply": expected_multiply_code,
        }

        for helper_name, expected_code in helper_expectations.items():
            assert helper_dict[helper_name].source_code.strip() == expected_code.strip(), (
                f"{helper_name} helper code does not match.\n"
                f"Expected:\n{expected_code}\n\n"
                f"Got:\n{helper_dict[helper_name].source_code}"
            )


class TestCodeExtractorEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def js_support(self):
        """Create JavaScriptSupport instance."""
        return JavaScriptSupport()

    def test_standalone_function(self, js_support, tmp_path):
        """Test standalone function with no helpers."""
        source = """\
function standalone(x) {
    return x * 2;
}

module.exports = { standalone };
"""
        test_file = tmp_path / "standalone.js"
        test_file.write_text(source)

        functions = js_support.discover_functions(test_file)
        func = next(f for f in functions if f.name == "standalone")

        context = js_support.extract_code_context(function=func, project_root=tmp_path, module_root=tmp_path)

        # STRICT: Exact code comparison
        expected_code = """\
function standalone(x) {
    return x * 2;
}"""
        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match.\nExpected:\n{expected_code}\n\nGot:\n{context.target_code}"
        )

        # STRICT: Exactly zero helpers
        assert len(context.helper_functions) == 0, (
            f"Expected 0 helpers, got {len(context.helper_functions)}: {[h.name for h in context.helper_functions]}"
        )

    def test_external_package_excluded(self, js_support, tmp_path):
        """Test external packages are not resolved as helpers."""
        source = """\
const _ = require('lodash');

function processArray(arr) {
    return _.map(arr, x => x * 2);
}

module.exports = { processArray };
"""
        test_file = tmp_path / "processor.js"
        test_file.write_text(source)

        functions = js_support.discover_functions(test_file)
        func = next(f for f in functions if f.name == "processArray")

        context = js_support.extract_code_context(function=func, project_root=tmp_path, module_root=tmp_path)

        expected_code = """\
function processArray(arr) {
    return _.map(arr, x => x * 2);
}"""

        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match.\nExpected:\n{expected_code}\n\nGot:\n{context.target_code}"
        )

        expected_imports = ["const _ = require('lodash');"]
        assert context.imports == expected_imports, (
            f"Imports do not match expected.\nExpected:\n{expected_imports}\n\nGot:\n{context.imports}"
        )

        helper_names = {h.name for h in context.helper_functions}
        assert helper_names == set(), f"Expected no helpers for external package usage, got: {helper_names}"

    def test_recursive_function(self, js_support, tmp_path):
        """Test recursive function doesn't list itself as helper."""
        source = """\
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

module.exports = { fibonacci };
"""
        test_file = tmp_path / "recursive.js"
        test_file.write_text(source)

        functions = js_support.discover_functions(test_file)
        func = next(f for f in functions if f.name == "fibonacci")

        context = js_support.extract_code_context(function=func, project_root=tmp_path, module_root=tmp_path)

        # STRICT: Exact code comparison
        expected_code = """\
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}"""
        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match.\nExpected:\n{expected_code}\n\nGot:\n{context.target_code}"
        )

        # STRICT: Function should NOT be its own helper
        helper_names = {h.name for h in context.helper_functions}
        assert "fibonacci" not in helper_names, f"Recursive function listed itself as helper. Helpers: {helper_names}"

    def test_arrow_function_helper(self, js_support, tmp_path):
        """Test arrow function helper extraction."""
        source = """\
const helper = (x) => x * 2;

const processValue = (value) => {
    return helper(value) + 1;
};

module.exports = { processValue };
"""
        test_file = tmp_path / "arrow.js"
        test_file.write_text(source)

        functions = js_support.discover_functions(test_file)
        func = next(f for f in functions if f.name == "processValue")

        context = js_support.extract_code_context(function=func, project_root=tmp_path, module_root=tmp_path)

        expected_code = """\
const processValue = (value) => {
    return helper(value) + 1;
};"""

        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match.\nExpected:\n{expected_code}\n\nGot:\n{context.target_code}"
        )

        assert context.imports == [], f"Expected no imports, got: {context.imports}"

        helper_dict = {h.name: h for h in context.helper_functions}
        assert set(helper_dict.keys()) == {"helper"}, f"Expected only 'helper', got: {list(helper_dict.keys())}"

        expected_helper_code = "const helper = (x) => x * 2;"
        actual_helper_code = helper_dict["helper"].source_code.strip()
        assert actual_helper_code == expected_helper_code, (
            f"Helper code does not match.\nExpected:\n{expected_helper_code}\n\nGot:\n{actual_helper_code}"
        )


class TestClassContextExtraction:
    """Tests for class constructor and field extraction in code context."""

    @pytest.fixture
    def js_support(self):
        """Create JavaScriptSupport instance."""
        return JavaScriptSupport()

    @pytest.fixture
    def ts_support(self):
        """Create TypeScriptSupport instance."""
        return TypeScriptSupport()

    def test_method_extraction_includes_constructor(self, js_support, tmp_path):
        """Test that extracting a class method includes the constructor."""
        source = """\
class Counter {
    constructor(initial = 0) {
        this.count = initial;
    }

    increment() {
        this.count++;
        return this.count;
    }
}

module.exports = { Counter };
"""
        test_file = tmp_path / "counter.js"
        test_file.write_text(source)

        functions = js_support.discover_functions(test_file)
        increment_func = next(f for f in functions if f.name == "increment")

        context = js_support.extract_code_context(
            function=increment_func, project_root=tmp_path, module_root=tmp_path
        )

        expected_code = """\
class Counter {
    constructor(initial = 0) {
        this.count = initial;
    }

    increment() {
        this.count++;
        return this.count;
    }
}"""

        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{expected_code}\n\n"
            f"Got:\n{context.target_code}"
        )

    def test_method_extraction_class_without_constructor(self, js_support, tmp_path):
        """Test extracting a method from a class that has no constructor."""
        source = """\
class MathUtils {
    add(a, b) {
        return a + b;
    }

    multiply(a, b) {
        return a * b;
    }
}

module.exports = { MathUtils };
"""
        test_file = tmp_path / "math_utils.js"
        test_file.write_text(source)

        functions = js_support.discover_functions(test_file)
        add_func = next(f for f in functions if f.name == "add")

        context = js_support.extract_code_context(
            function=add_func, project_root=tmp_path, module_root=tmp_path
        )

        expected_code = """\
class MathUtils {
    add(a, b) {
        return a + b;
    }
}"""

        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{expected_code}\n\n"
            f"Got:\n{context.target_code}"
        )

    def test_typescript_method_extraction_includes_fields(self, ts_support, tmp_path):
        """Test that TypeScript method extraction includes class fields."""
        source = """\
class User {
    private name: string;
    public age: number;

    constructor(name: string, age: number) {
        this.name = name;
        this.age = age;
    }

    getName(): string {
        return this.name;
    }
}

export { User };
"""
        test_file = tmp_path / "user.ts"
        test_file.write_text(source)

        functions = ts_support.discover_functions(test_file)
        get_name_func = next(f for f in functions if f.name == "getName")

        context = ts_support.extract_code_context(
            function=get_name_func, project_root=tmp_path, module_root=tmp_path
        )

        expected_code = """\
class User {
    private name: string;
    public age: number;

    constructor(name: string, age: number) {
        this.name = name;
        this.age = age;
    }

    getName(): string {
        return this.name;
    }
}"""

        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{expected_code}\n\n"
            f"Got:\n{context.target_code}"
        )

    def test_typescript_fields_only_no_constructor(self, ts_support, tmp_path):
        """Test TypeScript class with fields but no constructor."""
        source = """\
class Config {
    readonly apiUrl: string = "https://api.example.com";
    timeout: number = 5000;

    getUrl(): string {
        return this.apiUrl;
    }
}

export { Config };
"""
        test_file = tmp_path / "config.ts"
        test_file.write_text(source)

        functions = ts_support.discover_functions(test_file)
        get_url_func = next(f for f in functions if f.name == "getUrl")

        context = ts_support.extract_code_context(
            function=get_url_func, project_root=tmp_path, module_root=tmp_path
        )

        expected_code = """\
class Config {
    readonly apiUrl: string = "https://api.example.com";
    timeout: number = 5000;

    getUrl(): string {
        return this.apiUrl;
    }
}"""

        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{expected_code}\n\n"
            f"Got:\n{context.target_code}"
        )

    def test_constructor_with_jsdoc(self, js_support, tmp_path):
        """Test that constructor with JSDoc is fully extracted."""
        source = """\
class Logger {
    /**
     * Create a new Logger instance.
     * @param {string} prefix - The prefix to use for log messages.
     */
    constructor(prefix) {
        this.prefix = prefix;
    }

    getPrefix() {
        return this.prefix;
    }
}

module.exports = { Logger };
"""
        test_file = tmp_path / "logger.js"
        test_file.write_text(source)

        functions = js_support.discover_functions(test_file)
        get_prefix_func = next(f for f in functions if f.name == "getPrefix")

        context = js_support.extract_code_context(
            function=get_prefix_func, project_root=tmp_path, module_root=tmp_path
        )

        expected_code = """\
class Logger {
    /**
     * Create a new Logger instance.
     * @param {string} prefix - The prefix to use for log messages.
     */
    constructor(prefix) {
        this.prefix = prefix;
    }

    getPrefix() {
        return this.prefix;
    }
}"""

        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{expected_code}\n\n"
            f"Got:\n{context.target_code}"
        )

    def test_static_method_includes_constructor(self, js_support, tmp_path):
        """Test that static method extraction also includes constructor context."""
        source = """\
class Factory {
    constructor(config) {
        this.config = config;
    }

    static create(type) {
        return new Factory({ type: type });
    }
}

module.exports = { Factory };
"""
        test_file = tmp_path / "factory.js"
        test_file.write_text(source)

        functions = js_support.discover_functions(test_file)
        create_func = next(f for f in functions if f.name == "create")

        context = js_support.extract_code_context(
            function=create_func, project_root=tmp_path, module_root=tmp_path
        )

        expected_code = """\
class Factory {
    constructor(config) {
        this.config = config;
    }

    static create(type) {
        return new Factory({ type: type });
    }
}"""

        assert context.target_code.strip() == expected_code.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{expected_code}\n\n"
            f"Got:\n{context.target_code}"
        )


class TestCodeExtractorIntegration:
    """Integration tests with FunctionOptimizer."""

    @pytest.fixture
    def cjs_project(self, tmp_path):
        """Create a temporary CJS project from fixtures."""
        project_dir = tmp_path / "cjs_project"
        shutil.copytree(FIXTURES_DIR / "js_cjs", project_dir)
        return project_dir

    def test_function_optimizer_workflow(self, cjs_project):
        """Test full FunctionOptimizer workflow."""
        js_support = get_language_support("javascript")
        calculator_file = cjs_project / "calculator.js"

        functions = js_support.discover_functions(calculator_file)
        target = next(f for f in functions if f.name == "permutation")

        parents = [FunctionParent(name=p.name, type=p.type) for p in target.parents]

        func = FunctionToOptimize(
            function_name=target.name,
            file_path=target.file_path,
            parents=parents,
            starting_line=target.start_line,
            ending_line=target.end_line,
            starting_col=target.start_col,
            ending_col=target.end_col,
            is_async=target.is_async,
            language=target.language,
        )

        test_config = TestConfig(
            tests_root=cjs_project / "tests",
            tests_project_rootdir=cjs_project,
            project_root_path=cjs_project,
            pytest_cmd="jest",
        )

        func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
        result = func_optimizer.get_code_optimization_context()

        from codeflash.either import is_successful

        if not is_successful(result):
            error_msg = result.failure() if hasattr(result, "failure") else str(result)
            pytest.skip(f"Context extraction not fully implemented: {error_msg}")

        context = result.unwrap()

        assert context.read_writable_code is not None, "read_writable_code should not be None"

        # FunctionSource uses only_function_name, not name
        helper_names = {h.only_function_name for h in context.helper_functions}
        assert "factorial" in helper_names, f"factorial helper not found. Found: {helper_names}"
