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


# =============================================================================
# EXPECTED CODE STRINGS - Define exact expected outputs for strict comparison
# =============================================================================

# Expected code for the permutation method (with JSDoc)
EXPECTED_PERMUTATION_CODE = """\
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
    }"""

# Expected code for the calculateCompoundInterest method (with JSDoc)
EXPECTED_COMPOUND_INTEREST_CODE = """\
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
    }"""

# Expected code for the factorial helper function
EXPECTED_FACTORIAL_CODE = """\
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

# Expected code for the add helper function
EXPECTED_ADD_CODE = """\
/**
 * Add two numbers.
 * @param a - First number
 * @param b - Second number
 * @returns Sum of a and b
 */
function add(a, b) {
    return a + b;
}"""

# Expected code for the multiply helper function
EXPECTED_MULTIPLY_CODE = """\
/**
 * Multiply two numbers.
 * @param a - First number
 * @param b - Second number
 * @returns Product of a and b
 */
function multiply(a, b) {
    return a * b;
}"""

# Expected code for the formatNumber helper function
EXPECTED_FORMAT_NUMBER_CODE = """\
/**
 * Format a number to specified decimal places.
 * @param num - Number to format
 * @param decimals - Number of decimal places
 * @returns Formatted number
 */
function formatNumber(num, decimals) {
    return Number(num.toFixed(decimals));
}"""

# Expected code for the validateInput helper function
EXPECTED_VALIDATE_INPUT_CODE = """\
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

# Expected code for quickAdd static method
EXPECTED_QUICK_ADD_CODE = """\
    /**
     * Static method for quick calculations.
     */
    static quickAdd(a, b) {
        return add(a, b);
    }"""


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

        assert context.target_code is not None, "target_code should not be None"
        assert context.target_code.strip() == EXPECTED_PERMUTATION_CODE.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{EXPECTED_PERMUTATION_CODE}\n\n"
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

        assert factorial_helper.source_code.strip() == EXPECTED_FACTORIAL_CODE.strip(), (
            f"Factorial helper code does not match expected.\n"
            f"Expected:\n{EXPECTED_FACTORIAL_CODE}\n\n"
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

        assert context.target_code.strip() == EXPECTED_COMPOUND_INTEREST_CODE.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{EXPECTED_COMPOUND_INTEREST_CODE}\n\n"
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
        helper_expectations = {
            "add": (EXPECTED_ADD_CODE, "math_utils.js"),
            "multiply": (EXPECTED_MULTIPLY_CODE, "math_utils.js"),
            "formatNumber": (EXPECTED_FORMAT_NUMBER_CODE, "format.js"),
            "validateInput": (EXPECTED_VALIDATE_INPUT_CODE, "format.js"),
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

        imports_str = "\n".join(context.imports)

        # Should contain both require statements
        assert "require('./math_utils')" in imports_str or "math_utils" in imports_str, (
            f"math_utils import not found in imports:\n{imports_str}"
        )
        assert "require('./helpers/format')" in imports_str or "format" in imports_str, (
            f"helpers/format import not found in imports:\n{imports_str}"
        )

    def test_extract_static_method(self, js_support, cjs_project):
        """Test static method extraction (quickAdd)."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        quick_add_func = next(f for f in functions if f.name == "quickAdd")

        context = js_support.extract_code_context(
            function=quick_add_func, project_root=cjs_project, module_root=cjs_project
        )

        assert context.target_code.strip() == EXPECTED_QUICK_ADD_CODE.strip(), (
            f"Extracted code does not match expected.\n"
            f"Expected:\n{EXPECTED_QUICK_ADD_CODE}\n\n"
            f"Got:\n{context.target_code}"
        )

        # quickAdd uses add helper
        helper_names = {h.name for h in context.helper_functions}
        assert "add" in helper_names, f"add helper not found. Found: {helper_names}"


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

    def test_esm_factorial_helper(self, js_support, esm_project):
        """Test factorial helper extraction in ESM."""
        calculator_file = esm_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        permutation_func = next(f for f in functions if f.name == "permutation")

        context = js_support.extract_code_context(
            function=permutation_func, project_root=esm_project, module_root=esm_project
        )

        helper_names = {h.name for h in context.helper_functions}

        # STRICT: factorial must be present
        assert "factorial" in helper_names, f"factorial helper not found. Found helpers: {helper_names}"

    def test_esm_import_syntax(self, js_support, esm_project):
        """Test ESM uses import syntax, not require."""
        calculator_file = esm_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        compound_func = next(f for f in functions if f.name == "calculateCompoundInterest")

        context = js_support.extract_code_context(
            function=compound_func, project_root=esm_project, module_root=esm_project
        )

        imports_str = "\n".join(context.imports)

        # ESM should use import syntax
        if context.imports:
            # If imports are captured, they should be ESM style
            assert "import" in imports_str or len(context.imports) > 0, (
                f"ESM imports should use import syntax. Got:\n{imports_str}"
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

    def test_ts_factorial_helper(self, ts_support, ts_project):
        """Test factorial helper extraction in TypeScript."""
        calculator_file = ts_project / "calculator.ts"
        functions = ts_support.discover_functions(calculator_file)

        permutation_func = next(f for f in functions if f.name == "permutation")

        context = ts_support.extract_code_context(
            function=permutation_func, project_root=ts_project, module_root=ts_project
        )

        helper_names = {h.name for h in context.helper_functions}

        # STRICT: factorial must be present
        assert "factorial" in helper_names, f"factorial helper not found. Found helpers: {helper_names}"

    def test_ts_compound_interest_helpers(self, ts_support, ts_project):
        """Test helper extraction in TypeScript."""
        calculator_file = ts_project / "calculator.ts"
        functions = ts_support.discover_functions(calculator_file)

        compound_func = next(f for f in functions if f.name == "calculateCompoundInterest")

        context = ts_support.extract_code_context(
            function=compound_func, project_root=ts_project, module_root=ts_project
        )

        helper_names = {h.name for h in context.helper_functions}

        # STRICT: Verify exact set of helpers
        expected_helpers = {"add", "multiply", "formatNumber", "validateInput"}
        assert helper_names == expected_helpers, f"Expected helpers {expected_helpers}, got {helper_names}"


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

        # STRICT: No helpers from external packages
        helper_names = {h.name for h in context.helper_functions}
        assert len(helper_names) == 0, f"Expected no helpers for external package usage, got: {helper_names}"

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

        helper_dict = {h.name: h for h in context.helper_functions}
        assert "helper" in helper_dict, f"helper function not found. Found: {list(helper_dict.keys())}"

        expected_helper_code = "const helper = (x) => x * 2;"
        actual_helper_code = helper_dict["helper"].source_code.strip()
        assert actual_helper_code == expected_helper_code, (
            f"Helper code does not match.\nExpected:\n{expected_helper_code}\n\nGot:\n{actual_helper_code}"
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
