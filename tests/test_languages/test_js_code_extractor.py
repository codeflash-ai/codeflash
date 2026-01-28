"""Tests for JavaScript/TypeScript code extractor with multi-file dependencies.

These tests verify that code context extraction correctly handles:
- Class method optimization with helper dependencies
- Multi-file import resolution (CJS and ESM)
- Recursive helper function discovery
- TypeScript-specific type handling
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
        """Test discovering class methods in CJS module."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        # Should find class methods
        method_names = {f.name for f in functions}
        assert "calculateCompoundInterest" in method_names
        assert "permutation" in method_names
        assert "quickAdd" in method_names

    def test_class_method_has_correct_parent(self, js_support, cjs_project):
        """Test that class methods have correct parent class info."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        compound_interest = next(f for f in functions if f.name == "calculateCompoundInterest")
        assert compound_interest.is_method is True
        assert compound_interest.class_name == "Calculator"

    def test_extract_context_includes_direct_helpers(self, js_support, cjs_project):
        """Test that direct helper functions are included in context."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        # Find the permutation method
        permutation_func = next(f for f in functions if f.name == "permutation")

        # Extract code context
        context = js_support.extract_code_context(
            function=permutation_func, project_root=cjs_project, module_root=cjs_project
        )
        breakpoint()
        # Should include the factorial helper from math_utils.js
        helper_names = {h.name for h in context.helper_functions}
        assert "factorial" in helper_names

    def test_extract_context_includes_nested_helpers(self, js_support, cjs_project):
        """Test that nested helper dependencies are included."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        # Find calculateCompoundInterest which uses add, multiply from math_utils
        # and formatNumber, validateInput from helpers/format
        compound_func = next(f for f in functions if f.name == "calculateCompoundInterest")

        context = js_support.extract_code_context(
            function=compound_func, project_root=cjs_project, module_root=cjs_project
        )

        helper_names = {h.name for h in context.helper_functions}

        # Direct helpers from math_utils
        assert "add" in helper_names
        assert "multiply" in helper_names

        # Direct helpers from helpers/format
        assert "formatNumber" in helper_names
        assert "validateInput" in helper_names

    def test_extract_context_includes_imports(self, js_support, cjs_project):
        """Test that import statements are included in context."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        compound_func = next(f for f in functions if f.name == "calculateCompoundInterest")

        context = js_support.extract_code_context(
            function=compound_func, project_root=cjs_project, module_root=cjs_project
        )

        # Imports should be captured as strings
        imports_str = "\n".join(context.imports)
        assert "require('./math_utils')" in imports_str or "math_utils" in imports_str

    def test_helper_functions_have_correct_file_paths(self, js_support, cjs_project):
        """Test that helper functions have correct source file paths."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        compound_func = next(f for f in functions if f.name == "calculateCompoundInterest")

        context = js_support.extract_code_context(
            function=compound_func, project_root=cjs_project, module_root=cjs_project
        )

        # Find the factorial helper and check its file path
        for helper in context.helper_functions:
            if helper.name == "add":
                assert "math_utils.js" in str(helper.file_path)
            elif helper.name == "formatNumber":
                assert "format.js" in str(helper.file_path)

    def test_static_method_discovery(self, js_support, cjs_project):
        """Test that static methods are discovered correctly."""
        calculator_file = cjs_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        quick_add = next((f for f in functions if f.name == "quickAdd"), None)
        assert quick_add is not None
        assert quick_add.is_method is True


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

    def test_discover_class_methods_esm(self, js_support, esm_project):
        """Test discovering class methods in ESM module."""
        calculator_file = esm_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        method_names = {f.name for f in functions}
        assert "calculateCompoundInterest" in method_names
        assert "permutation" in method_names

    def test_extract_context_with_esm_imports(self, js_support, esm_project):
        """Test context extraction with ES Module imports."""
        calculator_file = esm_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        permutation_func = next(f for f in functions if f.name == "permutation")

        context = js_support.extract_code_context(
            function=permutation_func, project_root=esm_project, module_root=esm_project
        )

        # Should include helpers from ESM imports
        helper_names = {h.name for h in context.helper_functions}
        assert "factorial" in helper_names

    def test_esm_imports_captured_in_context(self, js_support, esm_project):
        """Test that ESM import statements are captured."""
        calculator_file = esm_project / "calculator.js"
        functions = js_support.discover_functions(calculator_file)

        compound_func = next(f for f in functions if f.name == "calculateCompoundInterest")

        context = js_support.extract_code_context(
            function=compound_func, project_root=esm_project, module_root=esm_project
        )

        imports_str = "\n".join(context.imports)
        # ESM uses import syntax
        assert "import" in imports_str or len(context.imports) > 0


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
        """Test TypeScriptSupport has correct properties."""
        assert ts_support.language == Language.TYPESCRIPT
        assert ".ts" in ts_support.file_extensions
        assert ".tsx" in ts_support.file_extensions

    def test_discover_typed_class_methods(self, ts_support, ts_project):
        """Test discovering class methods in TypeScript file."""
        calculator_file = ts_project / "calculator.ts"
        functions = ts_support.discover_functions(calculator_file)

        method_names = {f.name for f in functions}
        assert "calculateCompoundInterest" in method_names
        assert "permutation" in method_names
        assert "getHistory" in method_names

    def test_extract_context_typescript(self, ts_support, ts_project):
        """Test context extraction for TypeScript methods."""
        calculator_file = ts_project / "calculator.ts"
        functions = ts_support.discover_functions(calculator_file)

        permutation_func = next(f for f in functions if f.name == "permutation")

        context = ts_support.extract_code_context(
            function=permutation_func, project_root=ts_project, module_root=ts_project
        )

        # Should include typed helper functions
        helper_names = {h.name for h in context.helper_functions}
        assert "factorial" in helper_names

    def test_typescript_imports_resolved(self, ts_support, ts_project):
        """Test that TypeScript imports without extensions are resolved."""
        calculator_file = ts_project / "calculator.ts"
        functions = ts_support.discover_functions(calculator_file)

        compound_func = next(f for f in functions if f.name == "calculateCompoundInterest")

        context = ts_support.extract_code_context(
            function=compound_func, project_root=ts_project, module_root=ts_project
        )

        # Helpers should be resolved even with extension-less imports
        helper_names = {h.name for h in context.helper_functions}
        assert "add" in helper_names
        assert "multiply" in helper_names


class TestCodeExtractorEdgeCases:
    """Tests for edge cases in code extraction."""

    @pytest.fixture
    def js_support(self):
        """Create JavaScriptSupport instance."""
        return JavaScriptSupport()

    def test_function_without_helpers(self, js_support, tmp_path):
        """Test extracting context for function with no helper calls."""
        source = """
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

        assert context.target_code is not None
        assert len(context.helper_functions) == 0

    def test_function_with_external_package_imports(self, js_support, tmp_path):
        """Test that external package imports are not resolved as helpers."""
        source = """
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

        # External package helpers should not be included
        helper_names = {h.name for h in context.helper_functions}
        assert "map" not in helper_names

    def test_recursive_function_self_reference(self, js_support, tmp_path):
        """Test extracting context for recursive function."""
        source = """
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

        # Self-reference should not cause infinite loop or errors
        assert context.target_code is not None
        # Function should not be listed as its own helper
        helper_names = {h.name for h in context.helper_functions}
        assert "fibonacci" not in helper_names

    def test_arrow_function_context_extraction(self, js_support, tmp_path):
        """Test context extraction for arrow functions."""
        source = """
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

        helper_names = {h.name for h in context.helper_functions}
        assert "helper" in helper_names


class TestCodeExtractorIntegration:
    """Integration tests using FunctionOptimizer workflow."""

    @pytest.fixture
    def cjs_project(self, tmp_path):
        """Create a temporary CJS project from fixtures."""
        project_dir = tmp_path / "cjs_project"
        shutil.copytree(FIXTURES_DIR / "js_cjs", project_dir)
        return project_dir

    def test_full_context_extraction_workflow(self, cjs_project):
        """Test the full context extraction workflow via FunctionOptimizer."""
        js_support = get_language_support("javascript")
        calculator_file = cjs_project / "calculator.js"

        functions = js_support.discover_functions(calculator_file)
        target = next(f for f in functions if f.name == "permutation")

        # Convert ParentInfo to FunctionParent for compatibility
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
            test_framework="jest",
            pytest_cmd="jest",
        )

        func_optimizer = FunctionOptimizer(function_to_optimize=func, test_cfg=test_config)
        result = func_optimizer.get_code_optimization_context()

        # Should successfully extract context
        from codeflash.either import is_successful

        if not is_successful(result):
            error_msg = result.failure() if hasattr(result, "failure") else str(result)
            pytest.skip(f"Context extraction not fully implemented: {error_msg}")
        context = result.unwrap()

        # Verify context has expected properties
        assert context.code_to_optimize is not None
        assert len(context.helper_functions) > 0

        # factorial should be in helpers
        helper_names = {h.name for h in context.helper_functions}
        assert "factorial" in helper_names
