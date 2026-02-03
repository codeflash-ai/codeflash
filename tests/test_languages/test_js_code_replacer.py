"""Tests for JavaScript/TypeScript code replacement with import handling.

These tests verify that code replacement correctly handles:
- New imports added during optimization
- Import organization and merging
- CommonJS (require/module.exports) module syntax
- ES Modules (import/export) syntax
- TypeScript import handling
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from codeflash.code_utils.code_replacer import replace_function_definitions_for_language
from codeflash.languages.javascript.module_system import (
    ModuleSystem,
    convert_commonjs_to_esm,
    convert_esm_to_commonjs,
    detect_module_system,
    ensure_module_system_compatibility,
    get_import_statement,
)

from codeflash.languages.javascript.support import JavaScriptSupport, TypeScriptSupport
from codeflash.models.models import CodeStringsMarkdown


@pytest.fixture
def js_support():
    """Create a JavaScriptSupport instance."""
    return JavaScriptSupport()


@pytest.fixture
def ts_support():
    """Create a TypeScriptSupport instance."""
    return TypeScriptSupport()


@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project directory structure."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    return project_root



FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestModuleSystemDetection:
    """Tests for module system detection."""

    def test_detect_esm_from_package_json(self, tmp_path):
        """Test detecting ES Module from package.json type field."""
        package_json = tmp_path / "package.json"
        package_json.write_text('{"name": "test", "type": "module"}')

        result = detect_module_system(tmp_path)
        assert result == ModuleSystem.ES_MODULE, f"Expected ES_MODULE, got {result}"

    def test_detect_commonjs_from_package_json(self, tmp_path):
        """Test detecting CommonJS from package.json type field."""
        package_json = tmp_path / "package.json"
        package_json.write_text('{"name": "test", "type": "commonjs"}')

        result = detect_module_system(tmp_path)
        assert result == ModuleSystem.COMMONJS, f"Expected COMMONJS, got {result}"

    def test_detect_esm_from_mjs_extension(self, tmp_path):
        """Test detecting ES Module from .mjs extension."""
        test_file = tmp_path / "module.mjs"
        test_file.write_text("export function foo() {}")

        result = detect_module_system(tmp_path, file_path=test_file)
        assert result == ModuleSystem.ES_MODULE, f"Expected ES_MODULE for .mjs file, got {result}"

    def test_detect_commonjs_from_cjs_extension(self, tmp_path):
        """Test detecting CommonJS from .cjs extension."""
        test_file = tmp_path / "module.cjs"
        test_file.write_text("module.exports = { foo: () => {} };")

        result = detect_module_system(tmp_path, file_path=test_file)
        assert result == ModuleSystem.COMMONJS, f"Expected COMMONJS for .cjs file, got {result}"

    def test_detect_esm_from_import_syntax(self, tmp_path):
        """Test detecting ES Module from import/export syntax in file."""
        test_file = tmp_path / "module.js"
        source = """\
import { helper } from './helper.js';

export function process(x) {
    return helper(x);
}
"""
        test_file.write_text(source)

        result = detect_module_system(tmp_path, file_path=test_file)
        assert result == ModuleSystem.ES_MODULE, f"Expected ES_MODULE for file with import syntax, got {result}"

    def test_detect_commonjs_from_require_syntax(self, tmp_path):
        """Test detecting CommonJS from require/module.exports syntax."""
        test_file = tmp_path / "module.js"
        source = """\
const { helper } = require('./helper');

function process(x) {
    return helper(x);
}

module.exports = { process };
"""
        test_file.write_text(source)

        result = detect_module_system(tmp_path, file_path=test_file)
        assert result == ModuleSystem.COMMONJS, f"Expected COMMONJS for file with require syntax, got {result}"

    def test_detect_from_fixtures_cjs(self):
        """Test detection on actual CJS fixture."""
        cjs_dir = FIXTURES_DIR / "js_cjs"
        if not cjs_dir.exists():
            pytest.skip("CJS fixture not available")

        calculator_file = cjs_dir / "calculator.js"
        result = detect_module_system(cjs_dir, file_path=calculator_file)
        assert result == ModuleSystem.COMMONJS, f"Expected COMMONJS for CJS fixture, got {result}"

    def test_detect_from_fixtures_esm(self):
        """Test detection on actual ESM fixture."""
        esm_dir = FIXTURES_DIR / "js_esm"
        if not esm_dir.exists():
            pytest.skip("ESM fixture not available")

        package_json = esm_dir / "package.json"
        if not package_json.exists():
            package_json.write_text('{"name": "test", "type": "module"}')

        calculator_file = esm_dir / "calculator.js"
        result = detect_module_system(esm_dir, file_path=calculator_file)
        assert result == ModuleSystem.ES_MODULE, f"Expected ES_MODULE for ESM fixture, got {result}"


class TestCommonJSToESMConversion:
    """Tests for CommonJS to ES Module import conversion."""

    def test_convert_simple_require(self):
        """Test converting simple require to import - exact output."""
        code = "const lodash = require('lodash');"
        result = convert_commonjs_to_esm(code)

        expected = "import lodash from 'lodash';"
        assert result.strip() == expected, (
            f"CJS to ESM conversion failed.\nInput: {code}\nExpected: {expected}\nGot: {result}"
        )

    def test_convert_destructured_require(self):
        """Test converting destructured require to named import - exact output."""
        code = "const { map, filter } = require('lodash');"
        result = convert_commonjs_to_esm(code)

        expected = "import { map, filter } from 'lodash';"
        assert result.strip() == expected, (
            f"CJS to ESM conversion failed.\nInput: {code}\nExpected: {expected}\nGot: {result}"
        )

    def test_convert_relative_require_preserves_path(self):
        """Test that relative imports preserve the original path without adding extension."""
        code = "const { helper } = require('./utils');"
        result = convert_commonjs_to_esm(code)

        expected = "import { helper } from './utils';"
        assert result.strip() == expected, (
            f"CJS to ESM conversion failed.\nInput: {code}\nExpected: {expected}\nGot: {result}"
        )

    def test_convert_property_access_require(self):
        """Test converting property access require - exact output."""
        code = "const myHelper = require('./utils').helperFunction;"
        result = convert_commonjs_to_esm(code)

        expected = "import { helperFunction as myHelper } from './utils';"
        assert result.strip() == expected, (
            f"CJS to ESM conversion failed.\nInput: {code}\nExpected: {expected}\nGot: {result}"
        )

    def test_convert_default_property_access(self):
        """Test converting .default property access - exact output."""
        code = "const MyClass = require('./class').default;"
        result = convert_commonjs_to_esm(code)

        expected = "import MyClass from './class';"
        assert result.strip() == expected, (
            f"CJS to ESM conversion failed.\nInput: {code}\nExpected: {expected}\nGot: {result}"
        )

    def test_convert_multiple_requires(self):
        """Test converting multiple require statements - exact output."""
        code = """\
const { add, subtract } = require('./math');
const lodash = require('lodash');
const path = require('path');"""

        result = convert_commonjs_to_esm(code)

        expected = """\
import { add, subtract } from './math';
import lodash from 'lodash';
import path from 'path';"""

        assert result.strip() == expected.strip(), (
            f"CJS to ESM conversion failed.\nInput:\n{code}\n\nExpected:\n{expected}\n\nGot:\n{result}"
        )

    def test_preserves_function_code(self):
        """Test that non-require code is preserved exactly."""
        code = """\
const { add } = require('./math');

function calculate(x, y) {
    return add(x, y);
}

module.exports = { calculate };
"""
        result = convert_commonjs_to_esm(code)

        # The function body should be preserved exactly
        assert "function calculate(x, y) {" in result
        assert "return add(x, y);" in result


class TestESMToCommonJSConversion:
    """Tests for ES Module to CommonJS import conversion."""

    def test_convert_default_import(self):
        """Test converting default import to require - exact output."""
        code = "import lodash from 'lodash';"
        result = convert_esm_to_commonjs(code)

        expected = "const lodash = require('lodash');"
        assert result.strip() == expected, (
            f"ESM to CJS conversion failed.\nInput: {code}\nExpected: {expected}\nGot: {result}"
        )

    def test_convert_named_import(self):
        """Test converting named import to destructured require - exact output."""
        code = "import { map, filter } from 'lodash';"
        result = convert_esm_to_commonjs(code)

        expected = "const { map, filter } = require('lodash');"
        assert result.strip() == expected, (
            f"ESM to CJS conversion failed.\nInput: {code}\nExpected: {expected}\nGot: {result}"
        )

    def test_convert_relative_import_removes_extension(self):
        """Test that relative imports have .js extension removed - exact output."""
        code = "import { helper } from './utils.js';"
        result = convert_esm_to_commonjs(code)

        expected = "const { helper } = require('./utils');"
        assert result.strip() == expected, (
            f"ESM to CJS conversion failed.\nInput: {code}\nExpected: {expected}\nGot: {result}"
        )

    def test_convert_multiple_imports(self):
        """Test converting multiple import statements - exact output."""
        code = """\
import { add, subtract } from './math.js';
import lodash from 'lodash';
import path from 'path';"""

        result = convert_esm_to_commonjs(code)

        expected = """\
const { add, subtract } = require('./math');
const lodash = require('lodash');
const path = require('path');"""

        assert result.strip() == expected.strip(), (
            f"ESM to CJS conversion failed.\nInput:\n{code}\n\nExpected:\n{expected}\n\nGot:\n{result}"
        )

    def test_preserves_function_code(self):
        """Test that non-import code is preserved exactly."""
        code = """\
import { add } from './math.js';

export function calculate(x, y) {
    return add(x, y);
}
"""
        result = convert_esm_to_commonjs(code)

        # The function body should be preserved
        assert "function calculate(x, y)" in result
        assert "return add(x, y);" in result


class TestModuleSystemCompatibility:
    """Tests for module system compatibility."""

    def test_convert_mixed_code_to_esm(self):
        """Test converting mixed CJS/ESM code to pure ESM - exact output."""
        code = """\
import { existing } from './module.js';
const { helper } = require('./helpers');

function process() {
    return existing() + helper();
}
"""
        result = ensure_module_system_compatibility(code, ModuleSystem.ES_MODULE)

        # Should convert require to import
        assert "import { helper } from './helpers';" in result
        assert "require" not in result, f"require should be converted to import. Got:\n{result}"

    def test_convert_mixed_code_to_commonjs(self):
        """Test converting mixed ESM/CJS code to pure CommonJS - exact output."""
        code = """\
const { existing } = require('./module');
import { helper } from './helpers.js';

function process() {
    return existing() + helper();
}
"""
        result = ensure_module_system_compatibility(code, ModuleSystem.COMMONJS)

        # Should convert import to require
        assert "const { helper } = require('./helpers');" in result
        assert "import " not in result.split("\n")[0] or "import " not in result, (
            f"import should be converted to require. Got:\n{result}"
        )

    def test_pure_esm_unchanged(self):
        """Test that pure ESM code is unchanged when targeting ESM."""
        code = """\
import { add } from './math.js';

export function sum(a, b) {
    return add(a, b);
}
"""
        result = ensure_module_system_compatibility(code, ModuleSystem.ES_MODULE)
        assert result == code, f"Pure ESM code should be unchanged.\nExpected:\n{code}\n\nGot:\n{result}"

    def test_pure_commonjs_unchanged(self):
        """Test that pure CommonJS code is unchanged when targeting CommonJS."""
        code = """\
const { add } = require('./math');

function sum(a, b) {
    return add(a, b);
}

module.exports = { sum };
"""
        result = ensure_module_system_compatibility(code, ModuleSystem.COMMONJS)
        assert result == code, f"Pure CommonJS code should be unchanged.\nExpected:\n{code}\n\nGot:\n{result}"


class TestImportStatementGeneration:
    """Tests for generating import statements."""

    def test_generate_esm_named_import(self, tmp_path):
        """Test generating ESM named import statement - exact output."""
        target = tmp_path / "utils.js"
        source = tmp_path / "main.js"

        result = get_import_statement(ModuleSystem.ES_MODULE, target, source, imported_names=["helper", "process"])

        expected = "import { helper, process } from './utils';"
        assert result == expected, f"Import statement generation failed.\nExpected: {expected}\nGot: {result}"

    def test_generate_esm_default_import(self, tmp_path):
        """Test generating ESM default import statement - exact output."""
        target = tmp_path / "module.js"
        source = tmp_path / "main.js"

        result = get_import_statement(ModuleSystem.ES_MODULE, target, source)

        expected = "import module from './module';"
        assert result == expected, f"Import statement generation failed.\nExpected: {expected}\nGot: {result}"

    def test_generate_commonjs_named_require(self, tmp_path):
        """Test generating CommonJS destructured require - exact output."""
        target = tmp_path / "utils.js"
        source = tmp_path / "main.js"

        result = get_import_statement(ModuleSystem.COMMONJS, target, source, imported_names=["helper", "process"])

        expected = "const { helper, process } = require('./utils');"
        assert result == expected, f"Import statement generation failed.\nExpected: {expected}\nGot: {result}"

    def test_generate_commonjs_default_require(self, tmp_path):
        """Test generating CommonJS default require - exact output."""
        target = tmp_path / "module.js"
        source = tmp_path / "main.js"

        result = get_import_statement(ModuleSystem.COMMONJS, target, source)

        expected = "const module = require('./module');"
        assert result == expected, f"Import statement generation failed.\nExpected: {expected}\nGot: {result}"

    def test_generate_nested_path_import(self, tmp_path):
        """Test generating import for nested directory structure - exact path."""
        subdir = tmp_path / "src" / "utils"
        subdir.mkdir(parents=True)
        target = subdir / "helper.js"
        source = tmp_path / "main.js"

        result = get_import_statement(ModuleSystem.ES_MODULE, target, source, imported_names=["helper"])

        # Should contain the nested path
        assert "src/utils/helper" in result, f"Nested path not found in import.\nGot: {result}"
        assert "import { helper }" in result, f"Named import syntax not found.\nGot: {result}"

    def test_generate_parent_directory_import(self, tmp_path):
        """Test generating import that navigates to parent directory."""
        subdir = tmp_path / "src"
        subdir.mkdir()
        target = tmp_path / "shared" / "utils.js"
        target.parent.mkdir()
        source = subdir / "main.js"

        result = get_import_statement(ModuleSystem.ES_MODULE, target, source, imported_names=["helper"])

        # Should contain parent directory navigation
        assert "../shared/utils" in result, f"Parent directory path not found in import.\nGot: {result}"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_dynamic_import_preserved(self):
        """Test that dynamic imports are preserved during conversion."""
        code = """\
const { helper } = require('./utils');

async function loadModule() {
    const mod = await import('./dynamic-module.js');
    return mod.default;
}

module.exports = { loadModule };
"""
        result = convert_commonjs_to_esm(code)

        # Dynamic import should remain unchanged
        assert "await import('./dynamic-module.js')" in result, f"Dynamic import was modified.\nGot:\n{result}"
        # Static require should be converted
        assert "import { helper } from './utils.js';" in result, f"Static require was not converted.\nGot:\n{result}"

    def test_multiline_destructured_require(self):
        """Test conversion of multiline destructured require."""
        code = """\
const {
    helper1,
    helper2,
    helper3
} = require('./utils');
"""
        result = convert_commonjs_to_esm(code)

        # Should convert to import syntax
        assert "import" in result, f"Multiline require was not converted.\nGot:\n{result}"
        # All names should be present
        assert "helper1" in result
        assert "helper2" in result
        assert "helper3" in result

    def test_require_with_variable_unchanged(self):
        """Test that dynamic require with variable is unchanged."""
        code = """\
const moduleName = 'lodash';
const mod = require(moduleName);
"""
        result = convert_commonjs_to_esm(code)

        # Dynamic require with variable should be unchanged
        assert "require(moduleName)" in result, f"Dynamic require was incorrectly modified.\nGot:\n{result}"

    def test_empty_file_handling(self):
        """Test handling of empty file."""
        code = ""
        result_esm = convert_commonjs_to_esm(code)
        result_cjs = convert_esm_to_commonjs(code)

        assert result_esm == "", f"Empty file should remain empty after ESM conversion.\nGot: '{result_esm}'"
        assert result_cjs == "", f"Empty file should remain empty after CJS conversion.\nGot: '{result_cjs}'"

    def test_no_imports_file_preserved(self):
        """Test file with no imports is preserved exactly."""
        code = """\
function standalone() {
    return 42;
}

module.exports = { standalone };
"""
        result = convert_commonjs_to_esm(code)

        # Function should be preserved
        assert "function standalone()" in result
        assert "return 42;" in result


class TestIntegrationWithFixtures:
    """Integration tests using fixture files."""

    @pytest.fixture
    def cjs_project(self, tmp_path):
        """Create a temporary CJS project from fixtures."""
        project_dir = tmp_path / "cjs_project"
        if (FIXTURES_DIR / "js_cjs").exists():
            shutil.copytree(FIXTURES_DIR / "js_cjs", project_dir)
        return project_dir

    @pytest.fixture
    def esm_project(self, tmp_path):
        """Create a temporary ESM project from fixtures."""
        project_dir = tmp_path / "esm_project"
        if (FIXTURES_DIR / "js_esm").exists():
            shutil.copytree(FIXTURES_DIR / "js_esm", project_dir)
        return project_dir

    @pytest.fixture
    def ts_project(self, tmp_path):
        """Create a temporary TypeScript project from fixtures."""
        project_dir = tmp_path / "ts_project"
        if (FIXTURES_DIR / "ts").exists():
            shutil.copytree(FIXTURES_DIR / "ts", project_dir)
        return project_dir

    def test_cjs_fixture_detected_as_commonjs(self, cjs_project):
        """Test that CJS fixture is correctly detected as CommonJS."""
        if not cjs_project.exists():
            pytest.skip("CJS fixture not available")

        calculator_file = cjs_project / "calculator.js"
        if not calculator_file.exists():
            pytest.skip("Calculator file not available")

        result = detect_module_system(cjs_project, file_path=calculator_file)
        assert result == ModuleSystem.COMMONJS, f"Expected COMMONJS for CJS fixture, got {result}"

    def test_esm_fixture_detected_as_esmodule(self, esm_project):
        """Test that ESM fixture is correctly detected as ES Module."""
        if not esm_project.exists():
            pytest.skip("ESM fixture not available")

        package_json = esm_project / "package.json"
        if not package_json.exists():
            package_json.write_text('{"name": "test", "type": "module"}')

        calculator_file = esm_project / "calculator.js"
        if not calculator_file.exists():
            pytest.skip("Calculator file not available")

        result = detect_module_system(esm_project, file_path=calculator_file)
        assert result == ModuleSystem.ES_MODULE, f"Expected ES_MODULE for ESM fixture, got {result}"

    def test_ts_fixture_detected_correctly(self, ts_project):
        """Test that TypeScript fixture module detection works."""
        if not ts_project.exists():
            pytest.skip("TypeScript fixture not available")

        package_json = ts_project / "package.json"
        if not package_json.exists():
            package_json.write_text('{"name": "test", "type": "module"}')

        calculator_file = ts_project / "calculator.ts"
        if not calculator_file.exists():
            pytest.skip("Calculator file not available")

        result = detect_module_system(ts_project, file_path=calculator_file)
        assert result == ModuleSystem.ES_MODULE, f"Expected ES_MODULE for TypeScript with ESM config, got {result}"

    def test_cjs_fixture_conversion_removes_require(self, cjs_project):
        """Test converting CJS fixture code to ESM removes require."""
        if not cjs_project.exists():
            pytest.skip("CJS fixture not available")

        calculator_file = cjs_project / "calculator.js"
        if not calculator_file.exists():
            pytest.skip("Calculator file not available")

        original_code = calculator_file.read_text()
        esm_code = convert_commonjs_to_esm(original_code)

        # Verify conversion happened
        if "require(" in original_code:
            assert "require(" not in esm_code or "require('" not in esm_code, (
                f"require statements should be converted to import.\n"
                f"Original had require, converted still has require:\n{esm_code[:500]}"
            )

    def test_esm_fixture_conversion_removes_import(self, esm_project):
        """Test converting ESM fixture code to CommonJS removes import."""
        if not esm_project.exists():
            pytest.skip("ESM fixture not available")

        calculator_file = esm_project / "calculator.js"
        if not calculator_file.exists():
            pytest.skip("Calculator file not available")

        original_code = calculator_file.read_text()
        cjs_code = convert_esm_to_commonjs(original_code)

        # If original had imports, they should be converted
        if "import " in original_code:
            # Static imports at start of lines should be converted
            # Note: This is a basic check
            lines = cjs_code.strip().split("\n")
            import_lines = [l for l in lines if l.strip().startswith("import ")]
            assert len(import_lines) == 0, (
                f"import statements should be converted to require.\nFound import lines: {import_lines}"
            )

class TestSimpleFunctionReplacement:
    """Tests for simple function body replacement with strict assertions."""

    def test_replace_simple_function_body(self, js_support, temp_project):
        """Test replacing a simple function body preserves structure exactly."""
        original_source = """\
function add(a, b) {
    return a + b;
}
"""
        file_path = temp_project / "math.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        # Optimized version with different body
        optimized_code = """\
function add(a, b) {
    // Optimized: direct return
    return a + b;
}
"""

        result = js_support.replace_function(original_source, func, optimized_code)

        expected_result = """\
function add(a, b) {
    // Optimized: direct return
    return a + b;
}
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True

    def test_replace_function_with_multiple_statements(self, js_support, temp_project):
        """Test replacing function with complex multi-statement body."""
        original_source = """\
function processData(data) {
    const result = [];
    for (let i = 0; i < data.length; i++) {
        result.push(data[i] * 2);
    }
    return result;
}
"""
        file_path = temp_project / "processor.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        # Optimized version using map
        optimized_code = """\
function processData(data) {
    return data.map(x => x * 2);
}
"""

        result = js_support.replace_function(original_source, func, optimized_code)

        expected_result = """\
function processData(data) {
    return data.map(x => x * 2);
}
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True

    def test_replace_preserves_surrounding_code(self, js_support, temp_project):
        """Test that replacement preserves code before and after the function."""
        original_source = """\
const CONFIG = { debug: true };

function targetFunction(x) {
    console.log(x);
    return x * 2;
}

function otherFunction(y) {
    return y + 1;
}

module.exports = { targetFunction, otherFunction };
"""
        file_path = temp_project / "module.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        target_func = next(f for f in functions if f.function_name == "targetFunction")

        optimized_code = """\
function targetFunction(x) {
    return x << 1;
}
"""

        result = js_support.replace_function(original_source, target_func, optimized_code)

        expected_result = """\
const CONFIG = { debug: true };

function targetFunction(x) {
    return x << 1;
}

function otherFunction(y) {
    return y + 1;
}

module.exports = { targetFunction, otherFunction };
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True


class TestClassMethodReplacement:
    """Tests for class method replacement with strict assertions."""

    def test_replace_class_method_body(self, js_support, temp_project):
        """Test replacing a class method body preserves class structure."""
        original_source = """\
class Calculator {
    constructor(precision = 2) {
        this.precision = precision;
    }

    add(a, b) {
        const result = a + b;
        return Number(result.toFixed(this.precision));
    }

    subtract(a, b) {
        return a - b;
    }
}
"""
        file_path = temp_project / "calculator.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        add_method = next(f for f in functions if f.function_name == "add")

        # Optimized version provided in class context
        optimized_code = """\
class Calculator {
    constructor(precision = 2) {
        this.precision = precision;
    }

    add(a, b) {
        return +((a + b).toFixed(this.precision));
    }
}
"""

        result = js_support.replace_function(original_source, add_method, optimized_code)

        expected_result = """\
class Calculator {
    constructor(precision = 2) {
        this.precision = precision;
    }

    add(a, b) {
        return +((a + b).toFixed(this.precision));
    }

    subtract(a, b) {
        return a - b;
    }
}
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True

    def test_replace_method_calling_sibling_methods(self, js_support, temp_project):
        """Test replacing method that calls other methods in same class."""
        original_source = """\
class DataProcessor {
    constructor() {
        this.cache = new Map();
    }

    validate(data) {
        return data !== null && data !== undefined;
    }

    process(data) {
        if (!this.validate(data)) {
            throw new Error('Invalid data');
        }
        const result = [];
        for (let i = 0; i < data.length; i++) {
            result.push(data[i] * 2);
        }
        return result;
    }
}
"""
        file_path = temp_project / "processor.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        process_method = next(f for f in functions if f.function_name == "process")

        optimized_code = """\
class DataProcessor {
    constructor() {
        this.cache = new Map();
    }

    process(data) {
        if (!this.validate(data)) {
            throw new Error('Invalid data');
        }
        return data.map(x => x * 2);
    }
}
"""

        result = js_support.replace_function(original_source, process_method, optimized_code)

        expected_result = """\
class DataProcessor {
    constructor() {
        this.cache = new Map();
    }

    validate(data) {
        return data !== null && data !== undefined;
    }

    process(data) {
        if (!this.validate(data)) {
            throw new Error('Invalid data');
        }
        return data.map(x => x * 2);
    }
}
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True


class TestJSDocPreservation:
    """Tests for JSDoc comment handling during replacement."""

    def test_replace_preserves_jsdoc_above_function(self, js_support, temp_project):
        """Test that JSDoc comments above the function are preserved."""
        original_source = """\
/**
 * Calculates the sum of two numbers.
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} The sum
 */
function add(a, b) {
    const sum = a + b;
    return sum;
}
"""
        file_path = temp_project / "math.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        optimized_code = """\
/**
 * Calculates the sum of two numbers.
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} The sum
 */
function add(a, b) {
    return a + b;
}
"""

        result = js_support.replace_function(original_source, func, optimized_code)

        expected_result = """\
/**
 * Calculates the sum of two numbers.
 * @param {number} a - First number
 * @param {number} b - Second number
 * @returns {number} The sum
 */
function add(a, b) {
    return a + b;
}
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True

    def test_replace_class_method_with_jsdoc(self, js_support, temp_project):
        """Test replacing class method with JSDoc on both class and method."""
        original_source = """\
/**
 * A simple cache implementation.
 * @class Cache
 */
class Cache {
    constructor() {
        this.data = new Map();
    }

    /**
     * Gets a value from cache.
     * @param {string} key - The cache key
     * @returns {*} The cached value or undefined
     */
    get(key) {
        const entry = this.data.get(key);
        if (entry) {
            return entry.value;
        }
        return undefined;
    }
}
"""
        file_path = temp_project / "cache.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        get_method = next(f for f in functions if f.function_name == "get")

        optimized_code = """\
class Cache {
    constructor() {
        this.data = new Map();
    }

    /**
     * Gets a value from cache.
     * @param {string} key - The cache key
     * @returns {*} The cached value or undefined
     */
    get(key) {
        return this.data.get(key)?.value;
    }
}
"""

        result = js_support.replace_function(original_source, get_method, optimized_code)

        expected_result = """\
/**
 * A simple cache implementation.
 * @class Cache
 */
class Cache {
    constructor() {
        this.data = new Map();
    }

    /**
     * Gets a value from cache.
     * @param {string} key - The cache key
     * @returns {*} The cached value or undefined
     */
    get(key) {
        return this.data.get(key)?.value;
    }
}
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True


class TestAsyncFunctionReplacement:
    """Tests for async function replacement."""

    def test_replace_async_function_body(self, js_support, temp_project):
        """Test replacing async function preserves async keyword."""
        original_source = """\
async function fetchData(url) {
    const response = await fetch(url);
    const data = await response.json();
    return data;
}
"""
        file_path = temp_project / "api.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        optimized_code = """\
async function fetchData(url) {
    return (await fetch(url)).json();
}
"""

        result = js_support.replace_function(original_source, func, optimized_code)

        expected_result = """\
async function fetchData(url) {
    return (await fetch(url)).json();
}
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True

    def test_replace_async_class_method(self, js_support, temp_project):
        """Test replacing async class method."""
        original_source = """\
class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async get(endpoint) {
        const url = this.baseUrl + endpoint;
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error('Request failed');
        }
        const data = await response.json();
        return data;
    }
}
"""
        file_path = temp_project / "client.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        get_method = next(f for f in functions if f.function_name == "get")

        optimized_code = """\
class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async get(endpoint) {
        const response = await fetch(this.baseUrl + endpoint);
        if (!response.ok) throw new Error('Request failed');
        return response.json();
    }
}
"""

        result = js_support.replace_function(original_source, get_method, optimized_code)

        expected_result = """\
class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async get(endpoint) {
        const response = await fetch(this.baseUrl + endpoint);
        if (!response.ok) throw new Error('Request failed');
        return response.json();
    }
}
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True


class TestGeneratorFunctionReplacement:
    """Tests for generator function replacement."""

    def test_replace_generator_function_body(self, js_support, temp_project):
        """Test replacing generator function preserves generator syntax."""
        original_source = """\
function* range(start, end) {
    for (let i = start; i < end; i++) {
        yield i;
    }
}
"""
        file_path = temp_project / "generators.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        optimized_code = """\
function* range(start, end) {
    let i = start;
    while (i < end) yield i++;
}
"""

        result = js_support.replace_function(original_source, func, optimized_code)

        expected_result = """\
function* range(start, end) {
    let i = start;
    while (i < end) yield i++;
}
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True


class TestTypeScriptReplacement:
    """Tests for TypeScript-specific replacement."""

    def test_replace_typescript_function_with_types(self, ts_support, temp_project):
        """Test replacing TypeScript function preserves type annotations."""
        original_source = """\
function processArray(items: number[]): number {
    let sum = 0;
    for (let i = 0; i < items.length; i++) {
        sum += items[i];
    }
    return sum;
}
"""
        file_path = temp_project / "processor.ts"
        file_path.write_text(original_source, encoding="utf-8")

        functions = ts_support.discover_functions(file_path)
        func = functions[0]

        optimized_code = """\
function processArray(items: number[]): number {
    return items.reduce((a, b) => a + b, 0);
}
"""

        result = ts_support.replace_function(original_source, func, optimized_code)

        expected_result = """\
function processArray(items: number[]): number {
    return items.reduce((a, b) => a + b, 0);
}
"""
        assert result == expected_result
        assert ts_support.validate_syntax(result) is True

    def test_replace_typescript_class_method_with_generics(self, ts_support, temp_project):
        """Test replacing TypeScript generic class method."""
        original_source = """\
class Container<T> {
    private items: T[] = [];

    add(item: T): void {
        this.items.push(item);
    }

    getAll(): T[] {
        const result: T[] = [];
        for (let i = 0; i < this.items.length; i++) {
            result.push(this.items[i]);
        }
        return result;
    }
}
"""
        file_path = temp_project / "container.ts"
        file_path.write_text(original_source, encoding="utf-8")

        functions = ts_support.discover_functions(file_path)
        get_all_method = next(f for f in functions if f.function_name == "getAll")

        optimized_code = """\
class Container<T> {
    private items: T[] = [];

    getAll(): T[] {
        return [...this.items];
    }
}
"""

        result = ts_support.replace_function(original_source, get_all_method, optimized_code)

        expected_result = """\
class Container<T> {
    private items: T[] = [];

    add(item: T): void {
        this.items.push(item);
    }

    getAll(): T[] {
        return [...this.items];
    }
}
"""
        assert result == expected_result
        assert ts_support.validate_syntax(result) is True

    def test_replace_typescript_interface_typed_function(self, ts_support, temp_project):
        """Test replacing function that uses interfaces."""
        original_source = """\
interface User {
    id: string;
    name: string;
    email: string;
}

function createUser(name: string, email: string): User {
    const id = Math.random().toString(36).substring(2, 15);
    const user: User = {
        id: id,
        name: name,
        email: email
    };
    return user;
}
"""
        file_path = temp_project / "user.ts"
        file_path.write_text(original_source, encoding="utf-8")

        functions = ts_support.discover_functions(file_path)
        func = next(f for f in functions if f.function_name == "createUser")

        optimized_code = """\
function createUser(name: string, email: string): User {
    return {
        id: Math.random().toString(36).substring(2, 15),
        name,
        email
    };
}
"""

        result = ts_support.replace_function(original_source, func, optimized_code)

        expected_result = """\
interface User {
    id: string;
    name: string;
    email: string;
}

function createUser(name: string, email: string): User {
    return {
        id: Math.random().toString(36).substring(2, 15),
        name,
        email
    };
}
"""
        assert result == expected_result
        assert ts_support.validate_syntax(result) is True


class TestComplexReplacements:
    """Tests for complex replacement scenarios."""

    def test_replace_function_with_nested_functions(self, js_support, temp_project):
        """Test replacing function that contains nested function definitions."""
        original_source = """\
function processItems(items) {
    function helper(item) {
        return item * 2;
    }

    const results = [];
    for (let i = 0; i < items.length; i++) {
        results.push(helper(items[i]));
    }
    return results;
}
"""
        file_path = temp_project / "processor.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        process_func = next(f for f in functions if f.function_name == "processItems")

        optimized_code = """\
function processItems(items) {
    const helper = x => x * 2;
    return items.map(helper);
}
"""

        result = js_support.replace_function(original_source, process_func, optimized_code)

        expected_result = """\
function processItems(items) {
    const helper = x => x * 2;
    return items.map(helper);
}
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True

    def test_replace_multiple_methods_sequentially(self, js_support, temp_project):
        """Test replacing multiple methods in the same class sequentially."""
        original_source = """\
class MathUtils {
    static sum(arr) {
        let total = 0;
        for (let i = 0; i < arr.length; i++) {
            total += arr[i];
        }
        return total;
    }

    static average(arr) {
        if (arr.length === 0) return 0;
        let total = 0;
        for (let i = 0; i < arr.length; i++) {
            total += arr[i];
        }
        return total / arr.length;
    }
}
"""
        file_path = temp_project / "math.js"
        file_path.write_text(original_source, encoding="utf-8")

        # First replacement: sum method
        functions = js_support.discover_functions(file_path)
        sum_method = next(f for f in functions if f.function_name == "sum")

        optimized_sum = """\
class MathUtils {
    static sum(arr) {
        return arr.reduce((a, b) => a + b, 0);
    }
}
"""

        result = js_support.replace_function(original_source, sum_method, optimized_sum)

        expected_after_first = """\
class MathUtils {
    static sum(arr) {
        return arr.reduce((a, b) => a + b, 0);
    }

    static average(arr) {
        if (arr.length === 0) return 0;
        let total = 0;
        for (let i = 0; i < arr.length; i++) {
            total += arr[i];
        }
        return total / arr.length;
    }
}
"""
        assert result == expected_after_first
        assert js_support.validate_syntax(result) is True

    def test_replace_function_with_complex_destructuring(self, js_support, temp_project):
        """Test replacing function with complex parameter destructuring."""
        original_source = """\
function processConfig({ server: { host, port }, database: { url, poolSize } }) {
    const serverUrl = host + ':' + port;
    const dbConnection = url + '?poolSize=' + poolSize;
    return {
        server: serverUrl,
        db: dbConnection
    };
}
"""
        file_path = temp_project / "config.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        optimized_code = """\
function processConfig({ server: { host, port }, database: { url, poolSize } }) {
    return {
        server: `${host}:${port}`,
        db: `${url}?poolSize=${poolSize}`
    };
}
"""

        result = js_support.replace_function(original_source, func, optimized_code)

        expected_result = """\
function processConfig({ server: { host, port }, database: { url, poolSize } }) {
    return {
        server: `${host}:${port}`,
        db: `${url}?poolSize=${poolSize}`
    };
}
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True


class TestEdgeCases:
    """Tests for edge cases in code replacement."""

    def test_replace_minimal_function_body(self, js_support, temp_project):
        """Test replacing function with minimal body."""
        original_source = """\
function minimal() {
    return null;
}
"""
        file_path = temp_project / "minimal.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        optimized_code = """\
function minimal() {
    return { initialized: true, timestamp: Date.now() };
}
"""

        result = js_support.replace_function(original_source, func, optimized_code)

        expected_result = """\
function minimal() {
    return { initialized: true, timestamp: Date.now() };
}
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True

    def test_replace_single_line_function(self, js_support, temp_project):
        """Test replacing single-line function."""
        original_source = """\
function identity(x) { return x; }
"""
        file_path = temp_project / "utils.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        optimized_code = """\
function identity(x) { return x ?? null; }
"""

        result = js_support.replace_function(original_source, func, optimized_code)

        expected_result = """\
function identity(x) { return x ?? null; }
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True

    def test_replace_function_with_special_characters_in_strings(self, js_support, temp_project):
        """Test replacing function containing special characters in strings."""
        original_source = """\
function formatMessage(name) {
    const greeting = 'Hello, ' + name + '!';
    const special = "Contains \\"quotes\\" and \\n newlines";
    return greeting + ' ' + special;
}
"""
        file_path = temp_project / "formatter.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        optimized_code = """\
function formatMessage(name) {
    return `Hello, ${name}! Contains "quotes" and
 newlines`;
}
"""

        result = js_support.replace_function(original_source, func, optimized_code)

        expected_result = """\
function formatMessage(name) {
    return `Hello, ${name}! Contains "quotes" and
 newlines`;
}
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True

    def test_replace_function_with_regex(self, js_support, temp_project):
        """Test replacing function containing regex patterns."""
        original_source = """\
function validateEmail(email) {
    const pattern = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/;
    if (pattern.test(email)) {
        return true;
    }
    return false;
}
"""
        file_path = temp_project / "validator.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        func = functions[0]

        optimized_code = """\
function validateEmail(email) {
    return /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/.test(email);
}
"""

        result = js_support.replace_function(original_source, func, optimized_code)

        expected_result = """\
function validateEmail(email) {
    return /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$/.test(email);
}
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True


class TestModuleExportHandling:
    """Tests for proper handling of module exports during replacement."""

    def test_replace_exported_function_commonjs(self, js_support, temp_project):
        """Test replacing function in CommonJS module preserves exports."""
        original_source = """\
function helper(x) {
    return x * 2;
}

function main(data) {
    const results = [];
    for (let i = 0; i < data.length; i++) {
        results.push(helper(data[i]));
    }
    return results;
}

module.exports = { main, helper };
"""
        file_path = temp_project / "module.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        main_func = next(f for f in functions if f.function_name == "main")

        optimized_code = """\
function main(data) {
    return data.map(helper);
}
"""

        result = js_support.replace_function(original_source, main_func, optimized_code)

        expected_result = """\
function helper(x) {
    return x * 2;
}

function main(data) {
    return data.map(helper);
}

module.exports = { main, helper };
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True

    def test_replace_exported_function_esm(self, js_support, temp_project):
        """Test replacing function in ES Module preserves exports."""
        original_source = """\
export function helper(x) {
    return x * 2;
}

export function main(data) {
    const results = [];
    for (let i = 0; i < data.length; i++) {
        results.push(helper(data[i]));
    }
    return results;
}
"""
        file_path = temp_project / "module.js"
        file_path.write_text(original_source, encoding="utf-8")

        functions = js_support.discover_functions(file_path)
        main_func = next(f for f in functions if f.function_name == "main")

        optimized_code = """\
export function main(data) {
    return data.map(helper);
}
"""

        result = js_support.replace_function(original_source, main_func, optimized_code)

        expected_result = """\
export function helper(x) {
    return x * 2;
}

export function main(data) {
    return data.map(helper);
}
"""
        assert result == expected_result
        assert js_support.validate_syntax(result) is True


class TestSyntaxValidation:
    """Tests to ensure replaced code is always syntactically valid."""

    def test_all_replacements_produce_valid_syntax(self, js_support, temp_project):
        """Test that various replacements all produce valid JavaScript."""
        test_cases = [
            # (original, optimized, description)
            (
                "function f(x) { return x + 1; }",
                "function f(x) { return ++x; }",
                "increment replacement"
            ),
            (
                "function f(arr) { return arr.length > 0; }",
                "function f(arr) { return !!arr.length; }",
                "boolean conversion"
            ),
            (
                "function f(a, b) { if (a) { return a; } return b; }",
                "function f(a, b) { return a || b; }",
                "logical OR replacement"
            ),
        ]

        for i, (original, optimized, description) in enumerate(test_cases):
            file_path = temp_project / f"test_{i}.js"
            file_path.write_text(original, encoding="utf-8")

            functions = js_support.discover_functions(file_path)
            func = functions[0]

            result = js_support.replace_function(original, func, optimized)

            is_valid = js_support.validate_syntax(result)
            assert is_valid is True, f"Replacement '{description}' produced invalid syntax:\n{result}"


def test_code_replacer_for_class_method(ts_support, temp_project):
    original = """/**
 * DataProcessor class - demonstrates class method optimization in TypeScript.
 * Contains intentionally inefficient implementations for optimization testing.
 */

/**
 * A class for processing data arrays with various operations.
 */
export class DataProcessor<T> {
    private data: T[];

    /**
     * Create a DataProcessor instance.
     * @param data - Initial data array
     */
    constructor(data: T[] = []) {
        this.data = [...data];
    }

    /**
     * Find duplicates in the data array.
     * Intentionally inefficient implementation.
     * @returns Array of duplicate values
     */
    findDuplicates(): T[] {
        const duplicates: T[] = [];
        for (let i = 0; i < this.data.length; i++) {
            for (let j = i + 1; j < this.data.length; j++) {
                if (this.data[i] === this.data[j]) {
                    if (!duplicates.includes(this.data[i])) {
                        duplicates.push(this.data[i]);
                    }
                }
            }
        }
        return duplicates;
    }

    /**
     * Sort the data using bubble sort.
     * Intentionally inefficient implementation.
     * @returns Sorted copy of the data
     */
    sortData(): T[] {
        const result = [...this.data];
        const n = result.length;
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n - 1; j++) {
                if (result[j] > result[j + 1]) {
                    const temp = result[j];
                    result[j] = result[j + 1];
                    result[j + 1] = temp;
                }
            }
        }
        return result;
    }

    /**
     * Get unique values from the data.
     * Intentionally inefficient implementation.
     * @returns Array of unique values
     */
    getUnique(): T[] {
        const unique: T[] = [];
        for (let i = 0; i < this.data.length; i++) {
            let found = false;
            for (let j = 0; j < unique.length; j++) {
                if (unique[j] === this.data[i]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                unique.push(this.data[i]);
            }
        }
        return unique;
    }

    /**
     * Get the data array.
     * @returns The data array
     */
    getData(): T[] {
        return [...this.data];
    }
}
"""
    file_path = temp_project / "app.ts"
    file_path.write_text(original, encoding="utf-8")
    target_func = "findDuplicates"
    parent_class = "DataProcessor"

    functions = ts_support.discover_functions(file_path)
    # find function
    target_func_info = None
    for func in functions:
        if func.function_name == target_func and func.parents[0].name == parent_class:
            target_func_info = func
            break
    assert target_func_info is not None

    new_code = """```typescript:app.ts
class DataProcessor<T> {
   private data: T[];

   /**
    * Create a DataProcessor instance.
    * @param data - Initial data array
    */
   constructor(data: T[] = []) {
        this.data = [...data];
    }

    /**
     * Find duplicates in the data array.
     * Optimized O(n) implementation using Sets.
     * @returns Array of duplicate values
     */
    findDuplicates(): T[] {
        const seen = new Set<T>();
        const duplicates = new Set<T>();

        for (let i = 0, len = this.data.length; i < len; i++) {
            const item = this.data[i];
            if (seen.has(item)) {
                duplicates.add(item);
            } else {
                seen.add(item);
            }
        }

        return Array.from(duplicates);
    }
}
```
"""
    code_markdown = CodeStringsMarkdown.parse_markdown_code(new_code)
    replaced = replace_function_definitions_for_language([f"{parent_class}.{target_func}"], code_markdown, file_path, temp_project)
    assert replaced

    new_code = file_path.read_text()
    assert new_code == """/**
 * DataProcessor class - demonstrates class method optimization in TypeScript.
 * Contains intentionally inefficient implementations for optimization testing.
 */

/**
 * A class for processing data arrays with various operations.
 */
export class DataProcessor<T> {
    private data: T[];

    /**
     * Create a DataProcessor instance.
     * @param data - Initial data array
     */
    constructor(data: T[] = []) {
        this.data = [...data];
    }

    /**
     * Find duplicates in the data array.
     * Optimized O(n) implementation using Sets.
     * @returns Array of duplicate values
     */
    findDuplicates(): T[] {
        const seen = new Set<T>();
        const duplicates = new Set<T>();

        for (let i = 0, len = this.data.length; i < len; i++) {
            const item = this.data[i];
            if (seen.has(item)) {
                duplicates.add(item);
            } else {
                seen.add(item);
            }
        }

        return Array.from(duplicates);
    }

    /**
     * Sort the data using bubble sort.
     * Intentionally inefficient implementation.
     * @returns Sorted copy of the data
     */
    sortData(): T[] {
        const result = [...this.data];
        const n = result.length;
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n - 1; j++) {
                if (result[j] > result[j + 1]) {
                    const temp = result[j];
                    result[j] = result[j + 1];
                    result[j + 1] = temp;
                }
            }
        }
        return result;
    }

    /**
     * Get unique values from the data.
     * Intentionally inefficient implementation.
     * @returns Array of unique values
     */
    getUnique(): T[] {
        const unique: T[] = [];
        for (let i = 0; i < this.data.length; i++) {
            let found = false;
            for (let j = 0; j < unique.length; j++) {
                if (unique[j] === this.data[i]) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                unique.push(this.data[i]);
            }
        }
        return unique;
    }

    /**
     * Get the data array.
     * @returns The data array
     */
    getData(): T[] {
        return [...this.data];
    }
}
"""



class TestNewVariableFromOptimizedCode:
    """Tests for handling new variables introduced in optimized code."""

    def test_new_bound_method_variable_added_after_referenced_constant(self, ts_support, temp_project):
        """Test that a new variable binding a method is added after the constant it references.

        When optimized code introduces a new module-level variable (like `_has`) that
        references an existing constant (like `CODEFLASH_EMPLOYEE_GITHUB_IDS`), the
        replacement should:
        1. Add the new variable after the constant it references
        2. Replace the function with the optimized version
        """
        from codeflash.models.models import CodeStringsMarkdown, CodeString

        original_source = '''\
const CODEFLASH_EMPLOYEE_GITHUB_IDS = new Set([
  "1234",
]);

export function isCodeflashEmployee(userId: string): boolean {
  return CODEFLASH_EMPLOYEE_GITHUB_IDS.has(userId);
}
'''
        file_path = temp_project / "auth.ts"
        file_path.write_text(original_source, encoding="utf-8")

        # Optimized code introduces a bound method variable for performance
        optimized_code = '''const _has: (id: string) => boolean = CODEFLASH_EMPLOYEE_GITHUB_IDS.has.bind(
  CODEFLASH_EMPLOYEE_GITHUB_IDS
);

export function isCodeflashEmployee(userId: string): boolean {
  return _has(userId);
}
'''

        code_markdown = CodeStringsMarkdown(
            code_strings=[
                CodeString(
                    code=optimized_code,
                    file_path=Path("auth.ts"),
                    language="typescript"
                )
            ],
            language="typescript"
        )

        replaced = replace_function_definitions_for_language(
            ["isCodeflashEmployee"],
            code_markdown,
            file_path,
            temp_project,
        )

        assert replaced
        result = file_path.read_text()

        # Expected result for strict equality check
        expected_result = '''\
const CODEFLASH_EMPLOYEE_GITHUB_IDS = new Set([
  "1234",
]);

const _has: (id: string) => boolean = CODEFLASH_EMPLOYEE_GITHUB_IDS.has.bind(
  CODEFLASH_EMPLOYEE_GITHUB_IDS
);

export function isCodeflashEmployee(userId: string): boolean {
  return _has(userId);
}
'''
        assert result == expected_result, (
            f"Result does not match expected output.\n"
            f"Expected:\n{expected_result}\n\n"
            f"Got:\n{result}"
        )


class TestImportedTypeNotDuplicated:
    """Tests to ensure imported types are not duplicated during code replacement.

    When a type is already imported in the original file, it should NOT be
    added as a new declaration from the optimized code, even if the optimized
    code contains the type definition (because it was provided as context).

    See: https://github.com/codeflash-ai/appsmith/pull/20
    """

    def test_imported_interface_not_added_as_declaration(self, ts_support, temp_project):
        """Test that an imported interface is not duplicated in the output.

        When TreeNode is imported from another file and the optimized code
        contains the TreeNode interface definition (from read-only context),
        the replacement should NOT add the interface to the original file.
        """
        from codeflash.models.models import CodeStringsMarkdown, CodeString

        # Original source imports TreeNode
        original_source = """\
import type { TreeNode } from "./constants";

export function getNearestAbove(
    tree: Record<string, TreeNode>,
    effectedBoxId: string,
) {
    const aboves = tree[effectedBoxId].aboves;
    return aboves.reduce((prev: string[], next: string) => {
        if (!prev[0]) return [next];
        let nextBottomRow = tree[next].bottomRow;
        let prevBottomRow = tree[prev[0]].bottomRow;
        if (nextBottomRow > prevBottomRow) return [next];
        return prev;
    }, []);
}
"""
        file_path = temp_project / "helpers.ts"
        file_path.write_text(original_source, encoding="utf-8")

        # Optimized code includes the TreeNode interface (from read-only context)
        # This simulates what the AI might return when type definitions are included in context
        optimized_code_with_interface = """\
interface TreeNode {
    aboves: string[];
    belows: string[];
    topRow: number;
    bottomRow: number;
}

export function getNearestAbove(
    tree: Record<string, TreeNode>,
    effectedBoxId: string,
) {
    const aboves = tree[effectedBoxId].aboves;
    return aboves.reduce((prev: string[], next: string) => {
        if (!prev[0]) return [next];
        // Optimized: cache lookups
        const nextBottomRow = tree[next].bottomRow;
        const prevBottomRow = tree[prev[0]].bottomRow;
        return nextBottomRow > prevBottomRow ? [next] : prev;
    }, []);
}
"""

        code_markdown = CodeStringsMarkdown(
            code_strings=[
                CodeString(
                    code=optimized_code_with_interface,
                    file_path=Path("helpers.ts"),
                    language="typescript"
                )
            ],
            language="typescript"
        )

        replace_function_definitions_for_language(
            ["getNearestAbove"],
            code_markdown,
            file_path,
            temp_project,
        )

        result = file_path.read_text()

        # The TreeNode interface should NOT appear in the result
        # (it's already imported, so adding it would cause a duplicate)
        assert "interface TreeNode" not in result, (
            f"TreeNode interface should NOT be added to the file since it's already imported.\n"
            f"Result contains:\n{result}"
        )

        # The import should still be there
        assert 'import type { TreeNode } from "./constants"' in result, (
            f"Original import should be preserved.\nResult:\n{result}"
        )

        # The optimized function should be there
        assert "// Optimized: cache lookups" in result, (
            f"Optimized function should be in the result.\nResult:\n{result}"
        )

        # The result should be valid TypeScript
        assert ts_support.validate_syntax(result) is True

    def test_multiple_imported_types_not_duplicated(self, ts_support, temp_project):
        """Test that multiple imported types are not duplicated."""
        from codeflash.models.models import CodeStringsMarkdown, CodeString

        original_source = """\
import type { TreeNode, NodeSpace } from "./constants";
import { MAX_BOX_SIZE } from "./constants";

export function processNode(node: TreeNode, space: NodeSpace): number {
    return node.topRow + space.top;
}
"""
        file_path = temp_project / "processor.ts"
        file_path.write_text(original_source, encoding="utf-8")

        # Optimized code includes both interfaces
        optimized_code = """\
interface TreeNode {
    topRow: number;
    bottomRow: number;
}

interface NodeSpace {
    top: number;
    bottom: number;
}

export function processNode(node: TreeNode, space: NodeSpace): number {
    // Optimized
    return (node.topRow + space.top) | 0;
}
"""

        code_markdown = CodeStringsMarkdown(
            code_strings=[
                CodeString(
                    code=optimized_code,
                    file_path=Path("processor.ts"),
                    language="typescript"
                )
            ],
            language="typescript"
        )

        replace_function_definitions_for_language(
            ["processNode"],
            code_markdown,
            file_path,
            temp_project,
        )

        result = file_path.read_text()

        # Neither interface should be added
        assert "interface TreeNode" not in result
        assert "interface NodeSpace" not in result

        # Imports should be preserved
        assert 'import type { TreeNode, NodeSpace } from "./constants"' in result

        # Optimized code should be there
        assert "// Optimized" in result

        assert ts_support.validate_syntax(result) is True
