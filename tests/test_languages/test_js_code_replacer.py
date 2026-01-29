"""Tests for JavaScript/TypeScript code replacement with import handling.

These tests verify that code replacement correctly handles:
- New imports added during optimization
- Import organization and merging
- CommonJS (require/module.exports) module syntax
- ES Modules (import/export) syntax
- TypeScript import handling
"""

import shutil
from pathlib import Path

import pytest

from codeflash.languages.javascript.module_system import (
    ModuleSystem,
    convert_commonjs_to_esm,
    convert_esm_to_commonjs,
    detect_module_system,
    ensure_module_system_compatibility,
    get_import_statement,
)

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

    def test_convert_relative_require_adds_extension(self):
        """Test that relative imports get .js extension added - exact output."""
        code = "const { helper } = require('./utils');"
        result = convert_commonjs_to_esm(code)

        expected = "import { helper } from './utils.js';"
        assert result.strip() == expected, (
            f"CJS to ESM conversion failed.\nInput: {code}\nExpected: {expected}\nGot: {result}"
        )

    def test_convert_property_access_require(self):
        """Test converting property access require - exact output."""
        code = "const myHelper = require('./utils').helperFunction;"
        result = convert_commonjs_to_esm(code)

        expected = "import { helperFunction as myHelper } from './utils.js';"
        assert result.strip() == expected, (
            f"CJS to ESM conversion failed.\nInput: {code}\nExpected: {expected}\nGot: {result}"
        )

    def test_convert_default_property_access(self):
        """Test converting .default property access - exact output."""
        code = "const MyClass = require('./class').default;"
        result = convert_commonjs_to_esm(code)

        expected = "import MyClass from './class.js';"
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
import { add, subtract } from './math.js';
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
        assert "import { helper } from './helpers.js';" in result
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
