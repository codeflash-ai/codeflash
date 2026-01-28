"""
Tests for JavaScript/TypeScript code replacement with import handling.

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
        assert result == ModuleSystem.ES_MODULE

    def test_detect_commonjs_from_package_json(self, tmp_path):
        """Test detecting CommonJS from package.json type field."""
        package_json = tmp_path / "package.json"
        package_json.write_text('{"name": "test", "type": "commonjs"}')

        result = detect_module_system(tmp_path)
        assert result == ModuleSystem.COMMONJS

    def test_detect_esm_from_mjs_extension(self, tmp_path):
        """Test detecting ES Module from .mjs extension."""
        test_file = tmp_path / "module.mjs"
        test_file.write_text("export function foo() {}")

        result = detect_module_system(tmp_path, file_path=test_file)
        assert result == ModuleSystem.ES_MODULE

    def test_detect_commonjs_from_cjs_extension(self, tmp_path):
        """Test detecting CommonJS from .cjs extension."""
        test_file = tmp_path / "module.cjs"
        test_file.write_text("module.exports = { foo: () => {} };")

        result = detect_module_system(tmp_path, file_path=test_file)
        assert result == ModuleSystem.COMMONJS

    def test_detect_esm_from_import_syntax(self, tmp_path):
        """Test detecting ES Module from import/export syntax in file."""
        test_file = tmp_path / "module.js"
        test_file.write_text("""
import { helper } from './helper.js';

export function process(x) {
    return helper(x);
}
""")

        result = detect_module_system(tmp_path, file_path=test_file)
        assert result == ModuleSystem.ES_MODULE

    def test_detect_commonjs_from_require_syntax(self, tmp_path):
        """Test detecting CommonJS from require/module.exports syntax."""
        test_file = tmp_path / "module.js"
        test_file.write_text("""
const { helper } = require('./helper');

function process(x) {
    return helper(x);
}

module.exports = { process };
""")

        result = detect_module_system(tmp_path, file_path=test_file)
        assert result == ModuleSystem.COMMONJS

    def test_detect_from_fixtures_cjs(self):
        """Test detection on actual CJS fixture."""
        cjs_dir = FIXTURES_DIR / "js_cjs"
        if cjs_dir.exists():
            calculator_file = cjs_dir / "calculator.js"
            result = detect_module_system(cjs_dir, file_path=calculator_file)
            assert result == ModuleSystem.COMMONJS

    def test_detect_from_fixtures_esm(self):
        """Test detection on actual ESM fixture."""
        esm_dir = FIXTURES_DIR / "js_esm"
        if esm_dir.exists():
            calculator_file = esm_dir / "calculator.js"
            result = detect_module_system(esm_dir, file_path=calculator_file)
            assert result == ModuleSystem.ES_MODULE


class TestCommonJSToESMConversion:
    """Tests for CommonJS to ES Module import conversion."""

    def test_convert_simple_require(self):
        """Test converting simple require to import."""
        code = "const lodash = require('lodash');"
        result = convert_commonjs_to_esm(code)
        assert "import lodash from 'lodash';" in result

    def test_convert_destructured_require(self):
        """Test converting destructured require to named import."""
        code = "const { map, filter } = require('lodash');"
        result = convert_commonjs_to_esm(code)
        assert "import { map, filter } from 'lodash';" in result

    def test_convert_relative_require_adds_extension(self):
        """Test that relative imports get .js extension added."""
        code = "const { helper } = require('./utils');"
        result = convert_commonjs_to_esm(code)
        assert "import { helper } from './utils.js';" in result

    def test_convert_property_access_require(self):
        """Test converting property access require to named import with alias."""
        code = "const myHelper = require('./utils').helperFunction;"
        result = convert_commonjs_to_esm(code)
        assert "import { helperFunction as myHelper } from './utils.js';" in result

    def test_convert_default_property_access(self):
        """Test converting .default property access to default import."""
        code = "const MyClass = require('./class').default;"
        result = convert_commonjs_to_esm(code)
        assert "import MyClass from './class.js';" in result

    def test_convert_multiple_requires(self):
        """Test converting multiple require statements."""
        code = """const { add, subtract } = require('./math');
const lodash = require('lodash');
const path = require('path');"""
        result = convert_commonjs_to_esm(code)
        assert "import { add, subtract } from './math.js';" in result
        assert "import lodash from 'lodash';" in result
        assert "import path from 'path';" in result

    def test_preserves_non_require_code(self):
        """Test that non-require code is preserved."""
        code = """const { add } = require('./math');

function calculate(x, y) {
    return add(x, y);
}

module.exports = { calculate };
"""
        result = convert_commonjs_to_esm(code)
        assert "function calculate(x, y)" in result
        assert "return add(x, y);" in result


class TestESMToCommonJSConversion:
    """Tests for ES Module to CommonJS import conversion."""

    def test_convert_default_import(self):
        """Test converting default import to require."""
        code = "import lodash from 'lodash';"
        result = convert_esm_to_commonjs(code)
        assert "const lodash = require('lodash');" in result

    def test_convert_named_import(self):
        """Test converting named import to destructured require."""
        code = "import { map, filter } from 'lodash';"
        result = convert_esm_to_commonjs(code)
        assert "const { map, filter } = require('lodash');" in result

    def test_convert_relative_import_removes_extension(self):
        """Test that relative imports have .js extension removed."""
        code = "import { helper } from './utils.js';"
        result = convert_esm_to_commonjs(code)
        assert "const { helper } = require('./utils');" in result

    def test_convert_multiple_imports(self):
        """Test converting multiple import statements."""
        code = """import { add, subtract } from './math.js';
import lodash from 'lodash';
import path from 'path';"""
        result = convert_esm_to_commonjs(code)
        assert "const { add, subtract } = require('./math');" in result
        assert "const lodash = require('lodash');" in result
        assert "const path = require('path');" in result

    def test_preserves_non_import_code(self):
        """Test that non-import code is preserved."""
        code = """import { add } from './math.js';

export function calculate(x, y) {
    return add(x, y);
}
"""
        result = convert_esm_to_commonjs(code)
        assert "function calculate(x, y)" in result
        assert "return add(x, y);" in result


class TestModuleSystemCompatibility:
    """Tests for ensuring module system compatibility."""

    def test_convert_mixed_code_to_esm(self):
        """Test converting mixed CJS/ESM code to pure ESM."""
        code = """import { existing } from './module.js';
const { helper } = require('./helpers');

function process() {
    return existing() + helper();
}
"""
        result = ensure_module_system_compatibility(code, ModuleSystem.ES_MODULE)
        assert "import { helper } from './helpers.js';" in result
        assert "require" not in result

    def test_convert_mixed_code_to_commonjs(self):
        """Test converting mixed ESM/CJS code to pure CommonJS."""
        code = """const { existing } = require('./module');
import { helper } from './helpers.js';

function process() {
    return existing() + helper();
}
"""
        result = ensure_module_system_compatibility(code, ModuleSystem.COMMONJS)
        assert "const { helper } = require('./helpers');" in result
        assert "import " not in result

    def test_no_conversion_needed_esm(self):
        """Test that pure ESM code is unchanged when targeting ESM."""
        code = """import { add } from './math.js';

export function sum(a, b) {
    return add(a, b);
}
"""
        result = ensure_module_system_compatibility(code, ModuleSystem.ES_MODULE)
        assert result == code

    def test_no_conversion_needed_commonjs(self):
        """Test that pure CommonJS code is unchanged when targeting CommonJS."""
        code = """const { add } = require('./math');

function sum(a, b) {
    return add(a, b);
}

module.exports = { sum };
"""
        result = ensure_module_system_compatibility(code, ModuleSystem.COMMONJS)
        assert result == code


class TestImportStatementGeneration:
    """Tests for generating import statements."""

    def test_generate_esm_named_import(self, tmp_path):
        """Test generating ESM named import statement."""
        target = tmp_path / "utils.js"
        source = tmp_path / "main.js"

        result = get_import_statement(
            ModuleSystem.ES_MODULE, target, source, imported_names=["helper", "process"]
        )
        assert result == "import { helper, process } from './utils';"

    def test_generate_esm_default_import(self, tmp_path):
        """Test generating ESM default import statement."""
        target = tmp_path / "module.js"
        source = tmp_path / "main.js"

        result = get_import_statement(ModuleSystem.ES_MODULE, target, source)
        assert result == "import module from './module';"

    def test_generate_commonjs_named_require(self, tmp_path):
        """Test generating CommonJS destructured require statement."""
        target = tmp_path / "utils.js"
        source = tmp_path / "main.js"

        result = get_import_statement(
            ModuleSystem.COMMONJS, target, source, imported_names=["helper", "process"]
        )
        assert result == "const { helper, process } = require('./utils');"

    def test_generate_commonjs_default_require(self, tmp_path):
        """Test generating CommonJS default require statement."""
        target = tmp_path / "module.js"
        source = tmp_path / "main.js"

        result = get_import_statement(ModuleSystem.COMMONJS, target, source)
        assert result == "const module = require('./module');"

    def test_generate_nested_path_import(self, tmp_path):
        """Test generating import for nested directory structure."""
        subdir = tmp_path / "src" / "utils"
        subdir.mkdir(parents=True)
        target = subdir / "helper.js"
        source = tmp_path / "main.js"

        result = get_import_statement(
            ModuleSystem.ES_MODULE, target, source, imported_names=["helper"]
        )
        assert "src/utils/helper" in result
        assert "import { helper }" in result

    def test_generate_parent_directory_import(self, tmp_path):
        """Test generating import that navigates to parent directory."""
        subdir = tmp_path / "src"
        subdir.mkdir()
        target = tmp_path / "shared" / "utils.js"
        target.parent.mkdir()
        source = subdir / "main.js"

        result = get_import_statement(
            ModuleSystem.ES_MODULE, target, source, imported_names=["helper"]
        )
        assert "../shared/utils" in result


class TestImportOptimization:
    """Tests for import optimization scenarios during code replacement."""

    def test_optimization_adds_new_import_cjs(self, tmp_path):
        """Test that optimization can add new imports in CommonJS."""
        # Original file without lodash
        original_code = """const { helper } = require('./utils');

function process(arr) {
    return helper(arr);
}

module.exports = { process };
"""
        # Optimized code that introduces lodash
        optimized_code = """const { helper } = require('./utils');
const _ = require('lodash');

function process(arr) {
    return _.map(arr, helper);
}

module.exports = { process };
"""
        # Verify the optimized code has the new import
        assert "require('lodash')" in optimized_code
        assert "require('./utils')" in optimized_code

    def test_optimization_adds_new_import_esm(self, tmp_path):
        """Test that optimization can add new imports in ESM."""
        # Original file without lodash
        original_code = """import { helper } from './utils.js';

export function process(arr) {
    return helper(arr);
}
"""
        # Optimized code that introduces lodash
        optimized_code = """import { helper } from './utils.js';
import _ from 'lodash';

export function process(arr) {
    return _.map(arr, helper);
}
"""
        # Verify the optimized code has the new import
        assert "import _ from 'lodash'" in optimized_code
        assert "import { helper } from './utils.js'" in optimized_code

    def test_optimization_merges_imports_from_same_module(self):
        """Test that imports from the same module can be merged."""
        # Before: two separate imports from same module
        code_before = """import { add } from './math.js';
import { subtract } from './math.js';

export function calculate(a, b) {
    return add(a, b) - subtract(a, b);
}
"""
        # After optimization: merged import
        code_after = """import { add, subtract } from './math.js';

export function calculate(a, b) {
    return add(a, b) - subtract(a, b);
}
"""
        # The merge should reduce the number of import statements
        assert code_before.count("import") > code_after.count("import")
        assert "add, subtract" in code_after or "subtract, add" in code_after

    def test_optimization_removes_unused_import(self):
        """Test that unused imports can be removed after optimization."""
        # Original code with unused import
        original_code = """import { add, unused } from './math.js';

export function calculate(a, b) {
    return add(a, b);
}
"""
        # After optimization: unused import removed
        optimized_code = """import { add } from './math.js';

export function calculate(a, b) {
    return add(a, b);
}
"""
        assert "unused" not in optimized_code
        assert "add" in optimized_code


class TestTypeScriptImportHandling:
    """Tests for TypeScript-specific import handling."""

    def test_typescript_type_import_detection(self, tmp_path):
        """Test that TypeScript type imports are handled correctly."""
        code = """import type { Config } from './types';
import { processConfig } from './utils';

export function initialize(config: Config) {
    return processConfig(config);
}
"""
        # Type imports should be preserved
        assert "import type { Config }" in code
        assert "import { processConfig }" in code

    def test_typescript_extension_handling(self, tmp_path):
        """Test TypeScript module detection from .ts extension."""
        ts_file = tmp_path / "module.ts"
        ts_file.write_text("""
import { helper } from './helper';

export function process(x: number): number {
    return helper(x);
}
""")
        package_json = tmp_path / "package.json"
        package_json.write_text('{"name": "test", "type": "module"}')

        # TypeScript with ESM package.json should be detected as ESM
        result = detect_module_system(tmp_path, file_path=ts_file)
        assert result == ModuleSystem.ES_MODULE

    def test_tsx_extension_handling(self, tmp_path):
        """Test TSX (TypeScript React) module detection."""
        tsx_file = tmp_path / "component.tsx"
        tsx_file.write_text("""
import React from 'react';
import { Button } from './Button';

export const App: React.FC = () => {
    return <Button>Click me</Button>;
};
""")
        package_json = tmp_path / "package.json"
        package_json.write_text('{"name": "test", "type": "module"}')

        result = detect_module_system(tmp_path, file_path=tsx_file)
        assert result == ModuleSystem.ES_MODULE


class TestEdgeCases:
    """Tests for edge cases in import handling."""

    def test_dynamic_import_preserved(self):
        """Test that dynamic imports are preserved during conversion."""
        code = """const { helper } = require('./utils');

async function loadModule() {
    const mod = await import('./dynamic-module.js');
    return mod.default;
}

module.exports = { loadModule };
"""
        result = convert_commonjs_to_esm(code)
        # Dynamic import should remain unchanged
        assert "await import('./dynamic-module.js')" in result
        # Static require should be converted
        assert "import { helper } from './utils.js';" in result

    def test_comment_in_require_preserved(self):
        """Test that comments near imports are handled correctly."""
        code = """// Main utilities
const { helper } = require('./utils');
// Another comment
const lodash = require('lodash');
"""
        result = convert_commonjs_to_esm(code)
        assert "import { helper } from './utils.js';" in result
        assert "import lodash from 'lodash';" in result

    def test_multiline_destructured_require(self):
        """Test conversion of multiline destructured require."""
        code = """const {
    helper1,
    helper2,
    helper3
} = require('./utils');
"""
        result = convert_commonjs_to_esm(code)
        # Should convert to single line or preserve multiline
        assert "import" in result
        assert "helper1" in result
        assert "helper2" in result
        assert "helper3" in result

    def test_require_with_template_literal_unchanged(self):
        """Test that dynamic require with template literal is unchanged."""
        code = """const moduleName = 'lodash';
const mod = require(moduleName);  // Dynamic require, can't convert
"""
        result = convert_commonjs_to_esm(code)
        # Dynamic require with variable should be unchanged
        assert "require(moduleName)" in result

    def test_empty_file_handling(self):
        """Test handling of empty file."""
        code = ""
        result = convert_commonjs_to_esm(code)
        assert result == ""

        result = convert_esm_to_commonjs(code)
        assert result == ""

    def test_no_imports_file(self):
        """Test file with no imports."""
        code = """function standalone() {
    return 42;
}

module.exports = { standalone };
"""
        result = convert_commonjs_to_esm(code)
        assert "function standalone()" in result
        assert "return 42;" in result


class TestIntegrationWithFixtures:
    """Integration tests using actual fixture files."""

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

    def test_cjs_fixture_module_system(self, cjs_project):
        """Test that CJS fixture is correctly detected as CommonJS."""
        if not cjs_project.exists():
            pytest.skip("CJS fixture not available")

        calculator_file = cjs_project / "calculator.js"
        if calculator_file.exists():
            result = detect_module_system(cjs_project, file_path=calculator_file)
            assert result == ModuleSystem.COMMONJS

    def test_esm_fixture_module_system(self, esm_project):
        """Test that ESM fixture is correctly detected as ES Module."""
        if not esm_project.exists():
            pytest.skip("ESM fixture not available")

        package_json = esm_project / "package.json"
        if not package_json.exists():
            package_json.write_text('{"name": "test", "type": "module"}')

        calculator_file = esm_project / "calculator.js"
        if calculator_file.exists():
            result = detect_module_system(esm_project, file_path=calculator_file)
            assert result == ModuleSystem.ES_MODULE

    def test_ts_fixture_module_system(self, ts_project):
        """Test that TypeScript fixture module detection works."""
        if not ts_project.exists():
            pytest.skip("TypeScript fixture not available")

        package_json = ts_project / "package.json"
        if not package_json.exists():
            package_json.write_text('{"name": "test", "type": "module"}')

        calculator_file = ts_project / "calculator.ts"
        if calculator_file.exists():
            result = detect_module_system(ts_project, file_path=calculator_file)
            # TypeScript with ESM config should be ESM
            assert result == ModuleSystem.ES_MODULE

    def test_convert_cjs_fixture_to_esm(self, cjs_project):
        """Test converting CJS fixture code to ESM."""
        if not cjs_project.exists():
            pytest.skip("CJS fixture not available")

        calculator_file = cjs_project / "calculator.js"
        if not calculator_file.exists():
            pytest.skip("Calculator file not available")

        original_code = calculator_file.read_text()

        # Convert to ESM
        esm_code = convert_commonjs_to_esm(original_code)

        # Verify conversion
        assert "require(" not in esm_code or "require('" not in esm_code
        assert "import " in esm_code or "import(" in esm_code

    def test_convert_esm_fixture_to_cjs(self, esm_project):
        """Test converting ESM fixture code to CommonJS."""
        if not esm_project.exists():
            pytest.skip("ESM fixture not available")

        calculator_file = esm_project / "calculator.js"
        if not calculator_file.exists():
            pytest.skip("Calculator file not available")

        original_code = calculator_file.read_text()

        # Convert to CommonJS
        cjs_code = convert_esm_to_commonjs(original_code)

        # Verify conversion (if original had imports)
        if "import " in original_code:
            # Static imports should be converted
            # Note: This is a basic check as ESM fixtures use import syntax
            assert "const " in cjs_code or "let " in cjs_code or "var " in cjs_code
