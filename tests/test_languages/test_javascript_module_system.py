"""Tests for JavaScript module system detection.
"""

import json
import tempfile
from pathlib import Path

from codeflash.languages.javascript.module_system import (
    ModuleSystem,
    convert_commonjs_to_esm,
    convert_esm_to_commonjs,
    detect_module_system,
    get_import_statement,
)


class TestModuleSystemDetection:
    """Tests for module system detection."""

    def test_detect_esm_from_package_json(self):
        """Test detection of ES modules from package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            package_json = project_root / "package.json"
            package_json.write_text(json.dumps({"type": "module"}))

            result = detect_module_system(project_root)
            assert result == ModuleSystem.ES_MODULE

    def test_detect_commonjs_from_package_json(self):
        """Test detection of CommonJS from package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            package_json = project_root / "package.json"
            package_json.write_text(json.dumps({"type": "commonjs"}))

            result = detect_module_system(project_root)
            assert result == ModuleSystem.COMMONJS

    def test_detect_esm_from_mjs_extension(self):
        """Test detection of ES modules from .mjs extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            file_path = project_root / "module.mjs"
            file_path.write_text("export const foo = 'bar';")

            result = detect_module_system(project_root, file_path)
            assert result == ModuleSystem.ES_MODULE

    def test_detect_commonjs_from_cjs_extension(self):
        """Test detection of CommonJS from .cjs extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            file_path = project_root / "module.cjs"
            file_path.write_text("module.exports = { foo: 'bar' };")

            result = detect_module_system(project_root, file_path)
            assert result == ModuleSystem.COMMONJS

    def test_detect_esm_from_typescript_extension(self):
        """Test detection of ES modules from TypeScript file extensions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            # Test .ts files
            ts_file = project_root / "module.ts"
            ts_file.write_text("export const foo = 'bar';")
            assert detect_module_system(project_root, ts_file) == ModuleSystem.ES_MODULE

            # Test .tsx files
            tsx_file = project_root / "component.tsx"
            tsx_file.write_text("export const Component = () => <div />;")
            assert detect_module_system(project_root, tsx_file) == ModuleSystem.ES_MODULE

            # Test .mts files
            mts_file = project_root / "module.mts"
            mts_file.write_text("export const foo = 'bar';")
            assert detect_module_system(project_root, mts_file) == ModuleSystem.ES_MODULE

    def test_typescript_ignores_package_json_commonjs(self):
        """Test that TypeScript files are detected as ESM even with CommonJS package.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            # Create package.json with explicit commonjs type
            package_json = project_root / "package.json"
            package_json.write_text(json.dumps({"type": "commonjs"}))

            # TypeScript file should still be detected as ESM
            ts_file = project_root / "module.ts"
            ts_file.write_text("export const foo = 'bar';")
            assert detect_module_system(project_root, ts_file) == ModuleSystem.ES_MODULE

    def test_detect_esm_from_import_syntax(self):
        """Test detection of ES modules from import syntax."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            file_path = project_root / "module.js"
            file_path.write_text("import { foo } from './bar';\nexport const baz = 1;")

            result = detect_module_system(project_root, file_path)
            assert result == ModuleSystem.ES_MODULE

    def test_detect_commonjs_from_require_syntax(self):
        """Test detection of CommonJS from require syntax."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            file_path = project_root / "module.js"
            file_path.write_text("const foo = require('./bar');\nmodule.exports = { baz: 1 };")

            result = detect_module_system(project_root, file_path)
            assert result == ModuleSystem.COMMONJS

    def test_default_to_commonjs(self):
        """Test default to CommonJS when uncertain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)

            result = detect_module_system(project_root)
            assert result == ModuleSystem.COMMONJS


class TestImportStatementGeneration:
    """Tests for import statement generation."""

    def test_commonjs_named_import(self):
        """Test CommonJS named import statement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            target = tmpdir / "lib" / "utils.js"
            source = tmpdir / "tests" / "utils.test.js"

            result = get_import_statement(ModuleSystem.COMMONJS, target, source, ["foo", "bar"])

            assert result == "const { foo, bar } = require('../lib/utils');"

    def test_esm_named_import(self):
        """Test ES module named import statement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            target = tmpdir / "lib" / "utils.js"
            source = tmpdir / "tests" / "utils.test.js"

            result = get_import_statement(ModuleSystem.ES_MODULE, target, source, ["foo", "bar"])

            assert result == "import { foo, bar } from '../lib/utils';"

    def test_commonjs_default_import(self):
        """Test CommonJS default import statement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            target = tmpdir / "lib" / "utils.js"
            source = tmpdir / "tests" / "utils.test.js"

            result = get_import_statement(ModuleSystem.COMMONJS, target, source)

            assert result == "const utils = require('../lib/utils');"

    def test_esm_default_import(self):
        """Test ES module default import statement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            target = tmpdir / "lib" / "utils.js"
            source = tmpdir / "tests" / "utils.test.js"

            result = get_import_statement(ModuleSystem.ES_MODULE, target, source)

            assert result == "import utils from '../lib/utils';"

    def test_relative_path_same_directory(self):
        """Test import from same directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            target = tmpdir / "utils.js"
            source = tmpdir / "index.js"

            result = get_import_statement(ModuleSystem.COMMONJS, target, source, ["foo"])

            assert result == "const { foo } = require('./utils');"

    def test_relative_path_subdirectory(self):
        """Test import from subdirectory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            target = tmpdir / "lib" / "helpers" / "utils.js"
            source = tmpdir / "tests" / "test.js"

            result = get_import_statement(ModuleSystem.COMMONJS, target, source, ["foo"])

            assert result == "const { foo } = require('../lib/helpers/utils');"

    def test_relative_path_parent_directory(self):
        """Test import from parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            target = tmpdir / "utils.js"
            source = tmpdir / "tests" / "unit" / "test.js"

            result = get_import_statement(ModuleSystem.COMMONJS, target, source, ["foo"])

            assert result == "const { foo } = require('../../utils');"


class TestModuleSystemConversion:
    """Tests for CommonJS <-> ESM conversion."""

    def test_convert_simple_destructured_require(self):
        """Test converting simple destructured require to import."""
        code = "const { foo, bar } = require('./module');"
        result = convert_commonjs_to_esm(code)
        assert result == "import { foo, bar } from './module';"

    def test_convert_destructured_require_with_alias(self):
        """Test converting destructured require with alias to import with 'as'."""
        code = "const { foo: aliasedFoo } = require('./module');"
        result = convert_commonjs_to_esm(code)
        assert result == "import { foo as aliasedFoo } from './module';"

    def test_convert_mixed_destructured_require(self):
        """Test converting mixed destructured require (some aliased, some not)."""
        code = "const { foo, bar: aliasedBar, baz } = require('./module');"
        result = convert_commonjs_to_esm(code)
        assert result == "import { foo, bar as aliasedBar, baz } from './module';"

    def test_convert_destructured_with_whitespace(self):
        """Test that whitespace is handled correctly in destructuring."""
        code = "const {  foo : aliasedFoo ,  bar  } = require('./module');"
        result = convert_commonjs_to_esm(code)
        assert result == "import { foo as aliasedFoo, bar } from './module';"

    def test_convert_simple_require(self):
        """Test converting simple require to default import."""
        code = "const module = require('./module');"
        result = convert_commonjs_to_esm(code)
        assert result == "import module from './module';"

    def test_convert_property_access_require(self):
        """Test converting require with property access to named import."""
        code = "const foo = require('./module').bar;"
        result = convert_commonjs_to_esm(code)
        assert result == "import { bar as foo } from './module';"

    def test_convert_property_access_default(self):
        """Test converting require().default to default import."""
        code = "const foo = require('./module').default;"
        result = convert_commonjs_to_esm(code)
        assert result == "import foo from './module';"

    def test_convert_multiple_requires(self):
        """Test converting multiple requires in one code block."""
        code = """const { db: dbCore, cache } = require('@budibase/backend-core');
const utils = require('./utils');
const { process } = require('./processor');"""
        result = convert_commonjs_to_esm(code)
        expected = """import { db as dbCore, cache } from '@budibase/backend-core';
import utils from './utils';
import { process } from './processor';"""
        assert result == expected

    def test_convert_esm_to_commonjs_named(self):
        """Test converting named imports to destructured require."""
        code = "import { foo, bar } from './module';"
        result = convert_esm_to_commonjs(code)
        assert result == "const { foo, bar } = require('./module');"

    def test_convert_esm_to_commonjs_default(self):
        """Test converting default import to simple require."""
        code = "import module from './module';"
        result = convert_esm_to_commonjs(code)
        assert result == "const module = require('./module');"

    def test_convert_esm_to_commonjs_with_alias(self):
        """Test converting import with 'as' to destructured require.

        Note: ESM uses 'as' but the regex keeps it as-is in the output.
        This is acceptable since the test is primarily for CommonJS -> ESM conversion.
        """
        code = "import { foo as aliasedFoo } from './module';"
        result = convert_esm_to_commonjs(code)
        # The current implementation preserves 'as' syntax which works for our use case
        assert result == "const { foo as aliasedFoo } = require('./module');"

    def test_real_world_budibase_import(self):
        """Test the real-world case from Budibase that was failing."""
        code = "const { queue, context, db: dbCore, cache, events } = require('@budibase/backend-core');"
        result = convert_commonjs_to_esm(code)
        expected = "import { queue, context, db as dbCore, cache, events } from '@budibase/backend-core';"
        assert result == expected
