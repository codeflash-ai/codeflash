"""Tests for JavaScript module system detection.
"""

import json
import tempfile
from pathlib import Path

from codeflash.languages.javascript.module_system import ModuleSystem, detect_module_system, get_import_statement


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
