"""Tests for JavaScript/TypeScript import resolver.

These tests verify that the ImportResolver correctly resolves import paths
to actual file paths, enabling multi-file context extraction.
"""


import pytest

from codeflash.languages.javascript.import_resolver import HelperSearchContext, ImportResolver, MultiFileHelperFinder
from codeflash.languages.treesitter_utils import ImportInfo


class TestImportResolver:
    """Tests for ImportResolver class."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a temporary project structure."""
        # Create directories
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        lib_dir = src_dir / "lib"
        lib_dir.mkdir()
        utils_dir = src_dir / "utils"
        utils_dir.mkdir()

        # Create some test files
        (src_dir / "main.ts").write_text("export function main() {}")
        (src_dir / "helper.ts").write_text("export function helper() {}")
        (lib_dir / "math.ts").write_text("export function add() {}")
        (utils_dir / "index.ts").write_text("export function util() {}")

        return tmp_path

    @pytest.fixture
    def resolver(self, project_root):
        """Create an ImportResolver for the project."""
        return ImportResolver(project_root)

    def test_is_external_package_lodash(self, resolver):
        """Test that bare imports are detected as external."""
        assert resolver._is_external_package("lodash") is True

    def test_is_external_package_scoped(self, resolver):
        """Test that scoped packages are detected as external."""
        assert resolver._is_external_package("@company/utils") is True

    def test_is_external_package_react(self, resolver):
        """Test that react is detected as external."""
        assert resolver._is_external_package("react") is True

    def test_is_not_external_package_relative(self, resolver):
        """Test that relative imports are not external."""
        assert resolver._is_external_package("./utils") is False

    def test_is_not_external_package_parent_relative(self, resolver):
        """Test that parent relative imports are not external."""
        assert resolver._is_external_package("../lib/math") is False

    def test_resolve_relative_import_same_dir(self, resolver, project_root):
        """Test resolving ./helper from same directory."""
        source_file = project_root / "src" / "main.ts"
        import_info = ImportInfo(
            module_path="./helper",
            default_import=None,
            named_imports=[("helper", None)],
            namespace_import=None,
            is_type_only=False,
            start_line=1,
            end_line=1,
        )

        result = resolver.resolve_import(import_info, source_file)

        assert result is not None
        assert result.file_path == project_root / "src" / "helper.ts"
        assert result.module_path == "./helper"

    def test_resolve_relative_import_subdirectory(self, resolver, project_root):
        """Test resolving ./lib/math from parent directory."""
        source_file = project_root / "src" / "main.ts"
        import_info = ImportInfo(
            module_path="./lib/math",
            default_import=None,
            named_imports=[("add", None)],
            namespace_import=None,
            is_type_only=False,
            start_line=1,
            end_line=1,
        )

        result = resolver.resolve_import(import_info, source_file)

        assert result is not None
        assert result.file_path == project_root / "src" / "lib" / "math.ts"

    def test_resolve_index_file(self, resolver, project_root):
        """Test resolving ./utils to ./utils/index.ts."""
        source_file = project_root / "src" / "main.ts"
        import_info = ImportInfo(
            module_path="./utils",
            default_import=None,
            named_imports=[("util", None)],
            namespace_import=None,
            is_type_only=False,
            start_line=1,
            end_line=1,
        )

        result = resolver.resolve_import(import_info, source_file)

        assert result is not None
        assert result.file_path == project_root / "src" / "utils" / "index.ts"

    def test_resolve_external_package_returns_none(self, resolver, project_root):
        """Test that external package imports return None."""
        source_file = project_root / "src" / "main.ts"
        import_info = ImportInfo(
            module_path="lodash",
            default_import="_",
            named_imports=[],
            namespace_import=None,
            is_type_only=False,
            start_line=1,
            end_line=1,
        )

        result = resolver.resolve_import(import_info, source_file)

        assert result is None

    def test_resolve_nonexistent_file_returns_none(self, resolver, project_root):
        """Test that nonexistent file imports return None."""
        source_file = project_root / "src" / "main.ts"
        import_info = ImportInfo(
            module_path="./nonexistent",
            default_import=None,
            named_imports=[("foo", None)],
            namespace_import=None,
            is_type_only=False,
            start_line=1,
            end_line=1,
        )

        result = resolver.resolve_import(import_info, source_file)

        assert result is None

    def test_resolve_with_explicit_extension(self, resolver, project_root):
        """Test resolving import with explicit .ts extension."""
        source_file = project_root / "src" / "main.ts"
        import_info = ImportInfo(
            module_path="./helper.ts",
            default_import=None,
            named_imports=[("helper", None)],
            namespace_import=None,
            is_type_only=False,
            start_line=1,
            end_line=1,
        )

        result = resolver.resolve_import(import_info, source_file)

        assert result is not None
        assert result.file_path == project_root / "src" / "helper.ts"

    def test_resolved_import_contains_imported_names(self, resolver, project_root):
        """Test that ResolvedImport contains correct imported names."""
        source_file = project_root / "src" / "main.ts"
        import_info = ImportInfo(
            module_path="./helper",
            default_import="Helper",
            named_imports=[("foo", None), ("bar", "baz")],
            namespace_import=None,
            is_type_only=False,
            start_line=1,
            end_line=1,
        )

        result = resolver.resolve_import(import_info, source_file)

        assert result is not None
        assert "Helper" in result.imported_names
        assert "foo" in result.imported_names
        assert "baz" in result.imported_names  # alias is used
        assert result.is_default_import is True

    def test_namespace_import_detection(self, resolver, project_root):
        """Test that namespace imports are correctly detected."""
        source_file = project_root / "src" / "main.ts"
        import_info = ImportInfo(
            module_path="./helper",
            default_import=None,
            named_imports=[],
            namespace_import="utils",
            is_type_only=False,
            start_line=1,
            end_line=1,
        )

        result = resolver.resolve_import(import_info, source_file)

        assert result is not None
        assert result.is_namespace_import is True
        assert result.namespace_name == "utils"

    def test_caching_works(self, resolver, project_root):
        """Test that resolution results are cached."""
        source_file = project_root / "src" / "main.ts"
        import_info = ImportInfo(
            module_path="./helper",
            default_import=None,
            named_imports=[("helper", None)],
            namespace_import=None,
            is_type_only=False,
            start_line=1,
            end_line=1,
        )

        # First resolution
        result1 = resolver.resolve_import(import_info, source_file)
        # Second resolution should use cache
        result2 = resolver.resolve_import(import_info, source_file)

        assert result1 is not None
        assert result2 is not None
        assert result1.file_path == result2.file_path
        # Check cache was populated
        assert (source_file, "./helper") in resolver._resolution_cache


class TestMultiFileHelperFinder:
    """Tests for MultiFileHelperFinder class."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a temporary project with multi-file structure."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Main file that imports helper
        (src_dir / "main.ts").write_text("""
import { helperFunc } from './helper';

export function mainFunc() {
    return helperFunc() + 1;
}
""")

        # Helper file
        (src_dir / "helper.ts").write_text("""
export function helperFunc() {
    return 42;
}

export function unusedHelper() {
    return 0;
}
""")

        return tmp_path

    @pytest.fixture
    def resolver(self, project_root):
        """Create an ImportResolver."""
        return ImportResolver(project_root)

    @pytest.fixture
    def finder(self, project_root, resolver):
        """Create a MultiFileHelperFinder."""
        return MultiFileHelperFinder(project_root, resolver)

    def test_helper_search_context_defaults(self):
        """Test HelperSearchContext default values."""
        context = HelperSearchContext()
        assert context.visited_files == set()
        assert context.visited_functions == set()
        assert context.current_depth == 0
        assert context.max_depth == 2


class TestExportInfo:
    """Tests for ExportInfo parsing in TreeSitterAnalyzer."""

    @pytest.fixture
    def js_analyzer(self):
        """Create a JavaScript analyzer."""
        from codeflash.languages.treesitter_utils import TreeSitterAnalyzer, TreeSitterLanguage

        return TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)

    def test_find_named_export_function(self, js_analyzer):
        """Test finding export function declaration."""
        code = "export function helper() { return 1; }"
        exports = js_analyzer.find_exports(code)

        assert len(exports) == 1
        assert ("helper", None) in exports[0].exported_names
        assert exports[0].is_reexport is False

    def test_find_default_export_function(self, js_analyzer):
        """Test finding export default function."""
        code = "export default function myFunc() { return 1; }"
        exports = js_analyzer.find_exports(code)

        assert len(exports) == 1
        assert exports[0].default_export == "myFunc"

    def test_find_export_declaration(self, js_analyzer):
        """Test finding export { name }."""
        code = """
function helper() { return 1; }
export { helper };
"""
        exports = js_analyzer.find_exports(code)

        assert len(exports) == 1
        assert ("helper", None) in exports[0].exported_names

    def test_find_export_with_alias(self, js_analyzer):
        """Test finding export { name as alias }."""
        code = """
function helper() { return 1; }
export { helper as myHelper };
"""
        exports = js_analyzer.find_exports(code)

        assert len(exports) == 1
        assert ("helper", "myHelper") in exports[0].exported_names

    def test_find_reexport(self, js_analyzer):
        """Test finding re-export from another module."""
        code = "export { helper } from './other';"
        exports = js_analyzer.find_exports(code)

        assert len(exports) == 1
        assert exports[0].is_reexport is True
        assert exports[0].reexport_source == "./other"

    def test_find_export_const(self, js_analyzer):
        """Test finding export const declaration."""
        code = "export const myVar = 42;"
        exports = js_analyzer.find_exports(code)

        assert len(exports) == 1
        assert ("myVar", None) in exports[0].exported_names

    def test_is_function_exported_true(self, js_analyzer):
        """Test is_function_exported returns True for exported function."""
        code = "export function helper() { return 1; }"
        is_exported, export_name = js_analyzer.is_function_exported(code, "helper")

        assert is_exported is True
        assert export_name == "helper"

    def test_is_function_exported_false(self, js_analyzer):
        """Test is_function_exported returns False for non-exported function."""
        code = "function helper() { return 1; }"
        is_exported, export_name = js_analyzer.is_function_exported(code, "helper")

        assert is_exported is False
        assert export_name is None

    def test_is_function_exported_with_alias(self, js_analyzer):
        """Test is_function_exported returns alias name."""
        code = """
function helper() { return 1; }
export { helper as myHelper };
"""
        is_exported, export_name = js_analyzer.is_function_exported(code, "helper")

        assert is_exported is True
        assert export_name == "myHelper"

    def test_is_function_exported_default(self, js_analyzer):
        """Test is_function_exported returns 'default' for default export."""
        code = "export default function helper() { return 1; }"
        is_exported, export_name = js_analyzer.is_function_exported(code, "helper")

        assert is_exported is True
        assert export_name == "default"


class TestCommonJSRequire:
    """Tests for CommonJS require() import parsing."""

    @pytest.fixture
    def js_analyzer(self):
        """Create a JavaScript analyzer."""
        from codeflash.languages.treesitter_utils import TreeSitterAnalyzer, TreeSitterLanguage

        return TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)

    def test_require_default_import(self, js_analyzer):
        """Test const foo = require('./module')."""
        code = "const helper = require('./helper');"
        imports = js_analyzer.find_imports(code)

        assert len(imports) == 1
        assert imports[0].module_path == "./helper"
        assert imports[0].default_import == "helper"
        assert imports[0].named_imports == []

    def test_require_destructured_import(self, js_analyzer):
        """Test const { a, b } = require('./module')."""
        code = "const { foo, bar } = require('./helper');"
        imports = js_analyzer.find_imports(code)

        assert len(imports) == 1
        assert imports[0].module_path == "./helper"
        assert imports[0].default_import is None
        assert ("foo", None) in imports[0].named_imports
        assert ("bar", None) in imports[0].named_imports

    def test_require_destructured_with_alias(self, js_analyzer):
        """Test const { a: aliasA } = require('./module')."""
        code = "const { foo: myFoo, bar } = require('./helper');"
        imports = js_analyzer.find_imports(code)

        assert len(imports) == 1
        assert imports[0].module_path == "./helper"
        assert ("foo", "myFoo") in imports[0].named_imports
        assert ("bar", None) in imports[0].named_imports

    def test_require_property_access(self, js_analyzer):
        """Test const foo = require('./module').bar."""
        code = "const myFunc = require('./helper').helperFunc;"
        imports = js_analyzer.find_imports(code)

        assert len(imports) == 1
        assert imports[0].module_path == "./helper"
        assert imports[0].default_import is None
        # helperFunc is imported and assigned to myFunc
        assert ("helperFunc", "myFunc") in imports[0].named_imports

    def test_require_property_access_same_name(self, js_analyzer):
        """Test const foo = require('./module').foo."""
        code = "const helperFunc = require('./helper').helperFunc;"
        imports = js_analyzer.find_imports(code)

        assert len(imports) == 1
        assert imports[0].module_path == "./helper"
        # When var name equals property, no alias needed
        assert ("helperFunc", None) in imports[0].named_imports

    def test_require_external_package(self, js_analyzer):
        """Test require for external packages."""
        code = "const lodash = require('lodash');"
        imports = js_analyzer.find_imports(code)

        assert len(imports) == 1
        assert imports[0].module_path == "lodash"
        assert imports[0].default_import == "lodash"

    def test_require_side_effect_import(self, js_analyzer):
        """Test require('./module') without assignment."""
        code = "require('./side-effects');"
        imports = js_analyzer.find_imports(code)

        assert len(imports) == 1
        assert imports[0].module_path == "./side-effects"
        assert imports[0].default_import is None
        assert imports[0].named_imports == []


class TestCommonJSExports:
    """Tests for CommonJS module.exports parsing."""

    @pytest.fixture
    def js_analyzer(self):
        """Create a JavaScript analyzer."""
        from codeflash.languages.treesitter_utils import TreeSitterAnalyzer, TreeSitterLanguage

        return TreeSitterAnalyzer(TreeSitterLanguage.JAVASCRIPT)

    def test_module_exports_function(self, js_analyzer):
        """Test module.exports = function() {}."""
        code = "module.exports = function helper() { return 1; };"
        exports = js_analyzer.find_exports(code)

        assert len(exports) == 1
        assert exports[0].default_export == "helper"

    def test_module_exports_anonymous_function(self, js_analyzer):
        """Test module.exports = function() {} (anonymous)."""
        code = "module.exports = function() { return 1; };"
        exports = js_analyzer.find_exports(code)

        assert len(exports) == 1
        assert exports[0].default_export == "default"

    def test_module_exports_arrow_function(self, js_analyzer):
        """Test module.exports = () => {}."""
        code = "module.exports = () => { return 1; };"
        exports = js_analyzer.find_exports(code)

        assert len(exports) == 1
        assert exports[0].default_export == "default"

    def test_module_exports_identifier(self, js_analyzer):
        """Test module.exports = existingFunction."""
        code = """
function helper() { return 1; }
module.exports = helper;
"""
        exports = js_analyzer.find_exports(code)

        assert len(exports) == 1
        assert exports[0].default_export == "helper"

    def test_module_exports_object(self, js_analyzer):
        """Test module.exports = { foo, bar }."""
        code = """
function foo() {}
function bar() {}
module.exports = { foo, bar };
"""
        exports = js_analyzer.find_exports(code)

        # Should find the module.exports object
        module_export = [e for e in exports if e.exported_names]
        assert len(module_export) == 1
        assert ("foo", None) in module_export[0].exported_names
        assert ("bar", None) in module_export[0].exported_names

    def test_module_exports_object_with_rename(self, js_analyzer):
        """Test module.exports = { publicName: localFunc }."""
        code = """
function helper() {}
module.exports = { publicHelper: helper };
"""
        exports = js_analyzer.find_exports(code)

        module_export = [e for e in exports if e.exported_names]
        assert len(module_export) == 1
        # helper is exported as publicHelper
        assert ("helper", "publicHelper") in module_export[0].exported_names

    def test_module_exports_property(self, js_analyzer):
        """Test module.exports.foo = function() {}."""
        code = "module.exports.helper = function() { return 1; };"
        exports = js_analyzer.find_exports(code)

        assert len(exports) == 1
        assert ("helper", None) in exports[0].exported_names

    def test_exports_property(self, js_analyzer):
        """Test exports.foo = function() {}."""
        code = "exports.helper = function() { return 1; };"
        exports = js_analyzer.find_exports(code)

        assert len(exports) == 1
        assert ("helper", None) in exports[0].exported_names

    def test_module_exports_require_reexport(self, js_analyzer):
        """Test module.exports = require('./other')."""
        code = "module.exports = require('./other');"
        exports = js_analyzer.find_exports(code)

        assert len(exports) == 1
        assert exports[0].is_reexport is True
        assert exports[0].reexport_source == "./other"

    def test_is_function_exported_commonjs(self, js_analyzer):
        """Test is_function_exported works with CommonJS exports."""
        code = """
function helper() { return 1; }
module.exports = { helper };
"""
        is_exported, export_name = js_analyzer.is_function_exported(code, "helper")

        assert is_exported is True
        assert export_name == "helper"

    def test_is_function_exported_commonjs_property(self, js_analyzer):
        """Test is_function_exported with exports.foo pattern."""
        code = """
function helper() { return 1; }
exports.helper = helper;
"""
        is_exported, export_name = js_analyzer.is_function_exported(code, "helper")

        assert is_exported is True
        assert export_name == "helper"


class TestCommonJSImportResolver:
    """Tests for ImportResolver with CommonJS require() imports."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a temporary project structure with CommonJS files."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Create CommonJS module files
        (src_dir / "main.js").write_text("""
const helper = require('./helper');
const { add, subtract } = require('./math');

function main() {
    return helper.process() + add(1, 2);
}

module.exports = main;
""")

        (src_dir / "helper.js").write_text("""
function process() {
    return 42;
}

module.exports = { process };
""")

        (src_dir / "math.js").write_text("""
function add(a, b) { return a + b; }
function subtract(a, b) { return a - b; }

module.exports = { add, subtract };
""")

        return tmp_path

    @pytest.fixture
    def resolver(self, project_root):
        """Create an ImportResolver for the project."""
        return ImportResolver(project_root)

    def test_resolve_commonjs_default_require(self, resolver, project_root):
        """Test resolving const foo = require('./module')."""
        source_file = project_root / "src" / "main.js"
        import_info = ImportInfo(
            module_path="./helper",
            default_import="helper",
            named_imports=[],
            namespace_import=None,
            is_type_only=False,
            start_line=1,
            end_line=1,
        )

        result = resolver.resolve_import(import_info, source_file)

        assert result is not None
        assert result.file_path == project_root / "src" / "helper.js"

    def test_resolve_commonjs_destructured_require(self, resolver, project_root):
        """Test resolving const { a, b } = require('./module')."""
        source_file = project_root / "src" / "main.js"
        import_info = ImportInfo(
            module_path="./math",
            default_import=None,
            named_imports=[("add", None), ("subtract", None)],
            namespace_import=None,
            is_type_only=False,
            start_line=1,
            end_line=1,
        )

        result = resolver.resolve_import(import_info, source_file)

        assert result is not None
        assert result.file_path == project_root / "src" / "math.js"
        assert "add" in result.imported_names
        assert "subtract" in result.imported_names
