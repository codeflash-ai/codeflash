"""Comprehensive tests for JavaScript/TypeScript find references functionality.

These tests are inspired by real-world patterns found in the Appsmith codebase,
covering various import/export patterns, callback usage, memoization, and more.

Each test verifies:
1. The actual reference values (file, line, column, type, caller)
2. The formatted markdown output from _format_references_as_markdown
"""

import pytest
from pathlib import Path

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.javascript.find_references import (
    Reference,
    ReferenceFinder,
    ExportedFunction,
    ReferenceSearchContext,
    find_references,
)
from codeflash.languages.base import Language, FunctionInfo, ReferenceInfo
from codeflash.code_utils.code_extractor import _format_references_as_markdown
from codeflash.models.models import FunctionParent


def make_func(name: str, file_path: Path, class_name: str | None = None) -> FunctionToOptimize:
    """Helper to create FunctionToOptimize for testing."""
    parents = [FunctionParent(name=class_name, type="ClassDef")] if class_name else []
    return FunctionToOptimize(
        function_name=name,
        file_path=file_path,
        parents=parents,
        language="javascript",
    )


class TestReferenceFinder:
    """Tests for ReferenceFinder class."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a basic project structure."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        return tmp_path

    @pytest.fixture
    def finder(self, project_root):
        """Create a ReferenceFinder instance."""
        return ReferenceFinder(project_root)

    def test_init_default_exclude_patterns(self, project_root):
        """Test that default exclude patterns are set."""
        finder = ReferenceFinder(project_root)
        assert "node_modules" in finder.exclude_patterns
        assert "dist" in finder.exclude_patterns
        assert ".git" in finder.exclude_patterns

    def test_init_custom_exclude_patterns(self, project_root):
        """Test custom exclude patterns."""
        finder = ReferenceFinder(project_root, exclude_patterns=["custom_dir"])
        assert "custom_dir" in finder.exclude_patterns
        assert "node_modules" not in finder.exclude_patterns

    def test_should_exclude_node_modules(self, finder, project_root):
        """Test that node_modules files are excluded."""
        path = project_root / "node_modules" / "lodash" / "index.js"
        assert finder._should_exclude(path) is True

    def test_should_not_exclude_src(self, finder, project_root):
        """Test that src files are not excluded."""
        path = project_root / "src" / "utils.ts"
        assert finder._should_exclude(path) is False


class TestBasicNamedExports:
    """Tests for basic named export/import patterns.

    Inspired by Appsmith patterns like:
    import { getDynamicBindings, isDynamicValue } from "utils/DynamicBindingUtils";
    """

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with named export pattern."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        utils_dir = src_dir / "utils"
        utils_dir.mkdir()

        # Source file with named export
        (utils_dir / "DynamicBindingUtils.ts").write_text(
            'export function getDynamicBindings(value: string): string[] {\n'
            '    const regex = /{{([^}]+)}}/g;\n'
            '    return [];\n'
            '}\n'
        )

        # File that imports and uses the function
        (src_dir / "evaluator.ts").write_text(
            "import { getDynamicBindings } from './utils/DynamicBindingUtils';\n"
            '\n'
            'export function evaluate(expression: string) {\n'
            '    const bindings = getDynamicBindings(expression);\n'
            '    return bindings;\n'
            '}\n'
        )

        # Another file that uses the function
        (src_dir / "validator.ts").write_text(
            "import { getDynamicBindings } from './utils/DynamicBindingUtils';\n"
            '\n'
            'export function validateBindings(input: string) {\n'
            '    const bindings = getDynamicBindings(input);\n'
            '    return bindings.length > 0;\n'
            '}\n'
        )

        return tmp_path

    def test_find_named_export_references_values(self, project_root):
        """Test finding references to a named exported function with exact values."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils" / "DynamicBindingUtils.ts"

        refs = finder.find_references(make_func("getDynamicBindings", source_file))

        # Sort refs by file path for consistent ordering
        refs_sorted = sorted(refs, key=lambda r: (str(r.file_path), r.line))

        # Should find 2 references
        assert len(refs_sorted) == 2

        # Check evaluator.ts reference
        eval_ref = next(r for r in refs_sorted if "evaluator.ts" in str(r.file_path))
        assert eval_ref.line == 4
        assert eval_ref.reference_type == "call"
        assert eval_ref.caller_function == "evaluate"
        assert eval_ref.import_name == "getDynamicBindings"
        assert "getDynamicBindings(expression)" in eval_ref.context

        # Check validator.ts reference
        val_ref = next(r for r in refs_sorted if "validator.ts" in str(r.file_path))
        assert val_ref.line == 4
        assert val_ref.reference_type == "call"
        assert val_ref.caller_function == "validateBindings"
        assert val_ref.import_name == "getDynamicBindings"
        assert "getDynamicBindings(input)" in val_ref.context

    def test_format_references_as_markdown_named_exports(self, project_root):
        """Test _format_references_as_markdown output for named exports."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils" / "DynamicBindingUtils.ts"

        refs = finder.find_references(make_func("getDynamicBindings", source_file))

        # Convert to ReferenceInfo and sort for consistent ordering
        ref_infos = sorted([
            ReferenceInfo(
                file_path=r.file_path,
                line=r.line,
                column=r.column,
                end_line=r.end_line,
                end_column=r.end_column,
                context=r.context,
                reference_type=r.reference_type,
                import_name=r.import_name,
                caller_function=r.caller_function,
            )
            for r in refs
        ], key=lambda r: str(r.file_path))

        markdown = _format_references_as_markdown(ref_infos, source_file, project_root, Language.TYPESCRIPT)

        expected_markdown = (
            '```typescript:src/evaluator.ts\n'
            'function evaluate(expression: string) {\n'
            '    const bindings = getDynamicBindings(expression);\n'
            '    return bindings;\n'
            '}\n'
            '```\n'
            '```typescript:src/validator.ts\n'
            'function validateBindings(input: string) {\n'
            '    const bindings = getDynamicBindings(input);\n'
            '    return bindings.length > 0;\n'
            '}\n'
            '```\n'
        )
        assert markdown == expected_markdown


class TestDefaultExports:
    """Tests for default export/import patterns."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with default export pattern."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Source file with default export
        (src_dir / "helper.ts").write_text(
            'function processData(data: any[]) {\n'
            '    return data.filter(item => item.active);\n'
            '}\n'
            '\n'
            'export default processData;\n'
        )

        # File that imports the default export
        (src_dir / "main.ts").write_text(
            "import processData from './helper';\n"
            '\n'
            'export function handleData(items: any[]) {\n'
            '    const processed = processData(items);\n'
            '    return processed.length;\n'
            '}\n'
        )

        # File that imports with a different name
        (src_dir / "alternative.ts").write_text(
            "import myProcessor from './helper';\n"
            '\n'
            'export function process(items: any[]) {\n'
            '    return myProcessor(items);\n'
            '}\n'
        )

        return tmp_path

    def test_find_default_export_references_values(self, project_root):
        """Test finding references to a default exported function with exact values."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "helper.ts"

        refs = finder.find_references(make_func("processData", source_file))

        # Should find references in both files
        ref_files = {str(ref.file_path) for ref in refs}
        assert any("main.ts" in f for f in ref_files)
        assert any("alternative.ts" in f for f in ref_files)

        # Check main.ts reference (uses original name)
        main_ref = next(r for r in refs if "main.ts" in str(r.file_path))
        assert main_ref.line == 4
        assert main_ref.reference_type == "call"
        assert main_ref.caller_function == "handleData"
        assert main_ref.import_name == "processData"

        # Check alternative.ts reference (uses alias)
        alt_ref = next(r for r in refs if "alternative.ts" in str(r.file_path))
        assert alt_ref.line == 4
        assert alt_ref.reference_type == "call"
        assert alt_ref.caller_function == "process"
        assert alt_ref.import_name == "myProcessor"

    def test_format_references_as_markdown_default_exports(self, project_root):
        """Test markdown output for default exports with aliases."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "helper.ts"

        refs = finder.find_references(make_func("processData", source_file))
        ref_infos = sorted([
            ReferenceInfo(
                file_path=r.file_path, line=r.line, column=r.column,
                end_line=r.end_line, end_column=r.end_column, context=r.context,
                reference_type=r.reference_type, import_name=r.import_name,
                caller_function=r.caller_function,
            )
            for r in refs
        ], key=lambda r: str(r.file_path))

        markdown = _format_references_as_markdown(ref_infos, source_file, project_root, Language.TYPESCRIPT)

        expected_markdown = (
            '```typescript:src/alternative.ts\n'
            'function process(items: any[]) {\n'
            '    return myProcessor(items);\n'
            '}\n'
            '```\n'
            '```typescript:src/main.ts\n'
            'function handleData(items: any[]) {\n'
            '    const processed = processData(items);\n'
            '    return processed.length;\n'
            '}\n'
            '```\n'
        )
        assert markdown == expected_markdown


class TestReExports:
    """Tests for re-export patterns."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with re-export pattern."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        utils_dir = src_dir / "utils"
        utils_dir.mkdir()

        # Original function file
        (utils_dir / "filterUtils.ts").write_text(
            'export function filterBySearchTerm(items: any[], term: string) {\n'
            '    return items.filter(i => i.name.includes(term));\n'
            '}\n'
        )

        # Index file that re-exports
        (utils_dir / "index.ts").write_text(
            "export { filterBySearchTerm } from './filterUtils';\n"
        )

        # Consumer that imports from index
        (src_dir / "consumer.ts").write_text(
            "import { filterBySearchTerm } from './utils';\n"
            '\n'
            'export function searchItems(items: any[], query: string) {\n'
            '    return filterBySearchTerm(items, query);\n'
            '}\n'
        )

        return tmp_path

    def test_find_reexport_reference_values(self, project_root):
        """Test finding re-export references with exact values."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils" / "filterUtils.ts"

        refs = finder.find_references(make_func("filterBySearchTerm", source_file))

        # Should find re-export in index.ts
        reexport_refs = [r for r in refs if r.reference_type == "reexport"]
        assert len(reexport_refs) == 1
        assert "index.ts" in str(reexport_refs[0].file_path)
        assert reexport_refs[0].import_name == "filterBySearchTerm"

        # Should find call in consumer.ts (through re-export chain)
        call_refs = [r for r in refs if r.reference_type == "call"]
        assert len(call_refs) >= 1
        consumer_ref = next((r for r in call_refs if "consumer.ts" in str(r.file_path)), None)
        assert consumer_ref is not None
        assert consumer_ref.line == 4
        assert consumer_ref.caller_function == "searchItems"

    def test_format_references_as_markdown_reexports(self, project_root):
        """Test markdown output for re-exports."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils" / "filterUtils.ts"

        refs = finder.find_references(make_func("filterBySearchTerm", source_file))
        ref_infos = sorted([
            ReferenceInfo(
                file_path=r.file_path, line=r.line, column=r.column,
                end_line=r.end_line, end_column=r.end_column, context=r.context,
                reference_type=r.reference_type, import_name=r.import_name,
                caller_function=r.caller_function,
            )
            for r in refs
        ], key=lambda r: str(r.file_path))

        markdown = _format_references_as_markdown(ref_infos, source_file, project_root, Language.TYPESCRIPT)

        expected_markdown = (
            '```typescript:src/consumer.ts\n'
            'function searchItems(items: any[], query: string) {\n'
            '    return filterBySearchTerm(items, query);\n'
            '}\n'
            '```\n'
            '```typescript:src/utils/index.ts\n'
            "export { filterBySearchTerm } from './filterUtils';\n"
            '```\n'
        )
        assert markdown == expected_markdown


class TestCallbackPatterns:
    """Tests for functions passed as callbacks."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with callback patterns."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Helper function
        (src_dir / "transforms.ts").write_text(
            'export function normalizeItem(item: any) {\n'
            '    return { ...item, normalized: true };\n'
            '}\n'
        )

        # Consumer using callbacks
        (src_dir / "processor.ts").write_text(
            "import { normalizeItem } from './transforms';\n"
            '\n'
            'export function processItems(items: any[]) {\n'
            '    const normalized = items.map(normalizeItem);\n'
            '    return normalized;\n'
            '}\n'
        )

        return tmp_path

    def test_find_callback_references_values(self, project_root):
        """Test finding functions used as callbacks with exact values."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "transforms.ts"

        refs = finder.find_references(make_func("normalizeItem", source_file))

        # Should find the callback reference
        callback_refs = [r for r in refs if r.reference_type == "callback"]
        assert len(callback_refs) >= 1

        callback_ref = callback_refs[0]
        assert "processor.ts" in str(callback_ref.file_path)
        assert callback_ref.line == 4
        assert callback_ref.caller_function == "processItems"
        assert "items.map(normalizeItem)" in callback_ref.context

    def test_format_references_as_markdown_callbacks(self, project_root):
        """Test markdown output for callback patterns."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "transforms.ts"

        refs = finder.find_references(make_func("normalizeItem", source_file))
        ref_infos = [
            ReferenceInfo(
                file_path=r.file_path, line=r.line, column=r.column,
                end_line=r.end_line, end_column=r.end_column, context=r.context,
                reference_type=r.reference_type, import_name=r.import_name,
                caller_function=r.caller_function,
            )
            for r in refs
        ]

        markdown = _format_references_as_markdown(ref_infos, source_file, project_root, Language.TYPESCRIPT)

        expected_markdown = (
            '```typescript:src/processor.ts\n'
            'function processItems(items: any[]) {\n'
            '    const normalized = items.map(normalizeItem);\n'
            '    return normalized;\n'
            '}\n'
            '```\n'
        )
        assert expected_markdown == markdown


class TestAliasImports:
    """Tests for functions imported with aliases."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with alias import patterns."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Source file
        (src_dir / "utils.ts").write_text(
            'export function computeValue(input: number): number {\n'
            '    return input * 2;\n'
            '}\n'
        )

        # File using alias
        (src_dir / "consumer.ts").write_text(
            "import { computeValue as calculate } from './utils';\n"
            '\n'
            'export function processNumber(n: number) {\n'
            '    const result = calculate(n);\n'
            '    return result + 10;\n'
            '}\n'
        )

        return tmp_path

    def test_find_aliased_import_references_values(self, project_root):
        """Test finding references when function is imported with alias."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils.ts"

        refs = finder.find_references(make_func("computeValue", source_file))

        # Should find the reference even though it's called as "calculate"
        assert len(refs) == 1
        ref = refs[0]
        assert "consumer.ts" in str(ref.file_path)
        assert ref.line == 4
        assert ref.reference_type == "call"
        assert ref.import_name == "calculate"  # The aliased name
        assert ref.caller_function == "processNumber"
        assert "calculate(n)" in ref.context

    def test_format_references_as_markdown_aliases(self, project_root):
        """Test markdown output for aliased imports."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils.ts"

        refs = finder.find_references(make_func("computeValue", source_file))
        ref_infos = [
            ReferenceInfo(
                file_path=r.file_path, line=r.line, column=r.column,
                end_line=r.end_line, end_column=r.end_column, context=r.context,
                reference_type=r.reference_type, import_name=r.import_name,
                caller_function=r.caller_function,
            )
            for r in refs
        ]

        markdown = _format_references_as_markdown(ref_infos, source_file, project_root, Language.TYPESCRIPT)

        expected_markdown = (
            '```typescript:src/consumer.ts\n'
            'function processNumber(n: number) {\n'
            '    const result = calculate(n);\n'
            '    return result + 10;\n'
            '}\n'
            '```\n'
        )
        assert expected_markdown == markdown


class TestNamespaceImports:
    """Tests for namespace import patterns."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with namespace import patterns."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Source file with multiple exports
        (src_dir / "mathUtils.ts").write_text(
            'export function add(a: number, b: number): number {\n'
            '    return a + b;\n'
            '}\n'
        )

        # File using namespace import
        (src_dir / "calculator.ts").write_text(
            "import * as MathUtils from './mathUtils';\n"
            '\n'
            'export function calculate(a: number, b: number) {\n'
            '    return MathUtils.add(a, b);\n'
            '}\n'
        )

        return tmp_path

    def test_find_namespace_import_references_values(self, project_root):
        """Test finding references via namespace imports with exact values."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "mathUtils.ts"

        refs = finder.find_references(make_func("add", source_file))

        assert len(refs) == 1
        ref = refs[0]
        assert "calculator.ts" in str(ref.file_path)
        assert ref.line == 4
        assert ref.reference_type == "call"
        assert ref.import_name == "MathUtils.add"
        assert ref.caller_function == "calculate"
        assert "MathUtils.add(a, b)" in ref.context

    def test_format_references_as_markdown_namespace(self, project_root):
        """Test markdown output for namespace imports."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "mathUtils.ts"

        refs = finder.find_references(make_func("add", source_file))
        ref_infos = [
            ReferenceInfo(
                file_path=r.file_path, line=r.line, column=r.column,
                end_line=r.end_line, end_column=r.end_column, context=r.context,
                reference_type=r.reference_type, import_name=r.import_name,
                caller_function=r.caller_function,
            )
            for r in refs
        ]

        markdown = _format_references_as_markdown(ref_infos, source_file, project_root, Language.TYPESCRIPT)

        expected_markdown = (
            '```typescript:src/calculator.ts\n'
            'function calculate(a: number, b: number) {\n'
            '    return MathUtils.add(a, b);\n'
            '}\n'
            '```\n'
        )
        assert expected_markdown == markdown


class TestMemoizedFunctions:
    """Tests for memoized function patterns."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with memoized function patterns."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # Source file with function to be memoized
        (src_dir / "expensive.ts").write_text(
            'export function computeExpensive(x: number): number {\n'
            '    return x * x;\n'
            '}\n'
        )

        # File that memoizes the function
        (src_dir / "memoized.ts").write_text(
            "import memoize from 'micro-memoize';\n"
            "import { computeExpensive } from './expensive';\n"
            '\n'
            'export const memoizedCompute = memoize(computeExpensive);\n'
            '\n'
            'export function process(x: number) {\n'
            '    return computeExpensive(x) + memoizedCompute(x);\n'
            '}\n'
        )

        return tmp_path

    def test_find_memoized_function_references_values(self, project_root):
        """Test finding references to functions passed to memoize."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "expensive.ts"

        refs = finder.find_references(make_func("computeExpensive", source_file))

        # Should find memoize call and direct call
        assert len(refs) >= 2

        # Check for memoized reference
        memo_refs = [r for r in refs if r.reference_type == "memoized"]
        assert len(memo_refs) >= 1
        memo_ref = memo_refs[0]
        assert "memoized.ts" in str(memo_ref.file_path)
        assert "memoize(computeExpensive)" in memo_ref.context

        # Check for direct call
        call_refs = [r for r in refs if r.reference_type == "call"]
        assert len(call_refs) >= 1


class TestSameFileReferences:
    """Tests for references within the same file."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with same-file reference patterns."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # File with internal references
        (src_dir / "recursive.ts").write_text(
            'export function factorial(n: number): number {\n'
            '    if (n <= 1) return 1;\n'
            '    return n * factorial(n - 1);\n'
            '}\n'
        )

        return tmp_path

    def test_find_recursive_references_values(self, project_root):
        """Test finding recursive calls within same file with exact values."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "recursive.ts"

        refs = finder.find_references(make_func("factorial", source_file), include_definition=True)

        # Should find the recursive call
        call_refs = [r for r in refs if r.reference_type == "call"]
        assert len(call_refs) >= 1

        recursive_ref = call_refs[0]
        assert recursive_ref.line == 3
        assert recursive_ref.caller_function == "factorial"
        assert "factorial(n - 1)" in recursive_ref.context


class TestComplexMultiFileScenarios:
    """Tests for complex multi-file scenarios inspired by Appsmith."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a complex multi-file project structure."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "utils").mkdir()
        (src_dir / "components").mkdir()

        # Core utility function
        (src_dir / "utils" / "widgetUtils.ts").write_text(
            'export function isLargeWidget(type: string): boolean {\n'
            "    return ['TABLE', 'LIST'].includes(type);\n"
            '}\n'
        )

        # Re-export from index
        (src_dir / "utils" / "index.ts").write_text(
            "export { isLargeWidget } from './widgetUtils';\n"
        )

        # Component using the function via re-export
        (src_dir / "components" / "Widget.tsx").write_text(
            "import { isLargeWidget } from '../utils';\n"
            '\n'
            'export function Widget({ type }: { type: string }) {\n'
            '    const isLarge = isLargeWidget(type);\n'
            '    return isLarge;\n'
            '}\n'
        )

        return tmp_path

    def test_find_all_references_across_codebase_values(self, project_root):
        """Test finding all references to isLargeWidget with exact values."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils" / "widgetUtils.ts"

        refs = finder.find_references(make_func("isLargeWidget", source_file))

        # Should find re-export in index.ts
        reexport_refs = [r for r in refs if r.reference_type == "reexport"]
        assert len(reexport_refs) == 1
        assert "index.ts" in str(reexport_refs[0].file_path)

        # Should find call in Widget.tsx
        call_refs = [r for r in refs if r.reference_type == "call"]
        assert len(call_refs) >= 1
        widget_ref = next((r for r in call_refs if "Widget.tsx" in str(r.file_path)), None)
        assert widget_ref is not None
        assert widget_ref.line == 4
        assert widget_ref.caller_function == "Widget"

    def test_format_references_as_markdown_complex(self, project_root):
        """Test markdown output for complex multi-file scenario."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "utils" / "widgetUtils.ts"

        refs = finder.find_references(make_func("isLargeWidget", source_file))
        ref_infos = sorted([
            ReferenceInfo(
                file_path=r.file_path, line=r.line, column=r.column,
                end_line=r.end_line, end_column=r.end_column, context=r.context,
                reference_type=r.reference_type, import_name=r.import_name,
                caller_function=r.caller_function,
            )
            for r in refs
        ], key=lambda r: str(r.file_path))

        markdown = _format_references_as_markdown(ref_infos, source_file, project_root, Language.TYPESCRIPT)

        expected_markdown = (
            '```typescript:src/components/Widget.tsx\n'
            'function Widget({ type }: { type: string }) {\n'
            '    const isLarge = isLargeWidget(type);\n'
            '    return isLarge;\n'
            '}\n'
            '```\n'
            '```typescript:src/utils/index.ts\n'
            "export { isLargeWidget } from './widgetUtils';\n"
            '```\n'
        )
        assert markdown == expected_markdown


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project for edge case testing."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        return tmp_path

    def test_nonexistent_file(self, project_root):
        """Test handling of nonexistent source file."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "nonexistent.ts"

        refs = finder.find_references(make_func("someFunction", source_file))

        assert refs == []

    def test_non_exported_function(self, project_root):
        """Test handling of non-exported function."""
        # Create a file with non-exported function
        (project_root / "src" / "private.ts").write_text(
            'function internalHelper() {\n'
            '    return 42;\n'
            '}\n'
            '\n'
            'export function publicFunction() {\n'
            '    return internalHelper();\n'
            '}\n'
        )

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "private.ts"

        refs = finder.find_references(make_func("internalHelper", source_file))

        # Should only find internal reference
        assert all(r.file_path == source_file for r in refs)

    def test_empty_file(self, project_root):
        """Test handling of empty file."""
        (project_root / "src" / "empty.ts").write_text("")

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "empty.ts"

        refs = finder.find_references(make_func("anything", source_file))

        assert refs == []

    def test_format_references_empty_list(self, project_root):
        """Test _format_references_as_markdown with empty list."""
        markdown = _format_references_as_markdown([], project_root / "src" / "file.ts", project_root, Language.TYPESCRIPT)
        assert markdown == ""


class TestCommonJSPatterns:
    """Tests for CommonJS require/module.exports patterns."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project with CommonJS patterns."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        # CommonJS module
        (src_dir / "helpers.js").write_text(
            'function processConfig(config) {\n'
            '    return { ...config, processed: true };\n'
            '}\n'
            '\n'
            'module.exports = { processConfig };\n'
        )

        # Consumer using destructured require
        (src_dir / "main.js").write_text(
            "const { processConfig } = require('./helpers');\n"
            '\n'
            'function handleConfig(config) {\n'
            '    return processConfig(config);\n'
            '}\n'
            '\n'
            'module.exports = handleConfig;\n'
        )

        return tmp_path

    def test_find_commonjs_references_values(self, project_root):
        """Test finding CommonJS references with exact values."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "helpers.js"

        refs = finder.find_references(make_func("processConfig", source_file))

        assert len(refs) >= 1
        main_ref = next((r for r in refs if "main.js" in str(r.file_path)), None)
        assert main_ref is not None
        assert main_ref.line == 4
        assert main_ref.reference_type == "call"
        assert main_ref.caller_function == "handleConfig"

    def test_format_references_as_markdown_commonjs(self, project_root):
        """Test markdown output for CommonJS patterns."""
        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "helpers.js"

        refs = finder.find_references(make_func("processConfig", source_file))
        ref_infos = sorted([
            ReferenceInfo(
                file_path=r.file_path, line=r.line, column=r.column,
                end_line=r.end_line, end_column=r.end_column, context=r.context,
                reference_type=r.reference_type, import_name=r.import_name,
                caller_function=r.caller_function,
            )
            for r in refs
        ], key=lambda r: str(r.file_path))

        markdown = _format_references_as_markdown(ref_infos, source_file, project_root, Language.JAVASCRIPT)

        expected_markdown = (
            '```javascript:src/main.js\n'
            'function handleConfig(config) {\n'
            '    return processConfig(config);\n'
            '}\n'
            '```\n'
        )
        assert markdown == expected_markdown


class TestConvenienceFunction:
    """Tests for the find_references convenience function."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create a simple project."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        (src_dir / "utils.ts").write_text(
            'export function helper() {\n'
            '    return 42;\n'
            '}\n'
        )

        (src_dir / "main.ts").write_text(
            "import { helper } from './utils';\n"
            '\n'
            'export function main() {\n'
            '    return helper();\n'
            '}\n'
        )

        return tmp_path

    def test_find_references_function_values(self, project_root):
        """Test the find_references convenience function with exact values."""
        source_file = project_root / "src" / "utils.ts"

        refs = find_references(make_func("helper", source_file), project_root=project_root)

        assert len(refs) == 1
        ref = refs[0]
        assert "main.ts" in str(ref.file_path)
        assert ref.line == 4
        assert ref.reference_type == "call"
        assert ref.caller_function == "main"


class TestReferenceDataclass:
    """Tests for Reference dataclass."""

    def test_reference_creation(self, tmp_path):
        """Test creating a Reference object."""
        ref = Reference(
            file_path=tmp_path / "test.ts",
            line=10,
            column=5,
            end_line=10,
            end_column=15,
            context="const result = myFunction();",
            reference_type="call",
            import_name="myFunction",
            caller_function="processData",
        )

        assert ref.line == 10
        assert ref.column == 5
        assert ref.end_line == 10
        assert ref.end_column == 15
        assert ref.reference_type == "call"
        assert ref.import_name == "myFunction"
        assert ref.caller_function == "processData"
        assert ref.context == "const result = myFunction();"

    def test_reference_without_caller(self, tmp_path):
        """Test Reference with no caller function."""
        ref = Reference(
            file_path=tmp_path / "test.ts",
            line=1,
            column=0,
            end_line=1,
            end_column=10,
            context="export { fn } from './module';",
            reference_type="reexport",
            import_name="fn",
        )

        assert ref.caller_function is None


class TestExportedFunctionDataclass:
    """Tests for ExportedFunction dataclass."""

    def test_exported_function_named(self, tmp_path):
        """Test ExportedFunction for named export."""
        exp = ExportedFunction(
            function_name="myHelper",
            export_name="myHelper",
            is_default=False,
            file_path=tmp_path / "utils.ts",
        )

        assert exp.function_name == "myHelper"
        assert exp.export_name == "myHelper"
        assert exp.is_default is False
        assert exp.file_path == tmp_path / "utils.ts"

    def test_exported_function_default(self, tmp_path):
        """Test ExportedFunction for default export."""
        exp = ExportedFunction(
            function_name="processData",
            export_name="default",
            is_default=True,
            file_path=tmp_path / "processor.ts",
        )

        assert exp.function_name == "processData"
        assert exp.is_default is True
        assert exp.export_name == "default"


class TestReferenceSearchContext:
    """Tests for ReferenceSearchContext dataclass."""

    def test_context_defaults(self):
        """Test default values for ReferenceSearchContext."""
        ctx = ReferenceSearchContext()

        assert ctx.visited_files == set()
        assert ctx.max_files == 1000

    def test_context_custom_max_files(self):
        """Test custom max_files value."""
        ctx = ReferenceSearchContext(max_files=500)

        assert ctx.max_files == 500


class TestEdgeCasesAdvanced:
    """Advanced edge case tests to catch potential failures."""

    @pytest.fixture
    def project_root(self, tmp_path):
        """Create project for edge case testing."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        return tmp_path

    def test_circular_import_handling(self, project_root):
        """Test that circular imports don't cause infinite loops."""
        src_dir = project_root / "src"

        # Create circular import structure
        (src_dir / "a.ts").write_text(
            "import { funcB } from './b';\n"
            '\n'
            'export function funcA() {\n'
            '    return funcB() + 1;\n'
            '}\n'
        )

        (src_dir / "b.ts").write_text(
            "import { funcA } from './a';\n"
            '\n'
            'export function funcB() {\n'
            '    return 42;\n'
            '}\n'
            '\n'
            'export function callsA() {\n'
            '    return funcA();\n'
            '}\n'
        )

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "a.ts"

        # Should not hang or crash
        refs = finder.find_references(make_func("funcA", source_file))

        # Should find reference in b.ts
        b_refs = [r for r in refs if "b.ts" in str(r.file_path)]
        assert len(b_refs) >= 1
        assert b_refs[0].caller_function == "callsA"

    def test_syntax_error_graceful_handling(self, project_root):
        """Test that syntax errors in files are handled gracefully."""
        src_dir = project_root / "src"

        (src_dir / "valid.ts").write_text(
            'export function validFunction() {\n'
            '    return 42;\n'
            '}\n'
        )

        # Create a file with syntax error
        (src_dir / "invalid.ts").write_text(
            "import { validFunction } from './valid';\n"
            '\n'
            'export function broken( {\n'
            '    return validFunction(\n'
            '}\n'
        )

        finder = ReferenceFinder(project_root)
        source_file = project_root / "src" / "valid.ts"

        # Should not crash
        refs = finder.find_references(make_func("validFunction", source_file))
        assert isinstance(refs, list)
