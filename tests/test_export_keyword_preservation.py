"""Test that export keywords are preserved during code extraction."""

import pytest
from pathlib import Path
from codeflash.languages.javascript.support import JavaScriptSupport
from codeflash.models.function_types import FunctionToOptimize


class TestExportKeywordPreservation:
    """Test export keyword is preserved when extracting code context."""

    def test_export_class_includes_export_keyword(self, tmp_path: Path):
        """Test that exported classes include the export keyword in extracted code."""
        # Arrange: Create a test file with an exported class
        test_file = tmp_path / "test.ts"
        test_code = """export class WsContextCreator {
  public getMetadata(instance: any, methodName: string): any {
    return { test: true };
  }
}"""
        test_file.write_text(test_code)

        # Create a FunctionToOptimize for the class method
        from codeflash.models.function_types import FunctionParent
        function = FunctionToOptimize(
            function_name="getMetadata",
            file_path=test_file,
            parents=[FunctionParent(type="ClassDef", name="WsContextCreator")],
            starting_line=2,  # The method starts at line 2
            ending_line=4,    # The method ends at line 4
            starting_col=2,
            ending_col=3,
            is_async=False,
            is_method=True,
            language="typescript",
            doc_start_line=None,
        )

        # Act: Extract code context
        support = JavaScriptSupport()
        context = support.extract_code_context(function, tmp_path, tmp_path)

        # Assert: The extracted code should include the export keyword
        assert "export" in context.target_code, (
            f"Export keyword missing from extracted code. "
            f"Expected code to start with 'export class', but got:\n{context.target_code}"
        )
        assert "export class WsContextCreator" in context.target_code, (
            f"Expected 'export class WsContextCreator' in extracted code, but got:\n{context.target_code}"
        )

    def test_export_function_includes_export_keyword(self, tmp_path: Path):
        """Test that exported functions include the export keyword in extracted code."""
        # Arrange: Create a test file with an exported function
        test_file = tmp_path / "test.ts"
        test_code = """export function helperFunction(a: number, b: number): number {
  return a + b;
}"""
        test_file.write_text(test_code)

        # Create a FunctionToOptimize for the function
        function = FunctionToOptimize(
            function_name="helperFunction",
            file_path=test_file,
            parents=[],
            starting_line=1,
            ending_line=3,
            starting_col=0,
            ending_col=1,
            is_async=False,
            is_method=False,
            language="typescript",
            doc_start_line=None,
        )

        # Act: Extract code context
        support = JavaScriptSupport()
        context = support.extract_code_context(function, tmp_path, tmp_path)

        # Assert: The extracted code should include the export keyword
        assert "export" in context.target_code, (
            f"Export keyword missing from extracted code for function. "
            f"Expected code to start with 'export function', but got:\n{context.target_code}"
        )
        assert "export function helperFunction" in context.target_code, (
            f"Expected 'export function helperFunction' in extracted code, but got:\n{context.target_code}"
        )

    def test_export_const_arrow_function_includes_export(self, tmp_path: Path):
        """Test that exported const arrow functions include the export keyword."""
        # Arrange: Create a test file with an exported const arrow function
        test_file = tmp_path / "test.ts"
        test_code = """export const multiply = (a: number, b: number): number => {
  return a * b;
};"""
        test_file.write_text(test_code)

        # Create a FunctionToOptimize for the arrow function
        function = FunctionToOptimize(
            function_name="multiply",
            file_path=test_file,
            parents=[],
            starting_line=1,
            ending_line=3,
            starting_col=0,
            ending_col=2,
            is_async=False,
            is_method=False,
            language="typescript",
            doc_start_line=None,
        )

        # Act: Extract code context
        support = JavaScriptSupport()
        context = support.extract_code_context(function, tmp_path, tmp_path)

        # Assert: The extracted code should include the export keyword
        assert "export" in context.target_code, (
            f"Export keyword missing from exported const arrow function. "
            f"Expected code to start with 'export const', but got:\n{context.target_code}"
        )
        assert "export const multiply" in context.target_code, (
            f"Expected 'export const multiply' in extracted code, but got:\n{context.target_code}"
        )

    def test_non_exported_class_unchanged(self, tmp_path: Path):
        """Test that non-exported classes work correctly (baseline test)."""
        # Arrange: Create a test file with a NON-exported class
        test_file = tmp_path / "test.ts"
        test_code = """class InternalHelper {
  process(): void {
    console.log('internal');
  }
}"""
        test_file.write_text(test_code)

        # Create a FunctionToOptimize for the method
        from codeflash.models.function_types import FunctionParent
        function = FunctionToOptimize(
            function_name="process",
            file_path=test_file,
            parents=[FunctionParent(type="ClassDef", name="InternalHelper")],
            starting_line=2,
            ending_line=4,
            starting_col=2,
            ending_col=3,
            is_async=False,
            is_method=True,
            language="typescript",
            doc_start_line=None,
        )

        # Act: Extract code context
        support = JavaScriptSupport()
        context = support.extract_code_context(function, tmp_path, tmp_path)

        # Assert: The extracted code should NOT include export (it's not exported)
        # But it should include the class definition
        assert "class InternalHelper" in context.target_code, (
            f"Expected 'class InternalHelper' in extracted code, but got:\n{context.target_code}"
        )
        # Should not start with export
        stripped_code = context.target_code.lstrip()
        assert not stripped_code.startswith("export"), (
            f"Non-exported class should not start with 'export', but got:\n{context.target_code}"
        )
