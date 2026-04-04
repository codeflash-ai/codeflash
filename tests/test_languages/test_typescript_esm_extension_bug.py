"""Tests for TypeScript ESM .js extension bug fix.

Issue #7: add_js_extensions_to_relative_imports() adds .js to TypeScript imports,
breaking tests that run on TypeScript source with ts-jest.
"""

import pytest
from pathlib import Path
from codeflash.languages.javascript.support import JavaScriptSupport
from codeflash.models.function_types import FunctionToOptimize
from codeflash.verification.verification_utils import TestConfig
from codeflash.languages.javascript.module_system import ModuleSystem


class TestTypeScriptESMExtensionBug:
    """Test that TypeScript imports don't get .js extensions added."""

    def test_typescript_file_should_not_add_js_extensions(self, tmp_path):
        """TypeScript test files should not have .js extensions added to imports.

        When:
        - Source file is TypeScript (.ts or .tsx)
        - Test file is TypeScript (.test.ts or .test.tsx)
        - Project uses ESM module system

        Then:
        - Imports should NOT have .js extensions added
        - Because ts-jest runs on TypeScript source directly

        This prevents "Cannot find module './foo.js'" errors when the file is foo.ts.
        """
        # Setup: TypeScript source file
        source_file = tmp_path / "src" / "utils.ts"
        source_file.parent.mkdir(parents=True)
        source_file.write_text(
            """export function myFunction() {
  return 42;
}"""
        )

        # Create package.json with ESM type
        (tmp_path / "package.json").write_text('{"type": "module"}')

        # AI service generates test WITHOUT .js extension (correct!)
        generated_test = """import { myFunction } from '../src/utils';

describe('myFunction', () => {
  test('returns 42', () => {
    expect(myFunction()).toBe(42);
  });
});
"""

        # Setup test config
        lang_support = JavaScriptSupport()
        test_cfg = TestConfig(
            tests_root=tmp_path / "tests",
            project_root_path=tmp_path,
            tests_project_rootdir=tmp_path,
        )

        function_to_optimize = FunctionToOptimize(
            function_name="myFunction",
            file_path=str(source_file),
            line_number=1,
        )

        test_path = tmp_path / "tests" / "test_myFunction.test.ts"
        test_path.parent.mkdir(parents=True)

        # Process the test (this is where the bug happens)
        (
            final_generated,
            instrumented_behavior,
            instrumented_perf,
        ) = lang_support.process_generated_test_strings(
            generated_test_source=generated_test,
            instrumented_behavior_test_source=generated_test,  # Not instrumented yet
            instrumented_perf_test_source=generated_test,  # Not instrumented yet
            function_to_optimize=function_to_optimize,
            test_path=test_path,
            test_cfg=test_cfg,
            project_module_system=ModuleSystem.ES_MODULE,
        )

        # EXPECTED: TypeScript imports should NOT have .js extension
        # Because ts-jest runs on .ts source files directly
        assert "../src/utils.js" not in final_generated, (
            "TypeScript test should not have .js extension in import. "
            "ts-jest expects imports without .js when running on .ts source files."
        )
        assert "../src/utils" in final_generated or "../src/utils.ts" in final_generated, (
            "Import should be './utils' (no extension) or './utils.ts' (TS extension)"
        )

    def test_javascript_file_can_have_js_extensions(self, tmp_path):
        """JavaScript test files CAN have .js extensions (no harm in ESM).

        This test documents that adding .js to JavaScript imports is acceptable
        because .js files can import .js files.
        """
        # Setup: JavaScript source file
        source_file = tmp_path / "src" / "utils.js"
        source_file.parent.mkdir(parents=True)
        source_file.write_text(
            """export function myFunction() {
  return 42;
}"""
        )

        # Create package.json with ESM type
        (tmp_path / "package.json").write_text('{"type": "module"}')

        # AI service generates test
        generated_test = """import { myFunction } from '../src/utils';

describe('myFunction', () => {
  test('returns 42', () => {
    expect(myFunction()).toBe(42);
  });
});
"""

        # Setup test config
        lang_support = JavaScriptSupport()
        test_cfg = TestConfig(
            tests_root=tmp_path / "tests",
            project_root_path=tmp_path,
            tests_project_rootdir=tmp_path,
        )

        function_to_optimize = FunctionToOptimize(
            function_name="myFunction",
            file_path=str(source_file),
            line_number=1,
        )

        test_path = tmp_path / "tests" / "test_myFunction.test.js"
        test_path.parent.mkdir(parents=True)

        # Process the test
        (
            final_generated,
            instrumented_behavior,
            instrumented_perf,
        ) = lang_support.process_generated_test_strings(
            generated_test_source=generated_test,
            instrumented_behavior_test_source=generated_test,
            instrumented_perf_test_source=generated_test,
            function_to_optimize=function_to_optimize,
            test_path=test_path,
            test_cfg=test_cfg,
            project_module_system=ModuleSystem.ES_MODULE,
        )

        # For JavaScript, .js extension is OK (and required for ESM)
        # So we're fine either way
        # This test just documents the behavior - no assertion needed
        pass
