"""Test for false positive test discovery bug (Bug #4)."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.languages.javascript.support import TypeScriptSupport
from codeflash.models.models import CodePosition


def test_discover_tests_should_not_match_mocked_functions():
    """Test that functions mentioned only in mocks are not matched as test targets.

    Regression test for Bug #4: False positive test discovery due to substring matching.

    When a test file mocks a function (e.g., vi.mock("./restart-request.js", () => ({...}))),
    that function should NOT be considered as tested by that file, since it's only mocked,
    not actually called or tested.
    """
    support = TypeScriptSupport()

    with TemporaryDirectory() as tmpdir:
        test_root = Path(tmpdir)

        # Create a test file that MOCKS parseRestartRequestParams but doesn't test it
        test_file = test_root / "update.test.ts"
        test_file.write_text(
            '''
import { updateSomething } from "./update.js";

vi.mock("./restart-request.js", () => ({
  parseRestartRequestParams: (params: any) => ({ sessionKey: undefined }),
}));

describe("updateSomething", () => {
  it("should update successfully", () => {
    const result = updateSomething();
    expect(result).toBe(true);
  });
});
'''
        )

        # Source function that is only mocked, not tested
        source_function = FunctionToOptimize(
            qualified_name="parseRestartRequestParams",
            function_name="parseRestartRequestParams",
            file_path=test_root / "restart-request.ts",
            starting_line=1,
            ending_line=10,
            function_signature="",
            code_position=CodePosition(line_no=1, col_no=0),
            file_path_relative_to_project_root="restart-request.ts",
        )

        # Discover tests
        result = support.discover_tests(test_root, [source_function])

        # The bug: discovers update.test.ts as a test for parseRestartRequestParams
        # because "parseRestartRequestParams" appears as a substring in the mock
        # Expected: should NOT match (empty result)
        assert (
            source_function.qualified_name not in result or len(result[source_function.qualified_name]) == 0
        ), f"Should not match mocked function, but found: {result.get(source_function.qualified_name, [])}"


def test_discover_tests_should_match_actually_imported_functions():
    """Test that functions actually imported and tested ARE correctly matched.

    This is the positive case to ensure we don't break legitimate test discovery.
    """
    support = TypeScriptSupport()

    with TemporaryDirectory() as tmpdir:
        test_root = Path(tmpdir)

        # Create a test file that ACTUALLY imports and tests the function
        test_file = test_root / "restart-request.test.ts"
        test_file.write_text(
            '''
import { parseRestartRequestParams } from "./restart-request.js";

describe("parseRestartRequestParams", () => {
  it("should parse valid params", () => {
    const result = parseRestartRequestParams({ sessionKey: "abc" });
    expect(result.sessionKey).toBe("abc");
  });
});
'''
        )

        source_function = FunctionToOptimize(
            qualified_name="parseRestartRequestParams",
            function_name="parseRestartRequestParams",
            file_path=test_root / "restart-request.ts",
            starting_line=1,
            ending_line=10,
            function_signature="",
            code_position=CodePosition(line_no=1, col_no=0),
            file_path_relative_to_project_root="restart-request.ts",
        )

        result = support.discover_tests(test_root, [source_function])

        # Should match: function is imported and tested
        assert source_function.qualified_name in result, f"Should match imported function, but got: {result}"
        assert len(result[source_function.qualified_name]) > 0, "Should find at least one test"
