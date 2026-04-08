"""Test for coverage exclusion error message (Bug #5 regression test)."""

from pathlib import Path

from codeflash.models.function_types import FunctionToOptimize
from codeflash.models.models import CodePosition


def test_function_to_optimize_has_file_path_not_source_file_path():
    """Test that FunctionToOptimize has file_path attribute, not source_file_path.

    Regression test for Bug #5: Bug #1's fix used wrong attribute name 'source_file_path'
    instead of 'file_path', causing AttributeError when constructing coverage error messages.

    The bug occurred in function_optimizer.py lines 2797 and 2803:
        f"No coverage data found for {self.function_to_optimize.source_file_path}."

    This should be:
        f"No coverage data found for {self.function_to_optimize.file_path}."

    Trace ID: 5c4a75fb-d8eb-4f75-9e57-893f0c44b9c7
    """
    # Create a FunctionToOptimize object
    func = FunctionToOptimize(
        function_name="testFunc",
        file_path=Path("/workspace/target/src/test.ts"),
        starting_line=1,
        ending_line=10,
        code_position=CodePosition(line_no=1, col_no=0),
        file_path_relative_to_project_root="src/test.ts",
    )

    # Verify correct attribute exists
    assert hasattr(func, "file_path"), "FunctionToOptimize should have 'file_path' attribute"
    assert func.file_path == Path("/workspace/target/src/test.ts")

    # Verify wrong attribute does NOT exist
    assert not hasattr(
        func, "source_file_path"
    ), "FunctionToOptimize should NOT have 'source_file_path' attribute (it's a typo/bug)"

    # Verify we can access file_path in string formatting (like the bug location does)
    error_message = f"No coverage data found for {func.file_path}."
    assert "test.ts" in error_message
    # This should NOT raise AttributeError
