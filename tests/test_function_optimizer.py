import argparse
from pathlib import Path
import shutil
import tempfile

import pytest

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig

def test_bubble_sort_preserve_bad_formatting():
    """
    Test the bubble sort implementation in code_to_optimize/bubble_sort_preserve_bad_formatting_for_nonoptimized_code.py.

    This test sets the rubric for all other tests of formatting functionality.
    """
    with tempfile.TemporaryDirectory() as test_dir_str:
        test_dir = Path(test_dir_str)
        target_path = test_dir / "target.py"
        this_file = Path(__file__).resolve()
        repo_root_dir = this_file.parent.parent
        source_file = repo_root_dir / "code_to_optimize" / "bubble_sort_preserve_bad_formatting_for_nonoptimized_code.py"
        shutil.copy2(source_file, target_path)

        original_content = source_file.read_text()

        function_to_optimize = FunctionToOptimize(
            function_name="sorter",
            file_path=target_path,
            parents=[],
            starting_line=None,
            ending_line=None,
        )
        test_cfg = TestConfig(
            tests_root=test_dir,
            project_root_path=test_dir,
            test_framework="pytest",
            tests_project_rootdir=test_dir,
        )
        args = argparse.Namespace(
            disable_imports_sorting=False,
            formatter_cmds=["uvx ruff check --exit-zero --fix $file", "uvx ruff format $file"],
        )
        optimizer = FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=test_cfg,
            args=args,
        )

        preexisting_functions_by_filepath = {
            target_path: {("lol", tuple())},
        }

        # add a newline after the function definition
        target_content = target_path.read_text()
        target_content = target_content.replace("def sorter(arr):", "def sorter(arr):\n")
        assert target_content != original_content
        target_path.write_text(target_content)

        optimizer.reformat_code_and_helpers(
            preexisting_functions_by_filepath=preexisting_functions_by_filepath,
            helper_functions=[],
            fto_path=target_path,
            original_code=optimizer.function_to_optimize_source_code,
        )
        content = target_path.read_text()
        assert content == original_content
