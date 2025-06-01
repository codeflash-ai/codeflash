import argparse
from pathlib import Path
import tempfile

from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.optimization.function_optimizer import FunctionOptimizer
from codeflash.verification.verification_utils import TestConfig


def test_reformat_code_and_helpers():
    """
    reformat_code_and_helpers should only format the code that is optimized not the whole file, to avoid large diffing
    """
    with tempfile.TemporaryDirectory() as test_dir_str:
        test_dir = Path(test_dir_str)
        target_path = test_dir / "target.py"
        unformatted_code = """import sys


def lol():
    print(       "lol" )




class MyClass:
    def __init__(self, x=0):
        self.x = x

    def lol(self):
        print(       "lol" )

    def lol2    (self):
        print(       " lol2" )"""
        expected_code = """import sys


def lol():
    print(       "lol" )




class MyClass:
    def __init__(self, x=0):
        self.x = x

    def lol(self):
        print(       "lol" )

    def lol2(self):
        print(" lol2")
"""
        target_path.write_text(unformatted_code, encoding="utf-8")
        function_to_optimize = FunctionToOptimize(function_name="MyClass.lol2", parents=[], file_path=target_path)

        test_cfg = TestConfig(
            tests_root=test_dir,
            project_root_path=test_dir,
            test_framework="pytest",
            tests_project_rootdir=test_dir,
        )
        args = argparse.Namespace(
            disable_imports_sorting=False,
            formatter_cmds=[
                "ruff check --exit-zero --fix $file",
                "ruff format $file"
            ],
        )
        optimizer = FunctionOptimizer(
            function_to_optimize=function_to_optimize,
            test_cfg=test_cfg,
            args=args,
        )


        formatted_code,_ = optimizer.reformat_code_and_helpers(
            helper_functions=[],
            path=target_path,
            original_code=optimizer.function_to_optimize_source_code,
            opt_func_name=function_to_optimize.function_name
        )
        assert formatted_code == expected_code