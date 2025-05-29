import os
import pathlib
import tomlkit

from codeflash.code_utils.code_utils import add_addopts_to_pyproject
from end_to_end_test_utilities import CoverageExpectation, TestConfig, run_codeflash_command, run_with_retries


def run_test(expected_improvement_pct: int) -> bool:
    with add_addopts_to_pyproject():
        config = TestConfig(
            file_path="topological_sort.py",
            function_name="Graph.topologicalSort",
            test_framework="pytest",
            min_improvement_x=0.05,
            coverage_expectations=[
                CoverageExpectation(
                    function_name="Graph.topologicalSort",
                    expected_coverage=100.0,
                    expected_lines=[24, 25, 26, 27, 28, 29],
                )
            ],
        )
        cwd = (pathlib.Path(__file__).parent.parent.parent / "code_to_optimize").resolve()
        return_var = run_codeflash_command(cwd, config, expected_improvement_pct)
    return return_var


if __name__ == "__main__":
    exit(run_with_retries(run_test, int(os.getenv("EXPECTED_IMPROVEMENT_PCT", 5))))
