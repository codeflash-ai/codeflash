import os
import pathlib

from end_to_end_test_utilities import CoverageExpectation, TestConfig, run_codeflash_command, run_with_retries


def run_test(expected_improvement_pct: int) -> bool:
    config = TestConfig(
        file_path="topological_sort.py",
        function_name="Graph.topologicalSort",
        test_framework="pytest",
        min_improvement_x=0.05,
        coverage_expectations=[
            CoverageExpectation(
                function_name="Graph.topologicalSort", expected_coverage=100.0, expected_lines=[22, 23, 25, 26, 27, 29]
            )
        ],
    )
    cwd = (pathlib.Path(__file__).parent.parent.parent / "code_to_optimize").resolve()
    return run_codeflash_command(cwd, config, expected_improvement_pct)


if __name__ == "__main__":
    exit(run_with_retries(run_test, int(os.getenv("EXPECTED_IMPROVEMENT_PCT", 5))))
