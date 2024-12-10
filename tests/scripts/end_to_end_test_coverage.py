import os
import pathlib

from end_to_end_test_utilities import CoverageExpectation, TestConfig, run_codeflash_command, run_with_retries


def run_test(expected_improvement_pct: int) -> bool:
    config = TestConfig(
        file_path="bubble_sort.py",
        coverage_expectations=[
            CoverageExpectation(function_name="sorter_one_level_depth", expected_coverage=100.0, expected_lines=[2]),
            CoverageExpectation(function_name="add", expected_coverage=100.0, expected_lines=[44]),
            CoverageExpectation(function_name="multiply_and_add", expected_coverage=100.0, expected_lines=[41]),
        ],
    )
    cwd = (
        pathlib.Path(__file__).parent.parent.parent / "code_to_optimize" / "code_directories" / "my-best-repo"
    ).resolve()
    return run_codeflash_command(cwd, config, expected_improvement_pct)


if __name__ == "__main__":
    exit(run_with_retries(run_test, int(os.getenv("EXPECTED_IMPROVEMENT_PCT", 10))))
