import os
import pathlib

from end_to_end_test_utilities import CoverageExpectation, TestConfig, run_codeflash_command, run_with_retries


def run_test(expected_improvement_pct: int) -> bool:
    cwd = (pathlib.Path(__file__).parent.parent.parent / "code_to_optimize").resolve()
    config = TestConfig(
        file_path=pathlib.Path("bubble_sort.py"),
        function_name="sorter",
        benchmarks_root=cwd / "tests" / "pytest" / "benchmarks",
        test_framework="pytest",
        min_improvement_x=1.0,
        coverage_expectations=[
            CoverageExpectation(
                function_name="sorter", expected_coverage=100.0, expected_lines=[2, 3, 4, 5, 6, 7, 8, 9, 10]
            )
        ],
    )

    return run_codeflash_command(cwd, config, expected_improvement_pct)


if __name__ == "__main__":
    exit(run_with_retries(run_test, int(os.getenv("EXPECTED_IMPROVEMENT_PCT", 5))))
