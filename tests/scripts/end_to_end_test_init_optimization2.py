import os
import pathlib

from end_to_end_test_utilities import CoverageExpectation, TestConfig, run_codeflash_command, run_with_retries


def run_test(expected_improvement_pct: int) -> bool:
    config = TestConfig(
        file_path="quote_normalizer.py",
        function_name="QuoteNormalizer.normalize_quotes",
        test_framework="pytest",
        min_improvement_x=0.1,
        coverage_expectations=[
            CoverageExpectation(
                function_name="QuoteNormalizer.normalize_quotes", expected_coverage=100.0, expected_lines=[7, 8, 9, 10]
            )
        ],
    )
    cwd = (pathlib.Path(__file__).parent.parent.parent / "code_to_optimize").resolve()
    return run_codeflash_command(cwd, config, expected_improvement_pct)


if __name__ == "__main__":
    exit(run_with_retries(run_test, int(os.getenv("EXPECTED_IMPROVEMENT_PCT", 10))))
