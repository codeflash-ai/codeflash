import os
import pathlib

from end_to_end_test_utilities import TestConfig, run_codeflash_command, run_with_retries


def run_test(expected_improvement_pct: int) -> bool:
    config = TestConfig(
        file_path="bubble_sort.py", function_name="sorter", test_framework="unittest", min_improvement_x=3.0
    )
    cwd = (pathlib.Path(__file__).parent.parent.parent / "code_to_optimize").resolve()
    return run_codeflash_command(cwd, config, expected_improvement_pct)


if __name__ == "__main__":
    exit(run_with_retries(run_test, int(os.getenv("EXPECTED_IMPROVEMENT_PCT", 300))))
