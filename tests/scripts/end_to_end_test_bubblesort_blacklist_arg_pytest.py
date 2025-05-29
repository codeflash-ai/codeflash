import os
from pathlib import Path

from end_to_end_test_utilities import CoverageExpectation, TestConfig, run_codeflash_command, run_with_retries
import tomlkit


def run_test(expected_improvement_pct: int) -> bool:
    try:
        # Modify Pyproject file
        with Path.open((Path(__file__).parent.parent.parent / "pyproject.toml").resolve(), encoding="utf-8") as f:
            original_content = f.read()
            data = tomlkit.parse(original_content)
        data["tool"]["pytest"] = {}
        data["tool"]["pytest"]["ini_options"] = {}
        data["tool"]["pytest"]["ini_options"]["addopts"] = ["-n=auto", "-n", "1", "-n 1", "-n      1", "-n      auto"]
        with Path.open((Path(__file__).parent.parent.parent / "pyproject.toml").resolve(), "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(data))

        config = TestConfig(
            file_path="bubble_sort.py",
            function_name="sorter",
            test_framework="pytest",
            min_improvement_x=1.0,
            coverage_expectations=[
                CoverageExpectation(
                    function_name="sorter", expected_coverage=100.0, expected_lines=[2, 3, 4, 5, 6, 7, 8, 9, 10]
                )
            ],
        )
        cwd = (Path(__file__).parent.parent.parent / "code_to_optimize").resolve()
        return run_codeflash_command(
            cwd,
            config,
            expected_improvement_pct,
            ['print("codeflash stdout: Sorting list")', 'print(f"result: {arr}")'],
        )
    finally:
        with Path.open(Path("pyproject.toml"), "w", encoding="utf-8") as f:
            f.write(original_content)


if __name__ == "__main__":
    exit(run_with_retries(run_test, int(os.getenv("EXPECTED_IMPROVEMENT_PCT", 100))))
