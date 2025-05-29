import os
import pathlib
import tomlkit

from end_to_end_test_utilities import CoverageExpectation, TestConfig, run_codeflash_command, run_with_retries


def run_test(expected_improvement_pct: int) -> bool:
    try:
        # Modify Pyproject file
        with pathlib.Path.open((pathlib.Path(__file__).parent.parent.parent / "pyproject.toml").resolve(), encoding="utf-8") as f:
            original_content = f.read()
            data = tomlkit.parse(original_content)
        data["tool"]["pytest"] = {}
        data["tool"]["pytest"]["ini_options"] = {}
        data["tool"]["pytest"]["ini_options"]["addopts"] = ["-n=auto", "-n", "1", "-n 1", "-n      1", "-n      auto"]
        with pathlib.Path.open((pathlib.Path(__file__).parent.parent.parent / "pyproject.toml").resolve(), "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(data))
        config = TestConfig(
            file_path="topological_sort.py",
            function_name="Graph.topologicalSort",
            test_framework="pytest",
            min_improvement_x=0.05,
            coverage_expectations=[
                CoverageExpectation(
                    function_name="Graph.topologicalSort", expected_coverage=100.0, expected_lines=[24, 25, 26, 27, 28, 29]
                )
            ],
        )
        cwd = (pathlib.Path(__file__).parent.parent.parent / "code_to_optimize").resolve()
        return_var = run_codeflash_command(cwd, config, expected_improvement_pct)
    finally:
        with pathlib.Path.open(pathlib.Path("pyproject.toml"), "w", encoding="utf-8") as f:
            f.write(original_content)
    return return_var


if __name__ == "__main__":
    exit(run_with_retries(run_test, int(os.getenv("EXPECTED_IMPROVEMENT_PCT", 5))))
