"""End-to-end test for JavaScript CommonJS optimization.

Tests optimization of a simple recursive fibonacci function using CommonJS module format.
"""

import pathlib

from end_to_end_test_utilities_js import JSTestConfig, run_js_codeflash_command, run_with_retries


def run_test() -> bool:
    """Run the CommonJS fibonacci optimization test."""
    config = JSTestConfig(
        file_path=pathlib.Path("fibonacci.js"),
        function_name="fibonacci",
        min_improvement_x=0.5,  # Expect at least 50% improvement
        expected_improvement_pct=50,
        expected_test_files=1,
    )

    cwd = (
        pathlib.Path(__file__).parent.parent.parent
        / "code_to_optimize"
        / "js"
        / "code_to_optimize_js"
    ).resolve()

    return run_js_codeflash_command(cwd, config)


if __name__ == "__main__":
    exit(run_with_retries(run_test))
