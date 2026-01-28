"""End-to-end test for TypeScript class method optimization.

Tests optimization of class methods in TypeScript.
"""

import pathlib

from end_to_end_test_utilities_js import JSTestConfig, run_js_codeflash_command, run_with_retries


def run_test() -> bool:
    """Run the TypeScript class method optimization test."""
    config = JSTestConfig(
        file_path=pathlib.Path("data_processor.ts"),
        function_name="DataProcessor.findDuplicates",
        min_improvement_x=0.3,  # Expect at least 30% improvement
        expected_improvement_pct=30,
        expected_test_files=1,
    )

    cwd = (
        pathlib.Path(__file__).parent.parent.parent
        / "code_to_optimize"
        / "js"
        / "code_to_optimize_ts"
    ).resolve()

    return run_js_codeflash_command(cwd, config)


if __name__ == "__main__":
    exit(run_with_retries(run_test))
