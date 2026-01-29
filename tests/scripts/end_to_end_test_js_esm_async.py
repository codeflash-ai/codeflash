"""End-to-end test for JavaScript ES Modules async function optimization.

Tests optimization of async functions using ES Module format.
This tests the ESM module system with async/await code patterns.
"""

import pathlib

from end_to_end_test_utilities_js import JSTestConfig, run_js_codeflash_command, run_with_retries


def run_test() -> bool:
    """Run the ES Modules async function optimization test."""
    config = JSTestConfig(
        file_path=pathlib.Path("async_utils.js"),
        function_name="processItemsSequential",
        min_improvement_x=0.05,  # Async optimizations may have variable gains
        expected_improvement_pct=5,
        expected_test_files=1,
    )

    cwd = (
        pathlib.Path(__file__).parent.parent.parent
        / "code_to_optimize"
        / "js"
        / "code_to_optimize_js_esm"
    ).resolve()

    return run_js_codeflash_command(cwd, config)


if __name__ == "__main__":
    exit(run_with_retries(run_test))
