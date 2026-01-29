"""End-to-end test utilities for JavaScript/TypeScript optimization testing.

Similar to end_to_end_test_utilities.py but adapted for JS/TS projects.
"""

import logging
import os
import pathlib
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class JSTestConfig:
    """Configuration for a JavaScript/TypeScript e2e test."""

    # Path to the source file to optimize (relative to project root)
    file_path: pathlib.Path
    # Function name to optimize (optional - if not specified, optimizes all in file)
    function_name: Optional[str] = None
    # Minimum improvement multiplier (e.g., 0.5 = 50% faster)
    min_improvement_x: float = 0.1
    # Expected improvement percentage (optimization must exceed this)
    expected_improvement_pct: int = 10
    # Expected number of test files discovered
    expected_test_files: Optional[int] = None


def clear_codeflash_directory(cwd: pathlib.Path) -> None:
    """Clear the .codeflash directory to avoid stale state."""
    codeflash_dir = cwd / ".codeflash"
    if codeflash_dir.exists():
        shutil.rmtree(codeflash_dir)


def install_npm_dependencies(cwd: pathlib.Path) -> bool:
    """Install npm dependencies if needed."""
    node_modules = cwd / "node_modules"
    if not node_modules.exists():
        logging.info(f"Installing npm dependencies in {cwd}")
        result = subprocess.run(
            ["npm", "install"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logging.error(f"npm install failed: {result.stderr}")
            return False
    return True


def build_js_command(
    cwd: pathlib.Path,
    config: JSTestConfig,
) -> list[str]:
    """Build the codeflash CLI command for JS/TS optimization."""
    # JS projects are at code_to_optimize/js/code_to_optimize_*, which is 3 levels deep
    # So we need ../../../codeflash/main.py to get to the root
    python_path = "../../../codeflash/main.py"

    base_command = [
        "uv",
        "run",
        "--no-project",
        python_path,
        "--file",
        str(config.file_path),
        "--no-pr",
    ]

    if config.function_name:
        base_command.extend(["--function", config.function_name])

    return base_command


def validate_js_output(
    stdout: str,
    return_code: int,
    config: JSTestConfig,
) -> bool:
    """Validate the output of a JS/TS optimization run."""
    if return_code != 0:
        logging.error(f"Command returned exit code {return_code} instead of 0")
        return False

    if "⚡️ Optimization successful! 📄 " not in stdout:
        logging.error("Failed to find performance improvement message")
        return False

    improvement_match = re.search(r"📈 ([\d,]+)% (?:(\w+) )?improvement", stdout)
    if not improvement_match:
        logging.error("Could not find improvement percentage in output")
        return False

    improvement_pct = int(improvement_match.group(1).replace(",", ""))
    improvement_x = float(improvement_pct) / 100

    logging.info(f"Performance improvement: {improvement_pct}%; Rate: {improvement_x}x")

    if improvement_pct <= config.expected_improvement_pct:
        logging.error(
            f"Performance improvement {improvement_pct}% not above {config.expected_improvement_pct}%"
        )
        return False

    if improvement_x <= config.min_improvement_x:
        logging.error(
            f"Performance improvement rate {improvement_x}x not above {config.min_improvement_x}x"
        )
        return False

    if config.expected_test_files is not None:
        # Look for "Instrumented X existing unit test files" (the actual file count)
        test_files_match = re.search(r"Instrumented (\d+) existing unit test files?", stdout)
        if not test_files_match:
            logging.error("Could not find unit test file count in output")
            return False

        num_test_files = int(test_files_match.group(1))
        if num_test_files < config.expected_test_files:
            logging.error(
                f"Expected at least {config.expected_test_files} test files, found {num_test_files}"
            )
            return False

    logging.info(f"Success: Performance improvement is {improvement_pct}%")
    return True


def run_js_codeflash_command(
    cwd: pathlib.Path,
    config: JSTestConfig,
) -> bool:
    """Run codeflash optimization on a JavaScript/TypeScript project."""
    logging.basicConfig(level=logging.INFO)

    # Save original file contents for potential revert
    path_to_file = cwd / config.file_path
    file_contents = path_to_file.read_text("utf-8")

    # Clear any stale state
    clear_codeflash_directory(cwd)

    # Install dependencies if needed
    if not install_npm_dependencies(cwd):
        return False

    # Build and run command
    command = build_js_command(cwd, config)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    logging.info(f"Running: {' '.join(command)}")

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(cwd),
        env=env,
        encoding="utf-8",
    )

    output = []
    for line in process.stdout:
        logging.info(line.strip())
        output.append(line)

    return_code = process.wait()
    stdout = "".join(output)

    validated = validate_js_output(stdout, return_code, config)
    if not validated:
        # Revert file changes on failure
        path_to_file.write_text(file_contents, "utf-8")
        logging.info("Codeflash run did not meet expected requirements, reverting file changes.")
        return False

    return validated


def run_with_retries(test_func, *args, **kwargs) -> int:
    """Run a test function with retries on failure."""
    max_retries = int(os.getenv("MAX_RETRIES", 3))
    retry_delay = int(os.getenv("RETRY_DELAY", 5))

    log = logging.getLogger()
    log.setLevel(logging.DEBUG)

    for attempt in range(1, max_retries + 1):
        logging.info(f"\n=== Attempt {attempt} of {max_retries} ===")

        if test_func(*args, **kwargs):
            logging.info(f"Test passed on attempt {attempt}")
            return 0

        logging.error(f"Test failed on attempt {attempt}")

        if attempt < max_retries:
            logging.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            logging.error("Test failed after all retries")
            return 1

    return 1