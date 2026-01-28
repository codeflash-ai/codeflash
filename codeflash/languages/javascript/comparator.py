"""JavaScript test result comparison.

This module provides functionality to compare test results between
original and optimized JavaScript code using a Node.js comparison script.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from codeflash.cli_cmds.console import logger
from codeflash.models.models import TestDiff, TestDiffScope


def _get_compare_results_script(project_root: Path | None = None) -> Path | None:
    """Find the compare-results.js script from the installed codeflash npm package.

    Args:
        project_root: Project root directory where node_modules is installed.

    Returns:
        Path to compare-results.js if found, None otherwise.
    """
    search_dirs = []
    if project_root:
        search_dirs.append(project_root)
    search_dirs.append(Path.cwd())

    for base_dir in search_dirs:
        script_path = base_dir / "node_modules" / "codeflash" / "runtime" / "compare-results.js"
        if script_path.exists():
            return script_path

    return None


def compare_test_results(
    original_sqlite_path: Path,
    candidate_sqlite_path: Path,
    comparator_script: Path | None = None,
    project_root: Path | None = None,
) -> tuple[bool, list[TestDiff]]:
    """Compare JavaScript test results using the Node.js comparator.

    This function calls a Node.js script that:
    1. Reads serialized behavior data from both SQLite databases
    2. Deserializes using the codeflash serializer module
    3. Compares using the codeflash comparator module (handles Map, Set, Date, etc. natively)
    4. Returns comparison results as JSON

    Args:
        original_sqlite_path: Path to SQLite database with original code results.
        candidate_sqlite_path: Path to SQLite database with candidate code results.
        comparator_script: Optional path to the comparison script.
        project_root: Project root directory where node_modules is installed.

    Returns:
        Tuple of (all_equivalent, list of TestDiff objects).
    """
    script_path = comparator_script or _get_compare_results_script(project_root)

    if not script_path or not script_path.exists():
        logger.error(
            "JavaScript comparator script not found. "
            "Please ensure the 'codeflash' npm package is installed in your project."
        )
        return False, []

    if not original_sqlite_path.exists():
        logger.error(f"Original SQLite database not found: {original_sqlite_path}")
        return False, []

    if not candidate_sqlite_path.exists():
        logger.error(f"Candidate SQLite database not found: {candidate_sqlite_path}")
        return False, []

    # Determine working directory - should be where node_modules is installed
    # The script needs better-sqlite3 which is installed in the project's node_modules
    cwd = project_root or Path.cwd()

    # Set NODE_PATH to include the project's node_modules
    # This is needed because the script runs from the codeflash package directory,
    # but needs to resolve modules from the project's node_modules
    env = os.environ.copy()
    node_modules_path = cwd / "node_modules"
    if node_modules_path.exists():
        existing_node_path = env.get("NODE_PATH", "")
        if existing_node_path:
            env["NODE_PATH"] = f"{node_modules_path}:{existing_node_path}"
        else:
            env["NODE_PATH"] = str(node_modules_path)

    try:
        result = subprocess.run(
            ["node", str(script_path), str(original_sqlite_path), str(candidate_sqlite_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(cwd),
            env=env,
        )

        # Parse the JSON output first - errors are reported in JSON too
        try:
            if not result.stdout or not result.stdout.strip():
                logger.error("JavaScript comparator returned empty output")
                if result.stderr:
                    logger.error(f"stderr: {result.stderr}")
                return False, []
            comparison = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JavaScript comparator output: {e}")
            logger.error(f"stdout: {result.stdout[:500] if result.stdout else '(empty)'}")
            if result.stderr:
                logger.error(f"stderr: {result.stderr[:500]}")
            return False, []

        # Check for errors in the JSON response
        # Exit code 0 = equivalent, 1 = not equivalent, 2 = setup error
        if comparison.get("error"):
            logger.error(f"JavaScript comparator error: {comparison['error']}")
            return False, []

        # Check for unexpected exit codes (not 0 or 1)
        if result.returncode != 0 and result.returncode != 1:
            logger.error(f"JavaScript comparator failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"stderr: {result.stderr}")
            return False, []

        # Convert diffs to TestDiff objects
        test_diffs: list[TestDiff] = []
        for diff in comparison.get("diffs", []):
            scope_str = diff.get("scope", "return_value")
            scope = TestDiffScope.RETURN_VALUE
            if scope_str == "stdout":
                scope = TestDiffScope.STDOUT
            elif scope_str == "did_pass":
                scope = TestDiffScope.DID_PASS

            test_info = diff.get("test_info", {})
            # Build a test identifier string for JavaScript tests
            test_function_name = test_info.get("test_function_name", "unknown")
            function_getting_tested = test_info.get("function_getting_tested", "unknown")
            test_src_code = f"// Test: {test_function_name}\n// Testing function: {function_getting_tested}"

            test_diffs.append(
                TestDiff(
                    scope=scope,
                    original_value=diff.get("original"),
                    candidate_value=diff.get("candidate"),
                    test_src_code=test_src_code,
                    candidate_pytest_error=diff.get("candidate_error"),
                    original_pass=True,  # Assume passed if we got results
                    candidate_pass=diff.get("scope") != "missing",
                    original_pytest_error=None,
                )
            )

            logger.debug(
                f"JavaScript test diff:\n"
                f"  Test: {test_function_name}\n"
                f"  Function: {function_getting_tested}\n"
                f"  Scope: {scope_str}\n"
                f"  Original: {str(diff.get('original', 'N/A'))[:100]}\n"
                f"  Candidate: {str(diff.get('candidate', 'N/A'))[:100] if diff.get('candidate') else 'N/A'}"
            )

        equivalent = comparison.get("equivalent", False)

        logger.info(
            f"JavaScript comparison: {'equivalent' if equivalent else 'DIFFERENT'} "
            f"({comparison.get('total_invocations', 0)} invocations, {len(test_diffs)} diffs)"
        )

        return equivalent, test_diffs

    except subprocess.TimeoutExpired:
        logger.error("JavaScript comparator timed out")
        return False, []
    except FileNotFoundError:
        logger.error("Node.js not found. Please install Node.js to compare JavaScript test results.")
        return False, []
    except Exception as e:
        logger.error(f"Error running JavaScript comparator: {e}")
        return False, []
