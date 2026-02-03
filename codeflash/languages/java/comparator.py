"""Java test result comparison.

This module provides functionality to compare test results between
original and optimized Java code using the codeflash-runtime Comparator.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeflash.models.models import TestDiff

logger = logging.getLogger(__name__)


def _compare_json_values(json1: str | None, json2: str | None) -> bool:
    """Compare two JSON strings for semantic equality.

    This function parses JSON strings and compares the deserialized objects,
    handling differences in whitespace and key ordering.

    Args:
        json1: First JSON string (or None).
        json2: Second JSON string (or None).

    Returns:
        True if the JSON values are semantically equal, False otherwise.
    """
    # Handle None cases
    if json1 is None and json2 is None:
        return True
    if json1 is None or json2 is None:
        return False

    # Try exact string match first (fast path)
    if json1 == json2:
        return True

    # Parse and compare as JSON
    try:
        obj1 = json.loads(json1)
        obj2 = json.loads(json2)
        return obj1 == obj2
    except (json.JSONDecodeError, TypeError):
        # If JSON parsing fails, fall back to string comparison
        return json1 == json2


def _find_comparator_jar(project_root: Path | None = None) -> Path | None:
    """Find the codeflash-runtime JAR with the Comparator class.

    Args:
        project_root: Project root directory.

    Returns:
        Path to codeflash-runtime JAR if found, None otherwise.

    """
    search_dirs = []
    if project_root:
        search_dirs.append(project_root)
    search_dirs.append(Path.cwd())

    # Search for the JAR in common locations
    for base_dir in search_dirs:
        # Check in target directory (after Maven install)
        for jar_path in [
            base_dir / "target" / "dependency" / "codeflash-runtime-1.0.0.jar",
            base_dir / "target" / "codeflash-runtime-1.0.0.jar",
            base_dir / "lib" / "codeflash-runtime-1.0.0.jar",
            base_dir / ".codeflash" / "codeflash-runtime-1.0.0.jar",
        ]:
            if jar_path.exists():
                return jar_path

        # Check local Maven repository
        m2_jar = Path.home() / ".m2" / "repository" / "com" / "codeflash" / "codeflash-runtime" / "1.0.0" / "codeflash-runtime-1.0.0.jar"
        if m2_jar.exists():
            return m2_jar

    return None


def _find_java_executable() -> str | None:
    """Find the Java executable.

    Returns:
        Path to java executable, or None if not found.

    """
    import shutil

    # Check JAVA_HOME
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        java_path = Path(java_home) / "bin" / "java"
        if java_path.exists():
            return str(java_path)

    # Check PATH
    java_path = shutil.which("java")
    if java_path:
        return java_path

    return None


def compare_test_results(
    original_sqlite_path: Path,
    candidate_sqlite_path: Path,
    comparator_jar: Path | None = None,
    project_root: Path | None = None,
) -> tuple[bool, list]:
    """Compare Java test results using the codeflash-runtime Comparator.

    This function calls the Java Comparator CLI that:
    1. Reads serialized behavior data from both SQLite databases
    2. Deserializes using Gson
    3. Compares results using deep equality (handles Maps, Lists, arrays, etc.)
    4. Returns comparison results as JSON

    Args:
        original_sqlite_path: Path to SQLite database with original code results.
        candidate_sqlite_path: Path to SQLite database with candidate code results.
        comparator_jar: Optional path to the codeflash-runtime JAR.
        project_root: Project root directory.

    Returns:
        Tuple of (all_equivalent, list of TestDiff objects).

    """
    # Import lazily to avoid circular imports
    from codeflash.models.models import TestDiff, TestDiffScope

    java_exe = _find_java_executable()
    if not java_exe:
        logger.error("Java not found. Please install Java to compare test results.")
        return False, []

    jar_path = comparator_jar or _find_comparator_jar(project_root)
    if not jar_path or not jar_path.exists():
        logger.error(
            "codeflash-runtime JAR not found. "
            "Please ensure the codeflash-runtime is installed in your project."
        )
        return False, []

    if not original_sqlite_path.exists():
        logger.error(f"Original SQLite database not found: {original_sqlite_path}")
        return False, []

    if not candidate_sqlite_path.exists():
        logger.error(f"Candidate SQLite database not found: {candidate_sqlite_path}")
        return False, []

    cwd = project_root or Path.cwd()

    try:
        result = subprocess.run(
            [
                java_exe,
                "-cp",
                str(jar_path),
                "com.codeflash.Comparator",
                str(original_sqlite_path),
                str(candidate_sqlite_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(cwd),
        )

        # Parse the JSON output
        try:
            if not result.stdout or not result.stdout.strip():
                logger.error("Java comparator returned empty output")
                if result.stderr:
                    logger.error(f"stderr: {result.stderr}")
                return False, []

            comparison = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Java comparator output: {e}")
            logger.error(f"stdout: {result.stdout[:500] if result.stdout else '(empty)'}")
            if result.stderr:
                logger.error(f"stderr: {result.stderr[:500]}")
            return False, []

        # Check for errors in the JSON response
        if comparison.get("error"):
            logger.error(f"Java comparator error: {comparison['error']}")
            return False, []

        # Check for unexpected exit codes
        if result.returncode not in {0, 1}:
            logger.error(f"Java comparator failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"stderr: {result.stderr}")
            return False, []

        # Convert diffs to TestDiff objects
        test_diffs: list[TestDiff] = []
        for diff in comparison.get("diffs", []):
            scope_str = diff.get("scope", "return_value")
            scope = TestDiffScope.RETURN_VALUE
            if scope_str == "exception":
                scope = TestDiffScope.DID_PASS
            elif scope_str == "missing":
                scope = TestDiffScope.DID_PASS

            # Build test identifier
            method_id = diff.get("methodId", "unknown")
            call_id = diff.get("callId", 0)
            test_src_code = f"// Method: {method_id}\n// Call ID: {call_id}"

            test_diffs.append(
                TestDiff(
                    scope=scope,
                    original_value=diff.get("originalValue"),
                    candidate_value=diff.get("candidateValue"),
                    test_src_code=test_src_code,
                    candidate_pytest_error=diff.get("candidateError"),
                    original_pass=True,
                    candidate_pass=scope_str not in ("missing", "exception"),
                    original_pytest_error=diff.get("originalError"),
                )
            )

            logger.debug(
                f"Java test diff:\n"
                f"  Method: {method_id}\n"
                f"  Call ID: {call_id}\n"
                f"  Scope: {scope_str}\n"
                f"  Original: {str(diff.get('originalValue', 'N/A'))[:100]}\n"
                f"  Candidate: {str(diff.get('candidateValue', 'N/A'))[:100]}"
            )

        equivalent = comparison.get("equivalent", False)

        logger.info(
            f"Java comparison: {'equivalent' if equivalent else 'DIFFERENT'} "
            f"({comparison.get('totalInvocations', 0)} invocations, {len(test_diffs)} diffs)"
        )

        return equivalent, test_diffs

    except subprocess.TimeoutExpired:
        logger.error("Java comparator timed out")
        return False, []
    except FileNotFoundError:
        logger.error("Java not found. Please install Java to compare test results.")
        return False, []
    except Exception as e:
        logger.error(f"Error running Java comparator: {e}")
        return False, []


def compare_invocations_directly(
    original_results: dict,
    candidate_results: dict,
) -> tuple[bool, list]:
    """Compare test invocations directly from Python dictionaries.

    This is a fallback when the Java comparator is not available.
    It performs basic equality comparison on serialized JSON values.

    Args:
        original_results: Dict mapping call_id to result data from original code.
        candidate_results: Dict mapping call_id to result data from candidate code.

    Returns:
        Tuple of (all_equivalent, list of TestDiff objects).

    """
    # Import lazily to avoid circular imports
    from codeflash.models.models import TestDiff, TestDiffScope

    test_diffs: list[TestDiff] = []

    # Get all call IDs
    all_call_ids = set(original_results.keys()) | set(candidate_results.keys())

    for call_id in all_call_ids:
        original = original_results.get(call_id)
        candidate = candidate_results.get(call_id)

        if original is None and candidate is not None:
            # Candidate has extra invocation
            test_diffs.append(
                TestDiff(
                    scope=TestDiffScope.DID_PASS,
                    original_value=None,
                    candidate_value=candidate.get("result_json"),
                    test_src_code=f"// Call ID: {call_id}",
                    candidate_pytest_error=None,
                    original_pass=True,
                    candidate_pass=True,
                    original_pytest_error=None,
                )
            )
        elif original is not None and candidate is None:
            # Candidate missing invocation
            test_diffs.append(
                TestDiff(
                    scope=TestDiffScope.DID_PASS,
                    original_value=original.get("result_json"),
                    candidate_value=None,
                    test_src_code=f"// Call ID: {call_id}",
                    candidate_pytest_error="Missing invocation in candidate",
                    original_pass=True,
                    candidate_pass=False,
                    original_pytest_error=None,
                )
            )
        elif original is not None and candidate is not None:
            # Both have invocations - compare results
            orig_result = original.get("result_json")
            cand_result = candidate.get("result_json")
            orig_error = original.get("error_json")
            cand_error = candidate.get("error_json")

            # Check for exception differences
            if orig_error != cand_error:
                test_diffs.append(
                    TestDiff(
                        scope=TestDiffScope.DID_PASS,
                        original_value=orig_error,
                        candidate_value=cand_error,
                        test_src_code=f"// Call ID: {call_id}",
                        candidate_pytest_error=cand_error,
                        original_pass=orig_error is None,
                        candidate_pass=cand_error is None,
                        original_pytest_error=orig_error,
                    )
                )
            elif not _compare_json_values(orig_result, cand_result):
                # Results differ (using JSON-aware comparison)
                test_diffs.append(
                    TestDiff(
                        scope=TestDiffScope.RETURN_VALUE,
                        original_value=orig_result,
                        candidate_value=cand_result,
                        test_src_code=f"// Call ID: {call_id}",
                        candidate_pytest_error=None,
                        original_pass=True,
                        candidate_pass=True,
                        original_pytest_error=None,
                    )
                )

    equivalent = len(test_diffs) == 0

    logger.info(
        f"Python comparison: {'equivalent' if equivalent else 'DIFFERENT'} "
        f"({len(all_call_ids)} invocations, {len(test_diffs)} diffs)"
    )

    return equivalent, test_diffs
