from __future__ import annotations

import json
import reprlib
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from codeflash.cli_cmds.console import logger
from codeflash.code_utils.code_utils import shorten_pytest_error
from codeflash.models.models import TestDiff, TestDiffScope, TestResults, TestType, VerificationType
from codeflash.verification.comparator import comparator

if TYPE_CHECKING:
    from codeflash.models.models import TestResults

INCREASED_RECURSION_LIMIT = 5000

# Path to JavaScript comparison script (relative to codeflash package)
JAVASCRIPT_COMPARATOR_SCRIPT = (
    Path(__file__).parent.parent / "languages" / "javascript" / "runtime" / "codeflash-compare-results.js"
)

reprlib_repr = reprlib.Repr()
reprlib_repr.maxstring = 1500
test_diff_repr = reprlib_repr.repr


def compare_test_results(
    original_results: TestResults,
    candidate_results: TestResults,
    pass_fail_only: bool = False,  # noqa: FBT001, FBT002
) -> tuple[bool, list[TestDiff]]:
    # This is meant to be only called with test results for the first loop index
    if len(original_results) == 0 or len(candidate_results) == 0:
        return False, []  # empty test results are not equal
    original_recursion_limit = sys.getrecursionlimit()
    if original_recursion_limit < INCREASED_RECURSION_LIMIT:
        sys.setrecursionlimit(INCREASED_RECURSION_LIMIT)  # Increase recursion limit to avoid RecursionError
    test_ids_superset = original_results.get_all_unique_invocation_loop_ids().union(
        set(candidate_results.get_all_unique_invocation_loop_ids())
    )
    test_diffs: list[TestDiff] = []
    did_all_timeout: bool = True
    for test_id in test_ids_superset:
        original_test_result = original_results.get_by_unique_invocation_loop_id(test_id)
        cdd_test_result = candidate_results.get_by_unique_invocation_loop_id(test_id)

        if cdd_test_result is not None and original_test_result is None:
            continue
        # If helper function instance_state verification is not present, that's ok. continue
        if (
            original_test_result.verification_type
            and original_test_result.verification_type == VerificationType.INIT_STATE_HELPER
            and cdd_test_result is None
        ):
            continue
        if original_test_result is None or cdd_test_result is None:
            continue
        did_all_timeout = did_all_timeout and original_test_result.timed_out
        if original_test_result.timed_out:
            continue
        superset_obj = False
        if original_test_result.verification_type and (
            original_test_result.verification_type
            in {VerificationType.INIT_STATE_HELPER, VerificationType.INIT_STATE_FTO}
        ):
            superset_obj = True

        candidate_test_failures = candidate_results.test_failures
        original_test_failures = original_results.test_failures
        cdd_pytest_error = (
            candidate_test_failures.get(original_test_result.id.test_fn_qualified_name(), "")
            if candidate_test_failures
            else ""
        )
        if cdd_pytest_error:
            cdd_pytest_error = shorten_pytest_error(cdd_pytest_error)
        original_pytest_error = (
            original_test_failures.get(original_test_result.id.test_fn_qualified_name(), "")
            if original_test_failures
            else ""
        )
        if original_pytest_error:
            original_pytest_error = shorten_pytest_error(original_pytest_error)

        if not pass_fail_only and comparator(
            original_test_result.return_value, cdd_test_result.return_value, superset_obj=superset_obj
        ):
            test_diffs.append(
                TestDiff(
                    scope=TestDiffScope.RETURN_VALUE,
                    original_value=test_diff_repr(repr(original_test_result.return_value)),
                    candidate_value=test_diff_repr(repr(cdd_test_result.return_value)),
                    test_src_code=original_test_result.id.get_src_code(original_test_result.file_name),
                    candidate_pytest_error=cdd_pytest_error,
                    original_pass=original_test_result.did_pass,
                    candidate_pass=cdd_test_result.did_pass,
                    original_pytest_error=original_pytest_error,
                )
            )

            try:
                logger.debug(
                    f"File Name: {original_test_result.file_name}\n"
                    f"Test Type: {original_test_result.test_type}\n"
                    f"Verification Type: {original_test_result.verification_type}\n"
                    f"Invocation ID: {original_test_result.id}\n"
                    f"Original return value: {original_test_result.return_value}\n"
                    f"Candidate return value: {cdd_test_result.return_value}\n"
                )
            except Exception as e:
                logger.error(e)
        elif (
            not pass_fail_only
            and (original_test_result.stdout and cdd_test_result.stdout)
            and not comparator(original_test_result.stdout, cdd_test_result.stdout)
        ):
            test_diffs.append(
                TestDiff(
                    scope=TestDiffScope.STDOUT,
                    original_value=str(original_test_result.stdout),
                    candidate_value=str(cdd_test_result.stdout),
                    test_src_code=original_test_result.id.get_src_code(original_test_result.file_name),
                    candidate_pytest_error=cdd_pytest_error,
                    original_pass=original_test_result.did_pass,
                    candidate_pass=cdd_test_result.did_pass,
                    original_pytest_error=original_pytest_error,
                )
            )

        elif original_test_result.test_type in {
            TestType.EXISTING_UNIT_TEST,
            TestType.CONCOLIC_COVERAGE_TEST,
            TestType.GENERATED_REGRESSION,
            TestType.REPLAY_TEST,
        } and (cdd_test_result.did_pass != original_test_result.did_pass):
            test_diffs.append(
                TestDiff(
                    scope=TestDiffScope.DID_PASS,
                    original_value=str(original_test_result.did_pass),
                    candidate_value=str(cdd_test_result.did_pass),
                    test_src_code=original_test_result.id.get_src_code(original_test_result.file_name),
                    candidate_pytest_error=cdd_pytest_error,
                    original_pass=original_test_result.did_pass,
                    candidate_pass=cdd_test_result.did_pass,
                    original_pytest_error=original_pytest_error,
                )
            )

    sys.setrecursionlimit(original_recursion_limit)
    if did_all_timeout:
        return False, test_diffs
    return len(test_diffs) == 0, test_diffs


def compare_javascript_test_results(
    original_sqlite_path: Path, candidate_sqlite_path: Path, comparator_script: Path | None = None
) -> tuple[bool, list[TestDiff]]:
    """Compare JavaScript test results using the JavaScript comparator.

    This function calls a Node.js script that:
    1. Reads serialized behavior data from both SQLite databases
    2. Deserializes using codeflash-serializer.js
    3. Compares using codeflash-comparator.js (handles Map, Set, Date, etc. natively)
    4. Returns comparison results as JSON

    Args:
        original_sqlite_path: Path to SQLite database with original code results.
        candidate_sqlite_path: Path to SQLite database with candidate code results.
        comparator_script: Optional path to the comparison script.

    Returns:
        Tuple of (all_equivalent, list of TestDiff objects).

    """
    script_path = comparator_script or JAVASCRIPT_COMPARATOR_SCRIPT

    if not script_path.exists():
        logger.error(f"JavaScript comparator script not found: {script_path}")
        return False, []

    if not original_sqlite_path.exists():
        logger.error(f"Original SQLite database not found: {original_sqlite_path}")
        return False, []

    if not candidate_sqlite_path.exists():
        logger.error(f"Candidate SQLite database not found: {candidate_sqlite_path}")
        return False, []

    try:
        result = subprocess.run(
            ["node", str(script_path), str(original_sqlite_path), str(candidate_sqlite_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Parse the JSON output
        try:
            comparison = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JavaScript comparator output: {e}")
            logger.debug(f"stdout: {result.stdout}")
            logger.debug(f"stderr: {result.stderr}")
            return False, []

        # Check for errors
        if comparison.get("error"):
            logger.error(f"JavaScript comparator error: {comparison['error']}")
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

            test_diffs.append(
                TestDiff(
                    scope=scope,
                    original_value=diff.get("original"),
                    candidate_value=diff.get("candidate"),
                    test_src_code=None,  # JavaScript tests don't have Python source
                    candidate_pytest_error=None,
                    original_pass=True,  # Assume passed if we got results
                    candidate_pass=diff.get("scope") != "missing",
                    original_pytest_error=None,
                )
            )

            logger.debug(
                f"JavaScript test diff:\n"
                f"  Test: {test_info.get('test_function_name', 'unknown')}\n"
                f"  Function: {test_info.get('function_getting_tested', 'unknown')}\n"
                f"  Scope: {scope_str}\n"
                f"  Original: {diff.get('original', 'N/A')[:100]}\n"
                f"  Candidate: {diff.get('candidate', 'N/A')[:100] if diff.get('candidate') else 'N/A'}"
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
