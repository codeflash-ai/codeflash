from __future__ import annotations

from codeflash.code_utils.compat import LF
from codeflash.either import CodeflashError

_TEST_CONFIDENCE_ERROR = CodeflashError(
    "TEST_CONFIDENCE_THRESHOLD_NOT_MET_ERROR", "The threshold for test confidence was not met."
)


def shell_rc_permission_error(shell_rc_path: str, api_key_line: str) -> CodeflashError:
    return CodeflashError(
        "SHELL_RC_PERMISSION_ERROR",
        f"I tried adding your Codeflash API key to {{shell_rc_path}} - but seems like I don't have permissions to do so.{LF}"
        f"You'll need to open it yourself and add the following line:{LF}{LF}{{api_key_line}}{LF}",
        **locals(),
    )


def shell_rc_not_found_error(shell_rc_path: str, api_key_line: str) -> CodeflashError:
    return CodeflashError(
        "SHELL_RC_NOT_FOUND_ERROR",
        f"💡 I went to save your Codeflash API key to {{shell_rc_path}}, but noticed that it doesn't exist.{LF}"
        f"To ensure your Codeflash API key is automatically loaded into your environment at startup, you can create {{shell_rc_path}} and add the following line:{LF}"
        f"{LF}{{api_key_line}}{LF}",
        **locals(),
    )


def test_result_didnt_match_error() -> CodeflashError:
    return CodeflashError(
        "TEST_RESULT_DIDNT_MATCH_ERROR", "Test results did not match the test results of the original code."
    )


def function_optimization_attempted_error() -> CodeflashError:
    return CodeflashError(
        "FUNCTION_OPTIMIZATION_ATTEMPTED_ERROR", "Function optimization previously attempted, skipping."
    )


def baseline_establishment_failed_error(failure_msg: str) -> CodeflashError:
    return CodeflashError(
        "BASELINE_ESTABLISHMENT_FAILED_ERROR",
        "Failed to establish a baseline for the original code. {failure_msg}",
        **locals(),
    )


def no_tests_generated_error(function_name: str) -> CodeflashError:
    return CodeflashError("NO_TESTS_GENERATED_ERROR", "NO TESTS GENERATED for {function_name}", **locals())


def no_optimizations_generated_error(function_name: str) -> CodeflashError:
    return CodeflashError(
        "NO_OPTIMIZATIONS_GENERATED_ERROR", "NO OPTIMIZATIONS GENERATED for {function_name}", **locals()
    )


def no_best_optimization_found_error(function_name: str) -> CodeflashError:
    return CodeflashError(
        "NO_BEST_OPTIMIZATION_FOUND_ERROR", "No best optimizations found for function {function_name}", **locals()
    )


def code_context_extraction_failed_error(error: str) -> CodeflashError:
    return CodeflashError(
        "CODE_CONTEXT_EXTRACTION_FAILED_ERROR", "Failed to extract code context. Error: {error}.", **locals()
    )


def coverage_threshold_not_met_error() -> CodeflashError:
    return CodeflashError("COVERAGE_THRESHOLD_NOT_MET_ERROR", "The threshold for test coverage was not met.")


def test_confidence_threshold_not_met_error() -> CodeflashError:
    return _TEST_CONFIDENCE_ERROR


def behavioral_test_failure_error() -> CodeflashError:
    return CodeflashError(
        "BEHAVIORAL_TEST_FAILURE_ERROR",
        "Failed to establish a baseline for the original code - bevhavioral tests failed.",
    )
