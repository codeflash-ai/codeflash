"""Mixin: test generation, review, and repair API endpoints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import requests

from codeflash_python.api.types import FunctionRepairInfo, TestFileReview
from codeflash_python.code_utils.config_consts import PYTHON_VALID_TEST_FRAMEWORKS
from codeflash_python.telemetry.posthog_cf import ph
from codeflash_python.version import __version__ as codeflash_version

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash_core.models import FunctionToOptimize
else:
    _Base = object

logger = logging.getLogger("codeflash_python")


class AiServiceTestgenMixin(_Base):  # type: ignore[name-defined]
    def generate_regression_tests(
        self,
        source_code_being_tested: str,
        function_to_optimize: FunctionToOptimize,
        helper_function_names: list[str],
        module_path: Path,
        test_module_path: Path,
        test_framework: str,
        test_timeout: int,
        trace_id: str,
        test_index: int,
        *,
        language: str = "python",
        language_version: str | None = None,
        module_system: str | None = None,
        is_numerical_code: bool | None = None,
    ) -> tuple[str, str, str, str | None] | None:
        """Generate regression tests for the given function by making a request to the Django endpoint.

        Parameters
        ----------
        - source_code_being_tested (str): The source code of the function being tested.
        - function_to_optimize (FunctionToOptimize): The function to optimize.
        - helper_function_names (list[Source]): List of helper function names.
        - module_path (Path): The module path where the function is located.
        - test_module_path (Path): The module path for the test code.
        - test_framework (str): The test framework to use, e.g., "pytest".
        - test_timeout (int): The timeout for each test in seconds.
        - test_index (int): The index from 0-(n-1) if n tests are generated for a single trace_id
        - language (str): Programming language (e.g., "python")
        - language_version (str | None): Language version (e.g., "3.11.0")
        - module_system (str | None): Module system (None for Python)

        Returns
        -------
        - Dict[str, str] | None: The generated regression tests and instrumented tests, or None if an error occurred.

        """
        valid_frameworks = PYTHON_VALID_TEST_FRAMEWORKS
        assert test_framework in valid_frameworks, (
            f"Invalid test framework for python, got {test_framework} but expected one of {list(valid_frameworks)}"
        )

        payload: dict[str, Any] = {
            "source_code_being_tested": source_code_being_tested,
            "function_to_optimize": function_to_optimize,
            "helper_function_names": helper_function_names,
            "module_path": module_path,
            "test_module_path": test_module_path,
            "test_framework": test_framework,
            "test_timeout": test_timeout,
            "trace_id": trace_id,
            "test_index": test_index,
            "language": language,
            "codeflash_version": codeflash_version,
            "is_async": function_to_optimize.is_async,
            "call_sequence": self.get_next_sequence(),
            "is_numerical_code": is_numerical_code,
            "class_name": function_to_optimize.class_name,
            "qualified_name": function_to_optimize.qualified_name,
        }

        self.add_language_metadata(payload, language_version, module_system)

        # DEBUG: Print payload language field
        logger.debug("Sending testgen request with language='%s', framework='%s'", payload["language"], test_framework)
        try:
            response = self.make_ai_service_request("/testgen", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            from codeflash_python.telemetry.posthog_cf import ph

            logger.exception("Error generating tests: %s", e)
            ph("cli-testgen-error-caught", {"error": str(e)})
            return None

        # the timeout should be the same as the timeout for the AI service backend

        if response.status_code == 200:
            response_json = response.json()
            logger.debug("Generated tests for function %s", function_to_optimize.function_name)
            return (
                response_json["generated_tests"],
                response_json["instrumented_behavior_tests"],
                response_json["instrumented_perf_tests"],
                response_json.get("raw_generated_tests"),
            )
        self.log_error_response(response, "generating tests", "cli-testgen-error-response")
        return None

    def review_generated_tests(
        self,
        tests: list[dict[str, Any]],
        function_source_code: str,
        function_name: str,
        trace_id: str,
        coverage_summary: str = "",
        coverage_details: dict[str, Any] | None = None,
        language: str = "python",
    ) -> list[TestFileReview]:
        payload: dict[str, Any] = {
            "tests": tests,
            "function_source_code": function_source_code,
            "function_name": function_name,
            "trace_id": trace_id,
            "language": language,
            "codeflash_version": codeflash_version,
            "call_sequence": self.get_next_sequence(),
        }
        if coverage_summary:
            payload["coverage_summary"] = coverage_summary
        if coverage_details:
            payload["coverage_details"] = coverage_details
        self.add_language_metadata(payload)
        try:
            response = self.make_ai_service_request("/testgen_review", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.exception("Error reviewing generated tests: %s", e)
            ph("cli-testgen-review-error-caught", {"error": str(e)})
            return []

        if response.status_code == 200:
            data = response.json()
            return [
                TestFileReview(
                    test_index=r["test_index"],
                    functions_to_repair=[
                        FunctionRepairInfo(function_name=f["function_name"], reason=f.get("reason", ""))
                        for f in r.get("functions", [])
                    ],
                )
                for r in data.get("reviews", [])
            ]
        self.log_error_response(response, "reviewing generated tests", "cli-testgen-review-error-response")
        return []

    def repair_generated_tests(
        self,
        test_source: str,
        functions_to_repair: list[FunctionRepairInfo],
        function_source_code: str,
        function_to_optimize: FunctionToOptimize,
        helper_function_names: list[str],
        module_path: Path,
        test_module_path: Path,
        test_framework: str,
        test_timeout: int,
        trace_id: str,
        language: str = "python",
        coverage_details: dict[str, Any] | None = None,
        previous_repair_errors: dict[str, str] | None = None,
        module_source_code: str = "",
    ) -> tuple[str, str, str] | None:
        payload: dict[str, Any] = {
            "test_source": test_source,
            "functions_to_repair": [
                {"function_name": f.function_name, "reason": f.reason} for f in functions_to_repair
            ],
            "function_source_code": function_source_code,
            "function_to_optimize": function_to_optimize,
            "helper_function_names": helper_function_names,
            "module_path": module_path,
            "test_module_path": test_module_path,
            "test_framework": test_framework,
            "test_timeout": test_timeout,
            "trace_id": trace_id,
            "language": language,
            "codeflash_version": codeflash_version,
            "call_sequence": self.get_next_sequence(),
        }
        if module_source_code:
            payload["module_source_code"] = module_source_code
        if coverage_details:
            payload["coverage_details"] = coverage_details
        if previous_repair_errors:
            payload["previous_repair_errors"] = previous_repair_errors
        self.add_language_metadata(payload)
        try:
            response = self.make_ai_service_request("/testgen_repair", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.exception("Error repairing generated tests: %s", e)
            ph("cli-testgen-repair-error-caught", {"error": str(e)})
            return None

        if response.status_code == 200:
            data = response.json()
            return (data["generated_tests"], data["instrumented_behavior_tests"], data["instrumented_perf_tests"])
        self.log_error_response(response, "repairing generated tests", "cli-testgen-repair-error-response")
        return None
