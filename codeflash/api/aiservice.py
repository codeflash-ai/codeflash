from __future__ import annotations

import json
import os
import platform
import time
from itertools import count
from typing import TYPE_CHECKING, Any, cast

import requests
from pydantic.json import pydantic_encoder

from codeflash.cli_cmds.console import console, logger
from codeflash.code_utils.env_utils import get_codeflash_api_key
from codeflash.code_utils.git_utils import get_last_commit_author_if_pr_exists, get_repo_owner_and_name
from codeflash.code_utils.time_utils import humanize_runtime
from codeflash.models.ExperimentMetadata import ExperimentMetadata
from codeflash.models.models import (
    AIServiceRefinerRequest,
    CodeStringsMarkdown,
    OptimizationReviewResult,
    OptimizedCandidate,
    OptimizedCandidateSource,
)
from codeflash.telemetry.posthog_cf import ph
from codeflash.version import __version__ as codeflash_version

if TYPE_CHECKING:
    from pathlib import Path

    from codeflash.discovery.functions_to_optimize import FunctionToOptimize
    from codeflash.models.ExperimentMetadata import ExperimentMetadata
    from codeflash.models.models import (
        AIServiceAdaptiveOptimizeRequest,
        AIServiceCodeRepairRequest,
        AIServiceRefinerRequest,
    )
    from codeflash.result.explanation import Explanation


class AiServiceClient:
    def __init__(self) -> None:
        self.base_url = self.get_aiservice_base_url()
        self.headers = {"Authorization": f"Bearer {get_codeflash_api_key()}", "Connection": "close"}
        self.llm_call_counter = count(1)
        self.is_local = self.base_url == "http://localhost:8000"
        self.timeout: float | None = None if self.is_local else 90

    def get_next_sequence(self) -> int:
        """Get the next LLM call sequence number."""
        return next(self.llm_call_counter)

    def get_aiservice_base_url(self) -> str:
        if os.environ.get("CODEFLASH_AIS_SERVER", default="prod").lower() == "local":
            logger.info("Using local AI Service at http://localhost:8000")
            console.rule()
            return "http://localhost:8000"
        return "https://app.codeflash.ai"

    def make_ai_service_request(
        self,
        endpoint: str,
        method: str = "POST",
        payload: dict[str, Any] | list[dict[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
        """Make an API request to the given endpoint on the AI service.

        Args:
        ----
            endpoint: The endpoint to call, e.g., "/optimize"
            method: The HTTP method to use ('GET' or 'POST')
            payload: Optional JSON payload to include in the POST request body
            timeout: The timeout for the request in seconds

        Returns:
        -------
            The response object from the API

        Raises:
        ------
            requests.exceptions.RequestException: If the request fails

        """
        """Make an API request to the given endpoint on the AI service.

        :param endpoint: The endpoint to call, e.g., "/optimize".
        :param method: The HTTP method to use ('GET' or 'POST').
        :param payload: Optional JSON payload to include in the POST request body.
        :param timeout: The timeout for the request.
        :return: The response object from the API.
        """
        url = f"{self.base_url}/ai{endpoint}"
        if method.upper() == "POST":
            json_payload = json.dumps(payload, indent=None, default=pydantic_encoder)
            headers = {**self.headers, "Content-Type": "application/json"}
            response = requests.post(url, data=json_payload, headers=headers, timeout=timeout)
        else:
            response = requests.get(url, headers=self.headers, timeout=timeout)
        # response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        return response

    def _get_valid_candidates(
        self, optimizations_json: list[dict[str, Any]], source: OptimizedCandidateSource, language: str = "python"
    ) -> list[OptimizedCandidate]:
        candidates: list[OptimizedCandidate] = []
        for opt in optimizations_json:
            code = CodeStringsMarkdown.parse_markdown_code(opt["source_code"], expected_language=language)
            if not code.code_strings:
                continue
            candidates.append(
                OptimizedCandidate(
                    source_code=code,
                    explanation=opt["explanation"],
                    optimization_id=opt["optimization_id"],
                    source=source,
                    parent_id=opt.get("parent_id", None),
                    model=opt.get("model"),
                )
            )
        return candidates

    def optimize_code(  # noqa: D417
        self,
        source_code: str,
        dependency_code: str,
        trace_id: str,
        experiment_metadata: ExperimentMetadata | None = None,
        *,
        language: str = "python",
        language_version: str | None = None,
        is_async: bool = False,
        n_candidates: int = 5,
    ) -> list[OptimizedCandidate]:
        """Optimize the given code for performance by making a request to the Django endpoint.

        Parameters
        ----------
        - source_code (str): The code to optimize.
        - dependency_code (str): The dependency code used as read-only context for the optimization
        - trace_id (str): Trace id of optimization run
        - experiment_metadata (Optional[ExperimentalMetadata, None]): Any available experiment metadata for this optimization
        - language (str): Programming language ("python", "javascript", "typescript")
        - language_version (str | None): Language version (e.g., "3.11.0" for Python, "ES2022" for JS)
        - is_async (bool): Whether the function being optimized is async
        - n_candidates (int): Number of candidates to generate

        Returns
        -------
        - List[OptimizationCandidate]: A list of Optimization Candidates.

        """
        logger.info("Generating optimized candidates…")
        console.rule()
        start_time = time.perf_counter()
        git_repo_owner, git_repo_name = safe_get_repo_owner_and_name()

        # Build payload with language-specific fields
        payload: dict[str, Any] = {
            "source_code": source_code,
            "dependency_code": dependency_code,
            "trace_id": trace_id,
            "language": language,
            "experiment_metadata": experiment_metadata,
            "codeflash_version": codeflash_version,
            "current_username": get_last_commit_author_if_pr_exists(None),
            "repo_owner": git_repo_owner,
            "repo_name": git_repo_name,
            "is_async": is_async,
            "call_sequence": self.get_next_sequence(),
            "n_candidates": n_candidates,
        }

        # Add language-specific version fields
        # Always include python_version for backward compatibility with older backend
        payload["python_version"] = platform.python_version()
        if language == "python":
            pass  # python_version already set
        else:
            payload["language_version"] = language_version or "ES2022"

        # DEBUG: Print payload language field
        logger.debug(
            f"Sending optimize request with language='{payload['language']}' (type: {type(payload['language'])})"
        )
        logger.debug(f"Sending optimize request: trace_id={trace_id}, n_candidates={payload['n_candidates']}")

        try:
            response = self.make_ai_service_request("/optimize", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.exception(f"Error generating optimized candidates: {e}")
            ph("cli-optimize-error-caught", {"error": str(e)})
            console.rule()
            return []

        if response.status_code == 200:
            optimizations_json = response.json()["optimizations"]
            end_time = time.perf_counter()
            logger.debug(f"!lsp|Generating possible optimizations took {end_time - start_time:.2f} seconds.")
            logger.info(f"!lsp|Received {len(optimizations_json)} optimization candidates.")
            console.rule()
            return self._get_valid_candidates(optimizations_json, OptimizedCandidateSource.OPTIMIZE, language)
        try:
            error = response.json()["error"]
        except Exception:
            error = response.text
        logger.error(f"Error generating optimized candidates: {response.status_code} - {error}")
        ph("cli-optimize-error-response", {"response_status_code": response.status_code, "error": error})
        console.rule()
        return []

    # Backward-compatible alias
    def optimize_python_code(
        self,
        source_code: str,
        dependency_code: str,
        trace_id: str,
        experiment_metadata: ExperimentMetadata | None = None,
        *,
        is_async: bool = False,
        n_candidates: int = 5,
    ) -> list[OptimizedCandidate]:
        """Backward-compatible alias for optimize_code() with language='python'."""
        return self.optimize_code(
            source_code=source_code,
            dependency_code=dependency_code,
            trace_id=trace_id,
            experiment_metadata=experiment_metadata,
            language="python",
            is_async=is_async,
            n_candidates=n_candidates,
        )

    def optimize_python_code_line_profiler(  # noqa: D417
        self,
        source_code: str,
        dependency_code: str,
        trace_id: str,
        line_profiler_results: str,
        n_candidates: int,
        experiment_metadata: ExperimentMetadata | None = None,
    ) -> list[OptimizedCandidate]:
        """Optimize the given python code for performance using line profiler results.

        Parameters
        ----------
        - source_code (str): The python code to optimize.
        - dependency_code (str): The dependency code used as read-only context for the optimization
        - trace_id (str): Trace id of optimization run
        - line_profiler_results (str): Line profiler output to guide optimization
        - experiment_metadata (Optional[ExperimentalMetadata, None]): Any available experiment metadata for this optimization
        - n_candidates (int): Number of candidates to generate

        Returns
        -------
        - List[OptimizationCandidate]: A list of Optimization Candidates.

        """
        if line_profiler_results == "":
            logger.info("No LineProfiler results were provided, Skipping optimization.")
            return []

        logger.info("Generating optimized candidates with line profiler…")
        console.rule()

        payload = {
            "source_code": source_code,
            "dependency_code": dependency_code,
            "n_candidates": n_candidates,
            "line_profiler_results": line_profiler_results,
            "trace_id": trace_id,
            "python_version": platform.python_version(),
            "experiment_metadata": experiment_metadata,
            "codeflash_version": codeflash_version,
            "call_sequence": self.get_next_sequence(),
        }

        try:
            response = self.make_ai_service_request("/optimize-line-profiler", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.exception(f"Error generating optimized candidates: {e}")
            ph("cli-optimize-error-caught", {"error": str(e)})
            console.rule()
            return []

        if response.status_code == 200:
            optimizations_json = response.json()["optimizations"]
            logger.info(f"!lsp|Received {len(optimizations_json)} line profiler optimization candidates.")
            console.rule()
            return self._get_valid_candidates(optimizations_json, OptimizedCandidateSource.OPTIMIZE_LP)
        try:
            error = response.json()["error"]
        except Exception:
            error = response.text
        logger.error(f"Error generating optimized candidates: {response.status_code} - {error}")
        ph("cli-optimize-error-response", {"response_status_code": response.status_code, "error": error})
        console.rule()
        return []

    def adaptive_optimize(self, request: AIServiceAdaptiveOptimizeRequest) -> OptimizedCandidate | None:
        try:
            payload = {
                "trace_id": request.trace_id,
                "original_source_code": request.original_source_code,
                "candidates": request.candidates,
            }
            response = self.make_ai_service_request("/adaptive_optimize", payload=payload, timeout=self.timeout)
        except (requests.exceptions.RequestException, TypeError) as e:
            logger.exception(f"Error generating adaptive optimized candidates: {e}")
            ph("cli-optimize-error-caught", {"error": str(e)})
            return None

        if response.status_code == 200:
            fixed_optimization = response.json()
            console.rule()

            valid_candidates = self._get_valid_candidates([fixed_optimization], OptimizedCandidateSource.ADAPTIVE)
            if not valid_candidates:
                logger.error("Adaptive optimization failed to generate a valid candidate.")
                return None

            return valid_candidates[0]

        try:
            error = response.json()["error"]
        except Exception:
            error = response.text
        logger.error(f"Error generating optimized candidates: {response.status_code} - {error}")
        ph("cli-optimize-error-response", {"response_status_code": response.status_code, "error": error})
        return None

    def optimize_python_code_refinement(self, request: list[AIServiceRefinerRequest]) -> list[OptimizedCandidate]:
        """Optimize the given python code for performance by making a request to the Django endpoint.

        Args:
        request: A list of optimization candidate details for refinement

        Returns:
        -------
        - List[OptimizationCandidate]: A list of Optimization Candidates.

        """
        payload = [
            {
                "optimization_id": opt.optimization_id,
                "original_source_code": opt.original_source_code,
                "read_only_dependency_code": opt.read_only_dependency_code,
                "original_line_profiler_results": opt.original_line_profiler_results,
                "original_code_runtime": humanize_runtime(opt.original_code_runtime),
                "optimized_source_code": opt.optimized_source_code,
                "optimized_explanation": opt.optimized_explanation,
                "optimized_line_profiler_results": opt.optimized_line_profiler_results,
                "optimized_code_runtime": humanize_runtime(opt.optimized_code_runtime),
                "speedup": opt.speedup,
                "trace_id": opt.trace_id,
                "function_references": opt.function_references,
                "python_version": platform.python_version(),
                "call_sequence": self.get_next_sequence(),
            }
            for opt in request
        ]
        try:
            response = self.make_ai_service_request("/refinement", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.exception(f"Error generating optimization refinements: {e}")
            ph("cli-optimize-error-caught", {"error": str(e)})
            return []

        if response.status_code == 200:
            refined_optimizations = response.json()["refinements"]

            return self._get_valid_candidates(refined_optimizations, OptimizedCandidateSource.REFINE)

        try:
            error = response.json()["error"]
        except Exception:
            error = response.text
        logger.error(f"Error generating optimized candidates: {response.status_code} - {error}")
        ph("cli-optimize-error-response", {"response_status_code": response.status_code, "error": error})
        console.rule()
        return []

    def code_repair(self, request: AIServiceCodeRepairRequest) -> OptimizedCandidate | None:
        """Repair the optimization candidate that is not matching the test result of the original code.

        Args:
        request: candidate details for repair

        Returns:
        -------
        - OptimizedCandidate: new fixed candidate.

        """
        console.rule()
        try:
            payload = {
                "optimization_id": request.optimization_id,
                "original_source_code": request.original_source_code,
                "modified_source_code": request.modified_source_code,
                "trace_id": request.trace_id,
                "test_diffs": request.test_diffs,
                "language": request.language,
            }
            response = self.make_ai_service_request("/code_repair", payload=payload, timeout=self.timeout)
        except (requests.exceptions.RequestException, TypeError) as e:
            logger.exception(f"Error generating optimization repair: {e}")
            ph("cli-optimize-error-caught", {"error": str(e)})
            return None

        if response.status_code == 200:
            fixed_optimization = response.json()
            console.rule()

            valid_candidates = self._get_valid_candidates(
                [fixed_optimization], OptimizedCandidateSource.REPAIR, request.language
            )
            if not valid_candidates:
                logger.error("Code repair failed to generate a valid candidate.")
                return None

            return valid_candidates[0]

        try:
            error = response.json()["error"]
        except Exception:
            error = response.text
        logger.error(f"Error generating optimized candidates: {response.status_code} - {error}")
        ph("cli-optimize-error-response", {"response_status_code": response.status_code, "error": error})
        console.rule()
        return None

    def get_new_explanation(  # noqa: D417
        self,
        source_code: str,
        optimized_code: str,
        dependency_code: str,
        trace_id: str,
        original_line_profiler_results: str,
        optimized_line_profiler_results: str,
        original_code_runtime: str,
        optimized_code_runtime: str,
        speedup: str,
        annotated_tests: str,
        optimization_id: str,
        original_explanation: str,
        original_throughput: str | None = None,
        optimized_throughput: str | None = None,
        throughput_improvement: str | None = None,
        function_references: str | None = None,
        codeflash_version: str = codeflash_version,
    ) -> str:
        """Optimize the given python code for performance by making a request to the Django endpoint.

        Parameters
        ----------
        - source_code (str): The python code to optimize.
        - optimized_code (str): The python code generated by the AI service.
        - dependency_code (str): The dependency code used as read-only context for the optimization
        - original_line_profiler_results: str - line profiler results for the baseline code
        - optimized_line_profiler_results: str - line profiler results for the optimized code
        - original_code_runtime: str - runtime for the baseline code
        - optimized_code_runtime: str - runtime for the optimized code
        - speedup: str - speedup of the optimized code
        - annotated_tests: str - test functions annotated with runtime
        - optimization_id: str - unique id of opt candidate
        - original_explanation: str - original_explanation generated for the opt candidate
        - original_throughput: str | None - throughput for the baseline code (operations per second)
        - optimized_throughput: str | None - throughput for the optimized code (operations per second)
        - throughput_improvement: str | None - throughput improvement percentage
        - current codeflash version
        - function_references: str | None - where the function is called in the codebase

        Returns
        -------
        - List[OptimizationCandidate]: A list of Optimization Candidates.

        """
        payload = {
            "trace_id": trace_id,
            "source_code": source_code,
            "optimized_code": optimized_code,
            "original_line_profiler_results": original_line_profiler_results,
            "optimized_line_profiler_results": optimized_line_profiler_results,
            "original_code_runtime": original_code_runtime,
            "optimized_code_runtime": optimized_code_runtime,
            "speedup": speedup,
            "annotated_tests": annotated_tests,
            "optimization_id": optimization_id,
            "original_explanation": original_explanation,
            "dependency_code": dependency_code,
            "original_throughput": original_throughput,
            "optimized_throughput": optimized_throughput,
            "throughput_improvement": throughput_improvement,
            "function_references": function_references,
            "codeflash_version": codeflash_version,
            "call_sequence": self.get_next_sequence(),
        }
        logger.info("loading|Generating explanation")
        console.rule()
        try:
            response = self.make_ai_service_request("/explain", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.exception(f"Error generating explanations: {e}")
            ph("cli-optimize-error-caught", {"error": str(e)})
            return ""

        if response.status_code == 200:
            explanation: str = response.json()["explanation"]
            console.rule()
            return explanation
        try:
            error = response.json()["error"]
        except Exception:
            error = response.text
        logger.error(f"Error generating optimized candidates: {response.status_code} - {error}")
        ph("cli-optimize-error-response", {"response_status_code": response.status_code, "error": error})
        console.rule()
        return ""

    def generate_ranking(  # noqa: D417
        self,
        trace_id: str,
        diffs: list[str],
        optimization_ids: list[str],
        speedups: list[float],
        function_references: str | None = None,
    ) -> list[int] | None:
        """Optimize the given python code for performance by making a request to the Django endpoint.

        Parameters
        ----------
        - trace_id : unique uuid of function
        - diffs : list of unified diff strings of opt candidates
        - speedups : list of speedups of opt candidates
        - function_references : where the function is called in the codebase

        Returns
        -------
        - List[int]: Ranking of opt candidates in decreasing order

        """
        payload = {
            "trace_id": trace_id,
            "diffs": diffs,
            "speedups": speedups,
            "optimization_ids": optimization_ids,
            "python_version": platform.python_version(),
            "function_references": function_references,
        }
        logger.info("loading|Generating ranking")
        console.rule()
        try:
            response = self.make_ai_service_request("/rank", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.exception(f"Error generating ranking: {e}")
            ph("cli-optimize-error-caught", {"error": str(e)})
            return None

        if response.status_code == 200:
            ranking: list[int] = response.json()["ranking"]
            console.rule()
            return ranking
        try:
            error = response.json()["error"]
        except Exception:
            error = response.text
        logger.error(f"Error generating ranking: {response.status_code} - {error}")
        ph("cli-optimize-error-response", {"response_status_code": response.status_code, "error": error})
        console.rule()
        return None

    def log_results(  # noqa: D417
        self,
        function_trace_id: str,
        speedup_ratio: dict[str, float | None] | None,
        original_runtime: float | None,
        optimized_runtime: dict[str, float | None] | None,
        is_correct: dict[str, bool] | None,
        optimized_line_profiler_results: dict[str, str] | None,
        metadata: dict[str, Any] | None,
        optimizations_post: dict[str, str] | None = None,
    ) -> None:
        """Log features to the database.

        Parameters
        ----------
        - function_trace_id (str): The UUID.
        - speedup_ratio (Optional[Dict[str, float]]): The speedup.
        - original_runtime (Optional[Dict[str, float]]): The original runtime.
        - optimized_runtime (Optional[Dict[str, float]]): The optimized runtime.
        - is_correct (Optional[Dict[str, bool]]): Whether the optimized code is correct.
        - optimized_line_profiler_results: line_profiler results for every candidate mapped to their optimization_id
        - metadata: contains the best optimization id
        - optimizations_post - dict mapping opt id to code str after postprocessing

        """
        payload = {
            "trace_id": function_trace_id,
            "speedup_ratio": speedup_ratio,
            "original_runtime": original_runtime,
            "optimized_runtime": optimized_runtime,
            "is_correct": is_correct,
            "codeflash_version": codeflash_version,
            "optimized_line_profiler_results": optimized_line_profiler_results,
            "metadata": metadata,
            "optimizations_post": optimizations_post,
        }
        try:
            self.make_ai_service_request("/log_features", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.exception(f"Error logging features: {e}")

    def generate_regression_tests(  # noqa: D417
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
    ) -> tuple[str, str, str] | None:
        """Generate regression tests for the given function by making a request to the Django endpoint.

        Parameters
        ----------
        - source_code_being_tested (str): The source code of the function being tested.
        - function_to_optimize (FunctionToOptimize): The function to optimize.
        - helper_function_names (list[Source]): List of helper function names.
        - module_path (Path): The module path where the function is located.
        - test_module_path (Path): The module path for the test code.
        - test_framework (str): The test framework to use, e.g., "pytest", "jest".
        - test_timeout (int): The timeout for each test in seconds.
        - test_index (int): The index from 0-(n-1) if n tests are generated for a single trace_id
        - language (str): Programming language ("python", "javascript", "typescript")
        - language_version (str | None): Language version (e.g., "3.11.0" for Python, "ES2022" for JS)

        Returns
        -------
        - Dict[str, str] | None: The generated regression tests and instrumented tests, or None if an error occurred.

        """
        # Validate test framework based on language
        python_frameworks = ["pytest", "unittest"]
        javascript_frameworks = ["jest", "mocha", "vitest"]
        if language == "python":
            assert test_framework in python_frameworks, (
                f"Invalid test framework for Python, got {test_framework} but expected one of {python_frameworks}"
            )
        elif language in ("javascript", "typescript"):
            assert test_framework in javascript_frameworks, (
                f"Invalid test framework for JavaScript, got {test_framework} but expected one of {javascript_frameworks}"
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
        }

        # Add language-specific version fields
        # Always include python_version for backward compatibility with older backend
        payload["python_version"] = platform.python_version()
        if language == "python":
            pass  # python_version already set
        else:
            payload["language_version"] = language_version or "ES2022"

        # DEBUG: Print payload language field
        logger.debug(f"Sending testgen request with language='{payload['language']}', framework='{test_framework}'")
        try:
            response = self.make_ai_service_request("/testgen", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.exception(f"Error generating tests: {e}")
            ph("cli-testgen-error-caught", {"error": str(e)})
            return None

        # the timeout should be the same as the timeout for the AI service backend

        if response.status_code == 200:
            response_json = response.json()
            logger.debug(f"Generated tests for function {function_to_optimize.function_name}")
            return (
                response_json["generated_tests"],
                response_json["instrumented_behavior_tests"],
                response_json["instrumented_perf_tests"],
            )
        try:
            error = response.json()["error"]
            logger.error(f"Error generating tests: {response.status_code} - {error}")
            ph("cli-testgen-error-response", {"response_status_code": response.status_code, "error": error})
            return None  # noqa: TRY300
        except Exception:
            logger.error(f"Error generating tests: {response.status_code} - {response.text}")
            ph("cli-testgen-error-response", {"response_status_code": response.status_code, "error": response.text})
            return None

    def get_optimization_review(
        self,
        original_code: dict[Path, str],
        new_code: dict[Path, str],
        explanation: Explanation,
        existing_tests_source: str,
        generated_original_test_source: str,
        function_trace_id: str,
        coverage_message: str,
        replay_tests: str,
        concolic_tests: str,  # noqa: ARG002
        calling_fn_details: str,
        language: str = "python",
    ) -> OptimizationReviewResult:
        """Compute the optimization review of current Pull Request.

        Args:
        original_code: dict -> data structure mapping file paths to function definition for original code
        new_code: dict -> data structure mapping file paths to function definition for optimized code
        explanation: Explanation -> data structure containing runtime information
        existing_tests_source: str -> existing tests table
        generated_original_test_source: str -> annotated generated tests
        function_trace_id: str -> traceid of function
        coverage_message: str -> coverage information
        replay_tests: str -> replay test table
        root_dir: Path -> path of git directory
        concolic_tests: str -> concolic_tests (not used)
        calling_fn_details: str -> filenames and definitions of functions which call the function_to_optimize

        Returns:
        -------
        OptimizationReviewResult with review ('high', 'medium', 'low', or '') and explanation

        """
        original_code_str = "\n\n".join([original_code[p] for p in original_code])
        optimized_code_str = "\n\n".join([new_code[p] for p in new_code])

        logger.info("loading|Reviewing Optimization…")
        payload = {
            "original_code": original_code_str,
            "optimized_code": optimized_code_str,
            "explanation": explanation.raw_explanation_message,
            "existing_tests": existing_tests_source,
            "generated_tests": generated_original_test_source,
            "trace_id": function_trace_id,
            "coverage_message": coverage_message,
            "replay_tests": replay_tests,
            "speedup": f"{(100 * float(explanation.speedup)):.2f}%",
            "loop_count": explanation.winning_benchmarking_test_results.number_of_loops(),
            "benchmark_details": explanation.benchmark_details if explanation.benchmark_details else None,
            "optimized_runtime": humanize_runtime(explanation.best_runtime_ns),
            "original_runtime": humanize_runtime(explanation.original_runtime_ns),
            "codeflash_version": codeflash_version,
            "calling_fn_details": calling_fn_details,
            "language": language,
            "python_version": platform.python_version() if language == "python" else None,
            "call_sequence": self.get_next_sequence(),
        }
        console.rule()
        try:
            response = self.make_ai_service_request("/optimization_review", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.exception(f"Error generating optimization refinements: {e}")
            ph("cli-optimize-error-caught", {"error": str(e)})
            return OptimizationReviewResult(review="", explanation="")

        if response.status_code == 200:
            data = response.json()
            return OptimizationReviewResult(
                review=cast("str", data["review"]), explanation=cast("str", data.get("review_explanation", ""))
            )
        try:
            error = cast("str", response.json()["error"])
        except Exception:
            error = response.text
        logger.error(f"Error generating optimization review: {response.status_code} - {error}")
        ph("cli-optimize-error-response", {"response_status_code": response.status_code, "error": error})
        console.rule()
        return OptimizationReviewResult(review="", explanation="")

    def generate_workflow_steps(
        self,
        repo_files: dict[str, str],
        directory_structure: dict[str, Any],
        codeflash_config: dict[str, Any] | None = None,
    ) -> str | None:
        """Generate GitHub Actions workflow steps based on repository analysis.

        :param repo_files: Dictionary mapping file paths to their contents
        :param directory_structure: 2-level nested directory structure
        :param codeflash_config: Optional codeflash configuration
        :return: YAML string for workflow steps section, or None on error
        """
        payload = {
            "repo_files": repo_files,
            "directory_structure": directory_structure,
            "codeflash_config": codeflash_config,
        }

        logger.debug(
            f"[aiservice.py:generate_workflow_steps] Sending request to AI service with {len(repo_files)} files, "
            f"{len(directory_structure)} top-level directories"
        )

        try:
            response = self.make_ai_service_request("/workflow-gen", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            # AI service unavailable - this is expected, will fall back to static workflow
            logger.debug(
                f"[aiservice.py:generate_workflow_steps] Request exception (falling back to static workflow): {e}"
            )
            return None

        if response.status_code == 200:
            response_data = response.json()
            workflow_steps = cast("str", response_data.get("workflow_steps"))
            logger.debug(
                f"[aiservice.py:generate_workflow_steps] Successfully received workflow steps "
                f"({len(workflow_steps) if workflow_steps else 0} chars)"
            )
            return workflow_steps
        # AI service unavailable or endpoint not found - this is expected, will fall back to static workflow
        logger.debug(
            f"[aiservice.py:generate_workflow_steps] AI service returned status {response.status_code}, "
            f"falling back to static workflow generation"
        )
        try:
            error_response = response.json()
            error = cast("str", error_response.get("error", "Unknown error"))
            logger.debug(f"[aiservice.py:generate_workflow_steps] Error: {error}")
        except Exception:
            logger.debug("[aiservice.py:generate_workflow_steps] Could not parse error response")
        return None


class LocalAiServiceClient(AiServiceClient):
    """Client for interacting with the local AI service."""

    def get_aiservice_base_url(self) -> str:
        """Get the base URL for the local AI service."""
        return "http://localhost:8000"


def safe_get_repo_owner_and_name() -> tuple[str | None, str | None]:
    try:
        git_repo_owner, git_repo_name = get_repo_owner_and_name()
    except Exception as e:
        logger.warning(f"Could not determine repo owner and name: {e}")
        git_repo_owner, git_repo_name = None, None
    return git_repo_owner, git_repo_name
