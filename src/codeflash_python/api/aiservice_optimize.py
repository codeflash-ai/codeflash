"""Mixin: optimization-related API endpoints."""

from __future__ import annotations

import logging
import platform
from typing import TYPE_CHECKING, Any

import requests

from codeflash.models.models import OptimizedCandidateSource
from codeflash_python.code_utils.git_utils import get_last_commit_author_if_pr_exists, get_repo_owner_and_name
from codeflash_python.code_utils.time_utils import humanize_runtime
from codeflash_python.telemetry.posthog_cf import ph
from codeflash_python.version import __version__ as codeflash_version

if TYPE_CHECKING:
    from codeflash.models.models import OptimizedCandidate
    from codeflash_python.api.types import (
        AIServiceAdaptiveOptimizeRequest,
        AIServiceCodeRepairRequest,
        AIServiceRefinerRequest,
    )
    from codeflash_python.models.experiment_metadata import ExperimentMetadata
else:
    _Base = object

logger = logging.getLogger("codeflash_python")


def safe_get_repo_owner_and_name() -> tuple[str | None, str | None]:
    try:
        git_repo_owner, git_repo_name = get_repo_owner_and_name()
    except Exception as e:
        logger.warning("Could not determine repo owner and name: %s", e)
        git_repo_owner, git_repo_name = None, None
    return git_repo_owner, git_repo_name


class AiServiceOptimizeMixin(_Base):  # type: ignore[name-defined]
    def optimize_code(
        self,
        source_code: str,
        dependency_code: str,
        trace_id: str,
        experiment_metadata: ExperimentMetadata | None = None,
        *,
        language: str = "python",
        language_version: str | None = None,
        module_system: str | None = None,
        is_async: bool = False,
        n_candidates: int = 5,
        is_numerical_code: bool | None = None,
    ) -> list[OptimizedCandidate]:
        """Optimize the given code for performance by making a request to the Django endpoint.

        Parameters
        ----------
        - source_code (str): The code to optimize.
        - dependency_code (str): The dependency code used as read-only context for the optimization
        - trace_id (str): Trace id of optimization run
        - experiment_metadata (Optional[ExperimentalMetadata, None]): Any available experiment metadata for this optimization
        - language (str): Programming language (e.g., "python")
        - language_version (str | None): Language version (e.g., "3.11.0")
        - module_system (str | None): Module system (None for Python)
        - is_async (bool): Whether the function being optimized is async
        - n_candidates (int): Number of candidates to generate

        Returns
        -------
        - List[OptimizationCandidate]: A list of Optimization Candidates.

        """
        logger.info("Generating optimized candidates\u2026")
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
            "is_numerical_code": is_numerical_code,
        }

        self.add_language_metadata(payload, language_version, module_system)

        # DEBUG: Print payload language field
        logger.debug(
            "Sending optimize request with language='%s' (type: %s)", payload["language"], type(payload["language"])
        )
        logger.debug("Sending optimize request: trace_id=%s, n_candidates=%s", trace_id, payload["n_candidates"])

        try:
            response = self.make_ai_service_request("/optimize", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.exception("Error generating optimized candidates: %s", e)
            ph("cli-optimize-error-caught", {"error": str(e)})

            return []

        if response.status_code == 200:
            optimizations_json = response.json()["optimizations"]
            return self.get_valid_candidates(optimizations_json, OptimizedCandidateSource.OPTIMIZE, language)
        self.log_error_response(response, "generating optimized candidates", "cli-optimize-error-response")
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

    def get_jit_rewritten_code(self, source_code: str, trace_id: str) -> list[OptimizedCandidate]:
        """Rewrite the given python code for performance via jit compilation by making a request to the Django endpoint.

        Parameters
        ----------
        - source_code (str): The python code to optimize.
        - trace_id (str): Trace id of optimization run

        Returns
        -------
        - List[OptimizationCandidate]: A list of Optimization Candidates.

        """
        git_repo_owner, git_repo_name = safe_get_repo_owner_and_name()

        payload = {
            "source_code": source_code,
            "trace_id": trace_id,
            "dependency_code": "",  # dummy value to please the api endpoint
            "python_version": platform.python_version(),  # backward compat
            "current_username": get_last_commit_author_if_pr_exists(None),
            "repo_owner": git_repo_owner,
            "repo_name": git_repo_name,
        }

        try:
            response = self.make_ai_service_request("/rewrite_jit", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.exception("Error generating jit rewritten candidate: %s", e)
            ph("cli-jit-rewrite-error-caught", {"error": str(e)})
            return []

        if response.status_code == 200:
            optimizations_json = response.json()["optimizations"]
            return self.get_valid_candidates(optimizations_json, OptimizedCandidateSource.JIT_REWRITE)
        self.log_error_response(response, "generating jit rewritten candidate", "cli-jit-rewrite-error-response")
        return []

    def optimize_python_code_line_profiler(
        self,
        source_code: str,
        dependency_code: str,
        trace_id: str,
        line_profiler_results: str,
        n_candidates: int,
        experiment_metadata: ExperimentMetadata | None = None,
        is_numerical_code: bool | None = None,
        language: str = "python",
        language_version: str | None = None,
    ) -> list[OptimizedCandidate]:
        """Optimize code for performance using line profiler results.

        Parameters
        ----------
        - source_code (str): The code to optimize.
        - dependency_code (str): The dependency code used as read-only context for the optimization
        - trace_id (str): Trace id of optimization run
        - line_profiler_results (str): Line profiler output to guide optimization
        - experiment_metadata (Optional[ExperimentalMetadata, None]): Any available experiment metadata for this optimization
        - n_candidates (int): Number of candidates to generate
        - language (str): Programming language (e.g., "python")
        - language_version (str): Language version (e.g., "3.12.0")

        Returns
        -------
        - List[OptimizationCandidate]: A list of Optimization Candidates.

        """
        if line_profiler_results == "":
            logger.info("No LineProfiler results were provided, Skipping optimization.")
            return []

        logger.info("Generating optimized candidates with line profiler\u2026")

        payload = {
            "source_code": source_code,
            "dependency_code": dependency_code,
            "n_candidates": n_candidates,
            "line_profiler_results": line_profiler_results,
            "trace_id": trace_id,
            "language": language,
            "experiment_metadata": experiment_metadata,
            "codeflash_version": codeflash_version,
            "call_sequence": self.get_next_sequence(),
            "is_numerical_code": is_numerical_code,
        }
        self.add_language_metadata(payload, language_version)

        try:
            response = self.make_ai_service_request("/optimize-line-profiler", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.exception("Error generating optimized candidates: %s", e)
            ph("cli-optimize-error-caught", {"error": str(e)})

            return []

        if response.status_code == 200:
            optimizations_json = response.json()["optimizations"]
            return self.get_valid_candidates(optimizations_json, OptimizedCandidateSource.OPTIMIZE_LP)
        self.log_error_response(response, "generating optimized candidates", "cli-optimize-error-response")
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
            logger.exception("Error generating adaptive optimized candidates: %s", e)
            ph("cli-optimize-error-caught", {"error": str(e)})
            return None

        if response.status_code == 200:
            fixed_optimization = response.json()

            valid_candidates = self.get_valid_candidates([fixed_optimization], OptimizedCandidateSource.ADAPTIVE)
            if not valid_candidates:
                logger.error("Adaptive optimization failed to generate a valid candidate.")
                return None

            return valid_candidates[0]

        self.log_error_response(response, "generating optimized candidates", "cli-optimize-error-response")
        return None

    def optimize_code_refinement(self, request: list[AIServiceRefinerRequest]) -> list[OptimizedCandidate]:
        """Refine optimization candidates for improved performance.

        Refines optimization candidates with optional multi-file context for
        better understanding of imports and dependencies.

        Args:
            request: A list of optimization candidate details for refinement

        Returns:
            List of refined optimization candidates

        """
        payload: list[dict[str, Any]] = []
        for opt in request:
            item: dict[str, Any] = {
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
                "call_sequence": self.get_next_sequence(),
                # Multi-language support
                "language": opt.language,
            }

            self.add_language_metadata(item, opt.language_version)

            # Add multi-file context if provided
            if opt.additional_context_files:
                item["additional_context_files"] = opt.additional_context_files

            payload.append(item)

        try:
            response = self.make_ai_service_request("/refinement", payload=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            logger.exception("Error generating optimization refinements: %s", e)
            ph("cli-optimize-error-caught", {"error": str(e)})
            return []

        if response.status_code == 200:
            refined_optimizations = response.json()["refinements"]

            return self.get_valid_candidates(refined_optimizations, OptimizedCandidateSource.REFINE)

        self.log_error_response(response, "generating optimized candidates", "cli-optimize-error-response")

        return []

    # Alias for backward compatibility
    optimize_python_code_refinement = optimize_code_refinement

    def code_repair(self, request: AIServiceCodeRepairRequest) -> OptimizedCandidate | None:
        """Repair the optimization candidate that is not matching the test result of the original code.

        Args:
        request: candidate details for repair

        Returns:
        -------
        - OptimizedCandidate: new fixed candidate.

        """
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
            logger.exception("Error generating optimization repair: %s", e)
            ph("cli-optimize-error-caught", {"error": str(e)})
            return None

        if response.status_code == 200:
            fixed_optimization = response.json()

            valid_candidates = self.get_valid_candidates(
                [fixed_optimization], OptimizedCandidateSource.REPAIR, request.language
            )
            if not valid_candidates:
                logger.error("Code repair failed to generate a valid candidate.")
                return None

            return valid_candidates[0]

        self.log_error_response(response, "generating optimized candidates", "cli-optimize-error-response")

        return None
