from __future__ import annotations

import json
import logging
import os
import platform
from typing import Any

import requests
from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder

from codeflash.code_utils.env_utils import get_codeflash_api_key
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.models.ExperimentMetadata import ExperimentMetadata
from codeflash.telemetry.posthog import ph


@dataclass(frozen=True)
class OptimizedCandidate:
    source_code: str
    explanation: str
    optimization_id: str


class AiServiceClient:
    def __init__(self) -> None:
        self.base_url = self.get_aiservice_base_url()
        self.headers = {
            "Authorization": f"Bearer {get_codeflash_api_key()}",
            "Connection": "close",
        }

    def get_aiservice_base_url(self) -> str:
        if os.environ.get("CODEFLASH_AIS_SERVER", default="prod").lower() == "local":
            logging.info("Using local AI Service at http://localhost:8000")
            return "http://localhost:8000"
        return "https://app.codeflash.ai"

    def make_ai_service_request(
        self,
        endpoint: str,
        method: str = "POST",
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> requests.Response:
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

    def optimize_python_code(
        self,
        source_code: str,
        trace_id: str,
        num_candidates: int = 10,
        experiment_metadata: ExperimentMetadata | None = None,
    ) -> list[OptimizedCandidate]:
        """Optimize the given python code for performance by making a request to the Django endpoint.

        Parameters
        ----------
        - source_code (str): The python code to optimize.
        - num_variants (int): Number of optimization variants to generate. Default is 10.

        Returns
        -------
        - List[Optimization]: A list of Optimization objects.

        """
        payload = {
            "source_code": source_code,
            "num_variants": num_candidates,
            "trace_id": trace_id,
            "python_version": platform.python_version(),
            "experiment_metadata": experiment_metadata,
        }
        logging.info("Generating optimized candidates ...")
        try:
            response = self.make_ai_service_request(
                "/optimize",
                payload=payload,
                timeout=600,
            )
        except requests.exceptions.RequestException as e:
            logging.exception(f"Error generating optimized candidates: {e}")
            ph("cli-optimize-error-caught", {"error": str(e)})
            return []

        if response.status_code == 200:
            optimizations_json = response.json()["optimizations"]
            logging.info(f"Generated {len(optimizations_json)} candidates.")
            return [
                OptimizedCandidate(
                    source_code=opt["source_code"],
                    explanation=opt["explanation"],
                    optimization_id=opt["optimization_id"],
                )
                for opt in optimizations_json
            ]
        try:
            error = response.json()["error"]
        except Exception:
            error = response.text
        logging.error(f"Error generating optimized candidates: {response.status_code} - {error}")
        ph(
            "cli-optimize-error-response",
            {"response_status_code": response.status_code, "error": error},
        )
        return []

    def log_results(
        self,
        function_trace_id: str,
        speedup_ratio: dict[str, float] | None,
        original_runtime: float | None,
        optimized_runtime: dict[str, float] | None,
        is_correct: dict[str, bool] | None,
    ) -> None:
        """Log features to the database.

        Parameters
        ----------
        - function_trace_id (str): The UUID.
        - speedup_ratio (Optional[Dict[str, float]]): The speedup.
        - original_runtime (Optional[Dict[str, float]]): The original runtime.
        - optimized_runtime (Optional[Dict[str, float]]): The optimized runtime.
        - is_correct (Optional[Dict[str, bool]]): Whether the optimized code is correct.

        """
        payload = {
            "trace_id": function_trace_id,
            "speedup_ratio": speedup_ratio,
            "original_runtime": original_runtime,
            "optimized_runtime": optimized_runtime,
            "is_correct": is_correct,
        }
        try:
            self.make_ai_service_request("/log_features", payload=payload, timeout=5)
        except requests.exceptions.RequestException as e:
            logging.exception(f"Error logging features: {e}")

    def generate_regression_tests(
        self,
        source_code_being_tested: str,
        function_to_optimize: FunctionToOptimize,
        helper_function_names: list[str],
        module_path: str,
        test_module_path: str,
        test_framework: str,
        test_timeout: int,
        trace_id: str,
    ) -> tuple[str, str] | None:
        """Generate regression tests for the given function by making a request to the Django endpoint.

        Parameters
        ----------
        - source_code_being_tested (str): The source code of the function being tested.
        - function_to_optimize (FunctionToOptimize): The function to optimize.
        - helper_function_names (list[Source]): List of helper function names.
        - module_path (str): The module path where the function is located.
        - test_module_path (str): The module path for the test code.
        - test_framework (str): The test framework to use, e.g., "pytest".
        - test_timeout (int): The timeout for each test in seconds.

        Returns
        -------
        - Dict[str, str] | None: The generated regression tests and instrumented tests, or None if an error occurred.

        """
        assert test_framework in [
            "pytest",
            "unittest",
        ], f"Invalid test framework, got {test_framework} but expected 'pytest' or 'unittest'"
        payload = {
            "source_code_being_tested": source_code_being_tested,
            "function_to_optimize": function_to_optimize,
            "helper_function_names": helper_function_names,
            "module_path": module_path,
            "test_module_path": test_module_path,
            "test_framework": test_framework,
            "test_timeout": test_timeout,
            "trace_id": trace_id,
            "python_version": platform.python_version(),
        }
        try:
            response = self.make_ai_service_request("/testgen", payload=payload, timeout=600)
        except requests.exceptions.RequestException as e:
            logging.exception(f"Error generating tests: {e}")
            ph("cli-testgen-error-caught", {"error": str(e)})
            return None

        # the timeout should be the same as the timeout for the AI service backend

        if response.status_code == 200:
            response_json = response.json()
            logging.info(f"Generated tests for function {function_to_optimize.function_name}")
            return response_json["generated_tests"], response_json["instrumented_tests"]
        try:
            error = response.json()["error"]
            logging.error(f"Error generating tests: {response.status_code} - {error}")
            ph(
                "cli-testgen-error-response",
                {"response_status_code": response.status_code, "error": error},
            )
            return None
        except Exception:
            logging.error(f"Error generating tests: {response.status_code} - {response.text}")  # noqa: TRY400
            ph(
                "cli-testgen-error-response",
                {"response_status_code": response.status_code, "error": response.text},
            )
            return None


class LocalAiServiceClient(AiServiceClient):
    def get_aiservice_base_url(self) -> str:
        return "http://localhost:8000"
