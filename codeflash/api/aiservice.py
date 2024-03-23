import json
import logging
import os
from typing import Any, Dict, List, Tuple, Optional

import requests
from pydantic.json import pydantic_encoder

from codeflash.code_utils.env_utils import get_codeflash_api_key
from codeflash.discovery.functions_to_optimize import FunctionToOptimize
from codeflash.telemetry.posthog import ph

if os.environ.get("AIS_SERVER", default="prod").lower() == "local":
    AI_SERVICE_BASE_URL = "http://localhost:8000/"
    logging.info(f"Using local AI Service at {AI_SERVICE_BASE_URL}.")
else:
    AI_SERVICE_BASE_URL = "https://app.codeflash.ai"


def make_ai_service_request(
    endpoint: str,
    method: str = "POST",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = None,
) -> requests.Response:
    """
    Make an API request to the given endpoint on the AI service.
    :param endpoint: The endpoint to call, e.g., "/optimize".
    :param method: The HTTP method to use ('GET' or 'POST').
    :param payload: Optional JSON payload to include in the POST request body.
    :param timeout: The timeout for the request.
    :return: The response object from the API.
    """
    url = f"{AI_SERVICE_BASE_URL}/ai{endpoint}"
    ai_service_headers = {"Authorization": f"Bearer {get_codeflash_api_key()}"}
    if method.upper() == "POST":
        json_payload = json.dumps(payload, indent=None, default=pydantic_encoder)
        ai_service_headers["Content-Type"] = "application/json"
        response = requests.post(
            url, data=json_payload, headers=ai_service_headers, timeout=timeout
        )
    else:
        response = requests.get(url, headers=ai_service_headers, timeout=timeout)
    # response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    return response


def optimize_python_code(
    source_code: str, trace_id: str, num_variants: int = 10
) -> List[Tuple[Optional[str], Optional[str], Optional[str]]]:
    """
    Optimize the given python code for performance by making a request to the Django endpoint.

    Parameters:
    - source_code (str): The python code to optimize.
    - num_variants (int): Number of optimization variants to generate. Default is 10.

    Returns:
    - List[Tuple[str, str]]: A list of tuples where the first element is the optimized code and the second is the explanation.
    """
    payload = {"source_code": source_code, "num_variants": num_variants, "trace_id": trace_id}
    logging.info(f"Generating optimized candidates ...")
    try:
        response = make_ai_service_request("/optimize", payload=payload, timeout=600)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error generating optimized candidates: {e}")
        ph("cli-optimize-error-caught", {"error": str(e)})
        return [(None, None, None)]

    if response.status_code == 200:
        optimizations = response.json()["optimizations"]
        logging.info(f"Generated {len(optimizations)} candidates.")
        return [
            (opt["source_code"], opt["explanation"], opt["optimization_id"])
            for opt in optimizations
        ]
    else:
        try:
            error = response.json()["error"]
        except Exception as e:
            error = response.text
        logging.error(f"Error generating optimized candidates: {response.status_code} - {error}")
        ph(
            "cli-optimize-error-response",
            {"response_status_code": response.status_code, "error": error},
        )
        return [(None, None, None)]


def log_results(
    function_trace_id: str,
    speedup_ratio: Optional[Dict[str, float]],
    original_runtime: Optional[float],
    optimized_runtime: Optional[Dict[str, float]],
    is_correct: Optional[Dict[str, bool]],
):
    """
    Log features to the database.

    Parameters:
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
        make_ai_service_request("/log_features", payload=payload, timeout=5)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error logging features: {e}")


def generate_regression_tests(
    source_code_being_tested: str,
    function_to_optimize: FunctionToOptimize,
    dependent_function_names: list[str],
    module_path: str,
    test_module_path: str,
    test_framework: str,
    test_timeout: int,
    trace_id: str,
) -> Optional[Tuple[str, str]]:
    """
    Generate regression tests for the given function by making a request to the Django endpoint.

    Parameters:
    - source_code_being_tested (str): The source code of the function being tested.
    - function_to_optimize (FunctionToOptimize): The function to optimize.
    - dependent_function_names (list[Source]): List of dependent function names.
    - module_path (str): The module path where the function is located.
    - test_module_path (str): The module path for the test code.
    - test_framework (str): The test framework to use, e.g., "pytest".
    - test_timeout (int): The timeout for each test in seconds.

    Returns:
    - Dict[str, str] | None: The generated regression tests and instrumented tests, or None if an error occurred.
    """
    assert test_framework in [
        "pytest",
        "unittest",
    ], f"Invalid test framework, got {test_framework} but expected 'pytest' or 'unittest'"
    payload = {
        "source_code_being_tested": source_code_being_tested,
        "function_to_optimize": function_to_optimize,
        "dependent_function_names": dependent_function_names,
        "module_path": module_path,
        "test_module_path": test_module_path,
        "test_framework": test_framework,
        "test_timeout": test_timeout,
        "trace_id": trace_id,
    }
    try:
        response = make_ai_service_request("/testgen", payload=payload, timeout=600)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error generating tests: {e}")
        ph("cli-testgen-error-caught", {"error": str(e)})
        return None

    # the timeout should be the same as the timeout for the AI service backend

    if response.status_code == 200:
        response_json = response.json()
        logging.info(f"Generated tests for function {function_to_optimize.function_name}")
        return response_json["generated_tests"], response_json["instrumented_tests"]
    else:
        try:
            error = response.json()["error"]
            logging.error(f"Error generating tests: {response.status_code} - {error}")
            ph(
                "cli-testgen-error-response",
                {"response_status_code": response.status_code, "error": error},
            )
            return None
        except Exception as e:
            logging.error(f"Error generating tests: {response.status_code} - {response.text}")
            ph(
                "cli-testgen-error-response",
                {"response_status_code": response.status_code, "error": response.text},
            )
            return None
