import logging
import os
from typing import Any, Dict, List, Tuple, Optional

import requests
from pydantic import RootModel

from codeflash.analytics.posthog import ph
from codeflash.code_utils.env_utils import get_codeflash_api_key
from codeflash.discovery.functions_to_optimize import FunctionToOptimize

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

    Parameters:
    - endpoint (str): The endpoint to call, e.g., "/optimize".
    - method (str): The HTTP method to use, e.g., "POST".
    - data (Dict[str, Any]): The data to send in the request.

    Returns:
    - requests.Response: The response from the API.
    """
    url = f"{AI_SERVICE_BASE_URL}/ai{endpoint}"
    ai_service_headers = {"Authorization": f"Bearer {get_codeflash_api_key()}"}
    if method.upper() == "POST":
        response = requests.post(url, json=payload, headers=ai_service_headers, timeout=timeout)
    else:
        response = requests.get(url, headers=ai_service_headers, timeout=timeout)
    # response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    return response


def optimize_python_code(
    source_code: str, num_variants: int = 10
) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Optimize the given python code for performance by making a request to the Django endpoint.

    Parameters:
    - source_code (str): The python code to optimize.
    - num_variants (int): Number of optimization variants to generate. Default is 10.

    Returns:
    - List[Tuple[str, str]]: A list of tuples where the first element is the optimized code and the second is the explanation.
    """
    data = {"source_code": source_code, "num_variants": num_variants}
    logging.info(f"Generating optimized candidates ...")
    try:
        response = make_ai_service_request("/optimize", payload=data, timeout=600)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error generating optimized candidates: {e}")
        ph("cli-optimize-error-caught", {"error": str(e)})
        return [(None, None)]

    if response.status_code == 200:
        optimizations = response.json()
        return [(opt["source_code"], opt["explanation"]) for opt in optimizations]
    else:
        logging.error(
            f"Error generating optimized candidates: {response.status_code} {response.text}"
        )
        ph(
            "cli-optimize-error-response",
            {"response_status_code": response.status_code, "response_text": response.text},
        )
        return [(None, None)]


def generate_regression_tests(
    source_code_being_tested: str,
    function_to_optimize: FunctionToOptimize,
    dependent_function_names: list[str],
    module_path: str,
    test_module_path: str,
    test_framework: str,
    test_timeout: int,
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
    data = {
        "source_code_being_tested": source_code_being_tested,
        "function_to_optimize": RootModel[FunctionToOptimize](function_to_optimize).model_dump(
            mode="json"
        ),
        "dependent_function_names": dependent_function_names,
        "module_path": module_path,
        "test_module_path": test_module_path,
        "test_framework": test_framework,
        "test_timeout": test_timeout,
    }
    try:
        response = make_ai_service_request("/testgen", payload=data, timeout=600)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error generating tests: {e}")
        ph("cli-testgen-error-caught", {"error": str(e)})
        return None

    # the timeout should be the same as the timeout for the AI service backend

    if response.status_code == 200:
        response_json = response.json()
        return response_json["generated_tests"], response_json["instrumented_tests"]
    else:
        logging.error(f"Error generating tests: {response.status_code} {response.text}")
        ph(
            "cli-testgen-error-response",
            {"response_status_code": response.status_code, "response_text": response.text},
        )
        return None
