from typing import Any, Dict, List, Tuple, Optional, Union

import requests

from codeflash.code_utils.env_utils import get_codeflash_api_key

AI_SERVICE_BASE_URL = "https://app.codeflash.ai"

AI_SERVICE_HEADERS = {"Authorization": f"Bearer {get_codeflash_api_key()}"}


def make_ai_service_request(
    endpoint: str, method: str = "POST", payload: Optional[Dict[str, Any]] = None
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
    if method.upper() == "POST":
        response = requests.post(url, json=payload, headers=AI_SERVICE_HEADERS)
    else:
        response = requests.get(url, headers=AI_SERVICE_HEADERS)
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
    response = make_ai_service_request("/optimize", payload=data)

    if response.status_code == 200:
        optimizations = response.json()
        return [(opt["source_code"], opt["explanation"]) for opt in optimizations]
    else:
        print(f"Error: {response.status_code} {response.text}")
        return [(None, None)]


def generate_regression_tests(
    source_code_being_tested: str, function_name: str, test_framework: str
) -> Union[str, None]:
    """
    Generate regression tests for the given function by making a request to the Django endpoint.

    Parameters:
    - source_code_being_tested (str): The source code of the function being tested.
    - function_name (str): The name of the function being tested.
    - test_framework (str): The test framework to use, e.g., "pytest".

    Returns:
    - str | None: The generated regression tests.
    """
    assert test_framework in [
        "pytest",
        "unittest",
    ], f"Invalid test framework, got {test_framework} but expected 'pytest' or 'unittest' "
    data = {
        "source_code_being_tested": source_code_being_tested,
        "function_name": function_name,
        "test_framework": test_framework,
    }
    response = make_ai_service_request("/testgen", payload=data)

    if response.status_code == 200:
        return response.json()["code"]
    else:
        print(f"Error: {response.status_code} {response.text}")
        return None
