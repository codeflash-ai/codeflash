from typing import Optional, Dict, Any

import requests

from codeflash.code_utils.env_utils import get_codeflash_api_key
from codeflash.github.PrComment import PrComment

CFAPI_BASE_URL = "https://app.codeflash.ai"

CFAPI_HEADERS = {"Authorization": f"Bearer {get_codeflash_api_key()}"}


def make_cfapi_request(
    endpoint: str, method: str, payload: Optional[Dict[str, Any]] = None
) -> requests.Response:
    """
    Make an HTTP request using the specified method, URL, headers, and JSON payload.
    :param endpoint: The URL to send the request to.
    :param method: The HTTP method to use ('GET', 'POST', etc.).
    :param payload: Optional JSON payload to include in the request body.
    :return: The response object.
    """
    url = f"{CFAPI_BASE_URL}/cfapi{endpoint}"
    if method.upper() == "POST":
        response = requests.post(url, json=payload, headers=CFAPI_HEADERS)
    else:
        response = requests.get(url, headers=CFAPI_HEADERS)
    response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    return response


def suggest_changes(
    owner: str,
    repo: str,
    pr_number: int,
    file_changes: dict[str, dict[str, str]],
    pr_comment: PrComment,
    generated_tests: str,
) -> bool:
    payload = {
        "owner": owner,
        "repo": repo,
        "pullNumber": pr_number,
        "diffContents": file_changes,
        "prCommentFields": pr_comment.to_json(),
        "generatedTests": generated_tests,
    }
    response = make_cfapi_request(endpoint="/suggest-pr-changes", method="POST", payload=payload)
    return response.ok
