import logging
from typing import Optional, Dict, Any

import requests
from requests import Response

from codeflash.code_utils.env_utils import get_codeflash_api_key
from codeflash.github.PrComment import PrComment

CFAPI_BASE_URL = "https://app.codeflash.ai"
# CFAPI_BASE_URL = "http://localhost:3001"

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
    return response


def get_user_id() -> Optional[str]:
    """
    Retrieve the user's userid by making a request to the /cfapi/cli-get-user endpoint.
    :return: The userid or None if the request fails.
    """
    response = make_cfapi_request(endpoint="/cli-get-user", method="GET")
    if response.status_code == 200:
        return response.text
    else:
        logging.error(
            f"Failed to look up your userid; is your CF API key valid? ({response.reason})"
        )
        return None


def suggest_changes(
    owner: str,
    repo: str,
    pr_number: int,
    file_changes: dict[str, dict[str, str]],
    pr_comment: PrComment,
    generated_tests: str,
) -> Response:
    """
    Suggest changes to a pull request.
    Will make a review suggestion when possible;
    or create a new dependent pull request with the suggested changes.
    :param owner: The owner of the repository.
    :param repo: The name of the repository.
    :param pr_number: The number of the pull request.
    :param file_changes: A dictionary of file changes.
    :param pr_comment: The pull request comment object, containing the optimization explanation, best runtime, etc.
    :param generated_tests: The generated tests.
    :return: The response object.
    """
    payload = {
        "owner": owner,
        "repo": repo,
        "pullNumber": pr_number,
        "diffContents": file_changes,
        "prCommentFields": pr_comment.to_json(),
        "generatedTests": generated_tests,
    }
    response = make_cfapi_request(endpoint="/suggest-pr-changes", method="POST", payload=payload)
    return response


def create_pr(
    owner: str,
    repo: str,
    baseBranch: str,
    file_changes: dict[str, dict[str, str]],
    pr_comment: PrComment,
    generated_tests: str,
) -> Response:
    """
    Create a pull request, targeting the specified branch. (usually 'main')
    :param owner: The owner of the repository.
    :param repo: The name of the repository.
    :param targetBranch: The branch to target.
    :param file_changes: A dictionary of file changes.
    :param pr_comment: The pull request comment object, containing the optimization explanation, best runtime, etc.
    :param generated_tests: The generated tests.
    :return: The response object.
    """
    payload = {
        "owner": owner,
        "repo": repo,
        "baseBranch": baseBranch,
        "diffContents": file_changes,
        "prCommentFields": pr_comment.to_json(),
        "generatedTests": generated_tests,
    }
    response = make_cfapi_request(endpoint="/create-pr", method="POST", payload=payload)
    return response
