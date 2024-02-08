import json
import logging
import os
from functools import lru_cache
from typing import Optional, Dict, Any

import requests
from pydantic.json import pydantic_encoder
from requests import Response

from codeflash.code_utils.env_utils import get_codeflash_api_key
from codeflash.github.PrComment import PrComment

if os.environ.get("CFAPI_SERVER", default="prod").lower() == "local":
    CFAPI_BASE_URL = "http://localhost:3001"
    logging.info(f"Using local CF API at {CFAPI_BASE_URL}.")
else:
    CFAPI_BASE_URL = "https://app.codeflash.ai"


def make_cfapi_request(
    endpoint: str, method: str, payload: Optional[Dict[str, Any]] = None
) -> requests.Response:
    """
    Make an HTTP request using the specified method, URL, headers, and JSON payload.
    :param endpoint: The endpoint URL to send the request to.
    :param method: The HTTP method to use ('GET', 'POST', etc.).
    :param payload: Optional JSON payload to include in the POST request body.
    :return: The response object from the API.
    """
    url = f"{CFAPI_BASE_URL}/cfapi{endpoint}"
    cfapi_headers = {"Authorization": f"Bearer {get_codeflash_api_key()}"}
    if method.upper() == "POST":
        json_payload = json.dumps(payload, indent=None, default=pydantic_encoder)
        cfapi_headers["Content-Type"] = "application/json"
        response = requests.post(url, data=json_payload, headers=cfapi_headers)
    else:
        response = requests.get(url, headers=cfapi_headers)
    return response


@lru_cache(maxsize=1)
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
    base_branch: str,
    file_changes: dict[str, dict[str, str]],
    pr_comment: PrComment,
    generated_tests: str,
) -> Response:
    """
    Create a pull request, targeting the specified branch. (usually 'main')
    :param owner: The owner of the repository.
    :param repo: The name of the repository.
    :param base_branch: The base branch to target.
    :param file_changes: A dictionary of file changes.
    :param pr_comment: The pull request comment object, containing the optimization explanation, best runtime, etc.
    :param generated_tests: The generated tests.
    :return: The response object.
    """
    payload = {
        "owner": owner,
        "repo": repo,
        "baseBranch": base_branch,
        "diffContents": file_changes,
        "prCommentFields": pr_comment.to_json(),
        "generatedTests": generated_tests,
    }
    response = make_cfapi_request(endpoint="/create-pr", method="POST", payload=payload)
    return response


def check_github_app_installed_on_repo(owner: str, repo: str) -> Response:
    """
    Check if the Codeflash GitHub App is installed on the specified repository.
    :param owner: The owner of the repository.
    :param repo: The name of the repository.
    :return: The response object.
    """
    response = make_cfapi_request(
        endpoint=f"/is-github-app-installed?repo={repo}&owner={owner}",
        method="GET",
    )
    return response
