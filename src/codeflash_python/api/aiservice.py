from __future__ import annotations

import json
import logging
import os
from itertools import count
from typing import TYPE_CHECKING, Any

import requests
from pydantic.json import pydantic_encoder

from codeflash_python.api.aiservice_optimize import AiServiceOptimizeMixin
from codeflash_python.api.aiservice_results import AiServiceResultsMixin
from codeflash_python.api.aiservice_testgen import AiServiceTestgenMixin
from codeflash_python.code_utils.config_consts import PYTHON_LANGUAGE_VERSION
from codeflash_python.code_utils.env_utils import get_codeflash_api_key
from codeflash_python.models.models import CodeStringsMarkdown, OptimizedCandidate

if TYPE_CHECKING:
    from codeflash_python.models.models import OptimizedCandidateSource

logger = logging.getLogger("codeflash_python")


class AiServiceClient(AiServiceOptimizeMixin, AiServiceTestgenMixin, AiServiceResultsMixin):
    def __init__(self) -> None:
        self.base_url = self.get_aiservice_base_url()
        self.headers = {"Authorization": f"Bearer {get_codeflash_api_key()}", "Connection": "close"}
        self.llm_call_counter = count(1)
        self.is_local = self.base_url == "http://localhost:8000"
        self.timeout: float | None = 300 if self.is_local else 90

    def get_next_sequence(self) -> int:
        """Get the next LLM call sequence number."""
        return next(self.llm_call_counter)

    @staticmethod
    def add_language_metadata(
        payload: dict[str, Any],
        language_version: str | None = None,
        module_system: str | None = None,  # noqa: ARG004
    ) -> None:
        """Add language version metadata to an API payload."""
        if language_version is None:
            language_version = PYTHON_LANGUAGE_VERSION
        payload["language_version"] = language_version
        payload["python_version"] = language_version

    @staticmethod
    def log_error_response(response: requests.Response, action: str, ph_event: str) -> None:
        """Log and report an API error response."""
        from codeflash_python.telemetry.posthog_cf import ph

        try:
            error = response.json()["error"]
        except Exception:
            error = response.text
        logger.error("Error %s: %s - %s", action, response.status_code, error)
        ph(ph_event, {"response_status_code": response.status_code, "error": error})

    def get_aiservice_base_url(self) -> str:
        if os.environ.get("CODEFLASH_AIS_SERVER", default="prod").lower() == "local":
            logger.info("Using local AI Service at http://localhost:8000")

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
        url = f"{self.base_url}/ai{endpoint}"
        if method.upper() == "POST":
            json_payload = json.dumps(payload, indent=None, default=pydantic_encoder)
            headers = {**self.headers, "Content-Type": "application/json"}
            response = requests.post(url, data=json_payload, headers=headers, timeout=timeout)
        else:
            response = requests.get(url, headers=self.headers, timeout=timeout)
        # response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        return response

    def get_valid_candidates(
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


class LocalAiServiceClient(AiServiceClient):
    """Client for interacting with the local AI service."""

    def get_aiservice_base_url(self) -> str:
        """Get the base URL for the local AI service."""
        return "http://localhost:8000"
