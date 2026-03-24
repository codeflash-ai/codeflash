from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from codeflash_core.config import AIConfig
    from codeflash_core.models import Candidate, CodeContext

logger = logging.getLogger(__name__)


class AIClient:
    """Client for the Codeflash AI optimization service."""

    def __init__(self, config: AIConfig) -> None:
        self.base_url = config.base_url.rstrip("/")
        self.api_key = config.api_key
        self.timeout = config.timeout
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"

    def get_candidates(self, context: CodeContext) -> list[Candidate]:
        """Request optimization candidates from the AI service."""
        from codeflash_core.models import Candidate

        payload = {
            "function_name": context.target_function.qualified_name,
            "source_code": context.target_code,
            "helper_functions": [
                {"name": h.qualified_name, "source_code": h.source_code} for h in context.helper_functions
            ],
            "read_only_context": context.read_only_context,
            "imports": context.imports,
        }

        try:
            resp = self.session.post(f"{self.base_url}/ai/optimize", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            logger.exception("AI service request failed")
            return []

        candidates = []
        for item in data.get("candidates", []):
            candidates.append(Candidate(code=item["code"], explanation=item.get("explanation", "")))
        return candidates

    def close(self) -> None:
        self.session.close()
