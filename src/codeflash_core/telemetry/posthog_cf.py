from __future__ import annotations

import contextlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


class PostHogClient:
    """Wrapper around the PostHog analytics client."""

    instance: PostHogClient | None = None

    def __init__(self, api_key: str, enabled: bool = True) -> None:
        self.enabled = enabled and bool(api_key)
        self.ph: Any = None

        if self.enabled:
            try:
                from posthog import Posthog

                self.ph = Posthog(api_key, host="https://us.i.posthog.com")
            except Exception:
                logger.debug("Failed to initialize PostHog", exc_info=True)
                self.enabled = False

    @classmethod
    def initialize(cls, api_key: str, enabled: bool = True) -> PostHogClient:
        cls.instance = cls(api_key, enabled=enabled)
        return cls.instance

    def capture(self, distinct_id: str, event: str, properties: dict[str, Any] | None = None) -> None:
        if not self.enabled or self.ph is None:
            return
        try:
            self.ph.capture(distinct_id, event, properties=properties or {})
        except Exception:
            logger.debug("PostHog capture failed", exc_info=True)

    def shutdown(self) -> None:
        if self.ph is not None:
            with contextlib.suppress(Exception):
                self.ph.shutdown()
