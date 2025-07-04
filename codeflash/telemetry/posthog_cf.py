from __future__ import annotations

import logging
from typing import Any

from posthog import Posthog

from codeflash.api.cfapi import get_user_id
from codeflash.cli_cmds.console import logger
from codeflash.version import __version__

_posthog = None


def initialize_posthog(enabled: bool = True) -> None:  # noqa: FBT001, FBT002
    """Enable or disable PostHog.

    :param enabled: Whether to enable PostHog.
    """
    if not enabled:
        return

    global _posthog  # noqa: PLW0603
    _posthog = Posthog(project_api_key="phc_aUO790jHd7z1SXwsYCz8dRApxueplZlZWeDSpKc5hol", host="https://us.posthog.com")
    _posthog.log.setLevel(logging.CRITICAL)  # Suppress PostHog logging
    ph("cli-telemetry-enabled")


def ph(event: str, properties: dict[str, Any] | None = None) -> None:
    """Log an event to PostHog.

    :param event: The name of the event.
    :param properties: A dictionary of properties to attach to the event.
    """
    if _posthog is None:
        return

    properties = properties or {}
    properties.update({"cli_version": __version__})

    user_id = get_user_id()

    if user_id:
        _posthog.capture(distinct_id=user_id, event=event, properties=properties)
    else:
        logger.debug("Failed to log event to PostHog: User ID could not be retrieved.")
