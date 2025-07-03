from __future__ import annotations

import logging
from typing import Any

from posthog import Posthog

from codeflash.api.cfapi import get_user_id
from codeflash.cli_cmds.console import logger
from codeflash.version import __version__

_posthog = None


def initialize_posthog(enabled: bool = True) -> None:
    """Enable or disable PostHog.

    :param enabled: Whether to enable PostHog.
    """
    if not enabled:
        return

    global _posthog, _user_id  # noqa: PLW0603
    if _posthog is not None:
        return  # Fast re-init path

    _posthog = Posthog(project_api_key="phc_aUO790jHd7z1SXwsYCz8dRApxueplZlZWeDSpKc5hol", host="https://us.posthog.com")  # type: ignore[no-untyped-call]
    _posthog.log.setLevel(logging.CRITICAL)  # Suppress PostHog logging

    # Pre-fetch user_id for this session and cache it
    _user_id = get_user_id()

    ph("cli-telemetry-enabled")


def ph(event: str, properties: dict[str, Any] | None = None) -> None:
    """Log an event to PostHog.

    :param event: The name of the event.
    :param properties: A dictionary of properties to attach to the event.
    """
    if _posthog is None:
        return

    # Build the property dict only once per call
    props = {} if properties is None else dict(properties)
    props["cli_version"] = __version__

    # Use cached user_id if available, else fetch and memoize once per process run
    global _user_id  # noqa: PLW0603
    if _user_id is None:
        _user_id = get_user_id()
    user_id = _user_id

    if user_id:
        _posthog.capture(distinct_id=user_id, event=event, properties=props)  # type: ignore[no-untyped-call]
    else:
        logger.debug("Failed to log event to PostHog: User ID could not be retrieved.")


_user_id = None
