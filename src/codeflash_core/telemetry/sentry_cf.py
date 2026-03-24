from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def init_sentry(dsn: str, enabled: bool = True) -> None:
    """Initialize Sentry error tracking."""
    if not enabled or not dsn:
        return

    try:
        import sentry_sdk

        sentry_sdk.init(dsn=dsn, traces_sample_rate=0.0)
    except Exception:
        logger.debug("Failed to initialize Sentry", exc_info=True)
