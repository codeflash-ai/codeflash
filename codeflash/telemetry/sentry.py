import logging

import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration


def init_sentry(enabled: bool = False, exclude_errors: bool = False) -> None:  # noqa: FBT001, FBT002
    if enabled:
        sentry_logging = LoggingIntegration(
            level=logging.INFO,  # Capture info and above as breadcrumbs
            event_level=logging.CRITICAL  # Send only fatal errors as events if exclude_errors is True
            if exclude_errors
            else logging.ERROR,  # Otherwise, error logs will create sentry events
        )

        def before_breadcrumb(crumb, hint):
            # Filter out verbose debug logs to prevent payload size issues
            if crumb.get("category") == "logging" and crumb.get("level") == "debug":
                # Skip debug logs that contain repetitive patterns
                message = crumb.get("message", "")
                if any(
                    pattern in message
                    for pattern in [
                        "Ignoring test case",
                        "executing test run",
                        "Functions called in optimized entrypoint",
                    ]
                ):
                    return None
            return crumb

        sentry_sdk.init(
            dsn="https://4b9a1902f9361b48c04376df6483bc96@o4506833230561280.ingest.sentry.io/4506833262477312",
            integrations=[sentry_logging],
            # Set traces_sample_rate to 1.0 to capture 100%
            # of transactions for performance monitoring.
            traces_sample_rate=1.0,
            # Set profiles_sample_rate to 1.0 to profile 100%
            # of sampled transactions.
            # We recommend adjusting this value in production.
            profiles_sample_rate=1.0,
            ignore_errors=[KeyboardInterrupt],
            # Limit breadcrumbs to prevent payload size issues
            max_breadcrumbs=50,
            # Limit request body size to prevent 413 errors
            max_request_body_size="medium",  # "always", "never", "small", "medium"
            before_breadcrumb=before_breadcrumb,
        )
