import sentry_sdk


def init_sentry():
    sentry_sdk.init(
        dsn="https://4b9a1902f9361b48c04376df6483bc96@o4506833230561280.ingest.sentry.io/4506833262477312",
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
    )
