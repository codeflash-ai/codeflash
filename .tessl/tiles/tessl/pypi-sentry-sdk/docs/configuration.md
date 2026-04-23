# Configuration

SDK initialization and configuration management including DSN setup, sampling rates, integrations, transport options, and environment-specific settings.

## Capabilities

### SDK Initialization

Initialize the Sentry SDK with configuration options including DSN, sampling rates, integrations, and various client settings.

```python { .api }
def init(
    dsn: Optional[str] = None,
    integrations: Optional[List[Integration]] = None,
    default_integrations: bool = True,
    auto_enabling_integrations: bool = True,
    disabled_integrations: Optional[List[Union[type, str]]] = None,
    environment: Optional[str] = None,
    release: Optional[str] = None,
    traces_sample_rate: Optional[float] = None,
    traces_sampler: Optional[Callable[[Dict[str, Any]], float]] = None,
    profiles_sample_rate: Optional[float] = None,
    profiles_sampler: Optional[Callable[[Dict[str, Any]], float]] = None,
    max_breadcrumbs: int = 100,
    attach_stacktrace: bool = False,
    send_default_pii: bool = False,
    in_app_include: Optional[List[str]] = None,
    in_app_exclude: Optional[List[str]] = None,
    before_send: Optional[Callable[[Event, Hint], Optional[Event]]] = None,
    before_send_transaction: Optional[Callable[[Event, Hint], Optional[Event]]] = None,
    transport: Optional[Transport] = None,
    http_proxy: Optional[str] = None,
    https_proxy: Optional[str] = None,
    shutdown_timeout: int = 2,
    debug: bool = False,
    **kwargs
) -> None:
    """
    Initialize the Sentry SDK.
    
    Parameters:
    - dsn: Data Source Name for your Sentry project
    - integrations: List of integrations to enable
    - default_integrations: Enable default integrations (logging, stdlib, etc.)
    - auto_enabling_integrations: Enable auto-detected integrations (Django, Flask, etc.)
    - disabled_integrations: List of integration types/names to disable
    - environment: Environment name (e.g., 'production', 'staging')
    - release: Release version identifier
    - traces_sample_rate: Percentage of transactions to capture (0.0 to 1.0)
    - traces_sampler: Function to determine transaction sampling
    - profiles_sample_rate: Percentage of transactions to profile (0.0 to 1.0)
    - profiles_sampler: Function to determine profiling sampling
    - max_breadcrumbs: Maximum number of breadcrumbs to store
    - attach_stacktrace: Attach stack traces to non-exception events
    - send_default_pii: Send personally identifiable information
    - in_app_include: List of modules to consider "in-app"
    - in_app_exclude: List of modules to exclude from "in-app"
    - before_send: Event processor function for events
    - before_send_transaction: Event processor function for transactions
    - transport: Custom transport implementation
    - http_proxy: HTTP proxy URL
    - https_proxy: HTTPS proxy URL
    - shutdown_timeout: Timeout for SDK shutdown in seconds
    - debug: Enable debug logging
    """
```

**Usage Example:**

```python
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration
from sentry_sdk.integrations.redis import RedisIntegration

sentry_sdk.init(
    dsn="https://your-dsn@sentry.io/project-id",
    environment="production",
    release="my-app@1.0.0",
    traces_sample_rate=0.1,  # 10% of transactions
    profiles_sample_rate=0.1,  # 10% of transactions
    integrations=[
        DjangoIntegration(
            transaction_style='url',
            middleware_spans=True,
            signals_spans=True,
        ),
        RedisIntegration(),
    ],
    max_breadcrumbs=50,
    attach_stacktrace=True,
    send_default_pii=False,
    before_send=lambda event, hint: event if should_send_event(event) else None,
)
```

### Initialization Status

Check whether the SDK has been properly initialized and is ready to capture events.

```python { .api }
def is_initialized() -> bool:
    """
    Check if Sentry SDK has been initialized.
    
    Returns:
    bool: True if SDK is initialized and client is active, False otherwise
    """
```

**Usage Example:**

```python
import sentry_sdk

if not sentry_sdk.is_initialized():
    sentry_sdk.init(dsn="https://your-dsn@sentry.io/project-id")

# Now safe to use other SDK functions
sentry_sdk.capture_message("SDK is ready!")
```

## Configuration Options

### DSN (Data Source Name)

The DSN is the connection string that tells the SDK where to send events. It includes the protocol, public key, server address, and project ID.

**Format:** `https://{PUBLIC_KEY}@{HOSTNAME}/{PROJECT_ID}`

### Sampling

Control what percentage of events and transactions are sent to Sentry to manage volume and costs.

- **traces_sample_rate**: Float between 0.0 and 1.0 for uniform sampling
- **traces_sampler**: Function for custom sampling logic
- **profiles_sample_rate**: Float for profiling sampling (requires traces_sample_rate > 0)
- **profiles_sampler**: Function for custom profiling sampling logic

### Event Processing

- **before_send**: Function to filter or modify events before sending
- **before_send_transaction**: Function to filter or modify transactions before sending
- **in_app_include/exclude**: Control which modules are considered part of your application

### Integration Control

- **default_integrations**: Enable/disable standard integrations (logging, stdlib, etc.)
- **auto_enabling_integrations**: Enable/disable automatic framework detection
- **integrations**: Explicit list of integrations to enable
- **disabled_integrations**: List of integrations to disable

### Privacy and Security

- **send_default_pii**: Whether to send personally identifiable information
- **attach_stacktrace**: Add stack traces to non-exception events
- **max_breadcrumbs**: Limit breadcrumb storage (default: 100)

### Performance and Reliability

- **shutdown_timeout**: Time to wait for pending events during shutdown
- **transport**: Custom transport implementation for event delivery
- **http_proxy/https_proxy**: Proxy configuration for network requests