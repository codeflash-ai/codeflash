# Event Capture

Manual and automatic capture of exceptions, messages, and custom events with context preservation, filtering capabilities, and detailed metadata attachment.

## Capabilities

### Exception Capture

Capture exceptions with full stack traces, local variables, and contextual information for debugging and error tracking.

```python { .api }
def capture_exception(
    error: Optional[BaseException] = None,
    scope: Optional[Scope] = None,
    **scope_kwargs
) -> Optional[str]:
    """
    Capture an exception and send it to Sentry.
    
    Parameters:
    - error: Exception instance to capture (if None, captures current exception)
    - scope: Scope to use for this event (if None, uses current scope)
    - **scope_kwargs: Additional scope data (tags, extra, user, level, fingerprint)
    
    Returns:
    str: Event ID if event was sent, None if filtered or dropped
    """
```

**Usage Examples:**

```python
import sentry_sdk

# Capture current exception in except block
try:
    risky_operation()
except Exception:
    # Automatically captures current exception with full context
    event_id = sentry_sdk.capture_exception()
    print(f"Error reported with ID: {event_id}")

# Capture specific exception with additional context
try:
    process_user_data(user_id=123)
except ValueError as e:
    event_id = sentry_sdk.capture_exception(
        e,
        tags={"component": "data_processor"},
        extra={"user_id": 123, "operation": "data_processing"},
        level="error"
    )

# Capture exception outside of except block
def process_file(filename):
    try:
        return parse_file(filename)
    except Exception as e:
        # Capture and re-raise
        sentry_sdk.capture_exception(e)
        raise
```

### Message Capture

Capture log messages and custom events with structured data and severity levels for application monitoring.

```python { .api }
def capture_message(
    message: str,
    level: Optional[Union[str, LogLevel]] = None,
    scope: Optional[Scope] = None,
    **scope_kwargs
) -> Optional[str]:
    """
    Capture a message and send it to Sentry.
    
    Parameters:
    - message: Message text to capture
    - level: Log level ('debug', 'info', 'warning', 'error', 'fatal')
    - scope: Scope to use for this event (if None, uses current scope)
    - **scope_kwargs: Additional scope data (tags, extra, user, fingerprint)
    
    Returns:
    str: Event ID if event was sent, None if filtered or dropped
    """
```

**Usage Examples:**

```python
import sentry_sdk

# Simple message capture
sentry_sdk.capture_message("User logged in successfully")

# Message with level and context
sentry_sdk.capture_message(
    "Payment processing failed",
    level="error",
    tags={"payment_method": "credit_card"},
    extra={
        "user_id": "user_123",
        "amount": 99.99,
        "currency": "USD",
        "error_code": "CARD_DECLINED"
    }
)

# Critical system message
sentry_sdk.capture_message(
    "Database connection pool exhausted",
    level="fatal",
    tags={"component": "database", "severity": "critical"},
    fingerprint=["database", "connection-pool"]
)
```

### Custom Event Capture

Capture arbitrary events with full control over event structure, metadata, and processing for advanced use cases.

```python { .api }
def capture_event(
    event: Dict[str, Any],
    hint: Optional[Dict[str, Any]] = None,
    scope: Optional[Scope] = None,
    **scope_kwargs
) -> Optional[str]:
    """
    Capture a custom event and send it to Sentry.
    
    Parameters:
    - event: Event dictionary with message, level, and other properties
    - hint: Processing hints for event processors
    - scope: Scope to use for this event (if None, uses current scope)
    - **scope_kwargs: Additional scope data (tags, extra, user, level, fingerprint)
    
    Returns:
    str: Event ID if event was sent, None if filtered or dropped
    """
```

**Usage Examples:**

```python
import sentry_sdk

# Custom structured event
event = {
    "message": "Custom business logic event",
    "level": "info",
    "extra": {
        "business_process": "order_fulfillment",
        "step": "inventory_check",
        "result": "success",
        "processing_time_ms": 145
    },
    "tags": {
        "service": "inventory",
        "region": "us-west-2"
    }
}

event_id = sentry_sdk.capture_event(event)

# Event with processing hints
custom_event = {
    "message": "Security audit event",
    "level": "warning",
    "logger": "security.audit",
    "extra": {
        "action": "permission_denied",
        "resource": "/admin/users",
        "ip_address": "192.168.1.100"
    }
}

hint = {
    "should_capture": True,
    "security_event": True
}

sentry_sdk.capture_event(custom_event, hint=hint)
```

## Event Properties

### Event Structure

Events sent to Sentry follow a standard structure with these key fields:

- **message**: Human-readable description
- **level**: Severity level (trace, debug, info, warning, error, fatal)
- **logger**: Logger name or component identifier  
- **platform**: Platform identifier (python)
- **timestamp**: Event occurrence time
- **extra**: Additional metadata as key-value pairs
- **tags**: Indexed metadata for filtering and searching
- **user**: User context information
- **request**: HTTP request data (when available)
- **contexts**: Structured context objects (os, runtime, etc.)
- **breadcrumbs**: Trail of events leading to the error
- **fingerprint**: Custom grouping rules for the event

### Scope Integration

All capture functions inherit context from the current scope and support scope keyword arguments:

- **tags**: Key-value pairs for filtering (`tags={"component": "auth"}`)
- **extra**: Additional debug information (`extra={"user_id": 123}`)
- **user**: User identification (`user={"id": "123", "email": "user@example.com"}`)
- **level**: Override event severity (`level="error"`)
- **fingerprint**: Custom grouping (`fingerprint=["auth", "login-failed"]`)

### Automatic Context

The SDK automatically includes:

- **Stack traces**: For exceptions and when `attach_stacktrace=True`
- **Local variables**: When debug mode is enabled
- **Request data**: From web framework integrations
- **System context**: OS, runtime, and hardware information
- **Breadcrumbs**: Automatic trail from integrations and manual additions

## Error Handling

All capture functions are designed to never raise exceptions themselves, ensuring application stability:

```python
# These calls are safe even if Sentry is misconfigured
sentry_sdk.capture_exception()  # Returns None if disabled/failed
sentry_sdk.capture_message("test")  # Returns None if disabled/failed
sentry_sdk.capture_event({})  # Returns None if disabled/failed
```

Events may be filtered or dropped due to:
- SDK not initialized
- Rate limiting  
- Sampling configuration
- `before_send` filters
- Network connectivity issues

The return value indicates whether the event was successfully queued for sending (returns event ID) or filtered/dropped (returns None).