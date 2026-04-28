# Event Tracking

Core event capture functionality for tracking user actions, system events, and custom analytics data. PostHog's event tracking system supports rich properties, automatic context enrichment, and reliable delivery with batching and retry mechanisms.

## Capabilities

### Event Capture

Capture any user action or system event with optional properties, timestamps, and context information.

```python { .api }
def capture(event: str, **kwargs: OptionalCaptureArgs) -> Optional[str]:
    """
    Capture anything a user does within your system.

    Parameters:
    - event: str - The event name to specify the event
    - distinct_id: Optional[ID_TYPES] - Unique identifier for the user
    - properties: Optional[Dict[str, Any]] - Dict of event properties
    - timestamp: Optional[Union[datetime, str]] - When the event occurred
    - uuid: Optional[str] - Unique identifier for this specific event
    - groups: Optional[Dict[str, str]] - Dict of group types and IDs
    - send_feature_flags: Optional[Union[bool, SendFeatureFlagsOptions]] - Whether to include active feature flags
    - disable_geoip: Optional[bool] - Whether to disable GeoIP lookup

    Returns:
    Optional[str] - The event UUID if successful
    """
```

### Exception Capture

Automatically capture and track exceptions with full stack traces and context information.

```python { .api }
def capture_exception(exception: Optional[ExceptionArg] = None, **kwargs: OptionalCaptureArgs):
    """
    Capture exceptions that happen in your code.

    Parameters:
    - exception: Optional[ExceptionArg] - The exception to capture. If not provided, captures current exception via sys.exc_info()
    - All OptionalCaptureArgs parameters are supported

    Notes:
    - Idempotent - calling twice with same exception instance only tracks one occurrence
    - Automatically captures stack traces between raise and capture points
    - Context boundaries may truncate stack traces
    """
```

## Usage Examples

### Basic Event Tracking

```python
import posthog

# Configure PostHog
posthog.api_key = 'phc_your_project_api_key'

# Simple event
posthog.capture('user123', 'button_clicked')

# Event with properties
posthog.capture('user123', 'purchase_completed', {
    'product_id': 'abc123',
    'amount': 29.99,
    'currency': 'USD',
    'category': 'books'
})

# Event with groups
posthog.capture('user123', 'feature_used', {
    'feature_name': 'advanced_search'
}, groups={'company': 'acme_corp'})
```

### Context-Based Event Tracking

```python
import posthog

# Using context for automatic user identification
with posthog.new_context():
    posthog.identify_context('user123')
    posthog.tag('session_type', 'premium')
    
    # Event automatically includes user ID and tags
    posthog.capture('page_viewed', {'page': 'dashboard'})
    posthog.capture('button_clicked', {'button': 'export'})
    
    # Override context user for specific event
    posthog.capture('admin_action', {'action': 'user_reset'}, distinct_id='admin456')
```

### Exception Tracking

```python
import posthog

try:
    risky_operation()
except ValueError as e:
    # Capture specific exception
    posthog.capture_exception(e, properties={'operation': 'data_processing'})
    
try:
    another_operation()
except Exception:
    # Capture current exception automatically
    posthog.capture_exception(properties={'context': 'background_task'})
```

### Advanced Event Configuration

```python
import posthog
from datetime import datetime, timezone

# Event with timestamp and feature flags
posthog.capture(
    'user123',
    'api_call',
    properties={
        'endpoint': '/api/users',
        'method': 'POST',
        'response_time': 250
    },
    timestamp=datetime.now(timezone.utc),
    send_feature_flags=True,
    groups={'organization': 'org_456'}
)

# Event with custom UUID for deduplication
posthog.capture(
    'user123',
    'payment_processed',
    properties={'transaction_id': 'txn_789'},
    uuid='custom-event-uuid-123'
)
```

### Batching and Performance

```python
import posthog

# Configure batching behavior
posthog.flush_at = 50  # Send batch after 50 events
posthog.flush_interval = 2.0  # Send batch after 2 seconds

# Events are automatically batched
for i in range(100):
    posthog.capture(f'user_{i}', 'bulk_event', {'index': i})

# Force flush remaining events
posthog.flush()
```

## Error Handling

### Event Validation

Events are validated before sending:

- Event names must be non-empty strings
- Properties must be JSON-serializable
- Distinct IDs are automatically generated if not provided
- Invalid data is logged and dropped

### Retry Behavior

Failed events are automatically retried:

- Exponential backoff for temporary failures
- Maximum retry attempts configurable
- Failed events are logged for debugging
- Network errors trigger automatic retries

### Offline Support

The SDK handles offline scenarios gracefully:

- Events are queued when network is unavailable
- Automatic retry when connection is restored  
- Configurable queue size limits
- Events are persisted in memory queue

## Best Practices

### Event Naming

Use consistent, descriptive event names:

```python
# Good - verb + noun format
posthog.capture('user123', 'video_played')
posthog.capture('user123', 'purchase_completed')
posthog.capture('user123', 'signup_started')

# Avoid - unclear or inconsistent
posthog.capture('user123', 'click')  # Too vague
posthog.capture('user123', 'VIDEO_PLAY')  # Inconsistent case
```

### Property Structure

Keep properties flat and meaningful:

```python
# Good - flat structure with clear names
posthog.capture('user123', 'purchase_completed', {
    'product_id': 'abc123',
    'product_name': 'Premium Plan',
    'amount': 29.99,
    'currency': 'USD',
    'payment_method': 'stripe'
})

# Avoid - nested objects are flattened
posthog.capture('user123', 'purchase', {
    'product': {  # This gets flattened
        'id': 'abc123',
        'name': 'Premium Plan'
    }
})
```

### Context Usage

Use contexts for related event sequences:

```python
# Good - group related events in context
with posthog.new_context():
    posthog.identify_context('user123')
    posthog.tag('flow', 'onboarding')
    
    posthog.capture('onboarding_started')
    posthog.capture('form_step_completed', {'step': 1})
    posthog.capture('form_step_completed', {'step': 2})
    posthog.capture('onboarding_completed')
```