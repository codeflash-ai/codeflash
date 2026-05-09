# Structured Logging

OpenTelemetry-compatible structured logging interface with automatic correlation to Sentry events, performance data, and comprehensive context propagation for debugging and observability.

## Capabilities

### Structured Logging Functions

Send structured log messages with automatic correlation to Sentry events and spans using OpenTelemetry-compatible logging interfaces.

```python { .api }
def trace(template: str, **kwargs) -> None:
    """
    Send a trace-level structured log message.
    
    Parameters:
    - template: Log message template with placeholder formatting
    - **kwargs: Structured data to include in log context
    """

def debug(template: str, **kwargs) -> None:
    """
    Send a debug-level structured log message.
    
    Parameters:
    - template: Log message template with placeholder formatting
    - **kwargs: Structured data to include in log context
    """

def info(template: str, **kwargs) -> None:
    """
    Send an info-level structured log message.
    
    Parameters:
    - template: Log message template with placeholder formatting
    - **kwargs: Structured data to include in log context
    """

def warning(template: str, **kwargs) -> None:
    """
    Send a warning-level structured log message.
    
    Parameters:
    - template: Log message template with placeholder formatting
    - **kwargs: Structured data to include in log context
    """

def error(template: str, **kwargs) -> None:
    """
    Send an error-level structured log message.
    
    Parameters:
    - template: Log message template with placeholder formatting
    - **kwargs: Structured data to include in log context
    """

def fatal(template: str, **kwargs) -> None:
    """
    Send a fatal-level structured log message.
    
    Parameters:
    - template: Log message template with placeholder formatting
    - **kwargs: Structured data to include in log context
    """
```

**Usage Examples:**

```python
import sentry_sdk

# Initialize SDK first
sentry_sdk.init(dsn="your-dsn-here")

# Simple structured logging
sentry_sdk.logger.info("User login successful", user_id="user_123")
sentry_sdk.logger.warning("High memory usage detected", memory_percent=85.2)
sentry_sdk.logger.error("Database connection failed", 
                       host="db.example.com", 
                       port=5432, 
                       timeout=30)

# Template-based logging with structured data
sentry_sdk.logger.info(
    "Processing order {order_id} for user {user_id}",
    order_id="order_456",
    user_id="user_123",
    item_count=3,
    total_amount=99.99
)

# Debug logging with complex structured data
sentry_sdk.logger.debug(
    "Cache operation completed",
    operation="get",
    key="user_prefs:123",
    hit=True,
    latency_ms=1.2,
    cache_size=1024,
    metadata={
        "region": "us-west-2",
        "cluster": "cache-cluster-1"
    }
)

# Error logging with context
def process_payment(payment_data):
    try:
        result = payment_processor.charge(payment_data)
        sentry_sdk.logger.info(
            "Payment processed successfully",
            transaction_id=result.id,
            amount=payment_data.amount,
            currency=payment_data.currency,
            processing_time_ms=result.processing_time
        )
        return result
    except PaymentError as e:
        sentry_sdk.logger.error(
            "Payment processing failed",
            error_code=e.code,
            error_message=str(e),
            payment_method=payment_data.method,
            amount=payment_data.amount,
            retry_count=e.retry_count
        )
        raise
```

## Automatic Correlation

### Event Correlation

Structured logs automatically correlate with Sentry events and performance data:

- **Trace Correlation**: Logs include trace and span IDs when called within transactions
- **Event Context**: Logs inherit scope context (user, tags, extra data)
- **Error Association**: Error-level logs create Sentry events with structured data
- **Performance Integration**: Logs within spans include performance context

```python
import sentry_sdk

def process_user_request(user_id):
    with sentry_sdk.start_transaction(name="process_request", op="function"):
        # These logs will include transaction and span context
        sentry_sdk.logger.info("Starting request processing", user_id=user_id)
        
        with sentry_sdk.start_span(op="db.query", description="fetch user"):
            user = get_user(user_id)
            sentry_sdk.logger.debug("User data retrieved", 
                                  user_id=user_id, 
                                  user_type=user.type)
        
        # Error logs automatically create Sentry events
        if not user.is_active:
            sentry_sdk.logger.error("Inactive user attempted access",
                                  user_id=user_id,
                                  account_status=user.status,
                                  last_active=user.last_active)
            raise UserInactiveError()
        
        sentry_sdk.logger.info("Request processing completed", 
                             user_id=user_id,
                             processing_time_ms=get_processing_time())
```

### Scope Integration

Structured logs inherit and can modify scope context:

```python
# Set structured logging context at scope level
sentry_sdk.set_tag("component", "payment_processor")
sentry_sdk.set_extra("version", "2.1.0")

# All subsequent logs include scope context
sentry_sdk.logger.info("Service started")  # Includes component tag and version extra

# Temporarily modify context for specific logs
with sentry_sdk.new_scope() as scope:
    scope.set_tag("operation", "batch_process")
    sentry_sdk.logger.info("Batch processing started", batch_size=1000)
    # Log includes both component and operation tags
```

## Configuration Integration

### Log Level Control

Structured logging respects Sentry SDK configuration:

```python
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

# Configure structured logging behavior
sentry_sdk.init(
    dsn="your-dsn-here",
    integrations=[
        LoggingIntegration(
            level=logging.INFO,        # Breadcrumb level
            event_level=logging.ERROR  # Event creation level
        )
    ],
    debug=True  # Enable debug-level structured logs
)

# Structured logs follow the configured levels
sentry_sdk.logger.debug("Debug info")    # Only sent if debug=True
sentry_sdk.logger.info("Info message")   # Becomes breadcrumb
sentry_sdk.logger.error("Error occurred") # Creates Sentry event
```

### Custom Log Processing

```python
def custom_log_processor(record):
    """Custom processor for structured log records."""
    # Add custom fields to all structured logs
    record.update({
        "service": "payment-api",
        "environment": "production",
        "timestamp": datetime.utcnow().isoformat()
    })
    return record

# Apply custom processing (conceptual - actual implementation may vary)
sentry_sdk.logger.add_processor(custom_log_processor)

# All structured logs now include custom fields
sentry_sdk.logger.info("Operation completed", operation_id="op_123")
```

## Advanced Usage

### Performance Logging

Combine structured logging with performance monitoring:

```python
import time
import sentry_sdk

def monitored_operation(operation_name):
    start_time = time.time()
    
    sentry_sdk.logger.info("Operation starting", operation=operation_name)
    
    with sentry_sdk.start_span(op="custom", description=operation_name) as span:
        try:
            # Perform operation
            result = perform_complex_operation()
            
            duration = time.time() - start_time
            span.set_measurement("operation_duration", duration * 1000, "millisecond")
            
            sentry_sdk.logger.info(
                "Operation completed successfully",
                operation=operation_name,
                duration_ms=duration * 1000,
                result_size=len(result)
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            sentry_sdk.logger.error(
                "Operation failed",
                operation=operation_name,
                duration_ms=duration * 1000,
                error_type=type(e).__name__,
                error_message=str(e)
            )
            raise
```

### Business Logic Logging

Use structured logging for business intelligence and monitoring:

```python
def track_business_metrics(user_action, **metrics):
    """Track business metrics with structured logging."""
    sentry_sdk.logger.info(
        f"Business metric: {user_action}",
        metric_type="business_event",
        action=user_action,
        timestamp=datetime.utcnow().isoformat(),
        **metrics
    )

# Usage examples
track_business_metrics("user_signup", 
                      user_type="premium", 
                      referral_source="google_ads",
                      trial_length_days=14)

track_business_metrics("purchase_completed",
                      amount=99.99,
                      currency="USD", 
                      product_category="software",
                      payment_method="credit_card")

track_business_metrics("feature_usage",
                      feature="advanced_analytics",
                      usage_duration_minutes=45,
                      user_tier="enterprise")
```

## Best Practices

### Message Templates

Use consistent, searchable message templates:

```python
# Good: Consistent templates
sentry_sdk.logger.info("User action: {action}", action="login", user_id="123")
sentry_sdk.logger.info("User action: {action}", action="logout", user_id="123")

# Avoid: Variable messages that can't be grouped
sentry_sdk.logger.info(f"User {user_id} performed {action}")  # Hard to search/group
```

### Structured Data Organization

Organize structured data consistently:

```python
# Good: Consistent field naming and organization
sentry_sdk.logger.info("Database operation completed",
                      db_operation="select",
                      db_table="users", 
                      db_duration_ms=45,
                      db_rows_affected=1,
                      query_hash="abc123")

# Use nested structures for complex data
sentry_sdk.logger.info("API request processed",
                      request={
                          "method": "POST",
                          "path": "/api/users",
                          "duration_ms": 125
                      },
                      response={
                          "status_code": 200,
                          "size_bytes": 1024
                      })
```

### Log Level Guidelines

- **trace**: Very detailed debugging information
- **debug**: Development and troubleshooting information  
- **info**: Normal application flow and business events
- **warning**: Concerning situations that don't prevent operation
- **error**: Error conditions that should be investigated
- **fatal**: Critical failures that may cause application termination

Structured logging provides comprehensive observability with automatic correlation to Sentry's error tracking and performance monitoring capabilities.