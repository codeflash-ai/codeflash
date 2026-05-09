# Context and Metadata

Setting user information, tags, extra data, context objects, and breadcrumbs for enhanced debugging, event correlation, and structured data organization.

## Capabilities

### User Information

Associate user data with events for user-centric error tracking and debugging with support for multiple identification schemes.

```python { .api }
def set_user(value: Optional[Dict[str, Any]]) -> None:
    """
    Set user information for all subsequent events.
    
    Parameters:
    - value: User data dictionary or None to clear user data
      Common fields: id, email, username, ip_address, name, segment
    """
```

**Usage Examples:**

```python
import sentry_sdk

# Basic user identification
sentry_sdk.set_user({
    "id": "user_123",
    "email": "user@example.com",
    "username": "johndoe"
})

# Comprehensive user context
sentry_sdk.set_user({
    "id": "user_123",
    "email": "user@example.com", 
    "username": "johndoe",
    "name": "John Doe",
    "ip_address": "192.168.1.100",
    "segment": "premium",
    "subscription": "pro",
    "signup_date": "2023-01-15",
    "last_login": "2024-09-06T10:30:00Z"
})

# Clear user data (e.g., on logout)
sentry_sdk.set_user(None)

# Anonymous user with limited context
sentry_sdk.set_user({
    "id": "anonymous_456",
    "ip_address": "10.0.0.50",
    "segment": "trial"
})
```

### Tags

Set indexed metadata for filtering, searching, and organizing events in the Sentry interface.

```python { .api }
def set_tag(key: str, value: str) -> None:
    """
    Set a single tag for event filtering and grouping.
    
    Parameters:
    - key: Tag name (max 32 characters)
    - value: Tag value (max 200 characters, converted to string)
    """

def set_tags(tags: Dict[str, str]) -> None:
    """
    Set multiple tags at once.
    
    Parameters:
    - tags: Dictionary of tag key-value pairs
    """
```

**Usage Examples:**

```python
import sentry_sdk

# Single tag
sentry_sdk.set_tag("environment", "production")
sentry_sdk.set_tag("version", "1.2.3")
sentry_sdk.set_tag("feature_flag", "new_checkout")

# Multiple tags
sentry_sdk.set_tags({
    "service": "payment-processor",
    "region": "us-west-2", 
    "datacenter": "aws",
    "cluster": "prod-cluster-1",
    "component": "billing"
})

# Request-specific tags (often set by framework integrations)
sentry_sdk.set_tags({
    "http.method": "POST",
    "endpoint": "/api/v1/payments",
    "user_type": "premium",
    "api_version": "v1"
})

# Business context tags
sentry_sdk.set_tags({
    "tenant": "acme_corp",
    "subscription_tier": "enterprise",
    "feature_set": "advanced",
    "ab_test_group": "variant_b"
})
```

### Extra Data

Set additional debugging information that provides detailed context for troubleshooting events.

```python { .api }
def set_extra(key: str, value: Any) -> None:
    """
    Set extra debugging information for events.
    
    Parameters:
    - key: Extra data key name
    - value: Any serializable value (dict, list, string, number, etc.)
    """
```

**Usage Examples:**

```python
import sentry_sdk

# Simple key-value data
sentry_sdk.set_extra("request_id", "req_abc123")
sentry_sdk.set_extra("processing_time_ms", 245)
sentry_sdk.set_extra("retry_count", 3)

# Complex structured data
sentry_sdk.set_extra("request_payload", {
    "user_id": 123,
    "action": "create_order",
    "items": [
        {"product_id": "prod_1", "quantity": 2},
        {"product_id": "prod_2", "quantity": 1}
    ],
    "total_amount": 99.99,
    "currency": "USD"
})

# Configuration and environment details
sentry_sdk.set_extra("database_config", {
    "host": "db.example.com",
    "port": 5432,
    "database": "production",
    "pool_size": 10,
    "connection_timeout": 30
})

# Performance and resource data
sentry_sdk.set_extra("system_metrics", {
    "cpu_usage": 45.2,
    "memory_usage_mb": 1024,
    "disk_free_gb": 50.5,
    "active_connections": 25
})
```

### Context Objects

Set structured context data organized by context type for comprehensive environmental information.

```python { .api }
def set_context(key: str, value: Dict[str, Any]) -> None:
    """
    Set structured context information by category.
    
    Parameters:
    - key: Context category name (e.g., 'os', 'runtime', 'device', 'app')
    - value: Context data dictionary
    """
```

**Usage Examples:**

```python
import sentry_sdk

# Application context
sentry_sdk.set_context("app", {
    "name": "payment-service",
    "version": "2.1.4",
    "build_number": "1847",
    "commit_sha": "a1b2c3d4",
    "environment": "production",
    "region": "us-west-2"
})

# Runtime context
sentry_sdk.set_context("runtime", {
    "name": "python", 
    "version": "3.11.5",
    "implementation": "cpython",
    "build": "Python 3.11.5 (main, Aug 24 2023, 15:18:16)",
    "thread_count": 12,
    "gc_count": [145, 23, 2]
})

# Device/server context
sentry_sdk.set_context("device", {
    "type": "server",
    "arch": "x86_64",
    "cpu_count": 8,
    "memory_size": 16777216,  # bytes
    "storage_size": 107374182400,  # bytes
    "timezone": "UTC"
})

# Custom business context
sentry_sdk.set_context("business", {
    "tenant_id": "tenant_123",
    "organization": "Acme Corp",
    "subscription": "enterprise",
    "feature_flags": ["new_ui", "advanced_analytics"],
    "experiment_groups": ["checkout_v2", "pricing_test_b"]
})

# Request context (typically set by web framework integrations)
sentry_sdk.set_context("request", {
    "method": "POST",
    "url": "https://api.example.com/v1/orders",
    "query_string": "include_details=true",
    "headers": {
        "content-type": "application/json",
        "user-agent": "MyApp/1.0",
        "x-request-id": "req_abc123"
    }
})
```

### Breadcrumbs

Add trail of events leading up to an error or event for debugging context and timeline reconstruction.

```python { .api }
def add_breadcrumb(
    crumb: Optional[Dict[str, Any]] = None,
    hint: Optional[Dict[str, Any]] = None,
    **kwargs
) -> None:
    """
    Add a breadcrumb to the trail of events.
    
    Parameters:
    - crumb: Breadcrumb dictionary with message, level, category, data, etc.
    - hint: Processing hints for breadcrumb filters
    - **kwargs: Breadcrumb properties as keyword arguments
    """
```

**Usage Examples:**

```python
import sentry_sdk

# Simple message breadcrumb
sentry_sdk.add_breadcrumb(
    message="User started checkout process",
    level="info"
)

# Detailed breadcrumb with category and data
sentry_sdk.add_breadcrumb(
    message="Database query executed",
    category="db",
    level="info",
    data={
        "query": "SELECT * FROM orders WHERE user_id = %s",
        "duration_ms": 45,
        "rows_affected": 1
    }
)

# HTTP request breadcrumb
sentry_sdk.add_breadcrumb(
    message="API call to payment service",
    category="http",
    level="info",
    data={
        "url": "https://payments.example.com/v1/charge",
        "method": "POST",
        "status_code": 200,
        "response_time_ms": 250
    }
)

# Navigation breadcrumb (for web apps)
sentry_sdk.add_breadcrumb(
    message="User navigated to orders page",
    category="navigation", 
    level="info",
    data={
        "from": "/dashboard",
        "to": "/orders",
        "trigger": "menu_click"
    }
)

# User action breadcrumb
sentry_sdk.add_breadcrumb(
    message="Form submitted",
    category="user",
    level="info",
    data={
        "form": "checkout_form",
        "fields": ["email", "payment_method", "billing_address"],
        "validation_errors": []
    }
)

# System event breadcrumb
sentry_sdk.add_breadcrumb(
    message="Cache miss for user preferences",
    category="cache",
    level="warning",
    data={
        "key": "user_prefs_123",
        "ttl_remaining": 0,
        "hit_rate": 0.85
    }
)

# Using kwargs for simple breadcrumbs
sentry_sdk.add_breadcrumb(
    message="Configuration loaded",
    category="config",
    level="debug",
    config_source="environment_variables",
    config_count=25
)
```

## Breadcrumb Categories

### Standard Categories

- **default**: General application events
- **auth**: Authentication and authorization events
- **navigation**: Page/route changes and navigation
- **http**: HTTP requests and responses
- **db**: Database operations and queries
- **cache**: Cache operations (hits, misses, invalidation)
- **rpc**: Remote procedure calls and API interactions
- **user**: User actions and interactions
- **ui**: User interface events and state changes
- **system**: System-level events and resource usage

### Breadcrumb Levels

- **fatal**: Critical system failures
- **error**: Error conditions
- **warning**: Warning conditions
- **info**: Informational messages (default)
- **debug**: Debug information

## Data Organization Best Practices

### Tags vs Extra vs Context

**Use Tags for:**
- Filterable, searchable metadata
- Short string values (max 200 characters)
- Grouping and categorization
- Performance-critical indexing

**Use Extra for:**
- Detailed debugging information
- Complex data structures
- Large data that doesn't need indexing
- Request/response payloads

**Use Context for:**
- Structured environmental data
- Organized by logical categories
- Standard context types (app, runtime, device, etc.)
- Data that needs specific formatting in UI

### Scope-Level Data Setting

```python
# Global scope: Application-wide data
global_scope = sentry_sdk.get_global_scope()
global_scope.set_tag("service", "payment-processor")
global_scope.set_context("app", {"version": "1.2.3"})

# Isolation scope: Request/user-specific data
isolation_scope = sentry_sdk.get_isolation_scope()
isolation_scope.set_user({"id": "user_123"})
isolation_scope.set_tag("request_id", "req_456")

# Current scope: Local context data
current_scope = sentry_sdk.get_current_scope()
current_scope.set_extra("operation", "process_payment")
current_scope.add_breadcrumb(message="Payment validation started")
```

## Metadata Inheritance

Context, tags, and extra data follow scope inheritance rules:
1. Global scope data applies to all events
2. Isolation scope data applies to current request/user context  
3. Current scope data applies to local operations
4. Event-specific data takes highest precedence
5. Child scopes inherit parent data but can override values

This hierarchical system ensures proper context isolation while maintaining comprehensive debugging information across all events and performance monitoring data.