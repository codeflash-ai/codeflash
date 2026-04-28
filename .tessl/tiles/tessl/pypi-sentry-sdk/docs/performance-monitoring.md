# Performance Monitoring

Distributed tracing with transactions and spans for monitoring application performance, database queries, external service calls, and custom operations with automatic and manual instrumentation support.

## Capabilities

### Transaction Management

Create and manage top-level transactions representing complete operations like HTTP requests, background jobs, or business processes.

```python { .api }
def start_transaction(
    transaction: Optional[Union[Transaction, Dict[str, Any]]] = None,
    instrumenter: str = INSTRUMENTER.SENTRY,
    custom_sampling_context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Union[Transaction, NoOpSpan]:
    """
    Start a new transaction for performance monitoring.
    
    Parameters:
    - transaction: Transaction name or transaction object
    - instrumenter: Instrumentation source identifier
    - custom_sampling_context: Additional context for sampling decisions
    - **kwargs: Additional transaction properties (op, description, source, etc.)
    
    Returns:
    Transaction or NoOpSpan: Active transaction or no-op if disabled/sampled out
    """
```

**Usage Examples:**

```python
import sentry_sdk

# Simple transaction
with sentry_sdk.start_transaction(name="process_order", op="function") as transaction:
    # Process order logic
    validate_order()
    charge_payment()
    fulfill_order()

# Transaction with custom properties
transaction = sentry_sdk.start_transaction(
    name="data_pipeline",
    op="task",
    description="Process daily analytics batch",
    source="custom",
    data={"batch_size": 1000, "source": "analytics_db"}
)
try:
    process_analytics_batch()
finally:
    transaction.finish()

# HTTP request transaction (typically handled by web framework integrations)
def handle_api_request(request):
    with sentry_sdk.start_transaction(
        name=f"{request.method} {request.path}",
        op="http.server",
        source="route"
    ) as transaction:
        transaction.set_tag("http.method", request.method)
        transaction.set_tag("endpoint", request.path)
        return process_request(request)
```

### Span Creation

Create child spans within transactions to measure specific operations like database queries, API calls, or computational tasks.

```python { .api }
def start_span(
    instrumenter: str = INSTRUMENTER.SENTRY,
    **kwargs
) -> Span:
    """
    Start a new span within the current transaction.
    
    Parameters:
    - instrumenter: Instrumentation source identifier
    - **kwargs: Span properties (op, description, tags, data, etc.)
    
    Returns:
    Span: Active span for the operation
    """
```

**Usage Examples:**

```python
import sentry_sdk

def process_user_data(user_id):
    with sentry_sdk.start_transaction(name="process_user_data", op="function"):
        # Database query span
        with sentry_sdk.start_span(op="db.query", description="fetch user") as span:
            span.set_tag("db.table", "users")
            span.set_data("user_id", user_id)
            user = database.get_user(user_id)
        
        # API call span  
        with sentry_sdk.start_span(
            op="http.client",
            description="POST /api/enrichment"
        ) as span:
            span.set_tag("http.method", "POST")
            span.set_data("url", "https://api.example.com/enrichment")
            enriched_data = api_client.enrich_user_data(user)
        
        # Processing span
        with sentry_sdk.start_span(op="function", description="transform_data"):
            result = transform_user_data(enriched_data)
        
        return result
```

### Active Span Access

Get the currently active span for adding metadata, measurements, or creating child spans.

```python { .api }
def get_current_span(scope: Optional[Scope] = None) -> Optional[Span]:
    """
    Get the currently active span.
    
    Parameters:
    - scope: Scope to check for active span (uses current scope if None)
    
    Returns:
    Optional[Span]: Current span if active, None otherwise
    """
```

**Usage Examples:**

```python
import sentry_sdk

def database_operation(query):
    span = sentry_sdk.get_current_span()
    if span:
        span.set_tag("db.system", "postgresql")
        span.set_data("db.statement", query)
        span.set_data("db.operation", "select")
    
    return execute_query(query)

def add_custom_measurements():
    span = sentry_sdk.get_current_span()
    if span:
        span.set_measurement("memory_usage_mb", get_memory_usage(), "megabyte")
        span.set_measurement("cpu_usage_percent", get_cpu_usage(), "percent")
        span.set_data("custom_metric", calculate_business_metric())
```

### Distributed Tracing

Connect traces across service boundaries using W3C Trace Context headers for distributed system monitoring.

```python { .api }
def continue_trace(
    environ_or_headers: Union[Dict[str, str], Dict[str, Any]],
    op: Optional[str] = None,
    name: Optional[str] = None,
    source: Optional[str] = None,
    origin: str = "manual"
) -> Transaction:
    """
    Continue a distributed trace from incoming headers.
    
    Parameters:
    - environ_or_headers: WSGI environ dict or HTTP headers dict
    - op: Transaction operation type
    - name: Transaction name
    - source: Transaction source identifier
    - origin: Trace origin identifier
    
    Returns:
    Transaction: Connected transaction continuing the distributed trace
    """

def get_traceparent() -> Optional[str]:
    """
    Get W3C traceparent header value for outgoing requests.
    
    Returns:
    Optional[str]: Traceparent header value or None if no active transaction
    """

def get_baggage() -> Optional[str]:
    """
    Get W3C baggage header value for outgoing requests.
    
    Returns:
    Optional[str]: Baggage header value or None if no baggage data
    """
```

**Usage Examples:**

```python
import sentry_sdk
import requests

# Server: Continue trace from incoming request
def handle_incoming_request(request):
    # Extract trace context from headers
    transaction = sentry_sdk.continue_trace(
        request.headers,
        op="http.server",
        name=f"{request.method} {request.path}",
        source="route"
    )
    
    with transaction:
        return process_request(request)

# Client: Propagate trace to outgoing request
def make_api_call(url, data):
    headers = {}
    
    # Add trace headers for distributed tracing
    if traceparent := sentry_sdk.get_traceparent():
        headers["traceparent"] = traceparent
    
    if baggage := sentry_sdk.get_baggage():
        headers["baggage"] = baggage
    
    with sentry_sdk.start_span(op="http.client", description=f"POST {url}"):
        response = requests.post(url, json=data, headers=headers)
        return response.json()
```

### Automatic Tracing Decorator

Automatically create spans for function calls using the trace decorator.

```python { .api }
def trace(func: Callable) -> Callable:
    """
    Decorator to automatically create spans for function calls.
    
    The span will use the function name as description and 'function' as operation.
    Additional span data can be set within the decorated function.
    
    Parameters:
    - func: Function to wrap with automatic tracing
    
    Returns:
    Callable: Decorated function that creates spans automatically
    """
```

**Usage Examples:**

```python
import sentry_sdk

@sentry_sdk.trace
def process_payment(amount, currency):
    """This function will automatically create a span."""
    # Add custom span data
    span = sentry_sdk.get_current_span()
    if span:
        span.set_tag("payment.currency", currency)
        span.set_data("payment.amount", amount)
    
    return payment_processor.charge(amount, currency)

@sentry_sdk.trace
def calculate_analytics(dataset):
    """Complex calculation with automatic timing."""
    span = sentry_sdk.get_current_span()
    if span:
        span.set_data("dataset_size", len(dataset))
        span.set_tag("operation", "analytics")
    
    result = perform_complex_calculation(dataset)
    
    if span:
        span.set_data("result_count", len(result))
    
    return result

# Usage in transaction context
def process_order(order_id):
    with sentry_sdk.start_transaction(name="process_order", op="function"):
        payment_result = process_payment(100.0, "USD")  # Automatically traced
        analytics = calculate_analytics(order_data)      # Automatically traced
        return finalize_order(order_id, payment_result)
```

### Span Modification

Update properties of the currently active span without needing a direct reference to the span object.

```python { .api }
def update_current_span(
    op: Optional[str] = None,
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Union[str, int, float, bool]]] = None,
    data: Optional[Dict[str, Any]] = None  # Deprecated
) -> None:
    """
    Update the current active span with the provided parameters.
    
    Parameters:
    - op: The operation name for the span (e.g., "http.client", "db.query")
    - name: The human-readable name/description for the span
    - attributes: Key-value pairs to add as attributes to the span
    - data: Deprecated, use attributes instead
    """
```

**Usage Examples:**

```python
import sentry_sdk
from sentry_sdk.consts import OP

# Start a span and update it later
with sentry_sdk.start_span(op="function", name="process_data") as span:
    user_id = get_user_id()
    
    # Update span with additional context as we learn more
    sentry_sdk.update_current_span(
        name=f"process_data_for_user_{user_id}",
        attributes={
            "user_id": user_id,
            "batch_size": 50,
            "processing_type": "standard"
        }
    )
    
    # Process data...
    result = expensive_operation()
    
    # Update with results
    sentry_sdk.update_current_span(
        attributes={
            "result_count": len(result),
            "success": True
        }
    )
```

## Span Interface

### Span Properties and Methods

```python { .api }
class Span:
    def set_tag(self, key: str, value: str) -> None:
        """Set a tag on the span."""
    
    def set_data(self, key: str, value: Any) -> None:
        """Set structured data on the span."""
    
    def set_measurement(self, name: str, value: float, unit: str = "") -> None:
        """Set a performance measurement."""
    
    def set_status(self, status: str) -> None:
        """Set span status ('ok', 'cancelled', 'internal_error', etc.)."""
    
    def set_http_status(self, http_status: int) -> None:
        """Set HTTP status code and derive span status."""
    
    def finish(self, end_timestamp: Optional[datetime] = None) -> None:
        """Finish the span with optional custom end time."""
    
    def to_json(self) -> Dict[str, Any]:
        """Serialize span to JSON representation."""
    
    @property
    def span_id(self) -> str:
        """Unique span identifier."""
    
    @property
    def trace_id(self) -> str:
        """Trace identifier shared across distributed trace."""
    
    @property
    def parent_span_id(self) -> Optional[str]:
        """Parent span identifier."""
    
    @property
    def sampled(self) -> Optional[bool]:
        """Whether this span is sampled for tracing."""
```

### Transaction-Specific Interface

```python { .api }
class Transaction(Span):
    def set_name(self, name: str, source: Optional[str] = None) -> None:
        """Set transaction name and source."""
    
    @property
    def name(self) -> str:
        """Transaction name."""
    
    @property
    def source(self) -> str:
        """Transaction source ('custom', 'route', 'url', etc.)."""
```

## Performance Data

### Automatic Measurements

The SDK automatically collects:
- **Span duration**: Start and end timestamps
- **HTTP metrics**: Status codes, response sizes, request/response times
- **Database metrics**: Query timing, connection info, affected rows
- **Cache metrics**: Hit/miss ratios, operation timing
- **Queue metrics**: Job processing time, queue depth

### Custom Measurements

Add custom performance measurements:

```python
span = sentry_sdk.get_current_span()
if span:
    # Time-based measurements
    span.set_measurement("processing_time", 142.5, "millisecond")
    span.set_measurement("wait_time", 2.3, "second")
    
    # Size measurements
    span.set_measurement("payload_size", 1024, "byte")
    span.set_measurement("result_count", 50, "none")
    
    # Rate measurements  
    span.set_measurement("throughput", 1500, "per_second")
    span.set_measurement("error_rate", 0.02, "ratio")
```

## Integration with Scopes

Performance monitoring integrates with scope management:
- Transactions and spans inherit scope context (tags, user, extra data)
- Performance events include breadcrumbs and contextual information
- Scope modifications during spans affect the span's metadata
- Automatic correlation between errors and performance data