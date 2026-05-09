# Sentry SDK

The official Python SDK for Sentry, providing comprehensive error monitoring, performance tracking, and application observability. The SDK automatically captures exceptions, performance issues, and custom events from Python applications with extensive integration support for popular frameworks including Django, Flask, FastAPI, Celery, and 40+ other libraries.

## Package Information

- **Package Name**: sentry-sdk
- **Language**: Python
- **Installation**: `pip install sentry-sdk`

## Core Imports

```python
import sentry_sdk
```

For direct access to specific functions:

```python
from sentry_sdk import init, capture_exception, capture_message, start_transaction
```

## Basic Usage

```python
import sentry_sdk

# Initialize the SDK
sentry_sdk.init(
    dsn="https://your-dsn@sentry.io/project-id",
    # Set traces_sample_rate to 1.0 to capture 100% of transactions for performance monitoring
    traces_sample_rate=1.0,
)

# Automatic exception capture
try:
    division_by_zero = 1 / 0
except Exception as e:
    # Exception is automatically captured by Sentry
    pass

# Manual event capture
sentry_sdk.capture_message("Something important happened!")

# Manual exception capture with additional context
try:
    risky_operation()
except Exception as e:
    sentry_sdk.capture_exception(e)

# Performance monitoring
with sentry_sdk.start_transaction(name="my-transaction"):
    # Your application logic here
    process_data()
```

## Architecture

The Sentry SDK uses a modern scope-based architecture with three distinct scope levels:

- **Global Scope**: Process-wide data (release, environment, server details)
- **Isolation Scope**: Request/user-level data (user context, request details)
- **Current Scope**: Local thread/async context data (span-specific tags, breadcrumbs)

This design enables proper context isolation in concurrent environments while maintaining performance and providing flexible configuration options for different deployment scenarios.

## Capabilities

### Initialization and Configuration

SDK initialization and configuration management including DSN setup, sampling rates, integrations, and client options.

```python { .api }
def init(*args, **kwargs) -> None: ...
def is_initialized() -> bool: ...
```

[Configuration](./configuration.md)

### Event Capture

Manual and automatic capture of exceptions, messages, and custom events with context preservation and filtering capabilities.

```python { .api }
def capture_exception(error=None, scope=None, **scope_kwargs) -> Optional[str]: ...
def capture_message(message, level=None, scope=None, **scope_kwargs) -> Optional[str]: ...
def capture_event(event, hint=None, scope=None, **scope_kwargs) -> Optional[str]: ...
```

[Event Capture](./event-capture.md)

### Scope Management

Context management through a three-tier scope system providing isolation and data organization across global, request, and local levels.

```python { .api }
def get_global_scope() -> Scope: ...
def get_isolation_scope() -> Scope: ...
def get_current_scope() -> Scope: ...
def new_scope() -> ContextManager[Scope]: ...
def isolation_scope() -> ContextManager[Scope]: ...
```

[Scope Management](./scope-management.md)

### Context and Metadata

Setting user information, tags, extra data, context objects, and breadcrumbs for enhanced debugging and event correlation.

```python { .api }
def set_user(value) -> None: ...
def set_tag(key, value) -> None: ...
def set_tags(tags) -> None: ...
def set_extra(key, value) -> None: ...
def set_context(key, value) -> None: ...
def add_breadcrumb(crumb=None, hint=None, **kwargs) -> None: ...
```

[Context and Metadata](./context-metadata.md)

### Performance Monitoring

Distributed tracing with transactions and spans for monitoring application performance, database queries, and external service calls.

```python { .api }
def start_transaction(transaction=None, **kwargs) -> Union[Transaction, NoOpSpan]: ...
def start_span(**kwargs) -> Span: ...
def get_current_span(scope=None) -> Optional[Span]: ...
def continue_trace(environ_or_headers, op=None, name=None, source=None, origin="manual") -> Transaction: ...
def trace(func) -> Callable: ...
```

[Performance Monitoring](./performance-monitoring.md)

### Integrations

Framework and library integrations for automatic instrumentation including web frameworks, databases, HTTP clients, task queues, and AI/ML libraries.

```python { .api }
class Integration(ABC):
    identifier: str
    @staticmethod
    @abstractmethod
    def setup_once() -> None: ...
```

Available integrations include Django, Flask, FastAPI, Celery, SQLAlchemy, Redis, AWS Lambda, OpenAI, Anthropic, and 30+ others.

[Integrations](./integrations.md)

### Structured Logging

OpenTelemetry-compatible structured logging interface with automatic correlation to Sentry events and performance data.

```python { .api }
def trace(template, **kwargs) -> None: ...
def debug(template, **kwargs) -> None: ...
def info(template, **kwargs) -> None: ...
def warning(template, **kwargs) -> None: ...
def error(template, **kwargs) -> None: ...
def fatal(template, **kwargs) -> None: ...
```

[Structured Logging](./structured-logging.md)

### Cron Monitoring

Scheduled job monitoring with automatic check-ins, failure detection, and alerting for cron jobs and scheduled tasks.

```python { .api }
def monitor(monitor_slug: str = None, **monitor_config) -> Callable: ...
def capture_checkin(
    monitor_slug: str = None,
    check_in_id: str = None,
    status: MonitorStatus = None,
    duration: float = None,
    **monitor_config
) -> str: ...
```

[Cron Monitoring](./cron-monitoring.md)

### Profiling

CPU profiling capabilities for performance analysis with support for different scheduling backends and continuous profiling.

```python { .api }
def start_profiler() -> None: ...
def stop_profiler() -> None: ...
class Profile:
    def __init__(self, transaction: Transaction): ...
    def finish(self) -> None: ...
```

[Profiling](./profiling.md)

### AI Monitoring

AI-native performance tracking and observability for artificial intelligence workflows, including LLM calls, AI pipeline execution, and token usage monitoring.

```python { .api }
def ai_track(description: str, **span_kwargs) -> Callable[[F], F]: ...
def set_ai_pipeline_name(name: Optional[str]) -> None: ...
def get_ai_pipeline_name() -> Optional[str]: ...
def record_token_usage(
    span: Span,
    input_tokens: Optional[int] = None,
    input_tokens_cached: Optional[int] = None,
    output_tokens: Optional[int] = None,
    output_tokens_reasoning: Optional[int] = None,
    total_tokens: Optional[int] = None
) -> None: ...
```

[AI Monitoring](./ai-monitoring.md)

## Core Classes

### Scope

Context container for events, spans, and metadata with hierarchical inheritance and isolated modification capabilities.

```python { .api }
class Scope:
    @staticmethod
    def get_current_scope() -> Scope: ...
    @staticmethod
    def get_isolation_scope() -> Scope: ...
    @staticmethod
    def get_global_scope() -> Scope: ...
    
    def set_user(self, value) -> None: ...
    def set_tag(self, key, value) -> None: ...
    def set_extra(self, key, value) -> None: ...
    def set_context(self, key, value) -> None: ...
    def add_breadcrumb(self, crumb=None, hint=None, **kwargs) -> None: ...
    def start_transaction(self, **kwargs) -> Union[Transaction, NoOpSpan]: ...
    def start_span(self, **kwargs) -> Span: ...
```

### Client

Primary interface for event transport and SDK configuration with support for custom transports, event processing, and integration management.

```python { .api }
class Client:
    def __init__(self, *args, **kwargs): ...
    def is_active(self) -> bool: ...
    def capture_event(self, event, hint=None, scope=None) -> Optional[str]: ...
    def capture_exception(self, error=None, scope=None, **scope_kwargs) -> Optional[str]: ...
    def capture_message(self, message, level=None, scope=None, **scope_kwargs) -> Optional[str]: ...
    def flush(self, timeout=None, callback=None) -> bool: ...
    def close(self, timeout=None, callback=None) -> bool: ...
```

### Transport

Event delivery mechanism with support for HTTP transport, custom backends, and envelope-based event batching.

```python { .api }
class Transport(ABC):
    @abstractmethod
    def capture_envelope(self, envelope) -> None: ...

class HttpTransport(Transport):
    def __init__(self, options): ...
    def capture_envelope(self, envelope) -> None: ...
```

## Utility Functions

```python { .api }
def flush(timeout=None, callback=None) -> bool: ...
def last_event_id() -> Optional[str]: ...
def get_traceparent() -> Optional[str]: ...
def get_baggage() -> Optional[str]: ...
def set_measurement(name, value, unit="") -> None: ...  # Deprecated
def set_transaction_name(name, source=None) -> None: ...
def update_current_span(op=None, name=None, attributes=None, data=None) -> None: ...
```

## Constants and Types

```python { .api }
# Version
VERSION: str = "2.36.0"

# Enums
class LogLevel:
    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"

class MonitorStatus:
    OK = "ok"
    ERROR = "error"
    IN_PROGRESS = "in_progress"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

class SPANTEMPLATE:
    DEFAULT = "default"
    AI_AGENT = "ai_agent"
    AI_TOOL = "ai_tool"
    AI_CHAT = "ai_chat"

class INSTRUMENTER:
    SENTRY = "sentry"
    OTEL = "otel"

class EndpointType:
    ENVELOPE = "envelope"

class CompressionAlgo:
    GZIP = "gzip"
    BROTLI = "br"

# Configuration Constants
DEFAULT_MAX_VALUE_LENGTH: int = 100_000
DEFAULT_MAX_BREADCRUMBS: int = 100
DEFAULT_QUEUE_SIZE: int = 100

# Type definitions
Breadcrumb = Dict[str, Any]
Event = Dict[str, Any]
Hint = Dict[str, Any]
ExcInfo = Tuple[Optional[type], Optional[BaseException], Optional[types.TracebackType]]
```