# Integrations

Framework and library integrations for automatic instrumentation including web frameworks, databases, HTTP clients, task queues, AI/ML libraries, and 40+ other popular Python packages.

## Capabilities

### Integration System

The base integration system provides automatic setup and configuration for popular Python libraries and frameworks.

```python { .api }
class Integration(ABC):
    """Base class for all Sentry integrations."""
    
    identifier: str  # Unique integration identifier
    
    @staticmethod
    @abstractmethod
    def setup_once() -> None:
        """
        Initialize the integration. Called once when integration is enabled.
        This method performs monkey-patching, hooks installation, and other
        setup required for automatic instrumentation.
        """
```

### Web Framework Integrations

Automatic instrumentation for popular Python web frameworks with request tracking, error capture, and performance monitoring.

#### Django Integration

```python { .api }
class DjangoIntegration(Integration):
    def __init__(
        self,
        transaction_style: str = "url",
        middleware_spans: bool = True,
        signals_spans: bool = True,
        cache_spans: bool = True
    ): ...
```

**Usage:**

```python
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

sentry_sdk.init(
    dsn="your-dsn-here",
    integrations=[
        DjangoIntegration(
            transaction_style="url",    # 'url' or 'function_name'
            middleware_spans=True,      # Create spans for middleware
            signals_spans=True,         # Create spans for Django signals
            cache_spans=True           # Create spans for cache operations
        ),
    ]
)
```

#### Flask Integration

```python { .api }
class FlaskIntegration(Integration):
    def __init__(self, transaction_style: str = "endpoint"): ...
```

**Usage:**

```python
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn="your-dsn-here",
    integrations=[FlaskIntegration(transaction_style="endpoint")]
)
```

#### FastAPI Integration

```python { .api }
class FastApiIntegration(Integration):
    def __init__(self, transaction_style: str = "endpoint"): ...
```

### Database Integrations

Automatic instrumentation for database operations with query tracking, performance monitoring, and error capture.

#### SQLAlchemy Integration

```python { .api }
class SqlalchemyIntegration(Integration):
    def __init__(self): ...
```

**Usage:**

```python
import sentry_sdk
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

sentry_sdk.init(
    dsn="your-dsn-here",
    integrations=[SqlalchemyIntegration()]
)

# Automatic instrumentation for all SQLAlchemy operations
from sqlalchemy import create_engine
engine = create_engine("postgresql://user:pass@localhost/db")
# Database queries automatically tracked with performance data
```

#### AsyncPG Integration

```python { .api }
class AsyncPGIntegration(Integration):
    def __init__(self): ...
```

#### PyMongo Integration

```python { .api }
class PyMongoIntegration(Integration):
    def __init__(self): ...
```

### HTTP Client Integrations

Automatic instrumentation for HTTP clients with request/response tracking and distributed tracing support.

#### HTTP Client Instrumentation (via Stdlib)

HTTP client libraries like `requests` and `urllib3` are automatically instrumented through the StdlibIntegration which hooks into the standard library's `http.client` module.

#### HTTPX Integration

```python { .api }
class HttpxIntegration(Integration):
    def __init__(self): ...
```

**Usage:**

```python
import sentry_sdk
from sentry_sdk.integrations.httpx import HttpxIntegration

sentry_sdk.init(
    dsn="your-dsn-here",
    integrations=[HttpxIntegration()]
)

import httpx
# HTTP requests automatically tracked with distributed tracing
async with httpx.AsyncClient() as client:
    response = await client.get("https://api.example.com/data")
```

### Task Queue Integrations

Automatic instrumentation for background task processing with job tracking and error monitoring.

#### Celery Integration

```python { .api }
class CeleryIntegration(Integration):
    def __init__(
        self,
        monitor_beat_tasks: bool = False,
        propagate_traces: bool = True
    ): ...
```

**Usage:**

```python
import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration

sentry_sdk.init(
    dsn="your-dsn-here",
    integrations=[
        CeleryIntegration(
            monitor_beat_tasks=True,   # Monitor periodic tasks
            propagate_traces=True      # Enable distributed tracing
        )
    ]
)

# Celery tasks automatically tracked
from celery import Celery
app = Celery('tasks')

@app.task
def process_data(data_id):
    # Task execution automatically monitored
    return process(data_id)
```

#### RQ Integration

```python { .api }
class RqIntegration(Integration):
    def __init__(self): ...
```

#### Arq Integration

```python { .api }
class ArqIntegration(Integration):
    def __init__(self): ...
```

### AI/ML Library Integrations

Automatic instrumentation for AI and machine learning libraries with operation tracking and token usage monitoring.

#### OpenAI Integration

```python { .api }
class OpenAIIntegration(Integration):
    def __init__(
        self,
        include_prompts: bool = True,
        include_completions: bool = True
    ): ...
```

**Usage:**

```python
import sentry_sdk
from sentry_sdk.integrations.openai import OpenAIIntegration

sentry_sdk.init(
    dsn="your-dsn-here",
    integrations=[
        OpenAIIntegration(
            include_prompts=True,      # Include prompt data in spans
            include_completions=True   # Include completion data in spans
        )
    ]
)

import openai
# OpenAI API calls automatically tracked with token usage and timing
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

#### Anthropic Integration

```python { .api }
class AnthropicIntegration(Integration):
    def __init__(
        self,
        include_prompts: bool = True,
        include_completions: bool = True
    ): ...
```

#### LangChain Integration

```python { .api }
class LangchainIntegration(Integration):
    def __init__(self): ...
```

#### Hugging Face Hub Integration

```python { .api }
class HuggingfaceHubIntegration(Integration):
    def __init__(self): ...
```

### Cloud and Infrastructure Integrations

Integrations for cloud services and infrastructure components.

#### AWS Lambda Integration

```python { .api }
class AwsLambdaIntegration(Integration):
    def __init__(self, timeout_warning: bool = False): ...
```

#### Boto3 Integration

```python { .api }
class Boto3Integration(Integration):
    def __init__(self): ...
```

### ASGI/WSGI Integrations

Low-level integrations for ASGI and WSGI applications.

#### ASGI Integration

```python { .api }
class AsgiIntegration(Integration):
    def __init__(
        self,
        transaction_style: str = "endpoint",
        middleware_spans: bool = True
    ): ...
```

#### WSGI Integration

```python { .api }
class WsgiIntegration(Integration):
    def __init__(
        self,
        transaction_style: str = "endpoint"
    ): ...
```

### Logging Integrations

Enhanced logging with automatic error capture and log correlation.

#### Logging Integration

```python { .api }
class LoggingIntegration(Integration):
    def __init__(
        self,
        level: int = logging.INFO,
        event_level: int = logging.ERROR
    ): ...
```

**Usage:**

```python
import sentry_sdk
import logging
from sentry_sdk.integrations.logging import LoggingIntegration

sentry_sdk.init(
    dsn="your-dsn-here",
    integrations=[
        LoggingIntegration(
            level=logging.INFO,    # Capture info and above as breadcrumbs
            event_level=logging.ERROR  # Send error and above as events
        )
    ]
)

# Standard logging automatically integrated
logger = logging.getLogger(__name__)
logger.info("This becomes a breadcrumb")
logger.error("This becomes a Sentry event")
```

#### Loguru Integration

```python { .api }
class LoguruIntegration(Integration):
    def __init__(self): ...
```

## Default Integrations

### Automatically Enabled

These integrations are enabled by default when `default_integrations=True`:

- **ArgvIntegration**: Command-line arguments capture
- **AtexitIntegration**: Graceful shutdown handling
- **DedupeIntegration**: Duplicate event prevention
- **ExcepthookIntegration**: Uncaught exception handling
- **LoggingIntegration**: Standard library logging
- **ModulesIntegration**: Installed packages information
- **StdlibIntegration**: Standard library instrumentation
- **ThreadingIntegration**: Threading support

### Auto-Enabling Integrations

These integrations automatically enable when their corresponding libraries are detected:

Web Frameworks: Django, Flask, FastAPI, Starlette, Sanic, Tornado, Pyramid, Bottle, Falcon, Quart, Litestar

Databases: SQLAlchemy, AsyncPG, PyMongo, Redis, ClickHouse Driver

HTTP Clients: HTTPX, AIOHTTP

Task Queues: Celery, RQ, Arq, Huey, Dramatiq

AI/ML: OpenAI, Anthropic, Cohere, LangChain, Hugging Face Hub

Cloud: AWS Lambda (Boto3)

And many more...

## Integration Configuration

### Selective Integration Control

```python
import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration
from sentry_sdk.integrations.redis import RedisIntegration

sentry_sdk.init(
    dsn="your-dsn-here",
    # Enable only specific integrations
    integrations=[
        DjangoIntegration(),
        RedisIntegration(),
    ],
    default_integrations=False,      # Disable default integrations
    auto_enabling_integrations=False # Disable auto-detection
)
```

### Disable Specific Integrations

```python
import sentry_sdk
from sentry_sdk.integrations.httpx import HttpxIntegration

sentry_sdk.init(
    dsn="your-dsn-here",
    disabled_integrations=[HttpxIntegration]  # Disable httpx integration
)
```

## Custom Integration Development

### Creating Custom Integrations

```python
from sentry_sdk.integrations import Integration
from sentry_sdk.utils import capture_internal_exceptions

class MyCustomIntegration(Integration):
    identifier = "my_custom"
    
    def __init__(self, option1=True, option2="default"):
        self.option1 = option1
        self.option2 = option2
    
    @staticmethod
    def setup_once():
        # This method is called once when the integration is enabled
        import my_library
        
        original_method = my_library.important_function
        
        def sentry_wrapped_method(*args, **kwargs):
            with capture_internal_exceptions():
                # Add Sentry instrumentation
                with sentry_sdk.start_span(
                    op="my_library.operation",
                    description="important_function"
                ):
                    return original_method(*args, **kwargs)
        
        my_library.important_function = sentry_wrapped_method
```

**Usage:**

```python
sentry_sdk.init(
    dsn="your-dsn-here",
    integrations=[MyCustomIntegration(option1=False, option2="custom")]
)
```

Integration capabilities provide comprehensive automatic instrumentation across the Python ecosystem, enabling detailed observability and error tracking with minimal configuration effort.