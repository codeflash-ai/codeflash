# PostHog Python SDK

A comprehensive Python SDK for PostHog, providing developer-friendly integration of event tracking, feature flags, user identification, exception capture, and context management. The SDK offers both synchronous and asynchronous APIs with support for automatic session tracking, AI/LLM integrations (including OpenAI, Anthropic, Langchain), and comprehensive error handling with automatic retries, batching, offline queuing, and Django integration.

## Package Information

- **Package Name**: posthog
- **Language**: Python
- **Installation**: `pip install posthog`
- **Python Version**: 3.9+

## Core Imports

```python
import posthog
```

For direct client usage:

```python
from posthog import Posthog
```

For AI integrations:

```python
from posthog.ai.openai import OpenAI
from posthog.ai.anthropic import Anthropic
from posthog.ai.gemini import Client as GeminiClient, genai
from posthog.ai.langchain import CallbackHandler
```

For Django integration:

```python
from posthog.integrations.django import PosthogContextMiddleware
```

## Basic Usage

```python
import posthog

# Configure PostHog
posthog.api_key = 'phc_your_project_api_key'
posthog.host = 'https://app.posthog.com'  # or your self-hosted instance

# Track events
posthog.capture('user123', 'button_clicked', {
    'button_name': 'signup',
    'page': 'landing'
})

# Set user properties
posthog.set('user123', {
    'email': 'user@example.com',
    'plan': 'premium'
})

# Use feature flags
if posthog.feature_enabled('new-checkout', 'user123'):
    # Show new checkout flow
    pass

# Context-based tracking
with posthog.new_context():
    posthog.identify_context('user123')
    posthog.tag('session_type', 'premium')
    posthog.capture('purchase_completed', {'amount': 99.99})
```

## Architecture

PostHog Python SDK follows a context-aware architecture:

- **Global Module Interface**: Simplified API for quick integration using module-level functions
- **Client Instance**: Direct client instantiation for advanced configuration and multi-tenant usage
- **Context Management**: Thread-safe context system for automatic user identification and tagging
- **AI Integrations**: Wrapper clients for popular LLM providers with automatic usage tracking
- **Feature Flag System**: Local and remote evaluation with caching and fallback mechanisms

The SDK supports both fire-and-forget event tracking and comprehensive analytics workflows, with automatic batching, retry logic, and offline support for production deployments.

## Capabilities

### Event Tracking

Core event capture functionality for tracking user actions, system events, and custom analytics data with support for properties, timestamps, and automatic context enrichment.

```python { .api }
def capture(event: str, **kwargs: OptionalCaptureArgs) -> Optional[str]: ...
def capture_exception(exception: Optional[ExceptionArg] = None, **kwargs: OptionalCaptureArgs): ...
```

[Event Tracking](./event-tracking.md)

### User and Group Management

User identification, property management, and group associations for organizing users and tracking organizational-level data with support for both user and group properties.

```python { .api }
def set(**kwargs: OptionalSetArgs) -> Optional[str]: ...
def set_once(**kwargs: OptionalSetArgs) -> Optional[str]: ...
def group_identify(group_type: str, group_key: str, properties: Optional[Dict], timestamp: Optional[datetime], uuid: Optional[str], disable_geoip: Optional[bool]) -> Optional[str]: ...
def alias(previous_id: str, distinct_id: str, timestamp: Optional[datetime], uuid: Optional[str], disable_geoip: Optional[bool]) -> Optional[str]: ...
```

[User and Group Management](./user-group-management.md)

### Feature Flags

Feature flag evaluation system supporting boolean flags, multivariate testing, remote configuration, and both local and remote evaluation with caching and fallback support.

```python { .api }
def feature_enabled(key: str, distinct_id: str, groups: Optional[dict] = None, person_properties: Optional[dict] = None, group_properties: Optional[dict] = None, only_evaluate_locally: bool = False, send_feature_flag_events: bool = True, disable_geoip: Optional[bool] = None) -> bool: ...
def get_feature_flag(key: str, distinct_id: str, groups: Optional[dict] = None, person_properties: Optional[dict] = None, group_properties: Optional[dict] = None, only_evaluate_locally: bool = False, send_feature_flag_events: bool = True, disable_geoip: Optional[bool] = None) -> Optional[FeatureFlag]: ...
def get_all_flags(distinct_id: str, groups: Optional[dict] = None, person_properties: Optional[dict] = None, group_properties: Optional[dict] = None, only_evaluate_locally: bool = False, disable_geoip: Optional[bool] = None) -> Optional[dict[str, FeatureFlag]]: ...
```

[Feature Flags](./feature-flags.md)

### Context Management

Thread-safe context system for automatic user identification, session tracking, and property tagging with support for nested contexts and exception capture.

```python { .api }
def new_context(fresh: bool = False, capture_exceptions: bool = True): ...
def scoped(fresh: bool = False, capture_exceptions: bool = True): ...
def identify_context(distinct_id: str): ...
def set_context_session(session_id: str): ...
def tag(name: str, value: Any): ...
```

[Context Management](./context-management.md)

### Client Management

Client lifecycle management including initialization, configuration, batching control, and graceful shutdown with support for both global and instance-based usage.

```python { .api }
class Client:
    def __init__(self, project_api_key: str, host: Optional[str] = None, debug: bool = False, max_queue_size: int = 10000, send: bool = True, on_error: Optional[Callable] = None, flush_at: int = 100, flush_interval: float = 0.5, gzip: bool = False, max_retries: int = 3, sync_mode: bool = False, timeout: int = 15, thread: int = 1, poll_interval: int = 30, personal_api_key: Optional[str] = None, disabled: bool = False, disable_geoip: bool = True, historical_migration: bool = False, feature_flags_request_timeout_seconds: int = 3, super_properties: Optional[Dict] = None, enable_exception_autocapture: bool = False, log_captured_exceptions: bool = False, project_root: Optional[str] = None, privacy_mode: bool = False, before_send: Optional[Callable] = None, flag_fallback_cache_url: Optional[str] = None, enable_local_evaluation: bool = True): ...

def flush(): ...
def join(): ...
def shutdown(): ...
```

[Client Management](./client-management.md)

### AI Integrations

LLM provider integrations with automatic usage tracking, cost monitoring, and performance analytics for OpenAI, Anthropic, Gemini, and Langchain with support for both sync and async operations.

```python { .api }
# OpenAI Integration
class OpenAI: ...
class AsyncOpenAI: ...
class AzureOpenAI: ...
class AsyncAzureOpenAI: ...

# Anthropic Integration  
class Anthropic: ...
class AsyncAnthropic: ...
class AnthropicBedrock: ...
class AsyncAnthropicBedrock: ...

# Gemini Integration
class Client: ...  # Via posthog.ai.gemini
genai: _GenAI  # Module compatibility

# Langchain Integration
class CallbackHandler: ...
```

[AI Integrations](./ai-integrations.md)

### Django Integration

Django middleware for automatic request tracking with context management, session identification, and exception capture.

```python { .api }
class PosthogContextMiddleware:
    """
    Django middleware for automatic PostHog integration.
    
    Automatically wraps requests with PostHog contexts and extracts:
    - Session ID from X-POSTHOG-SESSION-ID header
    - Distinct ID from X-POSTHOG-DISTINCT-ID header
    - Request URL and method as properties
    - Automatic exception capture (configurable)
    
    Configurable via Django settings:
    - POSTHOG_MW_CAPTURE_EXCEPTIONS: Enable/disable exception capture
    - POSTHOG_MW_CLIENT: Custom client instance
    - POSTHOG_MW_EXTRA_TAGS: Function for additional context tags
    - POSTHOG_MW_REQUEST_FILTER: Function to filter requests
    - POSTHOG_MW_TAG_MAP: Function to modify tags before context
    """
```

## Types

### Core Types

```python { .api }
# Type aliases
FlagValue = Union[bool, str]
BeforeSendCallback = Callable[[dict[str, Any]], Optional[dict[str, Any]]]
ID_TYPES = Union[numbers.Number, str, UUID, int]
ExceptionArg = Union[BaseException, ExcInfo]

# Feature flag types
@dataclass(frozen=True)
class FeatureFlag:
    key: str
    enabled: bool
    variant: Optional[str]
    reason: Optional[FlagReason]
    metadata: Union[FlagMetadata, LegacyFlagMetadata]
    
    def get_value(self) -> FlagValue: ...

@dataclass(frozen=True)
class FeatureFlagResult:
    key: str
    enabled: bool
    variant: Optional[str]
    payload: Optional[Any]
    reason: Optional[str]
    
    def get_value(self) -> FlagValue: ...

@dataclass(frozen=True)
class FlagReason:
    code: str
    condition_index: Optional[int]
    description: str

@dataclass(frozen=True)
class FlagMetadata:
    id: int
    payload: Optional[str]
    version: int
    description: str

# Argument types
class OptionalCaptureArgs(TypedDict):
    distinct_id: NotRequired[Optional[ID_TYPES]]
    properties: NotRequired[Optional[Dict[str, Any]]]
    timestamp: NotRequired[Optional[Union[datetime, str]]]
    uuid: NotRequired[Optional[str]]
    groups: NotRequired[Optional[Dict[str, str]]]
    send_feature_flags: NotRequired[Optional[Union[bool, SendFeatureFlagsOptions]]]
    disable_geoip: NotRequired[Optional[bool]]

class OptionalSetArgs(TypedDict):
    distinct_id: NotRequired[Optional[ID_TYPES]]
    properties: NotRequired[Optional[Dict[str, Any]]]
    timestamp: NotRequired[Optional[Union[datetime, str]]]
    uuid: NotRequired[Optional[str]]
    disable_geoip: NotRequired[Optional[bool]]

class SendFeatureFlagsOptions(TypedDict, total=False):
    should_send: bool
    only_evaluate_locally: Optional[bool]
    person_properties: Optional[dict[str, Any]]
    group_properties: Optional[dict[str, dict[str, Any]]]
    flag_keys_filter: Optional[list[str]]
```

## Global Configuration

```python { .api }
# Core configuration
api_key: Optional[str]
host: Optional[str]
debug: bool
send: bool
sync_mode: bool
disabled: bool

# Advanced configuration
personal_api_key: Optional[str]
project_api_key: Optional[str]
poll_interval: int
disable_geoip: bool
feature_flags_request_timeout_seconds: int
super_properties: Optional[Dict]

# Exception handling
enable_exception_autocapture: bool
log_captured_exceptions: bool
project_root: Optional[str]

# Privacy and evaluation
privacy_mode: bool
enable_local_evaluation: bool

# Error handling
on_error: Optional[Callable]
```