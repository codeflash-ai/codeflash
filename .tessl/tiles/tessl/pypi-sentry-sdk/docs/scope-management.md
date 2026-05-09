# Scope Management

Context management through a three-tier scope system providing isolation and data organization across global, request, and local levels with hierarchical inheritance and thread-safe operations.

## Capabilities

### Scope Access

Access the three levels of scopes for reading and modifying context data with proper isolation guarantees.

```python { .api }
def get_global_scope() -> Scope:
    """
    Get the global scope containing process-wide data.
    
    The global scope contains data that applies to all events in the process,
    such as release version, environment, and server-specific context.
    
    Returns:
    Scope: The global scope instance
    """

def get_isolation_scope() -> Scope:
    """
    Get the isolation scope containing request/user-level data.
    
    The isolation scope isolates data between different requests, users,
    or logical operations. Integrations typically manage this scope.
    
    Returns:
    Scope: The current isolation scope instance
    """

def get_current_scope() -> Scope:
    """
    Get the current local scope containing thread/context-specific data.
    
    The current scope contains data specific to the current execution context,
    such as span-specific tags, local breadcrumbs, and temporary context.
    
    Returns:
    Scope: The current local scope instance
    """
```

**Usage Examples:**

```python
import sentry_sdk

# Set global application data
global_scope = sentry_sdk.get_global_scope()
global_scope.set_tag("service", "payment-processor")
global_scope.set_context("app", {
    "version": "1.2.3",
    "build": "abc123",
    "environment": "production"
})

# Set request-specific data (typically done by integrations)
isolation_scope = sentry_sdk.get_isolation_scope()
isolation_scope.set_user({"id": "user_123", "email": "user@example.com"})
isolation_scope.set_tag("request_id", "req_456")

# Set local context data
current_scope = sentry_sdk.get_current_scope()
current_scope.set_tag("operation", "process_payment")
current_scope.add_breadcrumb({
    "message": "Starting payment validation",
    "level": "info",
    "category": "payment"
})
```

### Scope Context Managers

Create isolated scope contexts for temporary modifications without affecting parent scopes.

```python { .api }
def new_scope() -> ContextManager[Scope]:
    """
    Create a new local scope context manager.
    
    The new scope inherits from the current scope but modifications
    are isolated and don't affect the parent scope.
    
    Returns:
    ContextManager[Scope]: Context manager yielding the new scope
    """

def isolation_scope() -> ContextManager[Scope]:
    """
    Create a new isolation scope context manager.
    
    Creates a fresh isolation scope for request/user isolation.
    Typically used by web framework integrations.
    
    Returns:
    ContextManager[Scope]: Context manager yielding the isolation scope
    """
```

**Usage Examples:**

```python
import sentry_sdk

# Temporary scope for specific operation
def process_batch(items):
    with sentry_sdk.new_scope() as scope:
        scope.set_tag("batch_size", len(items))
        scope.set_extra("batch_id", "batch_123")
        
        for item in items:
            try:
                process_item(item)
            except Exception:
                # Exception captured with batch context
                sentry_sdk.capture_exception()

# Isolation for user request (typically done by web frameworks)
def handle_request(request):
    with sentry_sdk.isolation_scope() as scope:
        scope.set_user({
            "id": request.user.id,
            "email": request.user.email,
            "ip_address": request.remote_addr
        })
        scope.set_tag("endpoint", request.path)
        
        # All events in this context include user data
        return process_request(request)
```

## Scope Hierarchy

### Inheritance Model

Scopes follow a hierarchical inheritance model where child scopes inherit data from parent scopes:

1. **Global Scope** (bottom layer): Process-wide data
2. **Isolation Scope** (middle layer): Request/user data  
3. **Current Scope** (top layer): Local context data

When an event is captured, data is merged from all three scopes with higher levels taking precedence for conflicting keys.

### Scope Levels

**Global Scope:**
- Release version and environment
- Server and deployment information
- Application-wide configuration
- Global tags and context

**Isolation Scope:**
- User identification and session data
- Request metadata and correlation IDs
- Per-request configuration overrides
- Request-specific context

**Current Scope:**
- Span and transaction-specific data
- Local breadcrumbs and temporary context
- Function-level tags and metadata
- Short-lived contextual information

## Scope Class Interface

### Core Scope Methods

```python { .api }
class Scope:
    # User and identification
    def set_user(self, value: Optional[Dict[str, Any]]) -> None:
        """Set user information for events."""
    
    # Tags for filtering and searching
    def set_tag(self, key: str, value: str) -> None:
        """Set a single tag key-value pair."""
    
    def set_tags(self, tags: Dict[str, str]) -> None:
        """Set multiple tags at once."""
    
    def remove_tag(self, key: str) -> None:
        """Remove a tag by key."""
    
    # Extra data for debugging
    def set_extra(self, key: str, value: Any) -> None:
        """Set extra debug information."""
    
    def remove_extra(self, key: str) -> None:
        """Remove extra data by key."""
    
    # Structured context objects
    def set_context(self, key: str, value: Dict[str, Any]) -> None:
        """Set structured context data."""
    
    def remove_context(self, key: str) -> None:
        """Remove context data by key."""
    
    # Breadcrumb trail
    def add_breadcrumb(
        self,
        crumb: Optional[Dict[str, Any]] = None,
        hint: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Add a breadcrumb to the trail."""
    
    def clear_breadcrumbs(self) -> None:
        """Clear all breadcrumbs."""
    
    # Event capture
    def capture_exception(
        self,
        error: Optional[BaseException] = None,
        **kwargs
    ) -> Optional[str]:
        """Capture exception with this scope's context."""
    
    def capture_message(
        self,
        message: str,
        level: Optional[str] = None,
        **kwargs
    ) -> Optional[str]:
        """Capture message with this scope's context."""
    
    # Scope management
    def fork(self) -> Scope:
        """Create a copy of this scope."""
    
    def clear(self) -> None:
        """Clear all data from this scope."""
    
    def update_from_scope(self, scope: Scope) -> None:
        """Update this scope with data from another scope."""
```

### Scope Properties

```python { .api }
class Scope:
    @property
    def level(self) -> Optional[str]:
        """Current log level."""
    
    @level.setter
    def level(self, value: Optional[str]) -> None:
        """Set log level."""
    
    @property
    def user(self) -> Optional[Dict[str, Any]]:
        """Current user data."""
    
    @user.setter
    def user(self, value: Optional[Dict[str, Any]]) -> None:
        """Set user data."""
    
    @property
    def transaction(self) -> Optional[str]:
        """Current transaction name."""
    
    @transaction.setter
    def transaction(self, value: Optional[str]) -> None:
        """Set transaction name."""
    
    @property
    def span(self) -> Optional[Span]:
        """Current active span."""
    
    @span.setter
    def span(self, value: Optional[Span]) -> None:
        """Set active span."""
```

## Best Practices

### Scope Usage Patterns

**Global Scope:**
- Set once during application initialization
- Use for data that never changes during process lifetime
- Avoid frequent modifications (performance impact)

**Isolation Scope:**
- Managed primarily by framework integrations
- Use for request/user/session boundaries
- Clear between logical operations

**Current Scope:**
- Use for temporary, local context
- Leverage context managers for automatic cleanup
- Safe for frequent modifications

### Context Isolation

```python
# Good: Proper isolation for concurrent operations
async def handle_multiple_requests():
    tasks = []
    for request_data in requests:
        task = asyncio.create_task(process_with_isolation(request_data))
        tasks.append(task)
    await asyncio.gather(*tasks)

async def process_with_isolation(request_data):
    with sentry_sdk.isolation_scope() as scope:
        scope.set_user(request_data.user)
        scope.set_tag("request_id", request_data.id)
        # Process request with isolated context
        return await process_request(request_data)
```

### Memory Management

Scopes automatically manage memory by:
- Limiting breadcrumb storage (configurable via `max_breadcrumbs`)
- Cleaning up temporary scopes when context managers exit
- Garbage collecting unused scope references
- Implementing efficient copy-on-write for scope inheritance