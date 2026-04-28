# Context Management

Thread-safe context system for automatic user identification, session tracking, and property tagging. PostHog's context management enables consistent user tracking across related operations while providing automatic exception capture and nested context support.

## Capabilities

### Context Creation

Create isolated contexts for grouping related operations with automatic cleanup and exception handling.

```python { .api }
def new_context(fresh: bool = False, capture_exceptions: bool = True):
    """
    Create a new context scope that will be active for the duration of the with block.

    Parameters:
    - fresh: bool - Whether to start with a fresh context (default: False)
    - capture_exceptions: bool - Whether to capture exceptions raised within the context (default: True)

    Returns:
    Context manager for use with 'with' statement

    Notes:
    - Creates isolated scope for user identification and tags
    - Automatically captures exceptions if enabled
    - Context data is inherited by child contexts unless fresh=True
    - Thread-safe for concurrent operations
    """
```

### Function-Level Contexts

Apply context management to individual functions using decorators for automatic scope management.

```python { .api }
def scoped(fresh: bool = False, capture_exceptions: bool = True):
    """
    Decorator that creates a new context for the function.

    Parameters:
    - fresh: bool - Whether to start with a fresh context (default: False)
    - capture_exceptions: bool - Whether to capture and track exceptions with posthog error tracking (default: True)

    Returns:
    Function decorator

    Notes:
    - Automatically creates context for decorated function
    - Cleans up context when function exits
    - Preserves function return values and exceptions
    - Suitable for background tasks, request handlers, etc.
    """
```

### User Identification

Associate contexts with specific users for automatic user tracking across all operations.

```python { .api }
def identify_context(distinct_id: str):
    """
    Identify the current context with a distinct ID.

    Parameters:
    - distinct_id: str - The distinct ID to associate with the current context and its children

    Notes:
    - Sets user ID for all subsequent operations in context
    - Inherited by child contexts
    - Used automatically by capture, set, and other operations
    - Overrides any parent context user ID
    """
```

### Session Management

Manage user sessions within contexts for tracking related user activities across time.

```python { .api }
def set_context_session(session_id: str):
    """
    Set the session ID for the current context.

    Parameters:
    - session_id: str - The session ID to associate with the current context and its children

    Notes:
    - Automatically included in all events as $session_id property
    - Inherited by child contexts
    - Used for session-based analytics and user journey tracking
    """
```

### Context Tagging

Add metadata and properties to contexts that are automatically included in all events and operations.

```python { .api }
def tag(name: str, value: Any):
    """
    Add a tag to the current context.

    Parameters:
    - name: str - The tag key
    - value: Any - The tag value (must be JSON serializable)

    Notes:
    - Tags are included in all subsequent events in the context
    - Inherited by child contexts
    - Useful for request IDs, feature flags, A/B test groups, etc.
    - Automatically merged with event properties
    """
```

## Usage Examples

### Basic Context Usage

```python
import posthog

# Configure PostHog
posthog.api_key = 'phc_your_project_api_key'

# Simple context with user identification
with posthog.new_context():
    posthog.identify_context('user123')
    
    # All events automatically include user123
    posthog.capture('page_viewed', {'page': 'dashboard'})
    posthog.capture('button_clicked', {'button': 'export'})
    posthog.set({'last_active': '2024-09-07'})

# Context with session tracking
with posthog.new_context():
    posthog.identify_context('user456')
    posthog.set_context_session('session_abc123')
    
    # Events include both user ID and session ID
    posthog.capture('session_started')
    posthog.capture('feature_used', {'feature': 'reports'})
    posthog.capture('session_ended')
```

### Context Tagging

```python
import posthog

# Context with multiple tags
with posthog.new_context():
    posthog.identify_context('user789')
    posthog.tag('request_id', 'req_12345')
    posthog.tag('user_segment', 'premium')
    posthog.tag('ab_test_group', 'variant_b')
    posthog.tag('feature_flag_new_ui', True)
    
    # All events automatically include these tags
    posthog.capture('api_request', {
        'endpoint': '/api/data',
        'method': 'GET'
    })
    
    # Tags are merged with explicit properties
    posthog.capture('error_occurred', {
        'error_type': 'validation',
        'error_code': 400
    })
    # Final event includes: user_id, request_id, user_segment, ab_test_group, 
    # feature_flag_new_ui, error_type, error_code
```

### Nested Contexts

```python
import posthog

# Parent context
with posthog.new_context():
    posthog.identify_context('user123')
    posthog.tag('request_type', 'api')
    
    posthog.capture('request_started')
    
    # Child context inherits parent data
    with posthog.new_context():
        posthog.tag('operation', 'data_processing')
        
        posthog.capture('processing_started')
        # Includes: user123, request_type=api, operation=data_processing
        
        # Another child context
        with posthog.new_context():
            posthog.tag('step', 'validation')
            
            posthog.capture('validation_completed')
            # Includes: user123, request_type=api, operation=data_processing, step=validation
    
    posthog.capture('request_completed')
    # Back to parent context: user123, request_type=api
```

### Fresh Contexts

```python
import posthog

# Parent context with data
with posthog.new_context():
    posthog.identify_context('user123')
    posthog.tag('parent_tag', 'value')
    
    # Fresh context ignores parent data
    with posthog.new_context(fresh=True):
        posthog.identify_context('admin456')
        posthog.tag('admin_operation', True)
        
        posthog.capture('admin_action')
        # Only includes: admin456, admin_operation=True
        # Does NOT include user123 or parent_tag
    
    # Back to parent context
    posthog.capture('user_action')
    # Includes: user123, parent_tag=value
```

### Function-Level Contexts

```python
import posthog

# Decorator for automatic context management
@posthog.scoped()
def process_user_request(user_id, request_data):
    posthog.identify_context(user_id)
    posthog.tag('operation', 'user_request')
    posthog.tag('request_id', request_data.get('id'))
    
    posthog.capture('request_processing_started')
    
    # Process request...
    result = handle_request(request_data)
    
    posthog.capture('request_processing_completed', {
        'success': result['success'],
        'processing_time': result['duration']
    })
    
    return result

# Background task with fresh context
@posthog.scoped(fresh=True)
def background_cleanup_task():
    posthog.identify_context('system')
    posthog.tag('task_type', 'cleanup')
    
    posthog.capture('cleanup_started')
    
    # Perform cleanup...
    cleaned_items = perform_cleanup()
    
    posthog.capture('cleanup_completed', {
        'items_cleaned': len(cleaned_items)
    })

# Usage
user_data = {'id': 'req_123', 'user': 'user456'}
process_user_request('user456', user_data)

# Run background task
background_cleanup_task()
```

### Exception Handling with Contexts

```python
import posthog

# Automatic exception capture (default behavior)
with posthog.new_context():
    posthog.identify_context('user123')
    posthog.tag('operation', 'risky_operation')
    
    try:
        posthog.capture('operation_started')
        
        # This exception is automatically captured by context
        raise ValueError("Something went wrong")
        
    except ValueError as e:
        # Exception was already captured by context
        posthog.capture('operation_failed', {
            'error_handled': True
        })

# Disable automatic exception capture
with posthog.new_context(capture_exceptions=False):
    posthog.identify_context('user456')
    
    try:
        dangerous_operation()
    except Exception as e:
        # Manual exception capture since auto-capture is disabled
        posthog.capture_exception(e)

# Function-level exception handling
@posthog.scoped(capture_exceptions=True)
def risky_function():
    posthog.identify_context('user789')
    posthog.capture('function_started')
    
    # Any exception here is automatically captured
    raise RuntimeError("Function failed")

try:
    risky_function()
except RuntimeError:
    # Exception was already captured by the scoped decorator
    pass
```

### Web Request Context Pattern

```python
import posthog
from flask import Flask, request, g

app = Flask(__name__)

@app.before_request
def before_request():
    # Create context for each request
    g.posthog_context = posthog.new_context()
    g.posthog_context.__enter__()
    
    # Set up request-level tracking
    posthog.tag('request_id', request.headers.get('X-Request-ID', 'unknown'))
    posthog.tag('user_agent', request.headers.get('User-Agent', 'unknown'))
    posthog.tag('endpoint', request.endpoint)
    
    # Identify user if available
    if hasattr(g, 'current_user') and g.current_user:
        posthog.identify_context(g.current_user.id)

@app.after_request
def after_request(response):
    # Tag response information
    posthog.tag('response_status', response.status_code)
    
    # Capture request completion
    posthog.capture('request_completed', {
        'method': request.method,
        'endpoint': request.endpoint,
        'status_code': response.status_code
    })
    
    # Clean up context
    if hasattr(g, 'posthog_context'):
        g.posthog_context.__exit__(None, None, None)
    
    return response

@app.route('/api/users/<user_id>')
def get_user(user_id):
    posthog.capture('user_profile_viewed', {'viewed_user_id': user_id})
    
    # Process request...
    return {'user': user_id}

# Each request automatically gets its own context with request-level tags
```

### Async Context Management

```python
import posthog
import asyncio

async def async_operation():
    # Contexts work with async operations
    with posthog.new_context():
        posthog.identify_context('async_user_123')
        posthog.tag('operation_type', 'async')
        
        posthog.capture('async_operation_started')
        
        # Simulate async work
        await asyncio.sleep(1)
        
        posthog.capture('async_operation_completed')

# Run async operation
asyncio.run(async_operation())

# Async function decorator
@posthog.scoped()
async def async_task(task_data):
    posthog.identify_context(task_data['user_id'])
    posthog.tag('task_id', task_data['id'])
    
    posthog.capture('async_task_started')
    
    await process_task(task_data)
    
    posthog.capture('async_task_completed')

# Usage
await async_task({'id': 'task_123', 'user_id': 'user_456'})
```

## Context Hierarchy and Inheritance

### Data Inheritance Rules

```python
import posthog

# Parent context
with posthog.new_context():
    posthog.identify_context('parent_user')
    posthog.tag('level', 'parent')
    posthog.tag('shared', 'parent_value')
    
    # Child context inherits parent data
    with posthog.new_context():
        posthog.tag('level', 'child')  # Overrides parent tag
        posthog.tag('child_only', 'child_value')  # New tag
        
        # Event includes:
        # - user: parent_user (inherited)
        # - level: child (overridden)
        # - shared: parent_value (inherited)
        # - child_only: child_value (new)
        posthog.capture('child_event')
    
    # Back to parent - child tags are gone
    # Event includes:
    # - user: parent_user
    # - level: parent
    # - shared: parent_value
    posthog.capture('parent_event')
```

### Context Isolation

```python
import posthog

# Sibling contexts are isolated
with posthog.new_context():
    posthog.identify_context('parent_user')
    
    # First child
    with posthog.new_context():
        posthog.tag('branch', 'first')
        posthog.capture('first_child_event')
    
    # Second child (no access to first child's data)
    with posthog.new_context():
        posthog.tag('branch', 'second')
        posthog.capture('second_child_event')  # Does NOT include branch=first
```

## Best Practices

### Context Scope Management

```python
# Good - Clear context boundaries
with posthog.new_context():
    posthog.identify_context('user123')
    process_user_request()

# Good - Function-level contexts for discrete operations
@posthog.scoped()
def handle_api_endpoint():
    posthog.identify_context(get_current_user())
    # Process request

# Avoid - Contexts that span too long
# with posthog.new_context():  # Don't do this
#     for user in all_users:  # This context lasts too long
#         process_user(user)
```

### Tag Naming and Organization

```python
# Good - Consistent, descriptive tag names
with posthog.new_context():
    posthog.tag('request_id', 'req_123')
    posthog.tag('user_segment', 'premium')
    posthog.tag('feature_flag_new_ui', True)
    posthog.tag('ab_test_variant', 'control')

# Avoid - Inconsistent or unclear tag names
with posthog.new_context():
    posthog.tag('reqId', 'req_123')  # Inconsistent naming
    posthog.tag('flag1', True)  # Unclear purpose
    posthog.tag('test', 'control')  # Too generic
```

### Exception Handling Strategy

```python
# Enable exception capture for user-facing operations
@posthog.scoped(capture_exceptions=True)
def user_request_handler():
    # Exceptions automatically captured
    pass

# Disable for internal/system operations where exceptions are expected
@posthog.scoped(capture_exceptions=False)
def system_health_check():
    # Handle exceptions manually
    try:
        check_system()
    except ExpectedError as e:
        # Don't capture expected errors
        pass
    except UnexpectedError as e:
        posthog.capture_exception(e)
```

### Thread Safety

PostHog contexts are thread-safe and work correctly in multi-threaded applications:

```python
import posthog
import threading

def worker_thread(worker_id):
    # Each thread gets its own context
    with posthog.new_context():
        posthog.identify_context(f'worker_{worker_id}')
        posthog.tag('thread_id', threading.current_thread().ident)
        
        posthog.capture('worker_started')
        # Do work...
        posthog.capture('worker_completed')

# Start multiple worker threads
threads = []
for i in range(5):
    t = threading.Thread(target=worker_thread, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
```