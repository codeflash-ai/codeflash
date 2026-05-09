# Client Management

Client lifecycle management including initialization, configuration, batching control, and graceful shutdown. PostHog's client management supports both global module-level usage and direct client instantiation for advanced configuration and multi-tenant applications.

## Capabilities

### Client Initialization

Create and configure PostHog client instances with comprehensive options for production deployments.

```python { .api }
class Client:
    def __init__(
        self,
        project_api_key: str,
        host: Optional[str] = None,
        debug: bool = False,
        max_queue_size: int = 10000,
        send: bool = True,
        on_error: Optional[Callable] = None,
        flush_at: int = 100,
        flush_interval: float = 0.5,
        gzip: bool = False,
        max_retries: int = 3,
        sync_mode: bool = False,
        timeout: int = 15,
        thread: int = 1,
        poll_interval: int = 30,
        personal_api_key: Optional[str] = None,
        disabled: bool = False,
        disable_geoip: bool = True,
        historical_migration: bool = False,
        feature_flags_request_timeout_seconds: int = 3,
        super_properties: Optional[Dict] = None,
        enable_exception_autocapture: bool = False,
        log_captured_exceptions: bool = False,
        project_root: Optional[str] = None,
        privacy_mode: bool = False,
        before_send: Optional[Callable] = None,
        flag_fallback_cache_url: Optional[str] = None,
        enable_local_evaluation: bool = True
    ):
        """
        Initialize a new PostHog client instance.

        Parameters:
        - project_api_key: str - The project API key
        - host: Optional[str] - The PostHog host URL (default: https://app.posthog.com)
        - debug: bool - Enable debug logging (default: False)
        - max_queue_size: int - Maximum events in queue before dropping (default: 10000)
        - send: bool - Whether to send events to PostHog (default: True)
        - on_error: Optional[Callable] - Error callback function
        - flush_at: int - Number of events to trigger batch send (default: 100)
        - flush_interval: float - Seconds between automatic flushes (default: 0.5)
        - gzip: bool - Enable gzip compression (default: False)
        - max_retries: int - Maximum retry attempts for failed requests (default: 3)
        - sync_mode: bool - Send events synchronously (default: False)
        - timeout: int - Request timeout in seconds (default: 15)
        - thread: int - Number of background threads (default: 1)
        - poll_interval: int - Feature flag polling interval in seconds (default: 30)
        - personal_api_key: Optional[str] - Personal API key for advanced features
        - disabled: bool - Disable all PostHog functionality (default: False)
        - disable_geoip: bool - Disable GeoIP lookup (default: True)
        - historical_migration: bool - Enable historical migration mode (default: False)
        - feature_flags_request_timeout_seconds: int - Feature flag request timeout (default: 3)
        - super_properties: Optional[Dict] - Properties added to all events
        - enable_exception_autocapture: bool - Auto-capture exceptions (default: False)
        - log_captured_exceptions: bool - Log captured exceptions (default: False)
        - project_root: Optional[str] - Project root for exception capture
        - privacy_mode: bool - Privacy mode for AI features (default: False)
        - before_send: Optional[Callable] - Event preprocessing callback
        - flag_fallback_cache_url: Optional[str] - Redis URL for flag caching
        - enable_local_evaluation: bool - Enable local flag evaluation (default: True)

        Notes:
        - project_api_key is required and should start with 'phc_'
        - host should include protocol (https://)
        - sync_mode blocks until events are sent
        - disabled=True makes all operations no-ops
        """

# Alias for backward compatibility
class Posthog(Client):
    pass
```

### Queue and Batch Management

Control event batching, queue management, and forced flushing for optimal performance and reliability.

```python { .api }
def flush():
    """
    Tell the client to flush all queued events.

    Notes:
    - Forces immediate sending of all queued events
    - Blocks until all events are sent or fail
    - Useful before application shutdown
    - Safe to call multiple times
    """

def join():
    """
    Block program until the client clears the queue. Used during program shutdown.

    Notes:
    - Waits for background threads to finish processing
    - Does not send events, use flush() first
    - Should be called before application exit
    - Use shutdown() for combined flush + join
    """

def shutdown():
    """
    Flush all messages and cleanly shutdown the client.

    Notes:
    - Combines flush() and join() operations
    - Recommended for application shutdown
    - Ensures all events are sent before exit
    - Stops background threads
    """
```

## Usage Examples

### Basic Client Setup

```python
import posthog

# Global configuration (simplest approach)
posthog.api_key = 'phc_your_project_api_key'
posthog.host = 'https://app.posthog.com'  # or your self-hosted instance

# Use module-level functions
posthog.capture('user123', 'event_name')
posthog.set('user123', {'property': 'value'})

# Shutdown when application exits
import atexit
atexit.register(posthog.shutdown)
```

### Direct Client Instantiation

```python
from posthog import Posthog

# Create client instance with custom configuration
client = Posthog(
    project_api_key='phc_your_project_api_key',
    host='https://app.posthog.com',
    debug=False,
    flush_at=50,
    flush_interval=1.0,
    max_retries=5,
    timeout=30
)

# Use client instance methods
client.capture('user123', 'event_name')
client.set('user123', {'property': 'value'})

# Shutdown client when done
client.shutdown()
```

### Production Configuration

```python
from posthog import Posthog
import logging

# Production-ready configuration
def create_posthog_client():
    return Posthog(
        project_api_key='phc_your_project_api_key',
        host='https://app.posthog.com',
        
        # Performance settings
        flush_at=200,           # Batch size
        flush_interval=2.0,     # Flush every 2 seconds
        max_queue_size=50000,   # Large queue for high traffic
        gzip=True,              # Compression for bandwidth
        max_retries=5,          # Retry failed requests
        timeout=30,             # Longer timeout for reliability
        
        # Feature flags
        personal_api_key='phc_your_personal_api_key',
        poll_interval=60,       # Check flags every minute
        enable_local_evaluation=True,
        
        # Error handling
        on_error=lambda error: logging.error(f"PostHog error: {error}"),
        
        # Privacy and debugging
        disable_geoip=True,
        debug=False,
        
        # Super properties for all events
        super_properties={
            'app_version': '1.2.3',
            'environment': 'production'
        }
    )

# Initialize client
posthog_client = create_posthog_client()
```

### Multi-Tenant Configuration

```python
from posthog import Posthog

class MultiTenantPostHog:
    def __init__(self):
        self.clients = {}
    
    def get_client(self, tenant_id: str) -> Posthog:
        if tenant_id not in self.clients:
            # Create client for new tenant
            self.clients[tenant_id] = Posthog(
                project_api_key=f'phc_tenant_{tenant_id}_key',
                host='https://app.posthog.com',
                flush_at=100,
                flush_interval=1.0,
                super_properties={
                    'tenant_id': tenant_id,
                    'app_name': 'multi-tenant-app'
                }
            )
        return self.clients[tenant_id]
    
    def shutdown_all(self):
        for client in self.clients.values():
            client.shutdown()

# Usage
multi_posthog = MultiTenantPostHog()

# Track events for different tenants
tenant_a_client = multi_posthog.get_client('tenant_a')
tenant_a_client.capture('user123', 'event_name')

tenant_b_client = multi_posthog.get_client('tenant_b')
tenant_b_client.capture('user456', 'event_name')

# Shutdown all clients
multi_posthog.shutdown_all()
```

### Development and Testing Configuration

```python
from posthog import Posthog
import os

def create_development_client():
    # Different config for development
    is_development = os.getenv('ENVIRONMENT') == 'development'
    is_testing = os.getenv('ENVIRONMENT') == 'testing'
    
    return Posthog(
        project_api_key=os.getenv('POSTHOG_API_KEY', 'phc_test_key'),
        host=os.getenv('POSTHOG_HOST', 'https://app.posthog.com'),
        
        # Development settings
        debug=is_development,
        send=not is_testing,    # Don't send events during tests
        disabled=is_testing,    # Disable completely during tests
        sync_mode=is_testing,   # Synchronous for testing
        
        # Faster flushing for development
        flush_at=10 if is_development else 100,
        flush_interval=0.1 if is_development else 0.5,
        
        # Development error handling
        on_error=lambda error: print(f"PostHog error: {error}") if is_development else None
    )

client = create_development_client()
```

### Advanced Batching Configuration

```python
from posthog import Posthog

# High-throughput application
high_volume_client = Posthog(
    project_api_key='phc_your_project_api_key',
    
    # Large batches for efficiency
    flush_at=500,
    flush_interval=5.0,
    max_queue_size=100000,
    
    # Multiple threads for processing
    thread=3,
    
    # Compression and timeouts
    gzip=True,
    timeout=60,
    max_retries=10
)

# Low-latency application
low_latency_client = Posthog(
    project_api_key='phc_your_project_api_key',
    
    # Small batches for low latency
    flush_at=10,
    flush_interval=0.1,
    
    # Synchronous mode for immediate sending
    sync_mode=True,
    timeout=5
)

# Offline-first application
offline_client = Posthog(
    project_api_key='phc_your_project_api_key',
    
    # Large queue for offline operation
    max_queue_size=1000000,
    flush_at=1000,
    flush_interval=30.0,
    
    # Aggressive retries
    max_retries=20,
    timeout=120
)
```

### Event Preprocessing

```python
from posthog import Posthog
import hashlib

def anonymize_sensitive_data(event):
    """Preprocessing function to anonymize sensitive data"""
    
    # Hash email addresses
    if 'properties' in event and 'email' in event['properties']:
        email = event['properties']['email']
        event['properties']['email_hash'] = hashlib.sha256(email.encode()).hexdigest()
        del event['properties']['email']
    
    # Remove PII fields
    sensitive_fields = ['ssn', 'credit_card', 'password']
    if 'properties' in event:
        for field in sensitive_fields:
            event['properties'].pop(field, None)
    
    # Add processing metadata
    event['properties']['_processed'] = True
    
    return event

client = Posthog(
    project_api_key='phc_your_project_api_key',
    before_send=anonymize_sensitive_data
)

# Events are automatically preprocessed
client.capture('user123', 'sensitive_event', {
    'email': 'user@example.com',  # Will be hashed
    'amount': 100
})
```

### Exception and Error Handling

```python
from posthog import Posthog
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def error_handler(error):
    """Custom error handler for PostHog errors"""
    logger.error(f"PostHog error: {error}")
    
    # Send to monitoring service
    # monitoring.capture_exception(error)
    
    # Optionally disable client on repeated errors
    # if isinstance(error, ConnectionError):
    #     client.disabled = True

client = Posthog(
    project_api_key='phc_your_project_api_key',
    on_error=error_handler,
    
    # Exception auto-capture for debugging
    enable_exception_autocapture=True,
    log_captured_exceptions=True,
    project_root='/app'
)

# Test error handling
try:
    risky_operation()
except Exception as e:
    # Exception is automatically captured if enable_exception_autocapture=True
    # Or capture manually:
    client.capture_exception(e)
```

### Graceful Shutdown Patterns

```python
from posthog import Posthog
import signal
import sys
import atexit

client = Posthog(project_api_key='phc_your_project_api_key')

def shutdown_handler(signum=None, frame=None):
    """Graceful shutdown handler"""
    print("Shutting down PostHog client...")
    client.shutdown()
    sys.exit(0)

# Register shutdown handlers
signal.signal(signal.SIGINT, shutdown_handler)   # Ctrl+C
signal.signal(signal.SIGTERM, shutdown_handler)  # Termination signal
atexit.register(client.shutdown)                 # Process exit

# Application code
def main():
    client.capture('app', 'started')
    
    # Application logic...
    
    # Events are automatically flushed on shutdown
    
if __name__ == '__main__':
    main()
```

### Performance Monitoring

```python
from posthog import Posthog
import time
import threading
from queue import Queue

class MonitoredPostHogClient:
    def __init__(self, **kwargs):
        self.client = Posthog(**kwargs)
        self.metrics = {
            'events_sent': 0,
            'events_failed': 0,
            'flush_count': 0,
            'queue_size': 0
        }
        
        # Override error handler to track failures
        original_error_handler = kwargs.get('on_error')
        def error_handler(error):
            self.metrics['events_failed'] += 1
            if original_error_handler:
                original_error_handler(error)
        
        self.client.on_error = error_handler
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_performance)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    def _monitor_performance(self):
        while True:
            # Monitor queue size
            if hasattr(self.client, 'queue'):
                self.metrics['queue_size'] = self.client.queue.qsize()
            
            # Log metrics every 60 seconds
            print(f"PostHog metrics: {self.metrics}")
            time.sleep(60)
    
    def capture(self, *args, **kwargs):
        result = self.client.capture(*args, **kwargs)
        if result:
            self.metrics['events_sent'] += 1
        return result
    
    def flush(self):
        self.client.flush()
        self.metrics['flush_count'] += 1
    
    def shutdown(self):
        self.client.shutdown()

# Usage
monitored_client = MonitoredPostHogClient(
    project_api_key='phc_your_project_api_key',
    flush_at=100,
    flush_interval=1.0
)

# Track events with monitoring
monitored_client.capture('user123', 'event_name')
```

## Configuration Best Practices

### Environment-Specific Settings

```python
import os
from posthog import Posthog

def create_client_for_environment():
    env = os.getenv('ENVIRONMENT', 'development')
    
    base_config = {
        'project_api_key': os.getenv('POSTHOG_API_KEY'),
        'host': os.getenv('POSTHOG_HOST', 'https://app.posthog.com'),
    }
    
    if env == 'production':
        return Posthog(
            **base_config,
            debug=False,
            send=True,
            flush_at=200,
            flush_interval=2.0,
            max_queue_size=50000,
            gzip=True,
            max_retries=5,
            enable_exception_autocapture=False  # Disable for production
        )
    
    elif env == 'staging':
        return Posthog(
            **base_config,
            debug=True,
            send=True,
            flush_at=50,
            flush_interval=1.0,
            enable_exception_autocapture=True
        )
    
    elif env == 'development':
        return Posthog(
            **base_config,
            debug=True,
            send=True,
            flush_at=10,
            flush_interval=0.5,
            sync_mode=True  # Immediate sending for development
        )
    
    else:  # testing
        return Posthog(
            **base_config,
            disabled=True,  # No events sent during tests
            sync_mode=True
        )

client = create_client_for_environment()
```

### Resource Management

```python
from posthog import Posthog
import contextlib

@contextlib.contextmanager
def posthog_client(**kwargs):
    """Context manager for automatic client cleanup"""
    client = Posthog(**kwargs)
    try:
        yield client
    finally:
        client.shutdown()

# Usage
with posthog_client(project_api_key='phc_key') as client:
    client.capture('user123', 'event_name')
    # Client automatically shuts down when exiting context
```

### Thread Safety

PostHog clients are thread-safe and can be shared across multiple threads:

```python
from posthog import Posthog
import threading
import queue

# Single client shared across threads
shared_client = Posthog(project_api_key='phc_your_project_api_key')

def worker_thread(thread_id, event_queue):
    while True:
        try:
            event_data = event_queue.get(timeout=1)
            shared_client.capture(
                event_data['user_id'],
                event_data['event_name'],
                event_data['properties']
            )
            event_queue.task_done()
        except queue.Empty:
            break

# Start multiple worker threads
event_queue = queue.Queue()
threads = []

for i in range(5):
    t = threading.Thread(target=worker_thread, args=(i, event_queue))
    t.start()
    threads.append(t)

# Add events to queue
for i in range(100):
    event_queue.put({
        'user_id': f'user_{i}',
        'event_name': 'thread_event',
        'properties': {'thread_id': i % 5}
    })

# Wait for completion
event_queue.join()

# Shutdown client
shared_client.shutdown()
```