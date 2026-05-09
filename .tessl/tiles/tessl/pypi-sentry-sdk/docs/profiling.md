# Profiling

CPU profiling capabilities for performance analysis with support for different scheduling backends, continuous profiling, transaction-based profiling, and comprehensive performance data collection.

## Capabilities

### Profile Session Management

Control profiling sessions with start/stop functions for manual profiling control and custom profiling workflows.

```python { .api }
def start_profiler() -> None:
    """
    Start a profiling session for the current process.
    
    Begins CPU profiling using the configured profiler backend.
    Must be called before any profiled operations.
    """

def stop_profiler() -> None:
    """
    Stop the current profiling session.
    
    Ends CPU profiling and processes collected profile data.
    Should be called after profiled operations complete.
    """

# Alternative names for compatibility
def start_profile_session() -> None:
    """Alias for start_profiler()."""

def stop_profile_session() -> None:
    """Alias for stop_profiler()."""
```

**Usage Examples:**

```python
import sentry_sdk

# Initialize SDK with profiling enabled
sentry_sdk.init(
    dsn="your-dsn-here",
    profiles_sample_rate=1.0,  # Profile 100% of transactions
)

# Manual profiling session
sentry_sdk.start_profiler()
try:
    # Code to profile
    perform_cpu_intensive_task()
    process_large_dataset()
    complex_algorithm()
finally:
    sentry_sdk.stop_profiler()

# Profiling with transaction context
with sentry_sdk.start_transaction(name="batch_processing", op="task"):
    sentry_sdk.start_profiler()
    try:
        process_batch_job()
    finally:
        sentry_sdk.stop_profiler()
```

### Transaction-Based Profiling

Automatic profiling within transactions when profiling is enabled through sampling configuration.

```python { .api }
class Profile:
    def __init__(self, transaction: Transaction):
        """
        Create a profile associated with a transaction.
        
        Parameters:
        - transaction: Transaction to associate profile data with
        """
    
    def finish(self) -> None:
        """
        Finish the profile and send data to Sentry.
        
        Processes collected profiling data and associates it with
        the transaction for performance analysis.
        """
```

**Usage Examples:**

```python
import sentry_sdk

# Automatic profiling through sampling
sentry_sdk.init(
    dsn="your-dsn-here",
    traces_sample_rate=1.0,    # Sample all transactions
    profiles_sample_rate=0.1,  # Profile 10% of sampled transactions
)

def cpu_intensive_function():
    # This function will be automatically profiled
    # when called within a sampled transaction
    for i in range(1000000):
        complex_calculation(i)

# Transaction with automatic profiling
with sentry_sdk.start_transaction(name="data_processing", op="function") as transaction:
    # If this transaction is selected for profiling,
    # all function calls will be profiled automatically
    load_data()
    cpu_intensive_function()  # Profile data collected here
    save_results()

# Manual profile management within transaction
def manual_profiling_example():
    with sentry_sdk.start_transaction(name="manual_profile", op="task") as transaction:
        # Create explicit profile instance
        profile = sentry_sdk.profiler.Profile(transaction)
        
        try:
            # Operations to profile
            expensive_operation_1()
            expensive_operation_2()
        finally:
            # Finish profile and send data
            profile.finish()
```

## Profiler Scheduling Backends

### Available Schedulers

Different profiler schedulers for various runtime environments and use cases.

```python { .api }
class ThreadScheduler:
    """Thread-based profiling scheduler for standard Python applications."""
    def __init__(self, frequency: int = 101): ...

class GeventScheduler:
    """Gevent-compatible profiling scheduler for gevent applications."""
    def __init__(self, frequency: int = 101): ...

class SleepScheduler:
    """Sleep-based profiling scheduler for minimal overhead profiling."""
    def __init__(self, frequency: int = 101): ...
```

**Configuration Examples:**

```python
import sentry_sdk
from sentry_sdk.profiler import ThreadScheduler, GeventScheduler

# Configure profiler for standard threading
sentry_sdk.init(
    dsn="your-dsn-here",
    profiles_sample_rate=0.1,
    _experiments={
        "profiler_scheduler": ThreadScheduler(frequency=101)  # 101 Hz sampling
    }
)

# Configure profiler for gevent applications
import gevent
sentry_sdk.init(
    dsn="your-dsn-here", 
    profiles_sample_rate=0.2,
    _experiments={
        "profiler_scheduler": GeventScheduler(frequency=201)  # 201 Hz sampling
    }
)
```

### Continuous Profiling

Background profiling that runs continuously throughout application lifetime.

```python
import sentry_sdk

# Enable continuous profiling
sentry_sdk.init(
    dsn="your-dsn-here",
    profiles_sample_rate=1.0,
    _experiments={
        "continuous_profiling_auto_start": True,
        "profiler_mode": "continuous"
    }
)

# Continuous profiling runs automatically in the background
# No manual start/stop required
def application_code():
    while True:
        handle_request()      # Profiled continuously
        process_background_tasks()  # Profiled continuously
        time.sleep(1)
```

## Profiling Configuration

### Sampling Configuration

Control when and how often profiling occurs through sampling rates and custom sampling functions.

```python
import sentry_sdk

# Percentage-based sampling
sentry_sdk.init(
    dsn="your-dsn-here",
    traces_sample_rate=1.0,      # Sample all transactions
    profiles_sample_rate=0.1,    # Profile 10% of transactions
)

# Custom profiling sampler
def custom_profiles_sampler(sampling_context):
    """Custom logic for profiling sampling decisions."""
    # Profile all transactions in development
    if sampling_context.get("environment") == "development":
        return 1.0
    
    # Profile high-value operations
    if sampling_context.get("transaction_context", {}).get("name", "").startswith("critical_"):
        return 0.5
    
    # Light profiling for regular operations
    return 0.05

sentry_sdk.init(
    dsn="your-dsn-here",
    traces_sample_rate=0.1,
    profiles_sampler=custom_profiles_sampler
)
```

### Performance Impact Control

Configure profiling to minimize performance impact on production applications.

```python
import sentry_sdk

# Low-impact production profiling
sentry_sdk.init(
    dsn="your-dsn-here",
    traces_sample_rate=0.01,     # Sample 1% of transactions
    profiles_sample_rate=0.1,    # Profile 10% of sampled transactions = 0.1% total
    _experiments={
        "profiler_scheduler_frequency": 51,  # Lower frequency = less overhead
        "max_profile_duration_ms": 30000,    # 30 second max profile duration
        "profile_timeout_warning": True      # Warn on long profiles
    }
)

# High-detail development profiling
sentry_sdk.init(
    dsn="your-dsn-here",
    traces_sample_rate=1.0,      # Sample all transactions
    profiles_sample_rate=1.0,    # Profile all transactions
    _experiments={
        "profiler_scheduler_frequency": 1001,  # High frequency sampling
        "max_profile_duration_ms": 300000,     # 5 minute max duration
        "include_local_variables": True        # Include variable values
    }
)
```

## Advanced Profiling Usage

### Custom Profiling Contexts

Combine profiling with custom contexts for targeted performance analysis.

```python
import sentry_sdk

def profile_critical_path(operation_name, **context):
    """Context manager for profiling critical code paths."""
    with sentry_sdk.start_transaction(
        name=operation_name, 
        op="performance_analysis"
    ) as transaction:
        
        # Add profiling context
        transaction.set_tag("profiling_target", "critical_path")
        for key, value in context.items():
            transaction.set_data(key, value)
        
        # Force profiling for this transaction
        sentry_sdk.start_profiler()
        try:
            yield transaction
        finally:
            sentry_sdk.stop_profiler()

# Usage
with profile_critical_path("payment_processing", 
                          user_tier="premium", 
                          amount=1000.0) as transaction:
    
    validate_payment_data()    # Profiled
    process_payment()          # Profiled  
    update_user_account()      # Profiled
    send_confirmation()        # Profiled
```

### Performance Regression Detection

Use profiling data to detect performance regressions and optimize hot paths.

```python
import sentry_sdk
import time

def benchmark_function(func, iterations=1000):
    """Benchmark a function with profiling data."""
    with sentry_sdk.start_transaction(
        name=f"benchmark_{func.__name__}", 
        op="benchmark"
    ) as transaction:
        
        transaction.set_data("iterations", iterations)
        transaction.set_tag("benchmark_type", "performance_test")
        
        sentry_sdk.start_profiler()
        start_time = time.time()
        
        try:
            for i in range(iterations):
                result = func()
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time = total_time / iterations
            
            transaction.set_measurement("total_time", total_time, "second")
            transaction.set_measurement("avg_time_per_call", avg_time * 1000, "millisecond")
            transaction.set_measurement("calls_per_second", iterations / total_time, "per_second")
            
            return {
                "total_time": total_time,
                "avg_time": avg_time,
                "calls_per_second": iterations / total_time
            }
            
        finally:
            sentry_sdk.stop_profiler()

# Usage
def cpu_bound_function():
    return sum(i * i for i in range(10000))

# Benchmark with profiling
results = benchmark_function(cpu_bound_function, iterations=100)
print(f"Average time per call: {results['avg_time']*1000:.2f}ms")
```

### Integration with Performance Monitoring

Combine profiling with transaction and span data for comprehensive performance analysis.

```python
import sentry_sdk

def analyze_database_performance():
    """Profile database operations with detailed span data."""
    with sentry_sdk.start_transaction(
        name="database_analysis", 
        op="performance.database"
    ) as transaction:
        
        sentry_sdk.start_profiler()
        
        try:
            # Profile individual database operations
            with sentry_sdk.start_span(op="db.query", description="user_lookup") as span:
                span.set_data("query", "SELECT * FROM users WHERE active = true")
                users = database.get_active_users()  # Profiled
                span.set_data("row_count", len(users))
            
            with sentry_sdk.start_span(op="db.bulk_insert", description="activity_log") as span:
                span.set_data("operation", "bulk_insert") 
                activities = []
                for user in users:
                    activities.append(create_activity_record(user))  # Profiled
                database.bulk_insert_activities(activities)  # Profiled
                span.set_data("insert_count", len(activities))
            
            with sentry_sdk.start_span(op="db.query", description="aggregate_stats") as span:
                span.set_data("query", "SELECT COUNT(*), AVG(score) FROM user_stats")
                stats = database.get_aggregate_stats()  # Profiled
                span.set_data("aggregation_result", stats)
                
        finally:
            sentry_sdk.stop_profiler()
            
        return {"user_count": len(users), "stats": stats}
```

## Profiling Best Practices

### Production Guidelines

- **Low sampling rates**: Use 1-10% profiling sampling in production
- **Time limits**: Set reasonable `max_profile_duration_ms` to prevent long-running profiles
- **Resource monitoring**: Monitor CPU and memory impact of profiling
- **Selective profiling**: Profile only critical code paths in production

### Development and Testing

- **High sampling**: Use 100% profiling sampling for development analysis
- **Comprehensive coverage**: Profile all major code paths during testing
- **Baseline establishment**: Create performance baselines for regression detection
- **Hot path identification**: Use profiling to identify CPU-intensive operations

### Performance Optimization Workflow

1. **Identify bottlenecks**: Use profiling to find slow functions
2. **Measure improvements**: Profile before and after optimizations  
3. **Regression testing**: Continuous profiling to catch performance regressions
4. **Resource usage**: Monitor profiling overhead and adjust sampling rates
5. **Long-term trends**: Analyze profiling data over time for performance trends

Profiling provides deep insights into application performance with flexible configuration options for development, testing, and production environments.