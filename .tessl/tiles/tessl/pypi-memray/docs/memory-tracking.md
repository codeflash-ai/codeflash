# Memory Tracking

Core functionality for tracking memory allocations in Python applications. The Tracker provides the primary interface for capturing allocation data with configurable options for different profiling scenarios.

## Capabilities

### Tracker Context Manager

The main interface for memory profiling that captures allocation events during execution. Can output to files or stream to sockets for live monitoring.

```python { .api }
class Tracker:
    def __init__(
        self, 
        file_name=None, 
        *, 
        destination=None, 
        native_traces=False,
        memory_interval_ms=10, 
        follow_fork=False, 
        trace_python_allocators=False,
        file_format=FileFormat.ALL_ALLOCATIONS
    ):
        """
        Initialize memory tracker.

        Parameters:
        - file_name: str or pathlib.Path, output file path
        - destination: FileDestination or SocketDestination, alternative to file_name
        - native_traces: bool, whether to capture native stack frames
        - memory_interval_ms: int, interval for RSS updates (default: 10ms)
        - follow_fork: bool, continue tracking in forked processes
        - trace_python_allocators: bool, whether to trace Python allocators separately
        - file_format: FileFormat, output format (ALL_ALLOCATIONS or AGGREGATED_ALLOCATIONS)
        """

    def __enter__(self) -> 'Tracker':
        """Activate memory tracking."""

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Deactivate memory tracking."""
```

Usage examples:

```python
# Basic file tracking
with memray.Tracker("profile.bin"):
    # Code to profile
    data = process_large_dataset()

# With native traces
with memray.Tracker("profile.bin", native_traces=True):
    import numpy as np
    array = np.zeros((1000, 1000))

# Using FileDestination with options
import memray
from memray import FileDestination

destination = FileDestination(
    path="profile.bin",
    overwrite=True,
    compress_on_exit=True
)

with memray.Tracker(destination=destination):
    # Code to profile
    pass

# Live streaming to socket
from memray import SocketDestination

destination = SocketDestination(server_port=12345, address="localhost")

with memray.Tracker(destination=destination):
    # Code being profiled in real-time
    process_data()
```

### Thread Tracing

Automatic thread tracking for multi-threaded applications.

```python { .api }
def start_thread_trace(frame, event, arg):
    """
    Thread tracing function for automatic thread tracking.
    
    Parameters:
    - frame: frame object
    - event: str, tracing event type
    - arg: event argument
    
    Returns:
    - start_thread_trace function (itself)
    """
```

### Utility Functions

Debug and configuration functions for memory tracking.

```python { .api }
def dump_all_records(file_name):
    """
    Debug function to dump all records from a memray file.
    
    Parameters:
    - file_name: str, path to memray capture file
    
    Returns:
    - None
    """

def set_log_level(level: int):
    """
    Configure log message threshold for memray.
    
    Parameters:
    - level: int, minimum log level (logging.WARNING by default)
    
    Returns:
    - None
    """
```

Usage example:

```python
import logging
import memray

# Set debug logging
memray.set_log_level(logging.DEBUG)

# Profile and dump for debugging
with memray.Tracker("debug.bin"):
    problematic_function()

# Dump all records for analysis
memray.dump_all_records("debug.bin")
```

## Destination Types

### FileDestination

Specifies file output for captured allocations with compression and overwrite options.

```python { .api }
class FileDestination:
    path: Union[pathlib.Path, str]
    overwrite: bool = False
    compress_on_exit: bool = True
```

### SocketDestination

Specifies socket streaming for live monitoring of allocations.

```python { .api }
class SocketDestination:
    server_port: int
    address: str = "127.0.0.1"
```

### Base Destination

Abstract base for all destination types.

```python { .api }
class Destination:
    pass
```