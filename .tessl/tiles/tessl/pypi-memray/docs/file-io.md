# File I/O and Data Analysis

Reading and analyzing captured memory profiling data. Includes file and socket readers for both offline analysis and live monitoring, with various methods to extract different views of allocation data.

```python
from typing import Optional, List, Tuple
```

## Capabilities

### FileReader

Context manager for reading and analyzing memray capture files. Provides multiple methods to extract different views of allocation data.

```python { .api }
class FileReader:
    def __init__(self, file_name, *, report_progress=False, max_memory_records=10000):
        """
        Initialize file reader for memray capture files.
        
        Parameters:
        - file_name: str or pathlib.Path, path to memray capture file
        - report_progress: bool, whether to show progress during reading
        - max_memory_records: int, maximum number of memory records to keep in memory
        """

    def close(self):
        """Close the file reader."""

    def get_allocation_records(self):
        """
        Get iterator over all allocation records.
        
        Returns:
        - Iterator[AllocationRecord]: All allocation records in chronological order
        """

    def get_high_watermark_allocation_records(self, merge_threads=True):
        """
        Get allocation records at peak memory usage.
        
        Parameters:
        - merge_threads: bool, whether to merge records across threads
        
        Returns:
        - Iterator[AllocationRecord]: Records at peak memory usage
        """

    def get_leaked_allocation_records(self, merge_threads=True):
        """
        Get allocation records for memory leaks.
        
        Parameters:
        - merge_threads: bool, whether to merge records across threads
        
        Returns:
        - Iterator[AllocationRecord]: Records for leaked allocations
        """

    def get_temporary_allocation_records(self, merge_threads=True, threshold=1):
        """
        Get allocation records for short-lived allocations.
        
        Parameters:
        - merge_threads: bool, whether to merge records across threads
        - threshold: int, minimum lifetime threshold in milliseconds
        
        Returns:
        - Iterator[AllocationRecord]: Records for temporary allocations
        """

    def get_temporal_allocation_records(self, merge_threads=True):
        """
        Get time-ordered allocation records.
        
        Parameters:
        - merge_threads: bool, whether to merge records across threads
        
        Returns:
        - Iterator[TemporalAllocationRecord]: Time-ordered temporal allocation records
        """

    def get_temporal_high_water_mark_allocation_records(self, merge_threads=True):
        """
        Get temporal records at peak memory usage.
        
        Parameters:
        - merge_threads: bool, whether to merge records across threads
        
        Returns:
        - Tuple[List[TemporalAllocationRecord], List[int]]: Temporal records at peak memory with thread IDs
        """

    def get_memory_snapshots(self):
        """
        Get iterator over memory snapshots.
        
        Returns:
        - Iterator[MemorySnapshot]: Memory usage snapshots over time
        """

    @property
    def metadata(self) -> 'Metadata':
        """File metadata and statistics."""

    @property
    def closed(self) -> bool:
        """Whether the file reader is closed."""
```

Usage examples:

```python
import memray

# Basic file reading
with memray.FileReader("profile.bin") as reader:
    print(f"Total allocations: {reader.metadata.total_allocations}")
    print(f"Peak memory: {reader.metadata.peak_memory}")
    
    # Iterate through all records
    for record in reader.get_allocation_records():
        print(f"Size: {record.size}, Thread: {record.thread_name}")

# Analyze memory leaks
with memray.FileReader("profile.bin") as reader:
    leaked_records = list(reader.get_leaked_allocation_records())
    print(f"Found {len(leaked_records)} leaked allocations")
    
    for record in leaked_records:
        stack_trace = record.stack_trace()
        print(f"Leaked {record.size} bytes at:")
        for frame in stack_trace:
            print(f"  {frame}")

# Analyze peak memory usage
with memray.FileReader("profile.bin") as reader:
    peak_records = list(reader.get_high_watermark_allocation_records())
    total_peak_size = sum(record.size for record in peak_records)
    print(f"Peak memory usage: {total_peak_size} bytes")

# Monitor memory over time
with memray.FileReader("profile.bin") as reader:
    snapshots = list(reader.get_memory_snapshots())
    for snapshot in snapshots[-10:]:  # Last 10 snapshots
        print(f"Time: {snapshot.time}ms, RSS: {snapshot.rss}, Heap: {snapshot.heap}")
```

### SocketReader

Context manager for reading allocations from live socket connections, enabling real-time monitoring of running processes.

```python { .api }
class SocketReader:
    def __init__(self, port: int):
        """
        Initialize socket reader for live monitoring.
        
        Parameters:
        - port: int, port number to connect to
        """

    def get_current_snapshot(self, *, merge_threads: bool):
        """
        Get current allocation snapshot from live process.
        
        Parameters:
        - merge_threads: bool, whether to merge records across threads
        
        Returns:
        - List[AllocationRecord]: Current allocation records
        """

    @property
    def command_line(self) -> str:
        """Command line of the tracked process."""

    @property
    def is_active(self) -> bool:
        """Whether the reader connection is active."""
    
    @property 
    def pid(self) -> Optional[int]:
        """Process ID of tracked process."""
    
    @property
    def has_native_traces(self) -> bool:
        """Whether native traces are enabled."""
```

Usage example:

```python
import memray
import time

# Connect to live profiling session
with memray.SocketReader(12345) as reader:
    print(f"Monitoring: {reader.command_line}")
    
    while True:
        try:
            snapshot = reader.get_current_snapshot(merge_threads=True)
            total_memory = sum(record.size for record in snapshot)
            print(f"Current memory usage: {total_memory} bytes ({len(snapshot)} allocations)")
            time.sleep(1)
        except KeyboardInterrupt:
            break
```

### Metadata Information

Comprehensive metadata about profiling sessions.

```python { .api }
class Metadata:
    start_time: datetime
    end_time: datetime
    total_allocations: int
    total_frames: int
    peak_memory: int
    command_line: str
    pid: int
    main_thread_id: int
    python_allocator: str
    has_native_traces: bool
    trace_python_allocators: bool
    file_format: FileFormat
```

Usage example:

```python
with memray.FileReader("profile.bin") as reader:
    meta = reader.metadata
    duration = meta.end_time - meta.start_time
    print(f"Profiling session: {duration.total_seconds():.2f} seconds")
    print(f"Process: {meta.command_line} (PID: {meta.pid})")
    print(f"Allocations: {meta.total_allocations}")
    print(f"Peak memory: {meta.peak_memory} bytes")
    print(f"Native traces: {'Yes' if meta.has_native_traces else 'No'}")
```

### Statistics Analysis

Compute statistics and summaries from capture files.

```python { .api }
def compute_statistics(file_name, *, report_progress=False, num_largest=5):
    """
    Compute comprehensive statistics from a memray capture file.
    
    Parameters:
    - file_name: str or pathlib.Path, path to memray capture file
    - report_progress: bool, whether to show progress during computation
    - num_largest: int, number of largest allocations to track
    
    Returns:
    - Stats: Statistics object with allocation summaries
    """
```

Usage example:

```python
import memray

# Compute statistics from a capture file
stats = memray.compute_statistics("profile.bin", num_largest=10)
print(f"Statistics computed for {stats.n_allocations} allocations")
```

### Utility Functions

Helper functions for formatting and debugging.

```python { .api }
def size_fmt(num_bytes):
    """
    Format byte size in human-readable format.
    
    Parameters:
    - num_bytes: int, number of bytes
    
    Returns:
    - str: Formatted size string (e.g., "1.5 MB")
    """

def get_symbolic_support():
    """
    Get level of symbolic debugging support available.
    
    Returns:
    - SymbolicSupport: Enum indicating support level
    """

def greenlet_trace(frame, event, arg):
    """
    Greenlet tracing function for async/coroutine tracking.
    
    Parameters:
    - frame: frame object
    - event: str, tracing event type  
    - arg: event argument
    
    Returns:
    - greenlet_trace function (itself)
    """
```

Usage example:

```python
import memray

# Format sizes for display
size_str = memray.size_fmt(1048576)  # "1.0 MB"

# Check symbolic support
support = memray.get_symbolic_support()
if support == memray.SymbolicSupport.TOTAL:
    print("Full symbolic debugging available")
```