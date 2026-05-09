# Memray

A high-performance memory profiler for Python applications that provides comprehensive tracking of memory allocations in Python code, native extension modules, and the Python interpreter itself. Memray features blazing fast profiling with minimal performance impact, native C/C++ call stack tracking, multiple report generation formats, and live monitoring capabilities.

## Package Information

- **Package Name**: memray
- **Language**: Python
- **Installation**: `pip install memray`

## Core Imports

```python
import memray
```

For specific components:

```python
from memray import Tracker, FileReader, AllocationRecord, TemporalAllocationRecord
from memray import compute_statistics, size_fmt, get_symbolic_support
from memray import MemrayError, MemrayCommandError
```

## Basic Usage

```python
import memray

# Profile a Python script
with memray.Tracker("profile.bin"):
    # Your code to profile
    data = [i ** 2 for i in range(100000)]
    result = sum(data)

# Analyze the results
with memray.FileReader("profile.bin") as reader:
    for record in reader.get_allocation_records():
        print(f"Address: {record.address}, Size: {record.size}")
```

Command-line profiling:

```bash
# Profile a script
memray run --output profile.bin script.py

# Generate flame graph report
memray flamegraph profile.bin
```

## Architecture

Memray's architecture consists of three main layers:

- **Tracking Layer**: High-performance C/C++ tracking hooks that intercept memory allocations with minimal overhead
- **Storage Layer**: Efficient binary format for storing allocation records with optional compression
- **Analysis Layer**: Python API for reading and analyzing captured data, plus CLI tools for report generation

The system supports both offline analysis of captured data and live monitoring of running processes, with native stack trace resolution for comprehensive profiling including C extensions.

## Capabilities

### Memory Tracking

Core functionality for tracking memory allocations in Python applications. The Tracker context manager provides the primary interface for capturing allocation data.

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
    ): ...
```

[Memory Tracking](./memory-tracking.md)

### File I/O and Data Analysis

Reading and analyzing captured memory profiling data. Includes file and socket readers for both offline analysis and live monitoring.

```python { .api }
class FileReader:
    def __init__(self, file_name, *, report_progress=False, max_memory_records=10000): ...
    def get_allocation_records(self): ...
    def get_high_watermark_allocation_records(self, merge_threads=True): ...
    def get_leaked_allocation_records(self, merge_threads=True): ...

class SocketReader:
    def __init__(self, port: int): ...
    def get_current_snapshot(self, *, merge_threads: bool): ...
```

[File I/O and Data Analysis](./file-io.md)

### CLI Commands

Command-line interface for profiling applications and generating reports. Provides tools for capturing profiles, generating visualizations, and analyzing memory usage patterns.

```bash
memray run [options] script.py
memray flamegraph [options] capture_file.bin
memray table [options] capture_file.bin
memray live [options] port
```

[CLI Commands](./cli-commands.md)

### IPython Integration

Enhanced integration with IPython and Jupyter notebooks for interactive memory profiling workflows.

```python { .api }
def load_ipython_extension(ipython): ...
```

[IPython Integration](./ipython-integration.md)

## Types and Data Structures

```python
from typing import List, Tuple, Union
from pathlib import Path
from datetime import datetime
```

```python { .api }
class AllocationRecord:
    tid: int
    address: int
    size: int
    @property
    def allocator(self) -> int  # Returns AllocatorType enum value
    stack_id: int
    n_allocations: int
    native_stack_id: int
    native_segment_generation: int
    thread_name: str

    def stack_trace(self, max_stacks=None): ...
    def native_stack_trace(self, max_stacks=None): ...
    def hybrid_stack_trace(self, max_stacks=None): ...

class MemorySnapshot:
    time: int
    rss: int
    heap: int

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

class AllocatorType:
    MALLOC = 1
    FREE = 2
    CALLOC = 3
    REALLOC = 4
    POSIX_MEMALIGN = 5
    ALIGNED_ALLOC = 6
    MEMALIGN = 7
    VALLOC = 8
    PVALLOC = 9
    MMAP = 10
    MUNMAP = 11
    PYMALLOC_MALLOC = 12
    PYMALLOC_CALLOC = 13
    PYMALLOC_REALLOC = 14
    PYMALLOC_FREE = 15

class FileFormat:
    ALL_ALLOCATIONS = 1
    AGGREGATED_ALLOCATIONS = 2

class Destination:
    pass

class FileDestination:
    path: Union[pathlib.Path, str]
    overwrite: bool = False
    compress_on_exit: bool = True

class SocketDestination:
    server_port: int
    address: str = "127.0.0.1"

class TemporalAllocationRecord:
    tid: int
    address: int
    size: int
    @property
    def allocator(self) -> int  # Returns AllocatorType enum value
    stack_id: int
    n_allocations: int
    native_stack_id: int
    native_segment_generation: int
    thread_name: str
    intervals: List[Interval]

    def stack_trace(self, max_stacks=None): ...
    def native_stack_trace(self, max_stacks=None): ...
    def hybrid_stack_trace(self, max_stacks=None): ...

class Interval:
    allocated_before_snapshot: bool
    deallocated_before_snapshot: bool
    n_allocations: int
    n_bytes: int

class MemrayError(Exception):
    pass

class MemrayCommandError(MemrayError):
    exit_code: int

class PymallocDomain:
    PYMALLOC_RAW = 1
    PYMALLOC_MEM = 2
    PYMALLOC_OBJECT = 3

class SymbolicSupport:
    NONE = 1
    FUNCTION_NAME_ONLY = 2
    TOTAL = 3

# Type aliases for stack trace elements  
PythonStackElement = Tuple[str, str, int]  # (filename, function_name, line_number)
NativeStackElement = Tuple[str, str, int]  # (filename, function_name, line_number)

# Constants
RTLD_NOW: int
RTLD_DEFAULT: int
```