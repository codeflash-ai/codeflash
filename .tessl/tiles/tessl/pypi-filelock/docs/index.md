# Filelock

A platform-independent file lock that supports the with-statement for coordinating access to shared resources across processes. Filelock provides synchronous and asynchronous file locking with platform-specific optimizations for Unix (fcntl), Windows (msvcrt), and a fallback soft lock implementation.

## Package Information

- **Package Name**: filelock
- **Language**: Python
- **Installation**: `pip install filelock`

## Core Imports

```python
from filelock import FileLock
```

For async usage:

```python
from filelock import AsyncFileLock
```

Import specific lock types:

```python
from filelock import SoftFileLock, UnixFileLock, WindowsFileLock
from filelock import AsyncSoftFileLock, AsyncUnixFileLock, AsyncWindowsFileLock
```

Import exceptions:

```python
from filelock import Timeout
```

Import version:

```python
from filelock import __version__
```

Import utility functions:

```python
from filelock._util import raise_on_not_writable_file, ensure_directory_exists
from filelock._unix import has_fcntl
```

## Basic Usage

```python
from filelock import FileLock
import time

# Create a file lock
lock = FileLock("my_file.lock")

# Use with context manager (recommended)
with lock:
    # Critical section - only one process can execute this at a time
    print("Lock acquired, doing work...")
    time.sleep(2)
    print("Work completed")
# Lock is automatically released

# Async usage
import asyncio
from filelock import AsyncFileLock

async def async_example():
    lock = AsyncFileLock("async_file.lock")
    
    async with lock:
        print("Async lock acquired")
        await asyncio.sleep(2)
        print("Async work completed")

# Manual acquire/release (not recommended)
lock = FileLock("manual.lock")
try:
    lock.acquire(timeout=10)
    print("Lock acquired manually")
finally:
    lock.release()
```

## Architecture

Filelock uses a hierarchical class design with platform-specific implementations:

- **BaseFileLock**: Abstract base class defining the common interface
- **Platform Implementations**: UnixFileLock (fcntl), WindowsFileLock (msvcrt), SoftFileLock (fallback)
- **FileLock Alias**: Automatically selects the best implementation for the current platform
- **Async Support**: BaseAsyncFileLock and async variants of all lock types
- **Context Management**: All locks support both synchronous `with` and asynchronous `async with` statements

The library automatically chooses the most appropriate locking mechanism based on the platform and available system features.

## Capabilities

### Platform-Specific File Locks

Cross-platform file locking with automatic platform detection. FileLock automatically selects the best available implementation for the current system.

```python { .api }
class FileLock(BaseFileLock):
    """Platform-specific file lock (alias for UnixFileLock/WindowsFileLock/SoftFileLock)."""
    
    def __init__(
        self,
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        mode: int = 0o644,
        thread_local: bool = True,
        *,
        blocking: bool = True,
        is_singleton: bool = False,
    ) -> None:
        """
        Create a new lock object.
        
        Args:
            lock_file: Path to the lock file
            timeout: Default timeout in seconds (-1 for no timeout)
            mode: File permissions for the lock file
            thread_local: Whether context should be thread local
            blocking: Whether the lock should be blocking
            is_singleton: If True, only one instance per lock file path
        """

class AsyncFileLock(BaseAsyncFileLock):
    """Platform-specific async file lock."""
    
    def __init__(
        self,
        lock_file: str | os.PathLike[str],
        timeout: float = -1,
        mode: int = 0o644,
        thread_local: bool = False,
        *,
        blocking: bool = True,
        is_singleton: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        run_in_executor: bool = True,
        executor: concurrent.futures.Executor | None = None,
    ) -> None:
        """
        Create a new async lock object.
        
        Args:
            lock_file: Path to the lock file
            timeout: Default timeout in seconds (-1 for no timeout)  
            mode: File permissions for the lock file
            thread_local: Whether context should be thread local
            blocking: Whether the lock should be blocking
            is_singleton: If True, only one instance per lock file path
            loop: Event loop to use
            run_in_executor: Whether to run in executor
            executor: Executor to use for blocking operations
        """
```

### Unix File Locks (fcntl-based)

Hard file locking using fcntl.flock for Unix-like systems (Linux, macOS, BSD).

```python { .api }
class UnixFileLock(BaseFileLock):
    """Uses fcntl.flock to hard lock the file on Unix systems."""

class AsyncUnixFileLock(UnixFileLock, BaseAsyncFileLock):
    """Async version of UnixFileLock."""
```

### Platform Availability Constants

Constants indicating platform-specific feature availability.

```python { .api }
has_fcntl: bool
# Boolean constant indicating if fcntl is available on the current system
# True on Unix-like systems with fcntl support, False on Windows or systems without fcntl
```

### Windows File Locks (msvcrt-based)

Hard file locking using msvcrt.locking for Windows systems.

```python { .api }
class WindowsFileLock(BaseFileLock):
    """Uses msvcrt.locking to hard lock the file on Windows systems."""

class AsyncWindowsFileLock(WindowsFileLock, BaseAsyncFileLock):
    """Async version of WindowsFileLock."""
```

### Soft File Locks (cross-platform fallback)

File existence-based locking that works on all platforms but provides softer guarantees.

```python { .api }
class SoftFileLock(BaseFileLock):
    """Simply watches the existence of the lock file."""

class AsyncSoftFileLock(SoftFileLock, BaseAsyncFileLock):
    """Async version of SoftFileLock."""
```

### Lock Acquisition and Release

Core methods for acquiring and releasing file locks with timeout support.

```python { .api }
class BaseFileLock:
    def acquire(
        self,
        timeout: float | None = None,
        poll_interval: float = 0.05,
        *,
        poll_intervall: float | None = None,
        blocking: bool | None = None,
    ) -> AcquireReturnProxy:
        """
        Try to acquire the file lock.
        
        Args:
            timeout: Maximum wait time in seconds (None uses default)
            poll_interval: Interval between acquisition attempts
            poll_intervall: Deprecated, use poll_interval instead
            blocking: Whether to block until acquired
            
        Returns:
            Context manager proxy for the lock
            
        Raises:
            Timeout: If lock cannot be acquired within timeout
        """
    
    def release(self, force: bool = False) -> None:
        """
        Release the file lock.
        
        Args:
            force: If True, ignore lock counter and force release
        """

class BaseAsyncFileLock:
    @property
    def run_in_executor(self) -> bool:
        """Whether operations run in an executor."""
    
    @property
    def executor(self) -> concurrent.futures.Executor | None:
        """The executor used for blocking operations."""
    
    @executor.setter
    def executor(self, value: concurrent.futures.Executor | None) -> None:
        """Set the executor for blocking operations."""
    
    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """The event loop associated with this lock."""
    
    async def acquire(
        self,
        timeout: float | None = None,
        poll_interval: float = 0.05,
        *,
        blocking: bool | None = None,
    ) -> AsyncAcquireReturnProxy:
        """
        Async version of acquire.
        
        Args:
            timeout: Maximum wait time in seconds (None uses default)
            poll_interval: Interval between acquisition attempts  
            blocking: Whether to block until acquired
            
        Returns:
            Async context manager proxy for the lock
            
        Raises:
            Timeout: If lock cannot be acquired within timeout
        """
    
    async def release(self, force: bool = False) -> None:
        """
        Async version of release.
        
        Args:
            force: If True, ignore lock counter and force release
        """
```

### Lock State and Properties

Properties and methods for inspecting lock state and configuration.

```python { .api }
class BaseFileLock:
    @property
    def is_locked(self) -> bool:
        """Whether the lock is currently held."""
    
    @property
    def lock_counter(self) -> int:
        """Number of times lock has been acquired (for reentrant locking)."""
        
    @property
    def lock_file(self) -> str:
        """Path to the lock file."""
        
    @property
    def timeout(self) -> float:
        """Default timeout value in seconds."""
        
    @timeout.setter
    def timeout(self, value: float | str) -> None:
        """Set the default timeout value."""
        
    @property
    def blocking(self) -> bool:
        """Whether locking is blocking by default."""
        
    @blocking.setter  
    def blocking(self, value: bool) -> None:
        """Set the default blocking behavior."""
        
    @property
    def mode(self) -> int:
        """File permissions for the lock file."""
        
    @property
    def is_singleton(self) -> bool:
        """Whether this lock uses singleton pattern."""
        
    def is_thread_local(self) -> bool:
        """Whether this lock uses thread-local context."""
```

### Context Manager Support

Built-in support for context managers enabling safe automatic lock release.

```python { .api }
class BaseFileLock:
    def __enter__(self) -> BaseFileLock:
        """Enter the context manager (acquire lock)."""
        
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Exit the context manager (release lock)."""

class BaseAsyncFileLock:
    async def __aenter__(self) -> BaseAsyncFileLock:
        """Enter the async context manager (acquire lock)."""
        
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Exit the async context manager (release lock)."""
    
    def __enter__(self) -> NoReturn:
        """Raises NotImplementedError - use async with instead."""
```

### Context Manager Proxies

Helper classes that provide context manager functionality while preventing double-acquisition.

```python { .api }
class AcquireReturnProxy:
    """Context manager returned by acquire() for safe lock handling."""
    
    def __init__(self, lock: BaseFileLock) -> None: ...
    
    def __enter__(self) -> BaseFileLock: ...
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...

class AsyncAcquireReturnProxy:
    """Async context manager returned by async acquire()."""
    
    def __init__(self, lock: BaseAsyncFileLock) -> None: ...
    
    async def __aenter__(self) -> BaseAsyncFileLock: ...
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...
```

### Exception Handling

Exception class for lock acquisition timeouts and error handling.

```python { .api }
class Timeout(TimeoutError):
    """Raised when the lock could not be acquired within the timeout period."""
    
    def __init__(self, lock_file: str) -> None:
        """
        Create a timeout exception.
        
        Args:
            lock_file: Path to the lock file that timed out
        """
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
    
    def __repr__(self) -> str:
        """Return detailed string representation of the exception."""
    
    def __reduce__(self) -> str | tuple[Any, ...]:
        """Support for pickling the exception."""
    
    @property
    def lock_file(self) -> str:
        """Path of the file lock that timed out."""

# Other exceptions that may be raised:
# ValueError: When invalid parameters are passed to singleton locks  
# NotImplementedError: When platform-specific features are unavailable
# PermissionError: When lock file cannot be written (via utility functions)
# IsADirectoryError: When lock path points to a directory (via utility functions)
```

### Utility Functions

Internal utility functions that are part of the public API for file and directory handling.

```python { .api }
def raise_on_not_writable_file(filename: str) -> None:
    """
    Raise an exception if attempting to open the file for writing would fail.
    
    This is done so files that will never be writable can be separated from files 
    that are writable but currently locked.
    
    Args:
        filename: Path to the file to check
        
    Raises:
        PermissionError: If file exists but is not writable
        IsADirectoryError: If path points to a directory (Unix/macOS)
        PermissionError: If path points to a directory (Windows)
    """

def ensure_directory_exists(filename: str | os.PathLike[str]) -> None:
    """
    Ensure the directory containing the file exists (create it if necessary).
    
    Args:
        filename: Path to the file whose parent directory should exist
    """
```

### Version Information

Package version information.

```python { .api }
__version__: str
# Version of the filelock package as a string
```

## Types

```python { .api }
import os
import asyncio
import concurrent.futures
from types import TracebackType
from typing import Union, Optional, Any, NoReturn

# Type aliases used in the API
PathLike = Union[str, os.PathLike[str]]
OptionalFloat = Optional[float] 
OptionalBool = Optional[bool]
OptionalLoop = Optional[asyncio.AbstractEventLoop]
OptionalExecutor = Optional[concurrent.futures.Executor]
OptionalExceptionType = Optional[type[BaseException]]
OptionalException = Optional[BaseException]
OptionalTraceback = Optional[TracebackType]
```

## Usage Examples

### Timeout Handling

```python
from filelock import FileLock, Timeout

lock = FileLock("resource.lock", timeout=5)

try:
    with lock:
        print("Got the lock!")
        time.sleep(10)  # Simulate work
except Timeout:
    print("Could not acquire lock within 5 seconds")
```

### Non-blocking Lock Attempts

```python
from filelock import FileLock

lock = FileLock("resource.lock", blocking=False)

try:
    with lock.acquire():
        print("Lock acquired immediately")
except Timeout:
    print("Lock is currently held by another process")
```

### Singleton Locks

```python
from filelock import FileLock

# All instances with same path share the same lock object
lock1 = FileLock("shared.lock", is_singleton=True)
lock2 = FileLock("shared.lock", is_singleton=True)

assert lock1 is lock2  # Same object
```

### Async Lock Usage

```python
import asyncio
from filelock import AsyncFileLock

async def worker(worker_id: int):
    lock = AsyncFileLock("shared_resource.lock")
    
    async with lock:
        print(f"Worker {worker_id} acquired lock")
        await asyncio.sleep(1)
        print(f"Worker {worker_id} releasing lock")

async def main():
    # Run multiple workers concurrently
    await asyncio.gather(
        worker(1),
        worker(2), 
        worker(3)
    )

asyncio.run(main())
```