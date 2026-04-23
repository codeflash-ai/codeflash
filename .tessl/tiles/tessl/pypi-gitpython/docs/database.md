# Database and Utilities

Infrastructure components for object database access and file locking mechanisms. These classes provide low-level access to Git's object storage and concurrent access control.

## Capabilities

### Object Database

Direct access to Git's object database with both high-level and command-line interfaces for reading and writing Git objects.

```python { .api }
from git.types import PathLike

class GitDB:
    """Git object database interface providing access to Git objects."""
    
    def __init__(self, root_path: PathLike): ...
    
    def stream(self, binsha: bytes): ...
    def info(self, binsha: bytes): ...
    def store(self, istream): ...

class GitCmdObjectDB:
    """Git object database using command-line interface for object operations."""
    
    def __init__(self, root_path: PathLike, git: "Git") -> None:
        """
        Initialize database with root path and Git command interface.
        
        Args:
            root_path: Path to Git object database
            git: Git command interface instance
        """
    
    def info(self, binsha: bytes):
        """Get object header information using git command."""
    
    def stream(self, binsha: bytes):
        """Get object data as stream using git command."""
```

### File Locking

File-based locking mechanisms for concurrent access control, essential for safe multi-process Git operations.

```python { .api }
class LockFile:
    """File-based locking mechanism for concurrent access control."""
    
    def __init__(self, file_path: PathLike): ...
    
    def acquire(self, fail_on_lock: bool = True) -> bool:
        """Acquire file lock."""
    
    def release(self) -> None:
        """Release file lock."""
    
    def __enter__(self): ...
    def __exit__(self, exc_type, exc_val, exc_tb): ...

class BlockingLockFile(LockFile):
    """Blocking file lock that waits until lock can be obtained or timeout."""
    
    def __init__(self, file_path: PathLike, check_interval: float = 0.3, max_time: float = float('inf')): ...
    
    def acquire(self, fail_on_lock: bool = True) -> bool:
        """Acquire lock, blocking until available or timeout."""
```

## Usage Examples

### Working with Object Database

```python
from git import Repo
from git.db import GitCmdObjectDB

repo = Repo('/path/to/repo')

# Access object database
odb = repo.odb

# Get object info
commit_sha = repo.head.commit.binsha
info = odb.info(commit_sha)
print(f"Object type: {info.type}")
print(f"Object size: {info.size}")

# Stream object data  
stream = odb.stream(commit_sha)
data = stream.read()
```

### Using File Locks

```python
from git.util import LockFile, BlockingLockFile
import time

# Basic file lock
lock_file = LockFile('/path/to/file.lock')
try:
    if lock_file.acquire():
        print("Lock acquired")
        # Perform operations requiring exclusive access
        time.sleep(1)
    else:
        print("Could not acquire lock")
finally:
    lock_file.release()

# Using context manager
with LockFile('/path/to/file.lock') as lock:
    # Operations here are protected by lock
    pass

# Blocking lock with timeout
blocking_lock = BlockingLockFile('/path/to/file.lock', max_time=30.0)
try:
    blocking_lock.acquire()  # Will wait up to 30 seconds
    # Protected operations
finally:
    blocking_lock.release()
```