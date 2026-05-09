# Index and Staging

Complete control over Git's staging area with support for adding, removing, and committing changes. Includes advanced staging operations and conflict resolution capabilities.

## Capabilities

### Index File Operations

```python { .api }
from typing import List, Callable
class IndexFile:
    def __init__(self, repo: "Repo", file_path: PathLike = None):
        """Initialize index file."""
    
    def add(self, items: list, force: bool = True, fprogress: Callable = None, path_rewriter: Callable = None, write: bool = True, write_extension_data: bool = False) -> "IndexFile":
        """Add files to index."""
    
    def remove(self, items: list, working_tree: bool = False, **kwargs) -> "IndexFile":
        """Remove files from index."""
    
    def commit(self, message: str, parent_commits: list = None, head: bool = True, author: "Actor" = None, committer: "Actor" = None, author_date: str = None, commit_date: str = None, skip_hooks: bool = False) -> "Commit":
        """Create commit from index."""
    
    def checkout(self, paths: list = None, index: bool = True, working_tree: bool = True, **kwargs) -> "IndexFile":
        """Checkout files from index."""
    
    def reset(self, commit: "Commit" = None, working_tree: bool = False, paths: list = None, **kwargs) -> "IndexFile":
        """Reset index."""

class IndexEntry:
    def __init__(self, binsha: bytes, mode: int, flags: int, path: str, stage: int = 0):
        """Index entry."""
    
    @property
    def binsha(self) -> bytes:
        """Binary SHA-1."""
    
    @property
    def hexsha(self) -> str:
        """Hex SHA-1."""
    
    @property
    def mode(self) -> int:
        """File mode."""
    
    @property
    def path(self) -> str:
        """File path."""
    
    @property
    def stage(self) -> int:
        """Merge stage (0=normal, 1=base, 2=ours, 3=theirs)."""

class BaseIndexEntry:
    """Base class for index entries with minimal required information."""
    
    def __init__(self, mode: int, binsha: bytes, flags: int, path: str):
        """
        Initialize base index entry.
        
        Args:
            mode: File mode
            binsha: Binary SHA-1 hash
            flags: Index flags  
            path: File path
        """
    
    @property
    def mode(self) -> int:
        """File mode."""
    
    @property
    def binsha(self) -> bytes:
        """Binary SHA-1."""
    
    @property
    def flags(self) -> int:
        """Index flags."""
    
    @property
    def path(self) -> str:
        """File path."""

class BlobFilter:
    """Filter predicate for selecting blobs by path patterns."""
    
    def __init__(self, paths: List[str]):
        """
        Initialize blob filter.
        
        Args:
            paths: List of paths to filter by
        """
    
    def __call__(self, path: str) -> bool:
        """Check if path matches filter."""

StageType = int  # 0=base, 1=ours, 2=theirs, 3=merge
```

## Usage Examples

```python
from git import Repo, Actor

repo = Repo('/path/to/repo')

# Stage files
repo.index.add(['file1.txt', 'file2.txt'])

# Commit changes
author = Actor("John Doe", "john@example.com")
commit = repo.index.commit("Added files", author=author)

# Remove files from staging
repo.index.remove(['old_file.txt'])

# Reset index
repo.index.reset()

# Check staging status
for item in repo.index.entries:
    print(f"Staged: {item}")
```