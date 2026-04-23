# Diff Operations

Comprehensive diff functionality for comparing commits, trees, and working directory state. Supports unified diff generation, patch creation, and change detection.

## Capabilities

```python { .api }
class Diff:
    def __init__(self, repo: "Repo", a_rawpath: bytes, b_rawpath: bytes, a_blob_id: str, b_blob_id: str, a_mode: int, b_mode: int, new_file: bool, deleted_file: bool, copied_file: bool, raw_rename_from: str, raw_rename_to: str, diff: str, change_type: str):
        """Initialize diff."""
    
    @property
    def a_path(self) -> str:
        """Path in A side."""
    
    @property
    def b_path(self) -> str:
        """Path in B side."""
    
    @property
    def change_type(self) -> str:
        """Change type (A/D/M/R/C/T/U)."""
    
    @property
    def diff(self) -> str:
        """Unified diff text."""

class DiffIndex(list):
    def iter_change_type(self, change_type: str) -> Iterator["Diff"]:
        """Iterate diffs of specific change type."""

class Diffable:
    def diff(self, other: Union["Diffable", str, None] = None, paths: Union[str, list] = None, create_patch: bool = False, **kwargs) -> "DiffIndex":
        """Create diff."""

NULL_TREE = object()  # Represents empty tree
INDEX = object()  # Represents index state

class DiffConstants:
    """Constants for diff operations."""
```

## Usage Examples

```python
from git import Repo

repo = Repo('/path/to/repo')

# Compare commits
commit1 = repo.commit('HEAD~1')
commit2 = repo.commit('HEAD')
diffs = commit1.diff(commit2)

for diff in diffs:
    print(f"{diff.change_type}: {diff.a_path} -> {diff.b_path}")
    if diff.diff:
        print(diff.diff)

# Compare working tree to index
index_diffs = repo.index.diff(None)

# Compare specific files
file_diffs = commit1.diff(commit2, paths=['README.md'])
```