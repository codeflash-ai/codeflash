# References and Branches

Management of Git references including branches, tags, HEAD, and remote references. Supports creation, deletion, and manipulation of all reference types with symbolic reference handling.

## Capabilities

### Reference Base Classes

```python { .api }
class Reference:
    def __init__(self, repo: "Repo", path: str, check_path: bool = True):
        """Initialize reference."""
    
    @property
    def name(self) -> str:
        """Reference name."""
    
    @property  
    def commit(self) -> "Commit":
        """Referenced commit."""
    
    def set_commit(self, commit: "Commit", logmsg: str = None) -> "Reference":
        """Set reference to commit."""
    
    def delete(self, repo: "Repo", *refs: "Reference") -> None:
        """Delete references."""

class SymbolicReference(Reference):
    def __init__(self, repo: "Repo", path: str):
        """Initialize symbolic reference."""
```

### Branch Management

```python { .api }
class Head(Reference):
    def checkout(self, force: bool = False, **kwargs) -> "HEAD":
        """Checkout this branch."""
    
    def reset(self, commit: "Commit" = None, index: bool = True, working_tree: bool = False, paths: list = None, **kwargs) -> "Head":
        """Reset branch to commit."""

class HEAD(SymbolicReference):
    def reset(self, commit: "Commit" = None, index: bool = True, working_tree: bool = False, paths: list = None, **kwargs) -> "HEAD":
        """Reset HEAD."""
```

### Tag Management

```python { .api }
class TagReference(Reference):
    @property
    def tag(self) -> "TagObject":
        """Tag object (for annotated tags)."""

class Tag(TagReference):
    def delete(self, repo: "Repo", *tags: "Tag") -> None:
        """Delete tags."""
```

### Remote References

```python { .api }
class RemoteReference(Reference):
    @property
    def remote_name(self) -> str:
        """Remote name."""
    
    @property
    def remote_head(self) -> str:
        """Remote branch name."""
```

### Reference Logs

```python { .api }
class RefLog:
    def __init__(self, filepath: str):
        """Initialize reflog."""
    
    def __iter__(self) -> Iterator["RefLogEntry"]:
        """Iterate reflog entries."""

class RefLogEntry:
    def __init__(self, from_sha: str, to_sha: str, actor: "Actor", time: int, tz_offset: int, message: str):
        """Initialize reflog entry."""
    
    @property
    def oldhexsha(self) -> str:
        """Previous SHA."""
    
    @property
    def newhexsha(self) -> str:
        """New SHA."""
```

## Usage Examples

```python
from git import Repo

repo = Repo('/path/to/repo')

# Work with branches
main_branch = repo.heads.main
feature_branch = repo.create_head('feature/new-feature')

# Checkout branch
feature_branch.checkout()

# Create and switch to new branch
new_branch = repo.create_head('hotfix', commit='HEAD~3')
new_branch.checkout()

# Work with tags
tag = repo.create_tag('v1.0.0', message='Release v1.0.0')
for tag in repo.tags:
    print(f"Tag: {tag.name}")

# Access remote references
for remote_ref in repo.remotes.origin.refs:
    print(f"Remote ref: {remote_ref.name}")
```