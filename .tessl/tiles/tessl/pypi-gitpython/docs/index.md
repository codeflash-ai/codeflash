# GitPython

GitPython is a comprehensive Python library for interacting with Git repositories, providing both high-level porcelain and low-level plumbing interfaces. It offers abstractions for Git objects (commits, trees, blobs, tags), repository management, branch and remote operations, and supports complex Git workflows including merging, rebasing, and conflict resolution.

## Package Information

- **Package Name**: GitPython
- **Language**: Python  
- **Installation**: `pip install GitPython`
- **Requirements**: Git 1.7.x or newer, Python >= 3.7

## Core Imports

```python
import git
```

Common import patterns:

```python
from git import Repo
from git import Repo, Actor, RemoteProgress
from git.exc import GitCommandError, InvalidGitRepositoryError
```

## Basic Usage

```python
from git import Repo

# Initialize a new repository
repo = Repo.init("/path/to/repo")

# Open an existing repository
repo = Repo("/path/to/existing/repo")

# Clone a repository
repo = Repo.clone_from("https://github.com/user/repo.git", "/local/path")

# Basic repository operations
print(f"Current branch: {repo.active_branch}")
print(f"Is dirty: {repo.is_dirty()}")
print(f"Untracked files: {repo.untracked_files}")

# Access commits
for commit in repo.iter_commits('main', max_count=10):
    print(f"{commit.hexsha[:7]} - {commit.message.strip()}")

# Stage and commit changes
repo.index.add(['file1.txt', 'file2.txt'])
repo.index.commit("Added files")

# Work with remotes
origin = repo.remotes.origin
origin.fetch()
origin.push()
```

## Architecture

GitPython provides a comprehensive object model that mirrors Git's internal structure:

- **Repository (`Repo`)**: Central interface providing access to all repository data and operations
- **Git Objects**: Complete object model including commits, trees, blobs, and tags with full metadata access
- **References**: Management of branches, tags, HEAD, and remote references with symbolic reference support
- **Index**: Full staging area control with fine-grained file operation support
- **Remotes**: Complete remote repository interaction including fetch, push, and progress reporting
- **Command Interface**: Direct access to Git command-line operations with comprehensive error handling

This design enables both high-level Git workflows and low-level repository manipulation, making it suitable for automation tools, CI/CD systems, code analysis tools, and any application requiring Git repository interaction.

## Capabilities

### Repository Management

Core repository operations including initialization, cloning, configuration, and state inspection. Provides the foundation for all Git operations through the central Repo interface.

```python { .api }
class Repo:
    def __init__(self, path: PathLike, odbt: Type[GitCmdObjectDB] = GitCmdObjectDB, search_parent_directories: bool = False, expand_vars: bool = True): ...
    @classmethod
    def init(cls, path: PathLike, mkdir: bool = True, odbt: Type[GitCmdObjectDB] = GitCmdObjectDB, expand_vars: bool = True, **kwargs) -> "Repo": ...
    @classmethod
    def clone_from(cls, url: str, to_path: PathLike, progress: RemoteProgress = None, env: dict = None, multi_options: list = None, **kwargs) -> "Repo": ...
    
    def is_dirty(self, index: bool = True, working_tree: bool = True, untracked_files: bool = False, submodules: bool = True, path: PathLike = None) -> bool: ...
    def iter_commits(self, rev: str = None, paths: str | list = None, **kwargs) -> Iterator["Commit"]: ...
```

[Repository Management](./repository.md)

### Git Objects

Access to Git's core object model including commits, trees, blobs, and tags. Provides complete metadata access and object traversal capabilities for repository analysis and manipulation.

```python { .api }
class Object:
    def __init__(self, repo: "Repo", binsha: bytes, mode: int = None, path: str = None): ...

class Commit(Object):
    def __init__(self, repo: "Repo", binsha: bytes, tree: "Tree" = None, author: Actor = None, authored_date: int = None, author_tz_offset: int = None, committer: Actor = None, committed_date: int = None, committer_tz_offset: int = None, message: str = None, parents: list = None, encoding: str = None): ...

class Tree(Object):
    def __init__(self, repo: "Repo", binsha: bytes, mode: int = None, path: str = None): ...

class Blob(Object):
    def __init__(self, repo: "Repo", binsha: bytes, mode: int = None, path: str = None): ...
```

[Git Objects](./objects.md)

### References and Branches

Management of Git references including branches, tags, HEAD, and remote references. Supports creation, deletion, and manipulation of all reference types with symbolic reference handling.

```python { .api }
class Reference:
    def __init__(self, repo: "Repo", path: str, check_path: bool = True): ...
    def set_commit(self, commit: "Commit", logmsg: str = None) -> "Reference": ...
    def delete(self, repo: "Repo", *refs: "Reference") -> None: ...

class Head(Reference):
    def checkout(self, force: bool = False, **kwargs) -> "HEAD": ...
    def reset(self, commit: "Commit" = None, index: bool = True, working_tree: bool = False, paths: list = None, **kwargs) -> "Head": ...

class Tag(Reference):
    def delete(self, repo: "Repo", *tags: "Tag") -> None: ...
```

[References and Branches](./references.md)

### Index and Staging

Complete control over Git's staging area with support for adding, removing, and committing changes. Includes advanced staging operations and conflict resolution capabilities.

```python { .api }
class IndexFile:
    def __init__(self, repo: "Repo", file_path: PathLike = None): ...
    def add(self, items: list, force: bool = True, fprogress: Callable = None, path_rewriter: Callable = None, write: bool = True, write_extension_data: bool = False) -> "IndexFile": ...
    def remove(self, items: list, working_tree: bool = False, **kwargs) -> "IndexFile": ...
    def commit(self, message: str, parent_commits: list = None, head: bool = True, author: Actor = None, committer: Actor = None, author_date: str = None, commit_date: str = None, skip_hooks: bool = False) -> "Commit": ...
```

[Index and Staging](./index-staging.md)

### Remote Operations

Remote repository interaction including fetch, push, pull operations with comprehensive progress reporting and authentication support. Handles multiple remotes and protocol support.

```python { .api }
class Remote:
    def __init__(self, repo: "Repo", name: str): ...
    def fetch(self, refspec: str = None, progress: RemoteProgress = None, **kwargs) -> list["FetchInfo"]: ...
    def push(self, refspec: str = None, progress: RemoteProgress = None, **kwargs) -> list["PushInfo"]: ...
    def pull(self, refspec: str = None, progress: RemoteProgress = None, **kwargs) -> list["FetchInfo"]: ...

class RemoteProgress:
    def update(self, op_code: int, cur_count: str | float, max_count: str | float = None, message: str = "") -> None: ...
```

[Remote Operations](./remote.md)

### Diff Operations

Comprehensive diff functionality for comparing commits, trees, and working directory state. Supports unified diff generation, patch creation, and change detection with file-level granularity.

```python { .api }
class Diff:
    def __init__(self, repo: "Repo", a_rawpath: bytes, b_rawpath: bytes, a_blob_id: str, b_blob_id: str, a_mode: int, b_mode: int, new_file: bool, deleted_file: bool, copied_file: bool, raw_rename_from: str, raw_rename_to: str, diff: str, change_type: str): ...

class DiffIndex(list):
    def __init__(self, repo: "Repo", *args): ...
    def iter_change_type(self, change_type: str) -> Iterator["Diff"]: ...

class Diffable:
    def diff(self, other: Union["Diffable", str, None] = None, paths: Union[str, list] = None, create_patch: bool = False, **kwargs) -> "DiffIndex": ...
```

[Diff Operations](./diff.md)

### Configuration Management

Access to Git configuration at repository, user, and system levels. Supports reading and writing configuration values with proper scope management and type conversion.

```python { .api }
class GitConfigParser:
    def __init__(self, file_or_files: Union[str, list, None] = None, read_only: bool = True, merge_includes: bool = True, config_level: str = None): ...
    def get_value(self, section: str, option: str, default: Any = None) -> Any: ...
    def set_value(self, section: str, option: str, value: Any) -> "GitConfigParser": ...
    def write(self) -> None: ...
```

[Configuration](./configuration.md)

### Command Interface

Direct access to Git command-line operations with comprehensive error handling, output parsing, and environment control. Enables custom Git operations not covered by high-level interfaces.

```python { .api }
class Git:
    def __init__(self, working_dir: PathLike = None): ...
    def execute(self, command: list, istream: BinaryIO = None, with_extended_output: bool = False, with_exceptions: bool = True, as_process: bool = False, output_stream: BinaryIO = None, stdout_as_string: bool = True, kill_after_timeout: int = None, with_stdout: bool = True, universal_newlines: bool = False, shell: bool = None, env: dict = None, max_chunk_size: int = io.DEFAULT_BUFFER_SIZE, **subprocess_kwargs) -> Union[str, tuple]: ...
```

[Command Interface](./command.md)

### Exception Handling

Comprehensive exception hierarchy for robust error handling across all Git operations. Includes specific exceptions for common Git scenarios and command failures.

```python { .api }
class GitError(Exception): ...
class InvalidGitRepositoryError(GitError): ...
class GitCommandError(GitError): ...
class GitCommandNotFound(GitError): ...
class CheckoutError(GitError): ...
class RepositoryDirtyError(GitError): ...
```

[Exception Handling](./exceptions.md)

### Database and Utilities

Infrastructure components for object database access and file locking mechanisms. Provides low-level access to Git's object storage and concurrent access control.

```python { .api }
class GitDB:
    """Git object database interface."""

class GitCmdObjectDB:
    """Git object database using command-line interface."""
    def __init__(self, root_path: PathLike, git: "Git") -> None: ...

class LockFile:
    """File-based locking mechanism for concurrent access control."""

class BlockingLockFile(LockFile):
    """Blocking file lock that waits until lock can be obtained."""
```

[Database and Utilities](./database.md)

## Types

```python { .api }
from typing import Union, Optional, List, Iterator, Callable, Any, BinaryIO
from pathlib import Path

PathLike = Union[str, Path]

class Actor:
    def __init__(self, name: str, email: str): ...
    
class Stats:
    def __init__(self, repo: "Repo", total: dict, files: dict): ...
    
    @property
    def total(self) -> dict:
        """Total statistics (files, insertions, deletions)."""
    
    @property
    def files(self) -> dict:
        """Per-file statistics."""


StageType = int  # 0=base, 1=ours, 2=theirs, 3=merge

# Utility functions
def refresh(path: Optional[PathLike] = None) -> None:
    """Refresh git executable path and global configuration."""

def remove_password_if_present(url: str) -> str:
    """Remove password from URL for safe logging."""

def safe_decode(data: bytes) -> str:
    """Safe decoding of bytes to string."""

def to_hex_sha(binsha: bytes) -> str:
    """Convert binary SHA to hexadecimal string."""

def rmtree(path: PathLike, ignore_errors: bool = False) -> None:
    """Remove directory tree with proper cleanup."""
```