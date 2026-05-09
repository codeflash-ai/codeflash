# Exception Handling

Comprehensive exception hierarchy for robust error handling across all Git operations. Includes specific exceptions for common Git scenarios and command failures.

## Capabilities

### Base Exceptions

```python { .api }
class GitError(Exception):
    """Base exception for all git operations."""

class InvalidGitRepositoryError(GitError):
    """Repository is not a valid git repository."""

class WorkTreeRepositoryUnsupported(GitError):
    """Operation not supported on worktree repositories."""

class NoSuchPathError(GitError):
    """Path does not exist in repository."""
```

### Command Exceptions

```python { .api }
class CommandError(GitError):
    """Git command execution failed."""

class GitCommandError(CommandError):
    """Git command returned non-zero exit code."""
    
    @property
    def status(self) -> int:
        """Command exit status."""
    
    @property
    def command(self) -> list[str]:
        """Failed command."""
    
    @property
    def stdout(self) -> str:
        """Command stdout."""
    
    @property
    def stderr(self) -> str:
        """Command stderr."""

class GitCommandNotFound(CommandError):
    """Git executable not found."""
```

### Security Exceptions

```python { .api }
class UnsafeProtocolError(GitError):
    """Unsafe protocol usage detected."""

class UnsafeOptionError(GitError):
    """Unsafe option usage detected."""
```

### Repository State Exceptions

```python { .api }
class CheckoutError(GitError):
    """Checkout operation failed."""

class RepositoryDirtyError(GitError):
    """Repository has uncommitted changes."""

class UnmergedEntriesError(GitError):
    """Index contains unmerged entries."""

class CacheError(GitError):
    """Object cache operation failed."""

class HookExecutionError(GitError):
    """Git hook execution failed."""
```

### Object Database Exceptions

```python { .api }
# Inherited from gitdb
class AmbiguousObjectName(GitError):
    """Object name matches multiple objects."""

class BadName(GitError):
    """Invalid object name."""

class BadObject(GitError):
    """Invalid or corrupted object."""

class BadObjectType(GitError):
    """Invalid object type."""

class InvalidDBRoot(GitError):
    """Invalid database root directory."""

class ODBError(GitError):
    """Object database error."""

class ParseError(GitError):
    """Object parsing failed."""

class UnsupportedOperation(GitError):
    """Operation not supported."""
```

## Usage Examples

```python
from git import Repo, GitCommandError, InvalidGitRepositoryError

try:
    repo = Repo('/invalid/path')
except InvalidGitRepositoryError as e:
    print(f"Not a git repository: {e}")

try:
    repo = Repo('/valid/repo')
    repo.git.execute(['invalid-command'])
except GitCommandError as e:
    print(f"Command failed: {e.command}")
    print(f"Exit code: {e.status}")
    print(f"Error: {e.stderr}")

# Handle specific scenarios
try:
    repo.heads.main.checkout()
except CheckoutError as e:
    print(f"Checkout failed: {e}")
    # Handle conflicts or dirty working tree

try:
    repo.index.commit("Test commit")
except RepositoryDirtyError:
    print("Cannot commit: repository has uncommitted changes")
```