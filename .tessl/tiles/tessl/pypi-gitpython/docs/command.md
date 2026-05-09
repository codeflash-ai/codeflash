# Command Interface

Direct access to Git command-line operations with comprehensive error handling, output parsing, and environment control. Enables custom Git operations not covered by high-level interfaces.

## Capabilities

```python { .api }
class Git:
    def __init__(self, working_dir: PathLike = None):
        """Initialize Git command interface."""
    
    def execute(self, command: list, istream: BinaryIO = None, with_extended_output: bool = False, with_exceptions: bool = True, as_process: bool = False, output_stream: BinaryIO = None, stdout_as_string: bool = True, kill_after_timeout: int = None, with_stdout: bool = True, universal_newlines: bool = False, shell: bool = None, env: dict = None, max_chunk_size: int = None, **subprocess_kwargs) -> Union[str, tuple]:
        """Execute git command."""
    
    def __getattr__(self, name: str) -> Callable:
        """Dynamic git command creation (e.g., git.status(), git.log())."""
    
    @property
    def working_dir(self) -> str:
        """Working directory for git commands."""
    
    @classmethod
    def refresh(cls, path: PathLike = None) -> bool:
        """Refresh git executable path."""

class GitMeta(type):
    """Metaclass for Git command interface."""
```

## Usage Examples

```python
from git import Git

# Create Git command interface
git = Git('/path/to/repo')

# Execute git commands
status_output = git.status('--porcelain')
log_output = git.log('--oneline', '-10')

# With custom options
result = git.execute(['rev-parse', 'HEAD'], with_extended_output=True)
stdout, stderr = result[:2]

# Custom environment
custom_env = {'GIT_AUTHOR_NAME': 'Custom Author'}
commit_output = git.commit('-m', 'Custom commit', env=custom_env)

# Stream processing
with open('output.txt', 'wb') as f:
    git.log('--stat', output_stream=f)
```