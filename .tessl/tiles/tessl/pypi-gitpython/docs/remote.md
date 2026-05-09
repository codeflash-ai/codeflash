# Remote Operations

Remote repository interaction including fetch, push, pull operations with comprehensive progress reporting and authentication support.

## Capabilities

### Remote Management

```python { .api }
class Remote:
    def __init__(self, repo: "Repo", name: str):
        """Initialize remote."""
    
    def fetch(self, refspec: str = None, progress: "RemoteProgress" = None, **kwargs) -> list["FetchInfo"]:
        """Fetch from remote."""
    
    def push(self, refspec: str = None, progress: "RemoteProgress" = None, **kwargs) -> list["PushInfo"]:
        """Push to remote."""
    
    def pull(self, refspec: str = None, progress: "RemoteProgress" = None, **kwargs) -> list["FetchInfo"]:
        """Pull from remote."""
    
    @property
    def name(self) -> str:
        """Remote name."""
    
    @property
    def url(self) -> str:
        """Remote URL."""

class RemoteProgress:
    def update(self, op_code: int, cur_count: Union[str, float], max_count: Union[str, float] = None, message: str = "") -> None:
        """Update progress."""

class FetchInfo:
    @property
    def ref(self) -> "Reference":
        """Updated reference."""
    
    @property
    def flags(self) -> int:
        """Fetch flags."""

class PushInfo:
    @property
    def flags(self) -> int:
        """Push flags."""
    
    @property
    def local_ref(self) -> "Reference":
        """Local reference."""
    
    @property
    def remote_ref(self) -> "RemoteReference":
        """Remote reference."""
```

## Usage Examples

```python
from git import Repo, RemoteProgress

class MyProgress(RemoteProgress):
    def update(self, op_code, cur_count, max_count=None, message=''):
        print(f'Progress: {cur_count}/{max_count} - {message}')

repo = Repo('/path/to/repo')

# Work with remotes
origin = repo.remotes.origin

# Fetch with progress
fetch_info = origin.fetch(progress=MyProgress())

# Push changes
push_info = origin.push()

# Add new remote
new_remote = repo.create_remote('upstream', 'https://github.com/user/repo.git')
```