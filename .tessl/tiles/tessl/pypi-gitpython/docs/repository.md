# Repository Management

Core repository operations providing the foundation for all Git interactions through the central `Repo` interface. Handles repository initialization, cloning, configuration access, and state inspection.

## Capabilities

### Repository Creation and Access

Create new repositories, clone from remotes, and open existing repositories with comprehensive path and configuration support.

```python { .api }
class Repo:
    def __init__(
        self, 
        path: PathLike = None, 
        odbt: Type[GitCmdObjectDB] = GitCmdObjectDB, 
        search_parent_directories: bool = False, 
        expand_vars: bool = True
    ):
        """
        Initialize a repository object.
        
        Args:
            path: Path to repository root or .git directory
            odbt: Object database type to use
            search_parent_directories: Search parent dirs for .git directory
            expand_vars: Expand environment variables in path
        """
    
    @classmethod
    def init(
        cls, 
        path: PathLike, 
        mkdir: bool = True, 
        odbt: Type[GitCmdObjectDB] = GitCmdObjectDB, 
        expand_vars: bool = True, 
        **kwargs
    ) -> "Repo":
        """
        Initialize a new git repository.
        
        Args:
            path: Path where repository should be created
            mkdir: Create directory if it doesn't exist
            odbt: Object database type
            expand_vars: Expand environment variables
            **kwargs: Additional git init options (bare, shared, etc.)
            
        Returns:
            New Repo instance
        """
    
    @classmethod
    def clone_from(
        cls, 
        url: str, 
        to_path: PathLike, 
        progress: "RemoteProgress" = None, 
        env: dict = None, 
        multi_options: list = None, 
        **kwargs
    ) -> "Repo":
        """
        Clone repository from URL.
        
        Args:
            url: Repository URL to clone from
            to_path: Local path to clone to
            progress: Progress reporter instance
            env: Environment variables for git command
            multi_options: Multiple option flags
            **kwargs: Additional clone options (branch, depth, etc.)
            
        Returns:
            Cloned Repo instance
        """
```

### Repository State Inspection

Query repository state including dirty status, active branch, untracked files, and commit history with filtering support.

```python { .api }
def is_dirty(
    self, 
    index: bool = True, 
    working_tree: bool = True, 
    untracked_files: bool = False, 
    submodules: bool = True, 
    path: PathLike = None
) -> bool:
    """
    Check if repository has uncommitted changes.
    
    Args:
        index: Check staged changes
        working_tree: Check working directory changes  
        untracked_files: Include untracked files
        submodules: Check submodules for changes
        path: Limit check to specific path
        
    Returns:
        True if repository has changes
    """

def iter_commits(
    self, 
    rev: str = None, 
    paths: Union[str, list] = None, 
    **kwargs
) -> Iterator["Commit"]:
    """
    Iterate over commits in repository.
    
    Args:
        rev: Starting revision (default: HEAD)
        paths: Limit to commits affecting these paths
        **kwargs: Additional options (max_count, skip, since, until, etc.)
        
    Returns:
        Iterator of Commit objects
    """

@property
def active_branch(self) -> "Head":
    """Current active branch."""

@property
def untracked_files(self) -> list[str]:
    """List of untracked file paths."""

@property
def heads(self) -> "IterableList[Head]":
    """All branch heads."""

@property
def tags(self) -> "IterableList[TagReference]":
    """All tags."""

@property
def remotes(self) -> "IterableList[Remote]":
    """All remotes."""
```

### Configuration Access

Access repository, user, and system Git configuration with read and write capabilities.

```python { .api }
def config_reader(self, config_level: str = "repository") -> "GitConfigParser":
    """
    Get configuration reader.
    
    Args:
        config_level: Configuration level ('repository', 'user', 'system')
        
    Returns:
        Read-only configuration parser
    """

def config_writer(self, config_level: str = "repository") -> "GitConfigParser":
    """
    Get configuration writer.
    
    Args:
        config_level: Configuration level to write to
        
    Returns:
        Configuration parser for writing
    """
```

### Repository Properties

Access to repository paths, state, and metadata.

```python { .api }
@property
def git_dir(self) -> str:
    """Path to .git directory."""

@property
def working_dir(self) -> Optional[str]:
    """Path to working directory (None for bare repos)."""

@property
def working_tree_dir(self) -> Optional[str]:
    """Alias for working_dir."""

@property
def bare(self) -> bool:
    """True if repository is bare."""

@property
def git(self) -> "Git":
    """Git command interface for this repository."""

@property
def index(self) -> "IndexFile":
    """Repository index/staging area."""

@property
def head(self) -> "HEAD":
    """HEAD reference."""

@property
def common_dir(self) -> str:
    """Path to common git directory."""
```

### Archive and Export

Create archives and export repository content.

```python { .api }
def archive(
    self, 
    ostream: BinaryIO, 
    treeish: str = None, 
    prefix: str = None, 
    **kwargs
) -> "Repo":
    """
    Create archive of repository content.
    
    Args:
        ostream: Output stream for archive
        treeish: Tree-ish to archive (default: HEAD)
        prefix: Prefix for archive entries
        **kwargs: Additional archive options
        
    Returns:
        Self for chaining
    """
```

## Usage Examples

### Basic Repository Operations

```python
from git import Repo

# Create new repository
repo = Repo.init('/path/to/new/repo')

# Open existing repository  
repo = Repo('/path/to/existing/repo')

# Clone repository
repo = Repo.clone_from(
    'https://github.com/user/repo.git',
    '/local/path',
    branch='main'
)

# Check repository state
if repo.is_dirty():
    print("Repository has uncommitted changes")

print(f"Active branch: {repo.active_branch}")
print(f"Untracked files: {repo.untracked_files}")
```

### Working with Commits

```python
# Get recent commits
for commit in repo.iter_commits('main', max_count=10):
    print(f"{commit.hexsha[:7]} - {commit.message.strip()}")

# Get commits for specific file
file_commits = list(repo.iter_commits(paths='README.md', max_count=5))

# Get commits in date range  
import datetime
since = datetime.datetime(2023, 1, 1)
recent_commits = list(repo.iter_commits(since=since))
```

### Configuration Management

```python
# Read configuration
config = repo.config_reader()
user_name = config.get_value('user', 'name')
user_email = config.get_value('user', 'email')

# Write configuration  
with repo.config_writer() as config:
    config.set_value('user', 'name', 'John Doe')
    config.set_value('user', 'email', 'john@example.com')
```

### Repository Inspection

```python
# Check various repository properties
print(f"Git directory: {repo.git_dir}")
print(f"Working directory: {repo.working_dir}")
print(f"Is bare: {repo.bare}")

# List branches and tags
for branch in repo.heads:
    print(f"Branch: {branch.name}")

for tag in repo.tags:
    print(f"Tag: {tag.name}")

# List remotes
for remote in repo.remotes:
    print(f"Remote: {remote.name} -> {remote.url}")
```