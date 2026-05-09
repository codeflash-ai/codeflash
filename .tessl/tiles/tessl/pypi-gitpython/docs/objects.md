# Git Objects

Access to Git's core object model including commits, trees, blobs, and tags. Provides complete metadata access, object traversal capabilities, and serialization support for repository analysis and manipulation.

## Capabilities

### Base Object Interface

Foundation class for all Git objects providing common functionality for object identification, serialization, and repository access.

```python { .api }
class Object:
    def __init__(self, repo: "Repo", binsha: bytes):
        """
        Initialize a git object.
        
        Args:
            repo: Repository containing the object
            binsha: Binary SHA-1 hash of the object
        """
    
    @property
    def hexsha(self) -> str:
        """Hexadecimal SHA-1 hash string."""
    
    @property
    def binsha(self) -> bytes:
        """Binary SHA-1 hash."""
    
    @property
    def type(self) -> str:
        """Object type string ('commit', 'tree', 'blob', 'tag')."""
    
    @property
    def size(self) -> int:
        """Size of object data in bytes."""
    
    @property
    def repo(self) -> "Repo":
        """Repository containing this object."""
    
    def __eq__(self, other: Any) -> bool:
        """Compare objects by SHA-1 hash."""
    
    def __hash__(self) -> int:
        """Hash based on SHA-1."""

class IndexObject(Object):
    """Base for objects that can be part of an index (trees, blobs, submodules)."""
    
    def __init__(self, repo: "Repo", binsha: bytes, mode: int = None, path: str = None):
        """
        Initialize index object.
        
        Args:
            repo: Repository containing object
            binsha: Binary SHA-1 hash
            mode: Unix file mode
            path: Path within repository
        """
    
    @property
    def mode(self) -> int:
        """Unix file mode."""
    
    @property
    def path(self) -> str:
        """Path within repository."""
    
    @property
    def name(self) -> str:
        """Object name (basename of path)."""
```

### Commit Objects

Represent Git commits with full metadata including author, committer, message, and parent relationships.

```python { .api }
class Commit(Object):
    def __init__(
        self, 
        repo: "Repo", 
        binsha: bytes, 
        tree: "Tree" = None,
        author: "Actor" = None, 
        authored_date: int = None,
        author_tz_offset: int = None,
        committer: "Actor" = None,
        committed_date: int = None, 
        committer_tz_offset: int = None,
        message: str = None,
        parents: list["Commit"] = None,
        encoding: str = None
    ):
        """
        Initialize commit object.
        
        Args:
            repo: Repository containing commit
            binsha: Binary SHA-1 hash
            tree: Tree object for this commit
            author: Author information
            authored_date: Author timestamp
            author_tz_offset: Author timezone offset
            committer: Committer information  
            committed_date: Commit timestamp
            committer_tz_offset: Committer timezone offset
            message: Commit message
            parents: Parent commits
            encoding: Text encoding
        """
    
    @property
    def message(self) -> str:
        """Commit message."""
    
    @property
    def author(self) -> "Actor":
        """Author information."""
    
    @property
    def committer(self) -> "Actor":
        """Committer information."""
    
    @property
    def authored_date(self) -> int:
        """Author timestamp as seconds since epoch."""
    
    @property
    def committed_date(self) -> int:
        """Commit timestamp as seconds since epoch."""
    
    @property
    def authored_datetime(self) -> datetime:
        """Author timestamp as datetime object."""
    
    @property
    def committed_datetime(self) -> datetime:
        """Commit timestamp as datetime object."""
    
    @property
    def author_tz_offset(self) -> int:
        """Author timezone offset in seconds."""
    
    @property  
    def committer_tz_offset(self) -> int:
        """Committer timezone offset in seconds."""
    
    @property
    def tree(self) -> "Tree":
        """Tree object for this commit."""
    
    @property
    def parents(self) -> tuple["Commit", ...]:
        """Parent commits."""
    
    @property
    def stats(self) -> "Stats":
        """Commit statistics (files changed, insertions, deletions)."""
    
    def iter_parents(self, paths: list[str] = None, **kwargs) -> Iterator["Commit"]:
        """
        Iterate parent commits.
        
        Args:
            paths: Limit to commits affecting these paths
            **kwargs: Additional git log options
            
        Returns:
            Iterator of parent commits
        """
    
    def iter_items(
        self, 
        repo: "Repo", 
        *args, 
        **kwargs
    ) -> Iterator["Commit"]:
        """
        Iterate commits.
        
        Args:
            repo: Repository to search
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Iterator of commits
        """
```

### Tree Objects

Represent Git tree objects (directories) with support for traversal and content access.

```python { .api }
class Tree(IndexObject):
    def __init__(self, repo: "Repo", binsha: bytes, mode: int = None, path: str = None):
        """
        Initialize tree object.
        
        Args:
            repo: Repository containing tree
            binsha: Binary SHA-1 hash
            mode: Unix file mode
            path: Path within repository
        """
    
    def __getitem__(self, item: Union[int, str]) -> IndexObject:
        """Get tree item by index or name."""
    
    def __contains__(self, item: str) -> bool:
        """Check if tree contains item."""
    
    def __len__(self) -> int:
        """Number of items in tree."""
    
    def __iter__(self) -> Iterator[IndexObject]:
        """Iterate over tree items."""
    
    @property
    def trees(self) -> list["Tree"]:
        """Subdirectories in this tree."""
    
    @property
    def blobs(self) -> list["Blob"]:
        """Files in this tree."""
    
    def traverse(
        self,
        predicate: Callable[[IndexObject, int], bool] = None,
        prune: Callable[[IndexObject, int], bool] = None,
        depth: int = -1,
        branch_first: bool = True,
        visit_once: bool = False,
        ignore_self: int = 1
    ) -> Iterator[IndexObject]:
        """
        Traverse tree recursively.
        
        Args:
            predicate: Filter function for items
            prune: Function to skip subtrees
            depth: Maximum depth (-1 for unlimited)
            branch_first: Traverse breadth-first vs depth-first
            visit_once: Visit each object only once
            ignore_self: Skip self in traversal
            
        Returns:
            Iterator of tree objects
        """

class TreeModifier:
    """Helper for modifying tree objects."""
    
    def __init__(self, cache: list = None):
        """
        Initialize tree modifier.
        
        Args:
            cache: Initial cache entries
        """
    
    def add(self, sha: Union[str, bytes], mode: int, name: str, force: bool = False) -> "TreeModifier":
        """
        Add entry to tree.
        
        Args:
            sha: Object SHA-1 hash
            mode: Unix file mode
            name: Entry name
            force: Overwrite existing entries
            
        Returns:
            Self for chaining
        """
    
    def write(self, repo: "Repo") -> "Tree":
        """
        Write modified tree to repository.
        
        Args:
            repo: Repository to write to
            
        Returns:
            New tree object
        """
```

### Blob Objects

Represent Git blob objects (files) with content access and metadata.

```python { .api }
class Blob(IndexObject):
    def __init__(self, repo: "Repo", binsha: bytes, mode: int = None, path: str = None):
        """
        Initialize blob object.
        
        Args:
            repo: Repository containing blob
            binsha: Binary SHA-1 hash  
            mode: Unix file mode
            path: Path within repository
        """
    
    @property
    def data_stream(self) -> "OStream":
        """Stream for reading blob data."""
    
    @property
    def mime_type(self) -> str:
        """MIME type of blob content."""
    
    @property 
    def encoding(self) -> Optional[str]:
        """Text encoding of blob (if text)."""
    
    def __str__(self) -> str:
        """Blob content as string."""
    
    @property
    def executable(self) -> bool:
        """True if blob has executable mode."""
```

### Tag Objects

Represent Git annotated tag objects with metadata and target object references.

```python { .api }
class TagObject(Object):
    def __init__(
        self,
        repo: "Repo", 
        binsha: bytes,
        object: Object = None,
        tag: str = None,
        tagger: "Actor" = None,
        tagged_date: int = None,
        tagger_tz_offset: int = None, 
        message: str = None
    ):
        """
        Initialize tag object.
        
        Args:
            repo: Repository containing tag
            binsha: Binary SHA-1 hash
            object: Tagged object
            tag: Tag name
            tagger: Tagger information
            tagged_date: Tag timestamp
            tagger_tz_offset: Tagger timezone offset
            message: Tag message
        """
    
    @property
    def object(self) -> Object:
        """Tagged object."""
    
    @property
    def tag(self) -> str:
        """Tag name."""
    
    @property
    def tagger(self) -> "Actor":
        """Tagger information."""
    
    @property
    def tagged_date(self) -> int:
        """Tag timestamp as seconds since epoch."""
    
    @property
    def tagged_datetime(self) -> datetime:
        """Tag timestamp as datetime object."""
    
    @property
    def tagger_tz_offset(self) -> int:
        """Tagger timezone offset in seconds."""
    
    @property
    def message(self) -> str:
        """Tag message."""
```

### Submodule Objects

Manage Git submodules with update and traversal capabilities.

```python { .api }
class Submodule(IndexObject):
    def __init__(self, repo: "Repo", binsha: bytes, mode: int = None, path: str = None, name: str = None, parent_commit: "Commit" = None, url: str = None, branch_path: str = None):
        """
        Initialize submodule.
        
        Args:
            repo: Parent repository
            binsha: Submodule commit SHA
            mode: File mode  
            path: Submodule path
            name: Submodule name
            parent_commit: Parent commit
            url: Submodule URL
            branch_path: Branch path
        """
    
    def update(
        self,
        recursive: bool = False,
        init: bool = True, 
        to_latest_revision: bool = False,
        progress: "UpdateProgress" = None,
        dry_run: bool = False,
        force: bool = False,
        keep_going: bool = False
    ) -> "Submodule":
        """
        Update submodule.
        
        Args:
            recursive: Update recursively
            init: Initialize if needed
            to_latest_revision: Update to latest
            progress: Progress reporter
            dry_run: Don't make changes
            force: Force update
            keep_going: Continue on errors
            
        Returns:
            Updated submodule
        """
    
    @property
    def module(self) -> "Repo":
        """Submodule repository."""
    
    @property
    def url(self) -> str:
        """Submodule URL."""

class UpdateProgress:
    """Progress reporter for submodule updates."""
    
    def update(
        self,
        op_code: int,
        cur_count: Union[str, float],
        max_count: Union[str, float] = None,
        message: str = ""
    ) -> None:
        """
        Update progress.
        
        Args:
            op_code: Operation code
            cur_count: Current progress
            max_count: Maximum progress  
            message: Progress message
        """

class RootModule(Submodule):
    """Virtual root of all submodules in repository for easier traversal."""
    
    def __init__(self, repo: "Repo"): ...
    
    def update(
        self,
        previous_commit: "Commit" = None,
        recursive: bool = True,
        force_remove: bool = False,
        init: bool = True,
        to_latest_revision: bool = False,
        progress: "RootUpdateProgress" = None,
        dry_run: bool = False,
        force_reset: bool = False,
        keep_going: bool = False
    ) -> "RootModule": ...

class RootUpdateProgress(UpdateProgress):
    """Extended progress reporter for root module operations with additional opcodes."""
    
    # Additional operation codes beyond UpdateProgress
    REMOVE: int
    PATHCHANGE: int
    BRANCHCHANGE: int
    URLCHANGE: int
```

## Usage Examples

### Working with Commits

```python
from git import Repo

repo = Repo('/path/to/repo')

# Get latest commit
commit = repo.head.commit

# Access commit metadata
print(f"SHA: {commit.hexsha}")
print(f"Author: {commit.author.name} <{commit.author.email}>")
print(f"Date: {commit.authored_datetime}")
print(f"Message: {commit.message}")

# Access commit tree and parents
tree = commit.tree
parents = commit.parents

# Get commit statistics
stats = commit.stats
print(f"Files changed: {stats.total['files']}")
print(f"Insertions: {stats.total['insertions']}")
print(f"Deletions: {stats.total['deletions']}")
```

### Traversing Trees

```python
# Get tree from commit
tree = commit.tree

# Access tree items
for item in tree:
    if item.type == 'tree':
        print(f"Directory: {item.path}")
    elif item.type == 'blob':
        print(f"File: {item.path} ({item.size} bytes)")

# Recursive traversal
for item in tree.traverse():
    print(f"{item.type}: {item.path}")

# Filter traversal
python_files = [item for item in tree.traverse() 
                if item.type == 'blob' and item.path.endswith('.py')]
```

### Working with Blobs

```python
# Get blob object
blob = tree['README.md']

# Access blob content
content = blob.data_stream.read().decode('utf-8')
print(content)

# Check if executable
if blob.executable:
    print("File is executable")

# Get MIME type
print(f"MIME type: {blob.mime_type}")
```

### Working with Tags

```python
# Access annotated tag
tag = repo.tags['v1.0.0']
if hasattr(tag, 'tag'):  # Annotated tag
    tag_obj = tag.tag
    print(f"Tag: {tag_obj.tag}")
    print(f"Tagger: {tag_obj.tagger}")
    print(f"Date: {tag_obj.tagged_datetime}")
    print(f"Message: {tag_obj.message}")
    print(f"Tagged object: {tag_obj.object.type}")
```

### Creating Objects

```python
from git import Actor
import time

# Create new commit
author = Actor("John Doe", "john@example.com")
committer = author

# Create tree modifier
from git.objects import TreeModifier
modifier = TreeModifier()

# Add files to tree
with open('new_file.txt', 'rb') as f:
    blob_data = f.read()

# Create blob and add to tree
blob = repo.odb.store(IStream('blob', len(blob_data), BytesIO(blob_data)))
modifier.add(blob.binsha, 0o100644, 'new_file.txt')

# Write new tree
new_tree = modifier.write(repo)

# Create new commit
new_commit = Commit.create_from_tree(
    repo,
    new_tree,
    message="Added new file",
    parent_commits=[repo.head.commit],
    author=author,
    committer=committer
)
```