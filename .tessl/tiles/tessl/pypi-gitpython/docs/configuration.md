# Configuration

Access to Git configuration at repository, user, and system levels. Supports reading and writing configuration values with proper scope management and type conversion.

## Capabilities

```python { .api }
class GitConfigParser:
    def __init__(self, file_or_files: Union[str, list, None] = None, read_only: bool = True, merge_includes: bool = True, config_level: str = None):
        """Initialize config parser."""
    
    def get_value(self, section: str, option: str, default: Any = None) -> Any:
        """Get configuration value."""
    
    def set_value(self, section: str, option: str, value: Any) -> "GitConfigParser":
        """Set configuration value."""
    
    def write(self) -> None:
        """Write configuration to file."""
    
    def read(self) -> None:
        """Read configuration from file."""
    
    def has_option(self, section: str, option: str) -> bool:
        """Check if option exists."""
    
    def remove_option(self, section: str, option: str) -> bool:
        """Remove configuration option."""
    
    def remove_section(self, section: str) -> bool:
        """Remove configuration section."""

class SectionConstraint:
    """Configuration section constraints."""
```

## Usage Examples

```python
from git import Repo

repo = Repo('/path/to/repo')

# Read configuration
config = repo.config_reader()
user_name = config.get_value('user', 'name')
user_email = config.get_value('user', 'email')

# Write configuration
with repo.config_writer() as config:
    config.set_value('user', 'name', 'John Doe')
    config.set_value('user', 'email', 'john@example.com')
    config.set_value('core', 'editor', 'vim')

# Access different config levels
user_config = repo.config_reader('user')
system_config = repo.config_reader('system')
```