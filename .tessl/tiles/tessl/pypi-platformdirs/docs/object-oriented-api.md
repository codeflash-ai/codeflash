# Object-Oriented Directory API

Object-based interface providing both string and Path object access to directories. This API enables reusable configuration and advanced features like directory iteration, making it ideal for applications that need to access multiple directories or require more complex directory management.

## Capabilities

### PlatformDirs Class

The main class providing comprehensive directory access through properties and methods. Automatically selects the appropriate platform implementation (Unix, Windows, macOS, or Android).

```python { .api }
class PlatformDirs:
    """
    Platform-specific directory paths for applications.
    
    Automatically detects the current platform and uses appropriate
    directory conventions (XDG on Unix, CSIDL on Windows, Apple guidelines on macOS).
    """
    
    def __init__(
        self,
        appname: str | None = None,
        appauthor: str | Literal[False] | None = None,
        version: str | None = None,
        roaming: bool = False,
        multipath: bool = False,
        opinion: bool = True,
        ensure_exists: bool = False,
    ) -> None:
        """
        Create a new platform directory instance.
        
        Parameters:
        - appname: Name of application
        - appauthor: Name of app author or distributing body (defaults to appname, False to disable)
        - version: Optional version path element (typically "<major>.<minor>")
        - roaming: Whether to use roaming appdata directory on Windows
        - multipath: Return entire list of directories (colon-separated on Unix) vs first item only
        - opinion: Flag indicating to use opinionated values
        - ensure_exists: Create directories on access if they don't exist
        """
    
    # Configuration properties
    appname: str | None
    appauthor: str | Literal[False] | None
    version: str | None
    roaming: bool
    multipath: bool
    opinion: bool
    ensure_exists: bool
```

### User Directory Properties

Properties returning user-specific directories as strings and Path objects.

```python { .api }
class PlatformDirs:
    @property
    def user_data_dir(self) -> str:
        """Data directory tied to the user."""
    
    @property
    def user_data_path(self) -> Path:
        """Data path tied to the user."""
    
    @property 
    def user_config_dir(self) -> str:
        """Config directory tied to the user."""
    
    @property
    def user_config_path(self) -> Path:
        """Config path tied to the user."""
    
    @property
    def user_cache_dir(self) -> str:
        """Cache directory tied to the user."""
    
    @property
    def user_cache_path(self) -> Path:
        """Cache path tied to the user."""
    
    @property
    def user_state_dir(self) -> str:
        """State directory tied to the user."""
    
    @property
    def user_state_path(self) -> Path:
        """State path tied to the user."""
    
    @property
    def user_log_dir(self) -> str:
        """Log directory tied to the user."""
    
    @property
    def user_log_path(self) -> Path:
        """Log path tied to the user."""
    
    @property
    def user_runtime_dir(self) -> str:
        """Runtime directory tied to the user."""
    
    @property
    def user_runtime_path(self) -> Path:
        """Runtime path tied to the user."""
```

### Site Directory Properties

Properties returning system-wide directories shared by all users.

```python { .api }
class PlatformDirs:
    @property
    def site_data_dir(self) -> str:
        """Data directory shared by users."""
    
    @property
    def site_data_path(self) -> Path:
        """Data path shared by users."""
    
    @property
    def site_config_dir(self) -> str:
        """Config directory shared by users."""
    
    @property
    def site_config_path(self) -> Path:
        """Config path shared by users."""
    
    @property
    def site_cache_dir(self) -> str:
        """Cache directory shared by users."""
    
    @property
    def site_cache_path(self) -> Path:
        """Cache path shared by users."""
    
    @property
    def site_runtime_dir(self) -> str:
        """Runtime directory shared by users."""
    
    @property
    def site_runtime_path(self) -> Path:
        """Runtime path shared by users."""
```

### Directory Iterator Methods

Methods for iterating over multiple related directories, useful for searching configuration or data files across user and system locations.

```python { .api }
class PlatformDirs:
    def iter_config_dirs(self) -> Iterator[str]:
        """Yield all user and site configuration directories."""
    
    def iter_config_paths(self) -> Iterator[Path]:
        """Yield all user and site configuration paths."""
    
    def iter_data_dirs(self) -> Iterator[str]:
        """Yield all user and site data directories."""
    
    def iter_data_paths(self) -> Iterator[Path]:
        """Yield all user and site data paths."""
    
    def iter_cache_dirs(self) -> Iterator[str]:
        """Yield all user and site cache directories."""
    
    def iter_cache_paths(self) -> Iterator[Path]:
        """Yield all user and site cache paths."""
    
    def iter_runtime_dirs(self) -> Iterator[str]:
        """Yield all user and site runtime directories."""
    
    def iter_runtime_paths(self) -> Iterator[Path]:
        """Yield all user and site runtime paths."""
```

### Platform-Specific Classes

Direct access to platform-specific implementations for advanced use cases.

```python { .api }
from platformdirs.unix import Unix
from platformdirs.windows import Windows  
from platformdirs.macos import MacOS
from platformdirs.android import Android

class Unix(PlatformDirsABC):
    """Unix/Linux implementation following XDG Base Directory Specification."""

class Windows(PlatformDirsABC):
    """Windows implementation using CSIDL constants and MSDN guidelines."""

class MacOS(PlatformDirsABC):
    """macOS implementation following Apple developer guidelines."""

class Android(PlatformDirsABC):
    """Android implementation for Android-specific directory conventions."""
```

## Usage Examples

```python
from platformdirs import PlatformDirs
from pathlib import Path

# Basic usage
dirs = PlatformDirs("MyApp", "MyCompany")
print(f"Data: {dirs.user_data_dir}")
print(f"Config: {dirs.user_config_dir}")
print(f"Cache: {dirs.user_cache_dir}")

# With version and auto-creation
dirs = PlatformDirs("MyApp", "MyCompany", version="2.1", ensure_exists=True)
config_path = dirs.user_config_path  # Path object, directory created if needed
data_path = dirs.user_data_path

# Create configuration file
config_file = config_path / "settings.json"
with open(config_file, 'w') as f:
    f.write('{"theme": "dark"}')

# Iterate over all possible config locations  
dirs = PlatformDirs("MyApp")
for config_dir in dirs.iter_config_dirs():
    config_file = Path(config_dir) / "settings.json"
    if config_file.exists():
        print(f"Found config: {config_file}")
        break

# Using multipath for Unix systems
dirs = PlatformDirs("MyApp", multipath=True)
# On Unix, may return colon-separated paths like "/home/user/.local/share/MyApp:/usr/local/share/MyApp"
all_data_dirs = dirs.site_data_dir

# Platform-specific usage
from platformdirs.unix import Unix
unix_dirs = Unix("MyApp", "MyCompany")
xdg_data_home = unix_dirs.user_data_dir  # Always uses XDG spec

# Backwards compatibility
from platformdirs import AppDirs  # Alias for PlatformDirs
dirs = AppDirs("MyApp", "MyCompany")  # Same as PlatformDirs
```

## Advanced Configuration

```python
from platformdirs import PlatformDirs

# Windows roaming profiles
dirs = PlatformDirs("MyApp", "MyCompany", roaming=True)
# Uses %APPDATA% instead of %LOCALAPPDATA% on Windows

# Disable appauthor on Windows
dirs = PlatformDirs("MyApp", appauthor=False)
# Windows path: %LOCALAPPDATA%\MyApp instead of %LOCALAPPDATA%\MyCompany\MyApp

# Opinion flag affects some directory choices
dirs = PlatformDirs("MyApp", opinion=False)
# May affect cache/log directory selection on some platforms

# Multiple versions can coexist
dirs_v1 = PlatformDirs("MyApp", version="1.0")
dirs_v2 = PlatformDirs("MyApp", version="2.0") 
# Each version gets separate directories
```