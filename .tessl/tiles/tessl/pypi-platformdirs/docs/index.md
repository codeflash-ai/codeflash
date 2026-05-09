# Platform Directories

A small Python package for determining appropriate platform-specific directories for storing application data, configuration files, cache, logs, and other resources. This library serves as a modern replacement for the appdirs library, providing cross-platform directory management with support for the XDG Base Directory Specification on Linux, Apple guidelines on macOS, MSDN recommendations on Windows, and Android-specific conventions.

## Package Information

- **Package Name**: platformdirs
- **Language**: Python
- **Installation**: `pip install platformdirs`
- **Python Requirements**: >=3.9

## Core Imports

```python
import platformdirs
```

For functional API:

```python
from platformdirs import user_data_dir, user_config_dir, user_cache_dir
```

For object-oriented API:

```python
from platformdirs import PlatformDirs
```

For version information:

```python
from platformdirs import __version__, __version_info__
```

## Basic Usage

```python
from platformdirs import user_data_dir, user_config_dir, user_cache_dir, PlatformDirs

# Functional API - simple directory retrieval
app_data = user_data_dir("MyApp", "MyCompany")
app_config = user_config_dir("MyApp", "MyCompany") 
app_cache = user_cache_dir("MyApp", "MyCompany")

print(f"Data: {app_data}")
print(f"Config: {app_config}")
print(f"Cache: {app_cache}")

# Object-oriented API - more flexible
dirs = PlatformDirs("MyApp", "MyCompany", version="1.0")
print(f"Data: {dirs.user_data_dir}")
print(f"Config: {dirs.user_config_dir}")
print(f"Logs: {dirs.user_log_dir}")

# Ensure directories exist when accessed
dirs_auto = PlatformDirs("MyApp", ensure_exists=True)
data_path = dirs_auto.user_data_path  # Creates directory if missing

# Check version information
from platformdirs import __version__, __version_info__
print(f"platformdirs version: {__version__}")
print(f"Version info: {__version_info__}")
```

## Architecture

platformdirs uses a polymorphic architecture that automatically selects the appropriate platform implementation:

- **PlatformDirsABC**: Abstract base class defining the complete API interface
- **Platform Implementations**: Unix (XDG), Windows (CSIDL), macOS (Apple guidelines), Android
- **Automatic Detection**: Runtime platform detection with fallback to Unix implementation
- **Dual APIs**: Both functional (single-use) and object-oriented (reusable) interfaces

The design follows platform conventions while providing a unified API, making it suitable for both simple scripts and complex applications requiring extensive directory management.

## Capabilities

### Functional Directory API

Simple functions that return directory paths as strings for common use cases. Each function accepts optional parameters for app identification and behavior customization.

```python { .api }
def user_data_dir(appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, roaming: bool = False, ensure_exists: bool = False) -> str: ...
def user_config_dir(appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, roaming: bool = False, ensure_exists: bool = False) -> str: ...
def user_cache_dir(appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, opinion: bool = True, ensure_exists: bool = False) -> str: ...
def site_data_dir(appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, multipath: bool = False, ensure_exists: bool = False) -> str: ...
```

[Functional API](./functional-api.md)

### Object-Oriented Directory API 

Object-based interface providing both string and Path object access to directories. Enables reusable configuration and advanced features like directory iteration.

```python { .api }
class PlatformDirs:
    def __init__(self, appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, roaming: bool = False, multipath: bool = False, opinion: bool = True, ensure_exists: bool = False) -> None: ...
    
    @property
    def user_data_dir(self) -> str: ...
    @property 
    def user_data_path(self) -> Path: ...
```

[Object-Oriented API](./object-oriented-api.md)

### User Media Directories

Access to standard user media directories like Documents, Downloads, Pictures, Videos, Music, and Desktop. These provide system-appropriate locations regardless of platform.

```python { .api }
def user_documents_dir() -> str: ...
def user_downloads_dir() -> str: ...
def user_pictures_dir() -> str: ...
def user_videos_dir() -> str: ...
def user_music_dir() -> str: ...
def user_desktop_dir() -> str: ...
```

[User Media Directories](./user-media-directories.md)

## Types

```python { .api }
from typing import Literal
from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Iterator
from platformdirs import PlatformDirsABC

class PlatformDirsABC(ABC):
    """Abstract base class for platform directories."""
    
    appname: str | None
    appauthor: str | Literal[False] | None  
    version: str | None
    roaming: bool
    multipath: bool
    opinion: bool
    ensure_exists: bool
    
    def __init__(self, appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, roaming: bool = False, multipath: bool = False, opinion: bool = True, ensure_exists: bool = False) -> None: ...

# Platform-specific implementations
class Unix(PlatformDirsABC): ...
class Windows(PlatformDirsABC): ...  
class MacOS(PlatformDirsABC): ...
class Android(PlatformDirsABC): ...

# Type aliases for backwards compatibility
AppDirs = PlatformDirs

# Version information
__version__: str
__version_info__: tuple[int, ...]
```