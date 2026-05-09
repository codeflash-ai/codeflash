# Functional Directory API

Simple functions that return directory paths as strings for common use cases. This API provides a straightforward way to get platform-appropriate directories without creating objects, making it ideal for simple scripts and one-time directory lookups.

## Capabilities

### User Directory Functions

Functions for accessing user-specific directories where applications store data, configuration, cache, logs, and state information.

```python { .api }
def user_data_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    roaming: bool = False,
    ensure_exists: bool = False,
) -> str:
    """
    Return data directory tied to the user.
    
    Parameters:
    - appname: Name of application
    - appauthor: Name of app author or distributing body (defaults to appname, False to disable)
    - version: Optional version path element (typically "<major>.<minor>")
    - roaming: Whether to use roaming appdata directory on Windows
    - ensure_exists: Create directory if it doesn't exist
    
    Returns:
    Platform-appropriate user data directory path
    
    Examples:
    - Unix: ~/.local/share/$appname/$version
    - Windows: %USERPROFILE%\AppData\Local\$appauthor\$appname
    - macOS: ~/Library/Application Support/$appname/$version
    """

def user_config_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    roaming: bool = False,
    ensure_exists: bool = False,
) -> str:
    """
    Return config directory tied to the user.
    
    Parameters:
    - appname: Name of application
    - appauthor: Name of app author or distributing body
    - version: Optional version path element
    - roaming: Whether to use roaming appdata directory on Windows
    - ensure_exists: Create directory if it doesn't exist
    
    Returns:
    Platform-appropriate user config directory path
    
    Examples:
    - Unix: ~/.config/$appname/$version
    - Windows: %USERPROFILE%\AppData\Local\$appauthor\$appname
    - macOS: ~/Library/Application Support/$appname/$version
    """

def user_cache_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    opinion: bool = True,
    ensure_exists: bool = False,
) -> str:
    """
    Return cache directory tied to the user.
    
    Parameters:
    - appname: Name of application
    - appauthor: Name of app author or distributing body
    - version: Optional version path element
    - opinion: Flag indicating to use opinionated values
    - ensure_exists: Create directory if it doesn't exist
    
    Returns:
    Platform-appropriate user cache directory path
    
    Examples:
    - Unix: ~/.cache/$appname/$version
    - Windows: %USERPROFILE%\AppData\Local\$appauthor\$appname\Cache
    - macOS: ~/Library/Caches/$appname/$version
    """

def user_state_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    roaming: bool = False,
    ensure_exists: bool = False,
) -> str:
    """
    Return state directory tied to the user.
    
    Parameters:
    - appname: Name of application
    - appauthor: Name of app author or distributing body
    - version: Optional version path element
    - roaming: Whether to use roaming appdata directory on Windows
    - ensure_exists: Create directory if it doesn't exist
    
    Returns:
    Platform-appropriate user state directory path
    
    Examples:
    - Unix: ~/.local/state/$appname/$version
    - Windows: %USERPROFILE%\AppData\Local\$appauthor\$appname
    - macOS: ~/Library/Application Support/$appname/$version
    """

def user_log_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    opinion: bool = True,
    ensure_exists: bool = False,
) -> str:
    """
    Return log directory tied to the user.
    
    Parameters:
    - appname: Name of application
    - appauthor: Name of app author or distributing body
    - version: Optional version path element
    - opinion: Flag indicating to use opinionated values
    - ensure_exists: Create directory if it doesn't exist
    
    Returns:
    Platform-appropriate user log directory path
    
    Examples:
    - Unix: ~/.local/state/$appname/$version/log
    - Windows: %USERPROFILE%\AppData\Local\$appauthor\$appname\Logs
    - macOS: ~/Library/Logs/$appname/$version
    """

def user_runtime_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    opinion: bool = True,
    ensure_exists: bool = False,
) -> str:
    """
    Return runtime directory tied to the user.
    
    Parameters:
    - appname: Name of application
    - appauthor: Name of app author or distributing body
    - version: Optional version path element
    - opinion: Flag indicating to use opinionated values
    - ensure_exists: Create directory if it doesn't exist
    
    Returns:
    Platform-appropriate user runtime directory path
    
    Examples:
    - Unix: /run/user/$uid/$appname/$version
    - Windows: %USERPROFILE%\AppData\Local\Temp\$appauthor\$appname
    - macOS: ~/Library/Caches/TemporaryItems/$appname/$version
    """
```

### Site Directory Functions

Functions for accessing system-wide directories shared by all users, typically used for application data, configuration, and cache that should be available system-wide.

```python { .api }
def site_data_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    multipath: bool = False,
    ensure_exists: bool = False,
) -> str:
    """
    Return data directory shared by users.
    
    Parameters:
    - appname: Name of application
    - appauthor: Name of app author or distributing body
    - version: Optional version path element
    - multipath: Return entire list of data dirs (colon-separated on Unix)
    - ensure_exists: Create directory if it doesn't exist
    
    Returns:
    Platform-appropriate site data directory path
    
    Examples:
    - Unix: /usr/local/share/$appname/$version
    - Windows: %ALLUSERSPROFILE%\$appauthor\$appname
    - macOS: /Library/Application Support/$appname/$version
    """

def site_config_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    multipath: bool = False,
    ensure_exists: bool = False,
) -> str:
    """
    Return config directory shared by users.
    
    Parameters:
    - appname: Name of application
    - appauthor: Name of app author or distributing body
    - version: Optional version path element
    - multipath: Return entire list of config dirs (colon-separated on Unix)
    - ensure_exists: Create directory if it doesn't exist
    
    Returns:
    Platform-appropriate site config directory path
    
    Examples:
    - Unix: /etc/$appname/$version
    - Windows: %ALLUSERSPROFILE%\$appauthor\$appname
    - macOS: /Library/Application Support/$appname/$version
    """

def site_cache_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    opinion: bool = True,
    ensure_exists: bool = False,
) -> str:
    """
    Return cache directory shared by users.
    
    Parameters:
    - appname: Name of application
    - appauthor: Name of app author or distributing body
    - version: Optional version path element
    - opinion: Flag indicating to use opinionated values
    - ensure_exists: Create directory if it doesn't exist
    
    Returns:
    Platform-appropriate site cache directory path
    
    Examples:
    - Unix: /var/cache/$appname/$version
    - Windows: %ALLUSERSPROFILE%\$appauthor\$appname\Cache
    - macOS: /Library/Caches/$appname/$version
    """

def site_runtime_dir(
    appname: str | None = None,
    appauthor: str | Literal[False] | None = None,
    version: str | None = None,
    opinion: bool = True,
    ensure_exists: bool = False,
) -> str:
    """
    Return runtime directory shared by users.
    
    Parameters:
    - appname: Name of application
    - appauthor: Name of app author or distributing body
    - version: Optional version path element
    - opinion: Flag indicating to use opinionated values
    - ensure_exists: Create directory if it doesn't exist
    
    Returns:
    Platform-appropriate site runtime directory path
    
    Examples:
    - Unix: /run/$appname/$version
    - Windows: %ALLUSERSPROFILE%\$appauthor\$appname
    - macOS: /var/run/$appname/$version
    """
```

### Path Versions

Each directory function has a corresponding path version that returns a `pathlib.Path` object instead of a string:

```python { .api }
def user_data_path(appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, roaming: bool = False, ensure_exists: bool = False) -> Path: ...
def user_config_path(appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, roaming: bool = False, ensure_exists: bool = False) -> Path: ...
def user_cache_path(appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, opinion: bool = True, ensure_exists: bool = False) -> Path: ...
def user_state_path(appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, roaming: bool = False, ensure_exists: bool = False) -> Path: ...
def user_log_path(appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, opinion: bool = True, ensure_exists: bool = False) -> Path: ...
def user_runtime_path(appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, opinion: bool = True, ensure_exists: bool = False) -> Path: ...
def site_data_path(appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, multipath: bool = False, ensure_exists: bool = False) -> Path: ...
def site_config_path(appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, multipath: bool = False, ensure_exists: bool = False) -> Path: ...
def site_cache_path(appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, opinion: bool = True, ensure_exists: bool = False) -> Path: ...
def site_runtime_path(appname: str | None = None, appauthor: str | Literal[False] | None = None, version: str | None = None, opinion: bool = True, ensure_exists: bool = False) -> Path: ...
```

## Usage Examples

```python
from platformdirs import user_data_dir, user_config_dir, site_data_dir

# Simple usage with just app name
data_dir = user_data_dir("MyApp")
config_dir = user_config_dir("MyApp")

# With app author for Windows compatibility
data_dir = user_data_dir("MyApp", "MyCompany")

# With version for multiple versions
data_dir = user_data_dir("MyApp", "MyCompany", version="2.1")

# Ensure directory exists when accessed
data_dir = user_data_dir("MyApp", ensure_exists=True)

# Get site-wide directories  
site_data = site_data_dir("MyApp", "MyCompany")

# Using Path objects for advanced path operations
from platformdirs import user_data_path
data_path = user_data_path("MyApp")
config_file = data_path / "config.json"
```