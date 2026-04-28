# User Media Directories

Access to standard user media directories like Documents, Downloads, Pictures, Videos, Music, and Desktop. These provide system-appropriate locations for user content regardless of platform, following OS conventions for where users typically store their personal files.

## Capabilities

### Media Directory Functions

Simple functions that return standard user media directory paths. These directories are user-specific and don't require application identification since they represent standard system locations.

```python { .api }
def user_documents_dir() -> str:
    """
    Return documents directory tied to the user.
    
    Returns:
    Platform-appropriate user documents directory path
    
    Examples:
    - Unix: ~/Documents
    - Windows: %USERPROFILE%\Documents
    - macOS: ~/Documents
    """

def user_downloads_dir() -> str:
    """
    Return downloads directory tied to the user.
    
    Returns:
    Platform-appropriate user downloads directory path
    
    Examples:
    - Unix: ~/Downloads
    - Windows: %USERPROFILE%\Downloads  
    - macOS: ~/Downloads
    """

def user_pictures_dir() -> str:
    """
    Return pictures directory tied to the user.
    
    Returns:
    Platform-appropriate user pictures directory path
    
    Examples:
    - Unix: ~/Pictures
    - Windows: %USERPROFILE%\Pictures
    - macOS: ~/Pictures
    """

def user_videos_dir() -> str:
    """
    Return videos directory tied to the user.
    
    Returns:
    Platform-appropriate user videos directory path
    
    Examples:
    - Unix: ~/Videos
    - Windows: %USERPROFILE%\Videos
    - macOS: ~/Movies
    """

def user_music_dir() -> str:
    """
    Return music directory tied to the user.
    
    Returns:
    Platform-appropriate user music directory path
    
    Examples:
    - Unix: ~/Music
    - Windows: %USERPROFILE%\Music
    - macOS: ~/Music
    """

def user_desktop_dir() -> str:
    """
    Return desktop directory tied to the user.
    
    Returns:
    Platform-appropriate user desktop directory path
    
    Examples:
    - Unix: ~/Desktop
    - Windows: %USERPROFILE%\Desktop
    - macOS: ~/Desktop
    """
```

### Media Directory Path Functions

Path versions that return `pathlib.Path` objects instead of strings for advanced path manipulation.

```python { .api }
def user_documents_path() -> Path:
    """Return documents path tied to the user."""

def user_downloads_path() -> Path:
    """Return downloads path tied to the user."""

def user_pictures_path() -> Path:
    """Return pictures path tied to the user."""

def user_videos_path() -> Path:
    """Return videos path tied to the user."""

def user_music_path() -> Path:
    """Return music path tied to the user."""

def user_desktop_path() -> Path:
    """Return desktop path tied to the user."""
```

### Object-Oriented Access

Media directories are also available through PlatformDirs properties for consistency with the OOP interface.

```python { .api }
class PlatformDirs:
    @property
    def user_documents_dir(self) -> str:
        """Documents directory tied to the user."""
    
    @property
    def user_documents_path(self) -> Path:
        """Documents path tied to the user."""
    
    @property
    def user_downloads_dir(self) -> str:
        """Downloads directory tied to the user."""
    
    @property
    def user_downloads_path(self) -> Path:
        """Downloads path tied to the user."""
    
    @property
    def user_pictures_dir(self) -> str:
        """Pictures directory tied to the user."""
    
    @property
    def user_pictures_path(self) -> Path:
        """Pictures path tied to the user."""
    
    @property
    def user_videos_dir(self) -> str:
        """Videos directory tied to the user."""
    
    @property
    def user_videos_path(self) -> Path:
        """Videos path tied to the user."""
    
    @property
    def user_music_dir(self) -> str:
        """Music directory tied to the user."""
    
    @property
    def user_music_path(self) -> Path:
        """Music path tied to the user."""
    
    @property
    def user_desktop_dir(self) -> str:
        """Desktop directory tied to the user."""
    
    @property
    def user_desktop_path(self) -> Path:
        """Desktop path tied to the user."""
```

## Usage Examples

```python
from platformdirs import (
    user_documents_dir, user_downloads_dir, user_pictures_dir,
    user_videos_dir, user_music_dir, user_desktop_dir,
    user_documents_path, PlatformDirs
)
from pathlib import Path

# Functional API - get standard user directories
docs = user_documents_dir()
downloads = user_downloads_dir()  
pictures = user_pictures_dir()
videos = user_videos_dir()
music = user_music_dir()
desktop = user_desktop_dir()

print(f"Documents: {docs}")
print(f"Downloads: {downloads}")
print(f"Pictures: {pictures}")

# Using Path objects for file operations
docs_path = user_documents_path()
my_file = docs_path / "MyDocument.txt"
with open(my_file, 'w') as f:
    f.write("Hello, World!")

# Object-oriented access
dirs = PlatformDirs()  # No app name needed for media dirs
downloads_path = dirs.user_downloads_path
pictures_path = dirs.user_pictures_path

# Organize files by type
def organize_downloads():
    downloads = user_downloads_path()
    pictures = user_pictures_path()
    videos = user_videos_path()
    music = user_music_path()
    
    for file in downloads.iterdir():
        if file.suffix.lower() in ['.jpg', '.png', '.gif', '.bmp']:
            file.rename(pictures / file.name)
        elif file.suffix.lower() in ['.mp4', '.avi', '.mkv', '.mov']:
            file.rename(videos / file.name)
        elif file.suffix.lower() in ['.mp3', '.wav', '.flac', '.ogg']:
            file.rename(music / file.name)

# Create a desktop shortcut file (Windows .lnk, Unix .desktop)
desktop_path = user_desktop_path()
if desktop_path.exists():
    shortcut = desktop_path / "MyApp.txt"
    with open(shortcut, 'w') as f:
        f.write("Link to my application")
```

## Platform Behavior

### Unix/Linux
- Follows XDG user directories specification when available
- Falls back to common home directory subdirectories
- Respects `$XDG_*_DIR` environment variables when set

### Windows  
- Uses Windows Known Folders (CSIDL constants)
- Typically located under `%USERPROFILE%`
- Localizes directory names based on system language

### macOS
- Follows Apple File System Programming Guide
- Uses appropriate directories under user home
- Videos directory maps to ~/Movies following macOS convention

### Android
- Maps to appropriate Android storage directories
- Considers both internal and external storage locations
- Handles Android permission model appropriately

## Notes

- Media directories are user-specific and don't require application parameters
- These directories typically exist by default on most systems
- Some directories may not exist on minimal/server installations
- Always check for directory existence before assuming it's available
- Consider using `ensure_exists=True` with PlatformDirs if you need to create parent directories