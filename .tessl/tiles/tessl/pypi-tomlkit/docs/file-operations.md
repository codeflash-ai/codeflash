# File Operations

High-level interface for reading and writing TOML files with automatic encoding handling and line ending preservation. The TOMLFile class provides a convenient wrapper for file-based TOML operations.

## Capabilities

### TOMLFile Class

A file wrapper that handles TOML document I/O with proper encoding and line ending detection.

```python { .api }
class TOMLFile:
    """
    Represents a TOML file with automatic encoding and line ending handling.
    """
    
    def __init__(self, path: StrPath) -> None:
        """
        Initialize TOMLFile with file path.
        
        Parameters:
        - path: File system path (string or PathLike object)
        """
    
    def read(self) -> TOMLDocument:
        """
        Read the file content as a TOMLDocument.
        
        Returns:
        TOMLDocument with preserved formatting
        
        Raises:
        FileNotFoundError: If file does not exist
        ParseError: If TOML content is invalid
        UnicodeDecodeError: If file encoding is invalid
        """
    
    def write(self, data: TOMLDocument) -> None:
        """
        Write TOMLDocument to the file.
        
        Parameters:
        - data: TOMLDocument to write
        
        Raises:
        PermissionError: If file cannot be written
        """
```

## Usage Examples

### Basic File Operations

```python
import tomlkit
from tomlkit.toml_file import TOMLFile

# Create TOMLFile instance
config_file = TOMLFile("config.toml")

# Read existing file
try:
    config = config_file.read()
    print(f"Title: {config['title']}")
    print(f"Version: {config['version']}")
except FileNotFoundError:
    print("Config file not found, creating new one")
    config = tomlkit.document()

# Modify configuration
config["title"] = "Updated Application"
config["version"] = "2.0.0"
config["last_modified"] = tomlkit.datetime("2023-06-15T10:30:00Z")

# Write back to file
config_file.write(config)
```

### Line Ending Preservation

```python
import tomlkit
from tomlkit.toml_file import TOMLFile

# TOMLFile automatically detects and preserves line endings
config_file = TOMLFile("config.toml")

# Read file (line endings detected automatically)
config = config_file.read()

# Modify content
config["new_setting"] = "value"

# Write preserves original line endings
# - Windows files keep \r\n
# - Unix files keep \n  
# - Mixed files are handled appropriately
config_file.write(config)
```

### Configuration Management

```python
import tomlkit
from tomlkit.toml_file import TOMLFile
from pathlib import Path

def load_config(config_path: str = "app.toml") -> tomlkit.TOMLDocument:
    """Load configuration with defaults."""
    config_file = TOMLFile(config_path)
    
    try:
        return config_file.read()
    except FileNotFoundError:
        # Create default configuration
        config = tomlkit.document()
        config["app"] = {
            "name": "MyApp",
            "version": "1.0.0",
            "debug": False
        }
        config["server"] = {
            "host": "localhost",
            "port": 8080,
            "workers": 1
        }
        config["database"] = {
            "url": "sqlite:///app.db",
            "echo": False
        }
        
        # Save default config
        config_file.write(config)
        return config

def update_config(updates: dict, config_path: str = "app.toml"):
    """Update configuration file with new values."""
    config_file = TOMLFile(config_path)
    config = config_file.read()
    
    # Apply updates while preserving structure
    for key, value in updates.items():
        if "." in key:
            # Handle nested keys like "server.port"
            parts = key.split(".")
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = tomlkit.table()
                current = current[part]
            current[parts[-1]] = value
        else:
            config[key] = value
    
    config_file.write(config)

# Usage
config = load_config()
update_config({
    "server.port": 9000,
    "app.debug": True,
    "new_feature.enabled": True
})
```

### Backup and Rotation

```python
import tomlkit
from tomlkit.toml_file import TOMLFile
from pathlib import Path
import shutil
from datetime import datetime

class TOMLFileManager:
    """Enhanced TOML file operations with backup support."""
    
    def __init__(self, path: str):
        self.file = TOMLFile(path)
        self.path = Path(path)
    
    def read_with_backup(self) -> tomlkit.TOMLDocument:
        """Read file, creating backup first."""
        if self.path.exists():
            backup_path = self.path.with_suffix('.toml.bak')
            shutil.copy2(self.path, backup_path)
        
        return self.file.read()
    
    def safe_write(self, data: tomlkit.TOMLDocument):
        """Write with temporary file for atomicity."""
        temp_path = self.path.with_suffix('.toml.tmp')
        temp_file = TOMLFile(temp_path)
        
        try:
            # Write to temporary file first
            temp_file.write(data)
            
            # Atomic rename
            temp_path.replace(self.path)
        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def rotate_backups(self, keep: int = 5):
        """Keep only the most recent backup files."""
        backup_pattern = f"{self.path.stem}.toml.bak*"
        backups = sorted(
            self.path.parent.glob(backup_pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Remove old backups
        for backup in backups[keep:]:
            backup.unlink()

# Usage
config_mgr = TOMLFileManager("important-config.toml")

# Safe operations
config = config_mgr.read_with_backup()
config["last_backup"] = tomlkit.datetime(datetime.now().isoformat())
config_mgr.safe_write(config)
config_mgr.rotate_backups(keep=3)
```

### Multi-file Configuration

```python
import tomlkit
from tomlkit.toml_file import TOMLFile
from pathlib import Path
from typing import Dict, Any

class ConfigSet:
    """Manage multiple related TOML configuration files."""
    
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def load_all(self) -> Dict[str, tomlkit.TOMLDocument]:
        """Load all TOML files in the configuration directory."""
        configs = {}
        for toml_file in self.config_dir.glob("*.toml"):
            name = toml_file.stem
            file_obj = TOMLFile(toml_file)
            try:
                configs[name] = file_obj.read()
            except Exception as e:
                print(f"Error loading {name}: {e}")
        return configs
    
    def save_config(self, name: str, config: tomlkit.TOMLDocument):
        """Save a configuration file."""
        file_path = self.config_dir / f"{name}.toml"
        config_file = TOMLFile(file_path)
        config_file.write(config)
    
    def merge_configs(self) -> tomlkit.TOMLDocument:
        """Merge all configuration files into one document."""
        merged = tomlkit.document()
        configs = self.load_all()
        
        for name, config in configs.items():
            merged[name] = config
        
        return merged

# Usage
config_set = ConfigSet("./configs/")

# Create individual config files
app_config = tomlkit.document()
app_config["name"] = "MyApp" 
app_config["version"] = "1.0.0"

db_config = tomlkit.document()
db_config["host"] = "localhost"
db_config["port"] = 5432

config_set.save_config("app", app_config)
config_set.save_config("database", db_config)

# Load and work with all configs
all_configs = config_set.load_all()
merged_config = config_set.merge_configs()
```

### Error Handling

```python
import tomlkit
from tomlkit.toml_file import TOMLFile
from tomlkit.exceptions import ParseError
import logging

def robust_config_read(file_path: str) -> tomlkit.TOMLDocument:
    """Read TOML config with comprehensive error handling."""
    config_file = TOMLFile(file_path)
    
    try:
        return config_file.read()
    except FileNotFoundError:
        logging.warning(f"Config file {file_path} not found, using defaults")
        return create_default_config()
    except ParseError as e:
        logging.error(f"Invalid TOML syntax in {file_path}: {e}")
        # Could prompt user or try to recover
        raise
    except PermissionError:
        logging.error(f"Permission denied reading {file_path}")
        raise
    except UnicodeDecodeError as e:
        logging.error(f"Encoding error in {file_path}: {e}")
        # Could try different encodings
        raise

def create_default_config() -> tomlkit.TOMLDocument:
    """Create a default configuration."""
    config = tomlkit.document()
    config["app"] = {"name": "DefaultApp", "version": "1.0.0"}
    return config

# Usage with error handling
try:
    config = robust_config_read("my-config.toml")
    print("Configuration loaded successfully")
except Exception as e:
    print(f"Failed to load configuration: {e}")
    config = create_default_config()
```