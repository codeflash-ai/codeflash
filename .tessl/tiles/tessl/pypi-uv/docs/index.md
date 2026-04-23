# UV

An extremely fast Python package and project manager, written in Rust. UV serves as a comprehensive, drop-in replacement for multiple Python development tools including pip, poetry, pipx, virtualenv, pyenv, and twine, offering 10-100x faster performance while providing a unified interface for the complete Python development lifecycle.

## Package Information

- **Package Name**: uv
- **Language**: Rust (CLI tool distributed as Python package)
- **Installation**: `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Platform Support**: macOS, Linux, Windows
- **Python Support**: >=3.8

## Core Usage

UV is primarily used as a command-line tool:

```bash
# Install uv
pip install uv

# Or via standalone installer
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Basic project workflow:

```bash
# Create a new project
uv init my-project
cd my-project

# Add dependencies
uv add requests numpy

# Install dependencies
uv sync

# Run a script
uv run python main.py

# Run a tool
uv tool run black .
```

## Architecture

UV provides a unified interface through multiple command namespaces:

- **Project Commands**: Complete project lifecycle management (init, add, remove, sync, run, lock, export)
- **Package Management**: Pip-compatible interface for environment management (pip install/uninstall/sync/compile)
- **Python Management**: Version discovery, installation, and switching (python install/list/find)
- **Tool Management**: Isolated tool installation and execution (tool install/run/list)
- **Build System**: Package building and publishing (build, publish)
- **Virtual Environments**: High-performance venv creation and management

This unified approach eliminates tool fragmentation while maintaining compatibility with existing Python ecosystems through pip-compatible interfaces and universal lockfile support.

## Capabilities

### Project Management

Complete project lifecycle management with pyproject.toml-based configuration, universal lockfiles, and workspace support similar to Cargo.

```bash { .api }
uv init [PATH]                    # Create new project
uv add PACKAGE...                 # Add dependencies
uv remove PACKAGE...              # Remove dependencies
uv sync                          # Update environment to match lockfile
uv run COMMAND                   # Run command in project environment
uv lock                          # Update project lockfile
uv export [--format FORMAT]      # Export lockfile to alternate formats
```

[Project Management](./project-management.md)

### Python Version Management

Automatic Python discovery, installation, and management with support for multiple implementations and versions.

```bash { .api }
uv python list                   # List available Python installations
uv python install VERSION       # Install Python version
uv python find [REQUEST]         # Find Python installation
uv python pin VERSION           # Pin project to Python version
```

[Python Management](./python-management.md)

### Package Installation (Pip-Compatible)

High-performance pip-compatible interface providing drop-in replacement for common pip workflows with universal resolution.

```bash { .api }
uv pip install PACKAGE...        # Install packages
uv pip uninstall PACKAGE...      # Uninstall packages
uv pip compile INPUT_FILE        # Compile requirements file
uv pip sync REQUIREMENTS_FILE    # Sync environment with requirements
uv pip list                      # List installed packages
uv pip freeze                    # Output installed packages in requirements format
```

[Pip Interface](./pip-interface.md)

### Tool Management

Isolated execution and installation of Python tools with automatic dependency resolution and global availability.

```bash { .api }
uv tool run TOOL [ARGS...]       # Run tool in isolated environment
uv tool install TOOL            # Install tool globally
uv tool list                     # List installed tools
uv tool uninstall TOOL          # Uninstall tool
```

[Tool Management](./tool-management.md)

### Virtual Environment Management

Fast virtual environment creation and management with automatic discovery and Python version selection.

```bash { .api }
uv venv [PATH]                   # Create virtual environment
```

[Virtual Environments](./virtual-environments.md)

### Build and Publishing

Package building into source distributions and wheels, plus publishing to PyPI and private indexes.

```bash { .api }
uv build [PATH]                  # Build packages
uv publish [DIST...]             # Publish to index
```

[Build System](./build-system.md)

### Authentication Management

Authentication handling for PyPI and private package indexes with secure credential storage.

```bash { .api }
uv auth login [SERVICE]          # Login to service
uv auth logout [SERVICE]         # Logout from service
uv auth token [SERVICE]          # Show authentication token
```

[Authentication](./authentication.md)

### Cache Management

Global cache management for packages and metadata with deduplication and cleanup utilities.

```bash { .api }
uv cache clean                   # Clear cache
uv cache prune                   # Prune unreachable cache objects
uv cache dir                     # Show cache directory
```

[Cache Management](./cache-management.md)

### Self Management

UV self-update and version management functionality.

```bash { .api }
uv self update                   # Update uv
uv self version                  # Show version information
```

[Self Management](./self-management.md)

### Utilities

Help system and completion utilities for enhanced user experience.

```bash { .api }
uv help [COMMAND]                # Display help for commands
uv generate-shell-completion     # Generate shell completion scripts
uv clean                         # Legacy alias for cache clean
```

## Global Options

All uv commands support these global options:

```bash { .api }
--help, -h                       # Show help
--version, -V                    # Show version
--quiet, -q                      # Quiet output (repeatable)
--verbose, -v                    # Verbose output (repeatable)
--color CHOICE                   # Color output (auto/always/never)
--no-color                       # Disable colors
--python VERSION                 # Python version to use
--config-file PATH               # Configuration file path
--no-config                      # Skip configuration discovery
--offline                        # Disable network access
--native-tls                     # Use system certificate store
--allow-insecure-host HOST       # Allow insecure connections
--directory PATH                 # Change directory before running
--project PATH                   # Project directory

# Exit codes:
# 0: Success
# 1: General error (command failed)
# 2: Usage error (invalid arguments)
```

## Configuration Files

UV uses these configuration files:

- **pyproject.toml**: Project configuration and dependencies
- **uv.toml**: UV-specific configuration
- **uv.lock**: Universal lockfile for dependencies
- **.python-version**: Python version specification

## Python Module Interface

When installed via pip, uv provides a Python module interface:

```python { .api }
# Available after: pip install uv
import uv

# Find uv binary location (returns pathlib.Path)
binary_path = uv.find_uv_bin()

# Module execution (delegates to native binary)
# python -m uv [commands...]
# Exit codes: 0 (success), 1 (error), 2 (usage error)

# Example usage:
import subprocess
result = subprocess.run(["python", "-m", "uv", "pip", "list"],
                       capture_output=True, text=True)
```

## Binary Entry Points

UV provides multiple entry points:

- **uv**: Main CLI interface with all functionality
- **uvx**: Shortcut for `uv tool run` - quickly run tools
- **uvw**: Windows-specific GUI wrapper

Example uvx usage:

```bash { .api }
uvx black .                      # Equivalent to: uv tool run black .
uvx ruff check                   # Equivalent to: uv tool run ruff check
```