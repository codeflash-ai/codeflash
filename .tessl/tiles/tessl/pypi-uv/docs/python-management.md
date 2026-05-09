# Python Management

UV provides comprehensive Python version discovery, installation, and management with support for multiple implementations (CPython, PyPy, GraalPy) and automatic downloads from official sources. It handles Python environments transparently while allowing explicit version control when needed.

## Capabilities

### Python Discovery and Listing

List and discover Python installations across the system with detailed version and implementation information.

```bash { .api }
uv python list
uv python ls                    # Alias for list
# Lists all discovered Python installations
# Shows version, implementation, and path information

# Options:
# --only-installed      # Show only uv-managed installations
# --format FORMAT       # Output format (text/json)
```

Usage examples:

```bash
# List all Python installations
uv python list

# List only uv-managed Python versions
uv python list --only-installed

# Get machine-readable output
uv python list --format json
```

### Python Installation

Download and install Python versions from official sources with automatic platform detection and optimization.

```bash { .api }
uv python install VERSION...
# Downloads and installs Python versions
# Supports multiple implementations and versions

# Version formats:
# 3.12                  # Latest patch of 3.12
# 3.12.1               # Specific version
# cpython@3.12         # CPython implementation
# pypy@3.10            # PyPy implementation
# graalpy@3.11         # GraalPy implementation
# cpython-3.12-macos-aarch64  # Platform-specific

# Options:
# --force              # Reinstall if already installed
# --python-preference  # Installation preference
```

Usage examples:

```bash
# Install latest Python 3.12
uv python install 3.12

# Install specific version
uv python install 3.12.1

# Install multiple versions
uv python install 3.11 3.12 3.13

# Install PyPy
uv python install pypy@3.10

# Force reinstall
uv python install 3.12 --force
```

### Python Discovery and Search

Find Python installations on the system based on version requests and implementation preferences.

```bash { .api }
uv python find [REQUEST]
# Finds Python installation matching request
# Uses discovery rules and preferences

# Request formats:
# 3.12                 # Version requirement
# >=3.11,<3.13        # Version range
# cpython             # Implementation
# cpython@3.12        # Implementation + version
# /path/to/python     # Specific executable

# Options:
# --format FORMAT     # Output format (text/json)
```

Usage examples:

```bash
# Find any Python 3.12
uv python find 3.12

# Find CPython 3.12
uv python find cpython@3.12

# Find Python with version range
uv python find ">=3.11,<3.13"

# Find specific executable
uv python find /usr/bin/python3
```

### Python Version Pinning

Pin projects to specific Python versions with .python-version file management.

```bash { .api }
uv python pin VERSION
# Creates/updates .python-version file
# Affects project's Python version resolution

# Options:
# --resolved          # Pin to resolved version
```

Usage examples:

```bash
# Pin to Python 3.12
uv python pin 3.12

# Pin to specific patch version
uv python pin 3.12.1

# Pin to resolved version
uv python pin 3.12 --resolved
```

### Python Directory Management

Show and manage Python installation directories and metadata.

```bash { .api }
uv python dir
# Shows Python installation directory
# Location where uv manages Python installations

# Options:
# --bin               # Show binary directory
```

Usage examples:

```bash
# Show Python installation directory
uv python dir

# Show Python binary directory
uv python dir --bin
```

### Python Version Upgrades

Upgrade installed Python versions to latest available releases.

```bash { .api }
uv python upgrade [VERSION...]
# Upgrades Python installations to latest versions
# Downloads and installs newer releases

# Options:
# --all               # Upgrade all installed versions
```

Usage examples:

```bash
# Upgrade specific version
uv python upgrade 3.12

# Upgrade all installed versions
uv python upgrade --all
```

### Python Uninstallation

Remove uv-managed Python installations to free disk space.

```bash { .api }
uv python uninstall VERSION...
# Removes uv-managed Python installations
# Does not affect system Python installations

# Options:
# --all               # Uninstall all managed versions
```

Usage examples:

```bash
# Uninstall specific version
uv python uninstall 3.11

# Uninstall multiple versions
uv python uninstall 3.10 3.11

# Uninstall all managed versions
uv python uninstall --all
```

## Python Discovery Rules

UV follows a systematic approach to finding Python installations:

1. **Virtual Environment**: Check active virtual environment or .venv in project hierarchy
2. **Project Pin**: Use .python-version file if present
3. **uv-managed**: Search uv-managed Python installations
4. **System PATH**: Search PATH environment variable for Python executables
5. **Platform-specific**: Check Windows registry, macOS framework locations
6. **Download**: Automatically download if enabled and version not found

## Python Version Request Formats

UV supports flexible Python version specification:

```bash { .api }
# Version patterns:
3                     # Latest Python 3.x
3.12                  # Latest Python 3.12.x
3.12.1               # Specific version 3.12.1

# Version specifiers:
>=3.11               # Minimum version
>=3.11,<3.13        # Version range
~=3.12.0            # Compatible release

# Implementation patterns:
cpython              # CPython implementation
cp                   # CPython (short)
pypy                 # PyPy implementation
graalpy              # GraalPy implementation

# Implementation + version:
cpython@3.12         # CPython 3.12
pypy@3.10           # PyPy 3.10
cpython3.12         # CPython 3.12 (alternate)
cp312               # CPython 3.12 (short)

# Platform-specific:
cpython-3.12.1-macos-aarch64-none
cpython-3.12.1-linux-x86_64-gnu
cpython-3.12.1-windows-x86_64-none

# Path-based:
/usr/bin/python3     # Absolute path
python3.12          # Executable name
/opt/python/        # Installation directory
```

## Global Python Configuration

Control Python behavior through global options and environment variables:

```bash { .api }
# Global options:
--python VERSION             # Specify Python version
--managed-python            # Require uv-managed Python
--no-managed-python         # Disable uv-managed Python
--no-python-downloads       # Disable automatic downloads

# Environment variables:
UV_PYTHON_PREFERENCE=managed    # Prefer managed Python
UV_PYTHON_DOWNLOADS=never      # Disable downloads
UV_MANAGED_PYTHON=true         # Force managed Python
```

## Python Version Configuration

Configure Python preferences in uv.toml:

```toml { .api }
[tool.uv]
python-preference = "managed"    # managed, system, or only-managed
python-downloads = "automatic"   # automatic, manual, or never

[tool.uv.sources]
python = "3.12"                 # Default Python version
```

Project-specific Python version in .python-version:

```text { .api }
3.12.1
```

## Supported Python Implementations

UV supports these Python implementations:

- **CPython**: Official Python implementation (most common)
- **PyPy**: Fast Python implementation with JIT compiler
- **GraalPy**: GraalVM-based Python implementation

Unsupported implementations are skipped during discovery, and requesting them results in an error.