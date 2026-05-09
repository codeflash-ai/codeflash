# Virtual Environment Management

UV provides fast virtual environment creation and management with automatic Python version selection, efficient package installations, and seamless integration with project workflows. Virtual environments are automatically discovered and activated when working with UV projects.

## Capabilities

### Virtual Environment Creation

Create virtual environments with automatic Python version selection and optimal configuration for performance and compatibility.

```bash { .api }
uv venv [PATH]
# Creates a virtual environment
# Default path is .venv in current directory

# Aliases:
# uv virtualenv [PATH]
# uv v [PATH]

# Options:
# --python VERSION        # Python version to use
# --system-site-packages  # Give access to system packages
# --seed                 # Install seed packages (pip, setuptools)
# --relocatable          # Make environment relocatable
# --prompt PROMPT        # Set environment prompt
# --clear               # Clear existing environment
# --symlinks            # Use symlinks instead of copies
# --copies              # Use copies instead of symlinks
```

Usage examples:

```bash
# Create virtual environment in .venv
uv venv

# Create in specific directory
uv venv myproject-env

# Create with specific Python version
uv venv --python 3.12

# Create with system packages access
uv venv --system-site-packages

# Create with custom prompt
uv venv --prompt "MyProject"

# Clear and recreate existing environment
uv venv --clear
```

### Environment Discovery

UV automatically discovers virtual environments using a hierarchical search strategy that prioritizes project-specific environments.

Discovery order:
1. **Active environment**: Currently activated virtual environment
2. **Project environment**: `.venv` directory in current or parent directories
3. **Custom environment**: `UV_PROJECT_ENVIRONMENT` environment variable location
4. **Conda environments**: Active conda environments when detected

Environment variables:
```bash { .api }
UV_PROJECT_ENVIRONMENT=env_name    # Custom environment name/path
VIRTUAL_ENV=/path/to/env          # Currently active environment
CONDA_DEFAULT_ENV=env_name        # Active conda environment
```

### Environment Activation

While UV automatically uses discovered environments, manual activation is still supported for shell integration:

```bash { .api }
# Bash/Zsh activation
source .venv/bin/activate

# Windows activation
.venv\Scripts\activate

# Fish shell
source .venv/bin/activate.fish

# PowerShell
.venv\Scripts\Activate.ps1
```

UV commands automatically use the discovered environment without requiring activation:

```bash
# These commands automatically use .venv if present:
uv pip install requests
uv pip list
uv run python script.py
```

### Environment Configuration

Configure virtual environment behavior through UV settings and project configuration.

Global configuration in `uv.toml`:
```toml { .api }
[tool.uv]
# Virtual environment settings
project-environment = ".venv"      # Default environment name
python-preference = "managed"      # Python selection preference
seed-packages = true               # Install seed packages by default

# Environment creation settings
venv-symlinks = true              # Use symlinks (Unix)
venv-system-site-packages = false # System packages access
```

Project-specific configuration in `pyproject.toml`:
```toml { .api }
[tool.uv]
# Project environment settings
virtual-env = ".venv"             # Virtual environment path
python = "3.12"                   # Required Python version

# Workspace settings for monorepos
[tool.uv.workspace]
virtual-env = ".venv"             # Shared workspace environment
```

### Environment Management

UV provides utilities for managing and inspecting virtual environments.

```bash { .api }
# Show environment information
python -m site                   # Show Python paths
python -c "import sys; print(sys.prefix)"  # Show environment path

# Environment inspection
uv pip list                      # List installed packages
uv pip show package             # Show package details
uv pip freeze                   # Export environment state
```

### Environment Migration

Migrate between virtual environments or recreate environments:

```bash
# Export current environment
uv pip freeze > requirements.txt

# Create new environment
uv venv new-env --clear

# Install packages in new environment
uv pip install -r requirements.txt

# Update project to use new environment
mv new-env .venv
```

### Environment Cleanup

Remove virtual environments and clean up associated resources:

```bash { .api }
# Remove virtual environment directory
rm -rf .venv

# Or on Windows:
rmdir /s .venv

# Recreate clean environment
uv venv --clear
```

## Environment Structure

UV virtual environments follow standard Python virtual environment structure:

```text { .api }
.venv/
├── bin/                    # Executables (Unix)
│   ├── activate           # Activation script
│   ├── python            # Python interpreter symlink
│   ├── pip               # Package installer
│   └── uv                # UV executable (if installed)
├── Scripts/               # Executables (Windows)
│   ├── activate.bat      # Activation script
│   ├── python.exe       # Python interpreter
│   └── pip.exe          # Package installer
├── include/              # Header files
├── lib/                  # Installed packages (Unix)
│   └── python3.x/
│       └── site-packages/
├── Lib/                  # Installed packages (Windows)
│   └── site-packages/
└── pyvenv.cfg           # Environment configuration
```

## Environment Configuration Files

Virtual environments use configuration files to define behavior:

### pyvenv.cfg
```ini { .api }
home = /usr/bin
include-system-site-packages = false
version = 3.12.1
executable = /usr/bin/python3.12
command = /usr/bin/python3.12 -m venv .venv
```

### activate script behavior
The activation script modifies the shell environment:
- Prepends virtual environment bin directory to PATH
- Sets VIRTUAL_ENV environment variable
- Updates PS1 prompt to show environment name
- Provides deactivate function to restore original environment

## Performance Optimizations

UV optimizes virtual environment operations:

- **Fast creation**: Leverages symlinks and efficient Python detection
- **Cached downloads**: Reuses downloaded packages across environments
- **Parallel operations**: Installs packages concurrently when possible
- **Smart linking**: Uses hardlinks when safe, symlinks when beneficial

## Platform-Specific Considerations

### Unix/Linux/macOS
- Uses symlinks by default for Python interpreter and libraries
- Activation script modifies PATH and environment variables
- Supports multiple Python installations and versions

### Windows
- Uses copies instead of symlinks for compatibility
- Provides batch and PowerShell activation scripts
- Integrates with Windows registry Python installations

### Conda Integration
UV recognizes and works with conda environments:
- Detects active conda environments
- Respects conda environment paths
- Allows UV operations within conda environments

## Environment Best Practices

1. **Project isolation**: Use one virtual environment per project
2. **Version control**: Add `.venv/` to `.gitignore`
3. **Documentation**: Include environment recreation instructions
4. **Dependency tracking**: Use `requirements.txt` or `pyproject.toml`
5. **Clean environments**: Recreate environments periodically
6. **Path management**: Avoid hardcoded paths to environment

## Troubleshooting

Common virtual environment issues and solutions:

### Environment not found
```bash
# Check current directory and parents for .venv
ls -la .venv
find . -name ".venv" -type d

# Create environment if missing
uv venv
```

### Wrong Python version
```bash
# Check Python version in environment
.venv/bin/python --version

# Recreate with specific Python
uv venv --python 3.12 --clear
```

### Package installation failures
```bash
# Check environment permissions
ls -la .venv/

# Recreate environment
uv venv --clear

# Use system packages if needed
uv venv --system-site-packages
```

### PATH issues
```bash
# Check if environment is in PATH
echo $PATH | grep .venv

# Activate environment manually
source .venv/bin/activate

# Or use UV commands directly
uv pip install package
```

## Integration with IDE and Tools

Popular development environments integrate with UV virtual environments:

- **VS Code**: Automatically detects `.venv` environments
- **PyCharm**: Configure interpreter to use `.venv/bin/python`
- **Vim/Neovim**: Use environment-specific Python for language servers
- **Jupyter**: Install `ipykernel` and register environment as kernel
- **Git hooks**: Use environment Python in pre-commit configurations