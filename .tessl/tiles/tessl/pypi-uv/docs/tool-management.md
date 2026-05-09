# Tool Management

UV provides isolated tool execution and installation capabilities, allowing you to run and install Python-packaged tools without affecting your project environment or system Python. Tools are installed in isolated environments with automatic dependency resolution and global availability.

## Capabilities

### Tool Execution

Run Python tools in isolated environments without permanent installation, automatically handling dependencies and environment setup.

```bash { .api }
uv tool run TOOL [ARGS...]
# Runs tool in isolated environment
# Downloads and installs tool temporarily if not installed

# Tool specification formats:
# tool                     # Latest version from PyPI
# tool@version            # Specific version
# tool==1.0.0             # Exact version constraint
# git+https://...         # Git repository
# path/to/local           # Local path
# tool --from package     # Tool from different package

# Options:
# --from PACKAGE          # Install tool from specific package
# --with DEPENDENCY       # Add temporary dependencies
# --python VERSION        # Python version to use
# --isolated             # Force isolated environment
# --no-project           # Ignore project configuration
```

Usage examples:

```bash
# Run black formatter
uv tool run black .

# Run specific version
uv tool run black@23.0.0 .

# Run with additional dependencies
uv tool run --with requests httpie http://example.com

# Run tool from different package
uv tool run --from django-admin django-admin startproject myproject

# Run from Git repository
uv tool run git+https://github.com/psf/black.git .
```

### Tool Installation

Install Python tools globally for persistent availability across projects and shell sessions.

```bash { .api }
uv tool install TOOL
# Installs tool globally in isolated environment
# Makes tool commands available in PATH

# Tool specification formats:
# tool                    # Latest version from PyPI
# tool@version           # Specific version
# tool==1.0.0            # Exact version constraint
# git+https://...        # Git repository
# path/to/local          # Local path

# Options:
# --from PACKAGE         # Install from specific package
# --with DEPENDENCY      # Include additional dependencies
# --python VERSION       # Python version to use
# --force               # Force reinstall if exists
# --editable           # Install as editable (local paths)
```

Usage examples:

```bash
# Install black formatter
uv tool install black

# Install specific version
uv tool install black@23.0.0

# Install with additional dependencies
uv tool install --with keyring twine

# Install from Git
uv tool install git+https://github.com/astral-sh/ruff.git

# Force reinstall
uv tool install black --force

# Install editable local tool
uv tool install --editable ./my-tool/
```

### Tool Listing and Information

List installed tools and show detailed information about tool installations.

```bash { .api }
uv tool list
uv tool ls                      # Alias for list
# Lists all installed tools with versions and entry points

# Options:
# --show-paths          # Show installation paths
# --format FORMAT       # Output format (text/json)
```

Usage examples:

```bash
# List all installed tools
uv tool list

# Show tool installation paths
uv tool list --show-paths

# Get machine-readable output
uv tool list --format json
```

### Tool Updates and Upgrades

Update tools to their latest versions with dependency resolution and conflict checking.

```bash { .api }
uv tool upgrade TOOL...
# Upgrades installed tools to latest versions

# Options:
# --all                 # Upgrade all installed tools
```

Usage examples:

```bash
# Upgrade specific tool
uv tool upgrade black

# Upgrade multiple tools
uv tool upgrade black ruff mypy

# Upgrade all installed tools
uv tool upgrade --all
```

### Tool Uninstallation

Remove installed tools and clean up their isolated environments.

```bash { .api }
uv tool uninstall TOOL...
# Uninstalls tools and removes their environments

# Options:
# --all                 # Uninstall all tools
```

Usage examples:

```bash
# Uninstall specific tool
uv tool uninstall black

# Uninstall multiple tools
uv tool uninstall black ruff mypy

# Uninstall all tools
uv tool uninstall --all
```

### PATH Management

Ensure tool binaries are available in the system PATH for seamless command-line access.

```bash { .api }
uv tool update-shell
# Updates shell configuration to include tool directory in PATH
# Modifies .bashrc, .zshrc, or equivalent shell config

# Alias:
uv tool ensurepath

# Options:
# --shell SHELL         # Specify shell type
```

Usage examples:

```bash
# Update shell PATH configuration
uv tool update-shell

# Update for specific shell
uv tool update-shell --shell zsh

# Alternative command name
uv tool ensurepath
```

### Tool Directory Management

Show and manage tool installation directories and configuration.

```bash { .api }
uv tool dir
# Shows tool installation directory
# Location where uv stores tool environments

# Options:
# --bin                 # Show binary directory
```

Usage examples:

```bash
# Show tool installation directory
uv tool dir

# Show tool binary directory
uv tool dir --bin
```

## Tool Discovery and Entry Points

UV automatically discovers and manages entry points from installed tools:

- **Console Scripts**: Primary command-line interfaces defined in package metadata
- **GUI Scripts**: Graphical applications (on platforms that support them)
- **Module Execution**: Tools that support `python -m module` execution

When you install a tool, UV:
1. Creates an isolated virtual environment for the tool
2. Installs the tool and its dependencies
3. Links entry points to the global tool binary directory
4. Makes commands available in PATH (after shell configuration)

## Tool Isolation Benefits

Each tool runs in its own isolated environment, providing:

- **Dependency Isolation**: No conflicts between tool dependencies
- **Version Isolation**: Multiple versions of tools can coexist
- **System Protection**: No impact on system or project Python environments
- **Clean Uninstalls**: Complete removal of tools and dependencies

## Tool Specification Formats

UV supports flexible tool specification:

```bash { .api }
# Package names:
black                    # Latest version
ruff                     # Latest version

# Version constraints:
black@23.0.0            # Specific version
black>=23.0.0           # Minimum version
black==23.0.0,<24.0.0   # Version range

# Git repositories:
git+https://github.com/psf/black.git
git+https://github.com/psf/black.git@main
git+https://github.com/psf/black.git@v23.0.0

# Local paths:
./my-tool               # Relative path
/path/to/tool          # Absolute path
~/dev/my-tool          # Home directory path

# URLs:
https://files.pythonhosted.org/packages/.../tool.whl

# Tool from different package:
--from django django-admin    # Install django-admin from django package
--from jupyter jupyter-lab    # Install jupyter-lab from jupyter package
```

## Common Tool Examples

Popular Python tools that work well with UV tool management:

```bash { .api }
# Code formatting and linting:
uv tool install black              # Code formatter
uv tool install ruff               # Fast linter and formatter
uv tool install mypy               # Type checker
uv tool install flake8             # Style guide enforcement
uv tool install isort              # Import sorter

# Development tools:
uv tool install poetry             # Dependency management
uv tool install pipx               # Tool installer (alternative to uv tool)
uv tool install pre-commit         # Git hooks framework
uv tool install tox                # Testing in multiple environments

# Publishing and packaging:
uv tool install twine              # PyPI package uploader
uv tool install build              # PEP 517/518 build frontend
uv tool install wheel              # Wheel package format

# Web development:
uv tool install django-admin       # Django admin command
uv tool install flask              # Flask web framework CLI
uv tool install cookiecutter       # Project templates

# Data science:
uv tool install jupyter            # Jupyter notebooks
uv tool install jupyterlab         # JupyterLab IDE
uv tool install datasette          # Data exploration tool

# Documentation:
uv tool install mkdocs             # Documentation generator
uv tool install sphinx             # Documentation tool
uv tool install pdoc               # API documentation
```

## Tool Environment Configuration

Configure tool behavior through UV settings:

```toml { .api }
[tool.uv]
# Tool installation settings
tool-python = "3.12"              # Default Python for tools
tool-upgrade = true                # Auto-upgrade tools

[tool.uv.tool-sources]
# Custom tool sources
black = { git = "https://github.com/psf/black.git" }
ruff = { path = "./local-ruff" }
```

## Integration with uvx

The `uvx` binary provides a shortcut for `uv tool run`:

```bash { .api }
# These are equivalent:
uvx black .
uv tool run black .

# Version specification:
uvx black@23.0.0 .
uv tool run black@23.0.0 .

# With additional dependencies:
uvx --with requests httpie http://example.com
uv tool run --with requests httpie http://example.com
```

## Tool Environment Variables

Control tool behavior with environment variables:

```bash { .api }
UV_TOOL_DIR=/custom/tools        # Custom tool directory
UV_TOOL_BIN_DIR=/custom/bin      # Custom binary directory
UV_NO_PROGRESS=1                 # Disable progress bars
```

## Troubleshooting Tool Installation

Common issues and solutions:

1. **Command not found**: Run `uv tool update-shell` to update PATH
2. **Tool conflicts**: Each tool is isolated, so conflicts shouldn't occur
3. **Version issues**: Use specific version constraints or `--force` to reinstall
4. **Permission errors**: UV installs to user directory, no sudo required
5. **Path issues**: Check `uv tool dir --bin` and ensure it's in PATH