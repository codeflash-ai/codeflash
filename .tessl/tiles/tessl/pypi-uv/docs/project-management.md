# Project Management

UV provides comprehensive project lifecycle management with pyproject.toml-based configuration, universal lockfiles, and Cargo-style workspace support. Projects use modern Python packaging standards with fast dependency resolution and cross-platform compatibility.

## Capabilities

### Project Initialization

Create new Python projects with pyproject.toml configuration and basic project structure.

```bash { .api }
uv init [PATH]
# Creates new project directory with:
# - pyproject.toml
# - README.md (unless --no-readme or --bare)
# - src/package/ directory structure (unless --bare)
# - .gitignore (unless --bare)
# - .python-version (unless --no-pin-python or --bare)

# Options:
# --name NAME              # Project name
# --bare                   # Only create pyproject.toml
# --package                # Create package structure (default)
# --no-package            # Don't set up as package
# --lib                    # Create library package
# --app                    # Create application package
# --script                 # Create PEP 723 script with inline metadata
# --description DESC       # Set project description
# --no-description        # Disable description for project
# --vcs {git,none}        # Initialize version control system (default: git)
# --build-backend BACKEND  # Initialize build backend (implies --package)
# --author-from {auto,git,none} # Fill authors field (default: auto)
# --no-readme             # Skip README.md creation
# --no-pin-python         # Don't pin Python version
# --no-workspace          # Avoid discovering workspace, create standalone
# --python VERSION        # Python version to use
```

Usage examples:

```bash
# Create new project in current directory
uv init

# Create project in specific directory
uv init my-project

# Create library with specific name
uv init --lib my-library --name mylib

# Create app with Python version
uv init my-app --app --python 3.12

# Create bare project (only pyproject.toml)
uv init --bare

# Create PEP 723 script
uv init script.py --script

# Create project with specific description and build backend
uv init my-lib --lib --description "My library" --build-backend setuptools
```

### Dependency Management

Add and remove project dependencies with automatic lockfile updates and conflict resolution.

```bash { .api }
uv add PACKAGE...
# Adds packages to pyproject.toml dependencies
# Updates uv.lock with resolved versions
# Installs packages in project environment

# Package specification formats:
# package                  # Latest version
# package==1.0.0          # Exact version
# package>=1.0.0          # Version constraint
# package[extra]          # With extra features
# git+https://...         # Git repository
# path/to/local           # Local path
# -e path/to/editable     # Editable install

# Options:
# --requirements, -r FILE  # Add packages from requirements file
# --constraints, -c FILE   # Apply version constraints from file
# --marker, -m MARKER      # Apply marker expression to packages
# --dev                    # Add to dev dependencies (alias for --group dev)
# --optional GROUP         # Add to optional dependency group
# --group GROUP            # Add to specified dependency group
# --editable               # Add as editable dependency
# --raw                    # Add dependency without bounds or to tool.uv.sources
# --bounds {exact,compatible,lowest-direct} # Version specifier type
# --rev REV                # Git commit to use
# --tag TAG                # Git tag to use
# --branch BRANCH          # Git branch to use
# --extra EXTRA            # Enable extras for dependency
# --no-sync                # Don't sync after adding
# --locked                 # Assert lockfile remains unchanged
# --frozen                 # Add without re-locking
# --active                 # Prefer active virtual environment
# --package PACKAGE        # Add to specific workspace package
# --script SCRIPT          # Add to PEP 723 script metadata
# --workspace              # Add as workspace member
# --no-workspace           # Don't add as workspace member
# --python VERSION         # Python version to use
```

```bash { .api }
uv remove PACKAGE...
# Removes packages from pyproject.toml
# Updates uv.lock
# Uninstalls from project environment

# Options:
# --dev                  # Remove from dev dependencies (alias for --group dev)
# --optional GROUP       # Remove from optional dependency group
# --group GROUP          # Remove from specified dependency group
# --no-sync             # Don't sync after removing
# --locked              # Assert lockfile remains unchanged
# --frozen              # Remove without re-locking
# --active              # Prefer active virtual environment
# --package PACKAGE     # Remove from specific workspace package
# --script SCRIPT       # Remove from PEP 723 script metadata
# --python VERSION      # Python version to use
```

Usage examples:

```bash
# Add production dependencies
uv add requests numpy pandas

# Add development dependencies
uv add --dev pytest black ruff

# Add with version constraints
uv add "django>=4.0,<5.0"

# Add with extras
uv add "fastapi[all]"

# Add git dependency
uv add git+https://github.com/user/repo.git

# Add git dependency with specific branch
uv add git+https://github.com/user/repo.git --branch feature

# Add from requirements file
uv add -r requirements.txt

# Add to optional dependency group
uv add --optional test pytest pytest-cov

# Add to custom group
uv add --group docs mkdocs sphinx

# Add with marker
uv add --marker "sys_platform == 'win32'" pywin32

# Add to workspace package
uv add --package my-package requests

# Add to script metadata
uv add --script myscript.py requests

# Remove dependencies
uv remove requests
uv remove --dev pytest

# Remove from specific group
uv remove --group docs sphinx

# Remove from optional dependency group
uv remove --optional test pytest-cov

# Remove from workspace package
uv remove --package my-package requests
```

### Environment Synchronization

Update project environment to match the lockfile, ensuring consistent dependency versions across environments.

```bash { .api }
uv sync
# Synchronizes environment with uv.lock
# Installs/updates/removes packages as needed
# Creates virtual environment if missing

# Options:
# --dev                   # Include dev dependencies
# --all-extras           # Include all optional groups
# --extra EXTRA          # Include specific extras
# --only-dev             # Install only dev dependencies
# --frozen               # Use exact lockfile versions
# --no-install-project   # Don't install project itself
# --no-editable          # Install project non-editably
# --python VERSION       # Python version to use
```

Usage examples:

```bash
# Sync production dependencies
uv sync

# Sync with dev dependencies
uv sync --dev

# Sync specific extras
uv sync --extra testing --extra docs

# Sync only dev dependencies
uv sync --only-dev

# Sync with exact lockfile versions
uv sync --frozen
```

### Command Execution

Run commands and scripts in the project environment with automatic environment activation and path management.

```bash { .api }
uv run COMMAND [ARGS...]
# Executes command in project environment
# Activates virtual environment automatically
# Supports scripts defined in pyproject.toml

# Options:
# --python VERSION       # Python version to use
# --with PACKAGE         # Add temporary dependencies
# --no-sync             # Skip environment sync
# --isolated            # Run in isolated environment
# --frozen              # Use exact lockfile versions
```

Usage examples:

```bash
# Run Python script
uv run python main.py

# Run Python module
uv run python -m pytest

# Run project script (defined in pyproject.toml)
uv run dev

# Run with temporary dependencies
uv run --with requests python fetch_data.py

# Run without syncing
uv run --no-sync python quick_script.py
```

### Lockfile Management

Create and update universal lockfiles for reproducible dependency resolution across platforms and Python versions.

```bash { .api }
uv lock
# Updates uv.lock with current dependencies
# Resolves all transitive dependencies
# Generates cross-platform lock entries

# Options:
# --frozen              # Don't update dependencies
# --upgrade             # Upgrade all dependencies
# --upgrade-package PKG # Upgrade specific package
# --python VERSION      # Target Python version
```

Usage examples:

```bash
# Update lockfile
uv lock

# Lock without updating dependencies
uv lock --frozen

# Upgrade all dependencies
uv lock --upgrade

# Upgrade specific packages
uv lock --upgrade-package requests --upgrade-package numpy
```

### Export and Integration

Export project dependencies to various formats for integration with other tools and deployment pipelines.

```bash { .api }
uv export
# Exports dependencies from lockfile
# Supports multiple output formats

# Options:
# --format FORMAT        # Output format (requirements-txt, etc.)
# --output-file FILE     # Output file path
# --dev                  # Include dev dependencies
# --all-extras          # Include all extras
# --extra EXTRA         # Include specific extra
# --no-hashes          # Exclude hashes from output
# --frozen             # Use exact lockfile versions
```

Usage examples:

```bash
# Export to requirements.txt
uv export --format requirements-txt --output-file requirements.txt

# Export with dev dependencies
uv export --dev --format requirements-txt > requirements-dev.txt

# Export specific extras
uv export --extra testing --format requirements-txt
```

### Version Management

Manage project version information with automatic updates across project files.

```bash { .api }
uv version [VERSION]
# Shows or updates project version
# Updates pyproject.toml and other version files

# Options:
# --bump LEVEL          # Bump version (major/minor/patch)
```

Usage examples:

```bash
# Show current version
uv version

# Set specific version
uv version 1.2.3

# Bump patch version
uv version --bump patch

# Bump minor version
uv version --bump minor
```

### Dependency Tree Visualization

Display project dependency relationships and version information in tree format.

```bash { .api }
uv tree
# Shows dependency tree for project
# Displays version information and relationships

# Options:
# --depth DEPTH         # Maximum depth to display
# --prune PACKAGE       # Prune specific packages
# --package PACKAGE     # Show tree for specific package
# --universal           # Show universal tree
```

Usage examples:

```bash
# Show full dependency tree
uv tree

# Show tree with limited depth
uv tree --depth 2

# Show tree for specific package
uv tree --package requests

# Show universal tree
uv tree --universal
```

### Code Formatting

Format Python code in the project using integrated formatters with consistent configuration.

```bash { .api }
uv format [PATH...]
# Formats Python code in project
# Uses project configuration from pyproject.toml

# Options:
# --check               # Check formatting without changes
# --diff                # Show formatting changes
# --config-file FILE    # Configuration file path
```

Usage examples:

```bash
# Format entire project
uv format

# Format specific files
uv format src/main.py tests/

# Check formatting without changes
uv format --check

# Show formatting differences
uv format --diff
```

## Project Configuration

UV projects use standard pyproject.toml configuration:

```toml { .api }
[project]
name = "my-project"
version = "0.1.0"
description = "Project description"
dependencies = [
    "requests>=2.25.0",
    "numpy>=1.20.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=22.0.0",
]
testing = [
    "pytest-cov>=4.0.0",
]

[project.scripts]
dev = "my_project.dev:main"
start = "my_project:main"

[tool.uv]
dev-dependencies = [
    "ruff>=0.1.0",
]

[tool.uv.workspace]
members = ["packages/*"]
```

## Workspace Support

UV supports Cargo-style workspaces for managing multiple packages:

```toml { .api }
# Root pyproject.toml
[tool.uv.workspace]
members = [
    "packages/*",
    "tools/cli",
]
exclude = [
    "packages/experimental",
]

# Package pyproject.toml
[project]
name = "package-name"

[tool.uv.sources]
shared-utils = { workspace = true }
```