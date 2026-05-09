# Pip Interface

UV provides a high-performance, pip-compatible interface that serves as a drop-in replacement for common pip workflows. The `uv pip` commands offer 10-100x faster performance while maintaining compatibility with existing requirements files, constraints, and pip conventions.

## Capabilities

### Package Installation

Install Python packages from PyPI, Git repositories, local paths, and other sources with fast dependency resolution.

```bash { .api }
uv pip install PACKAGE...
# Installs packages into current environment
# Supports all pip package specification formats

# Package specification formats:
# package                    # Latest version
# package==1.0.0            # Exact version
# package>=1.0.0,<2.0.0     # Version constraints
# package[extra]            # With optional dependencies
# git+https://github.com/user/repo.git  # Git repository
# git+https://github.com/user/repo.git@branch  # Git branch/tag
# https://example.com/package.whl  # Direct URL
# ./path/to/package         # Local path
# -e ./path/to/package      # Editable install

# Options:
# -r, --requirement FILE    # Install from requirements file
# -c, --constraint FILE     # Apply constraints file
# -e, --editable           # Install as editable
# --extra EXTRA            # Install optional dependencies
# --all-extras             # Install all optional dependencies
# --no-deps                # Don't install dependencies
# --force-reinstall        # Force reinstall packages
# --no-build-isolation     # Disable build isolation
# --no-index              # Ignore package index
# --find-links URL        # Additional package locations
# --index-url URL         # Base URL for package index
# --extra-index-url URL   # Extra package index URLs
# --trusted-host HOST     # Trust host for HTTP
# --pre                   # Include pre-release versions
# --python VERSION        # Target Python version
```

Usage examples:

```bash
# Install single package
uv pip install requests

# Install with version constraints
uv pip install "django>=4.0,<5.0"

# Install from requirements file
uv pip install -r requirements.txt

# Install with extras
uv pip install "fastapi[all]"

# Install from Git
uv pip install git+https://github.com/psf/requests.git

# Install editable local package
uv pip install -e ./my-package

# Install with constraints
uv pip install -r requirements.txt -c constraints.txt
```

### Package Uninstallation

Remove packages and their dependencies from the current environment.

```bash { .api }
uv pip uninstall PACKAGE...
# Uninstalls packages from current environment

# Options:
# -r, --requirement FILE    # Uninstall from requirements file
# -y, --yes                # Don't prompt for confirmation
```

Usage examples:

```bash
# Uninstall single package
uv pip uninstall requests

# Uninstall multiple packages
uv pip uninstall requests urllib3

# Uninstall from requirements file
uv pip uninstall -r requirements.txt

# Uninstall without confirmation
uv pip uninstall -y requests
```

### Requirements Compilation

Compile requirements.in files into locked requirements.txt files with resolved dependencies and hashes.

```bash { .api }
uv pip compile INPUT_FILE...
# Compiles requirements files with locked versions
# Resolves all transitive dependencies

# Input formats:
# requirements.in          # Basic requirements file
# pyproject.toml          # Extract from project dependencies
# setup.py                # Extract from setup.py
# setup.cfg               # Extract from setup.cfg

# Options:
# -o, --output-file FILE   # Output file (default: requirements.txt)
# --upgrade               # Upgrade all dependencies
# --upgrade-package PKG   # Upgrade specific packages
# -c, --constraint FILE   # Apply constraints
# --extra EXTRA          # Include optional dependencies
# --all-extras           # Include all optional dependencies
# --annotation-style STYLE  # Annotation style (line/split)
# --header               # Include header in output
# --no-header           # Exclude header from output
# --index-url URL       # Base package index URL
# --extra-index-url URL # Extra package index URLs
# --find-links URL      # Additional package locations
# --no-index           # Ignore package index
# --generate-hashes    # Generate hashes for packages
# --python VERSION     # Target Python version
# --resolution MODE    # Resolution strategy (highest/lowest-direct)
```

Usage examples:

```bash
# Compile requirements.in to requirements.txt
uv pip compile requirements.in

# Compile with custom output file
uv pip compile requirements.in -o prod-requirements.txt

# Compile pyproject.toml dependencies
uv pip compile pyproject.toml

# Upgrade all dependencies
uv pip compile requirements.in --upgrade

# Upgrade specific packages
uv pip compile requirements.in --upgrade-package requests

# Include optional dependencies
uv pip compile requirements.in --extra dev

# Generate with hashes
uv pip compile requirements.in --generate-hashes
```

### Environment Synchronization

Synchronize the current environment to exactly match a requirements file, adding and removing packages as needed.

```bash { .api }
uv pip sync REQUIREMENTS_FILE...
# Synchronizes environment with requirements file
# Installs missing packages, removes extras

# Options:
# -c, --constraint FILE    # Apply constraints file
# --reinstall             # Reinstall all packages
# --find-links URL        # Additional package locations
# --index-url URL         # Base package index URL
# --extra-index-url URL   # Extra package index URLs
# --no-index             # Ignore package index
# --trusted-host HOST    # Trust host for HTTP
# --python VERSION       # Target Python version
```

Usage examples:

```bash
# Sync with requirements file
uv pip sync requirements.txt

# Sync with constraints
uv pip sync requirements.txt -c constraints.txt

# Reinstall all packages during sync
uv pip sync requirements.txt --reinstall

# Sync multiple requirements files
uv pip sync requirements.txt dev-requirements.txt
```

### Package Listing and Inspection

List installed packages and show detailed package information.

```bash { .api }
uv pip list
uv pip ls                       # Alias for list
# Lists installed packages in tabular format

# Options:
# --format FORMAT         # Output format (columns/freeze/json)
# --exclude PACKAGE       # Exclude packages from output
# --outdated             # Show outdated packages only
```

```bash { .api }
uv pip freeze
# Lists installed packages in requirements format
# Compatible with pip freeze output

# Options:
# --exclude-editable     # Exclude editable packages
```

```bash { .api }
uv pip show PACKAGE...
# Shows detailed information about packages

# Options:
# --files                # Show installed files
```

Usage examples:

```bash
# List all packages in table format
uv pip list

# List packages in freeze format
uv pip freeze

# List in JSON format
uv pip list --format json

# Show outdated packages
uv pip list --outdated

# Show package details
uv pip show requests

# Show package with files
uv pip show requests --files
```

### Dependency Tree Visualization

Display dependency relationships in tree format for better understanding of package dependencies.

```bash { .api }
uv pip tree
# Shows dependency tree for installed packages

# Options:
# --depth DEPTH          # Maximum display depth
# --prune PACKAGE        # Prune specific packages from tree
# --package PACKAGE      # Show tree for specific package only
# --reverse             # Show reverse dependencies
```

Usage examples:

```bash
# Show full dependency tree
uv pip tree

# Show tree with limited depth
uv pip tree --depth 2

# Show dependencies for specific package
uv pip tree --package requests

# Show what depends on a package
uv pip tree --reverse --package urllib3
```

### Dependency Validation

Check installed packages for dependency conflicts and compatibility issues.

```bash { .api }
uv pip check
# Verifies installed packages have compatible dependencies
# Reports conflicts and missing dependencies

# Exit codes:
# 0: No issues found
# 1: Issues found
```

Usage examples:

```bash
# Check for dependency conflicts
uv pip check

# Use in CI/CD pipelines
uv pip check && echo "Dependencies OK"
```

## Index and Repository Configuration

Configure package indexes and repositories for private packages and mirrors:

```bash { .api }
# Index options (available for install, compile, sync):
--index-url URL              # Primary package index
--extra-index-url URL        # Additional package indexes
--find-links URL             # Local package directories or URLs
--no-index                   # Disable all package indexes
--trusted-host HOST          # Trust HTTP connections to host
```

Common configurations:

```bash
# Use private PyPI mirror
uv pip install requests --index-url https://pypi.company.com/simple/

# Use multiple indexes
uv pip install requests --extra-index-url https://test.pypi.org/simple/

# Use local package directory
uv pip install mypackage --find-links ./dist/

# Trust internal host
uv pip install requests --trusted-host internal.pypi.org
```

## Requirements File Format

UV supports standard pip requirements file syntax with extensions:

```text { .api }
# requirements.txt
requests>=2.25.0
numpy==1.21.0
pandas>=1.3.0,<2.0.0

# With extras
fastapi[all]>=0.68.0

# Git dependencies
git+https://github.com/user/repo.git@v1.0.0#egg=package

# Local paths
-e ./local-package

# URLs
https://files.pythonhosted.org/packages/.../package.whl

# Include other files
-r dev-requirements.txt

# Constraints files
-c constraints.txt

# Index configuration
--index-url https://pypi.org/simple/
--extra-index-url https://test.pypi.org/simple/
--find-links https://download.pytorch.org/whl/torch_stable.html
--trusted-host download.pytorch.org
```

## Constraint Files

Constraint files limit package versions without requiring installation:

```text { .api }
# constraints.txt
# Pin transitive dependencies
urllib3==1.26.12
certifi==2022.9.24

# Version bounds
cryptography>=3.0,<4.0
```

Usage with constraints:

```bash
uv pip install -r requirements.txt -c constraints.txt
uv pip compile requirements.in -c constraints.txt
```

## Resolution Strategies

UV supports different dependency resolution modes:

```bash { .api }
--resolution MODE
# highest (default): Prefer highest versions
# lowest-direct: Use lowest versions for direct dependencies
```

This helps test compatibility with minimum supported versions:

```bash
# Test with minimum versions
uv pip compile requirements.in --resolution lowest-direct
```

## Build Configuration

Control package building and compilation:

```bash { .api }
# Build options:
--no-build-isolation        # Disable build environment isolation
--no-binary PACKAGE         # Disable binary packages
--only-binary PACKAGE       # Only use binary packages
--config-settings KEY=VALUE # Pass settings to build backend
```

Examples:

```bash
# Force source builds
uv pip install numpy --no-binary numpy

# Only use wheels
uv pip install --only-binary :all: numpy

# Pass build settings
uv pip install package --config-settings="--build-option=--debug"
```