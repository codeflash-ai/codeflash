# Build System

UV provides comprehensive package building and publishing capabilities, supporting modern Python packaging standards (PEP 517/518) with fast builds, multiple output formats, and seamless integration with PyPI and private indexes.

## Capabilities

### Package Building

Build Python packages into distributable formats with automatic dependency resolution and platform-specific optimizations.

```bash { .api }
uv build [PATH]
# Builds Python packages from source
# Default PATH is current directory

# Build targets:
# - Source distribution (sdist)
# - Binary distribution (wheel)
# - Both (default behavior)

# Options:
# --sdist                  # Build source distribution only
# --wheel                  # Build wheel only
# --out-dir DIR           # Output directory (default: dist/)
# --config-setting KEY=VALUE  # Pass settings to build backend
# --no-build-isolation    # Disable build isolation
# --skip-dependency-check # Skip build dependency verification
# --python VERSION        # Target Python version
```

Usage examples:

```bash
# Build both sdist and wheel (default)
uv build

# Build only source distribution
uv build --sdist

# Build only wheel
uv build --wheel

# Build to custom directory
uv build --out-dir packages/

# Build specific project directory
uv build ./my-package/

# Pass settings to build backend
uv build --config-setting="--global-option=--with-cython"
```

### Package Publishing

Upload built distributions to PyPI, TestPyPI, or private package indexes with authentication and verification.

```bash { .api }
uv publish [DIST...]
# Publishes distributions to package index
# Default: publishes all files in dist/ directory

# Target formats:
# - Source distributions (.tar.gz)
# - Wheel distributions (.whl)
# - Both

# Options:
# --repository URL         # Repository URL (default: PyPI)
# --repository-url URL     # Alternate repository URL
# --username USERNAME      # Authentication username
# --password PASSWORD      # Authentication password
# --token TOKEN           # API token for authentication
# --config-file FILE      # Configuration file path
# --skip-existing         # Skip files that already exist
# --verify-ssl           # Verify SSL certificates (default: true)
# --no-verify-ssl        # Disable SSL verification
# --cert FILE            # Client certificate file
# --client-cert FILE     # Client certificate file
# --verbose              # Verbose output
```

Usage examples:

```bash
# Publish all distributions in dist/
uv publish

# Publish specific files
uv publish dist/mypackage-1.0.0.tar.gz dist/mypackage-1.0.0-py3-none-any.whl

# Publish to TestPyPI
uv publish --repository testpypi

# Publish with API token
uv publish --token $PYPI_TOKEN

# Publish with username/password
uv publish --username myuser --password mypass

# Publish to private index
uv publish --repository-url https://private.pypi.org/simple/

# Skip existing files
uv publish --skip-existing
```

### Build Configuration

Configure build behavior through project settings and build system integration.

Build system configuration in `pyproject.toml`:
```toml { .api }
[build-system]
requires = ["hatchling"]              # Build dependencies
build-backend = "hatchling.build"    # Build backend

[project]
name = "my-package"
version = "1.0.0"
description = "Package description"
authors = [
    {name = "Author Name", email = "author@example.com"}
]
dependencies = [
    "requests>=2.25.0",
]
requires-python = ">=3.8"

[project.optional-dependencies]
dev = ["pytest", "black", "ruff"]
docs = ["mkdocs", "mkdocs-material"]

[project.scripts]
my-tool = "my_package.cli:main"

[project.gui-scripts]
my-gui = "my_package.gui:main"

[project.entry-points."my_package.plugins"]
plugin1 = "my_package.plugins:plugin1"
```

UV-specific build configuration:
```toml { .api }
[tool.uv]
# Build settings
build-backend = "hatchling"          # Preferred build backend
build-isolation = true               # Enable build isolation
build-verbosity = 1                  # Build verbosity level

# Source distribution settings
include-patterns = [
    "src/**/*.py",
    "tests/**/*.py",
    "README.md",
    "LICENSE",
]
exclude-patterns = [
    "**/__pycache__",
    "**/*.pyc",
    ".git/**",
]
```

### Build Backend Integration

UV integrates with popular Python build backends and tools:

#### Hatchling
```toml { .api }
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.sdist]
include = [
    "/src",
    "/tests",
    "/README.md",
]

[tool.hatch.build.targets.wheel]
packages = ["src/my_package"]
```

#### Setuptools
```toml { .api }
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
packages = ["my_package"]
package-dir = {"" = "src"}
```

#### PDM-Backend
```toml { .api }
[build-system]
requires = ["pdm-backend"]
build-backend = "pdm.backend"

[tool.pdm.build]
includes = ["src"]
excludes = ["tests"]
```

#### Flit
```toml { .api }
[build-system]
requires = ["flit_core >=3.2,<4"]
build-backend = "flit_core.buildapi"

[project]
name = "my-package"
authors = [{name = "Author", email = "author@example.com"}]
dynamic = ["version", "description"]
```

### Build Isolation

UV uses build isolation by default to ensure reproducible builds:

1. **Isolated environment**: Creates temporary virtual environment
2. **Build dependencies**: Installs only declared build requirements
3. **Clean builds**: No interference from existing packages
4. **Reproducibility**: Consistent results across different machines

Disable isolation for debugging:
```bash
uv build --no-build-isolation
```

### Multi-platform Building

Build packages for multiple platforms and Python versions:

```bash { .api }
# Build with specific Python version
uv build --python 3.12

# Build configuration for multiple platforms
[tool.cibuildwheel]
build = "cp38-* cp39-* cp310-* cp311-* cp312-*"
skip = "*-win32 *-manylinux_i686"

# Build with cibuildwheel integration
uv tool run cibuildwheel --platform linux
```

### Package Metadata

Configure package metadata for proper distribution:

```toml { .api }
[project]
name = "my-package"
version = "1.0.0"
description = "Short package description"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "Author Name", email = "author@example.com"}
]
maintainers = [
    {name = "Maintainer", email = "maintainer@example.com"}
]
keywords = ["keyword1", "keyword2"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

[project.urls]
Homepage = "https://github.com/user/repo"
Documentation = "https://docs.example.com"
Repository = "https://github.com/user/repo"
"Bug Tracker" = "https://github.com/user/repo/issues"
```

### Publishing Configuration

Configure publishing targets and authentication:

```toml { .api }
# .pypirc configuration file
[distutils]
index-servers =
    pypi
    testpypi
    private

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi-token-here

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = testpypi-token-here

[private]
repository = https://private.pypi.org/simple/
username = user
password = pass
```

Environment variables for authentication:
```bash { .api }
TWINE_USERNAME=__token__
TWINE_PASSWORD=pypi-token
TWINE_REPOSITORY_URL=https://upload.pypi.org/legacy/
```

### Build Artifacts

UV build produces standard Python distribution artifacts:

#### Source Distribution (.tar.gz)
Contains:
- Source code files
- Package metadata (PKG-INFO)
- Build scripts and configuration
- Documentation and data files

#### Wheel Distribution (.whl)
Contains:
- Compiled Python code (.pyc files)
- Package metadata (METADATA, WHEEL)
- Entry points and console scripts
- Data files and resources

### Build Verification

Verify built packages before publishing:

```bash { .api }
# Check package metadata
uv tool run twine check dist/*

# Test installation from built wheel
uv pip install dist/package-1.0.0-py3-none-any.whl

# Test package functionality
uv run python -c "import package; package.test()"
```

### Continuous Integration

Integrate building and publishing into CI/CD pipelines:

```yaml { .api }
# GitHub Actions example
name: Build and Publish
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    - name: Install UV
      run: pip install uv
    - name: Build package
      run: uv build
    - name: Check built package
      run: uv tool run twine check dist/*
    - name: Publish to PyPI
      if: startsWith(github.ref, 'refs/tags/')
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
      run: uv publish
```

### Build Troubleshooting

Common build issues and solutions:

#### Missing build dependencies
```bash
# Install build dependencies manually
uv pip install build setuptools wheel

# Or use build isolation (default)
uv build
```

#### Build backend errors
```bash
# Check build backend configuration
cat pyproject.toml | grep -A 5 "\[build-system\]"

# Try different backend
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

#### Permission issues
```bash
# Check output directory permissions
ls -la dist/

# Use custom output directory
uv build --out-dir ./packages/
```

#### Publishing failures
```bash
# Verify credentials
uv publish --repository testpypi

# Check package name availability
uv tool run twine check dist/*

# Skip existing files
uv publish --skip-existing
```