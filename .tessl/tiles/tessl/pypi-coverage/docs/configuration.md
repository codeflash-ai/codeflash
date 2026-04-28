# Configuration Management

Comprehensive configuration system supporting multiple file formats (pyproject.toml, .coveragerc), command-line options, and programmatic configuration. Manages source inclusion/exclusion, measurement options, and output settings.

## Capabilities

### Configuration Methods

Get and set configuration options programmatically through the Coverage class.

```python { .api }
def get_option(self, option_name: str):
    """
    Get the value of a configuration option.
    
    Parameters:
    - option_name (str): Configuration option name in section:key format
    
    Returns:
        Value of the configuration option
    """

def set_option(self, option_name: str, value) -> None:
    """
    Set the value of a configuration option.
    
    Parameters:
    - option_name (str): Configuration option name in section:key format
    - value: New value for the configuration option
    """
```

Usage example:

```python
import coverage

cov = coverage.Coverage()

# Get configuration values
branch_enabled = cov.get_option('run:branch')
source_dirs = cov.get_option('run:source')
omit_patterns = cov.get_option('run:omit')

print(f"Branch coverage: {branch_enabled}")
print(f"Source directories: {source_dirs}")

# Set configuration values
cov.set_option('run:branch', True)
cov.set_option('run:source', ['src/', 'lib/'])
cov.set_option('run:omit', ['*/tests/*', '*/migrations/*'])
cov.set_option('report:precision', 2)
```

### Configuration File Reading

Read configuration from various file formats and sources.

```python { .api }
def read_coverage_config(
    config_file=True,
    warn=None,
    debug=None,
    check_preimported=False,
    **kwargs
):
    """
    Read coverage configuration from files and arguments.
    
    Parameters:
    - config_file (str | bool): Path to config file, True for auto-detection, False to disable
    - warn (callable | None): Function to call with warning messages
    - debug (callable | None): Function for debug output
    - check_preimported (bool): Check for already imported modules
    - **kwargs: Additional configuration options
    
    Returns:
        CoverageConfig: Configuration object
    """
```

Usage example:

```python
from coverage.config import read_coverage_config

# Read from default locations (.coveragerc, pyproject.toml, setup.cfg)
config = read_coverage_config()

# Read from specific file
config = read_coverage_config(config_file='custom.ini')

# Read with additional options
config = read_coverage_config(
    config_file='pyproject.toml',
    branch=True,
    source=['src/']
)

# Disable config file reading
config = read_coverage_config(config_file=False, branch=True)
```

## Configuration File Formats

### pyproject.toml

Modern Python project configuration using TOML format:

```toml
[tool.coverage.run]
source = ["src"]
branch = true
omit = [
    "*/tests/*",
    "*/venv/*",
    "*/.tox/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError"
]
show_missing = true
skip_covered = true
precision = 2

[tool.coverage.html]
directory = "htmlcov"
show_contexts = true

[tool.coverage.xml]
output = "coverage.xml"

[tool.coverage.json]
output = "coverage.json"
pretty_print = true
```

### .coveragerc

Legacy INI-style configuration file:

```ini
[run]
source = src
branch = True
omit = 
    */tests/*
    */venv/*
    */.tox/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
show_missing = True
skip_covered = True
precision = 2

[html]
directory = htmlcov
show_contexts = True

[xml]
output = coverage.xml

[json]
output = coverage.json
pretty_print = True
```

### setup.cfg

Configuration can also be placed in setup.cfg:

```ini
[coverage:run]
source = src
branch = True
omit = 
    */tests/*
    */venv/*

[coverage:report]
show_missing = True
skip_covered = True
```

## Configuration Sections

### [run] Section

Controls coverage measurement behavior:

```python { .api }
# Measurement options
branch = True | False              # Enable branch coverage
source = ["dir1", "dir2"]         # Source directories/files to measure
omit = ["pattern1", "pattern2"]   # File patterns to omit
include = ["pattern1"]            # File patterns to include explicitly

# Data file options
data_file = ".coverage"           # Coverage data file path
parallel = True | False           # Enable parallel data collection
context = "context_name"          # Static context label

# Execution options
cover_pylib = True | False        # Measure Python standard library
timid = True | False              # Use slower but more compatible tracer
concurrency = ["thread", "multiprocessing"]  # Concurrency libraries

# Plugin options
plugins = ["plugin1", "plugin2"] # Coverage plugins to load
```

### [report] Section

Controls text report generation:

```python { .api }
# Output options
show_missing = True | False       # Show missing line numbers
skip_covered = True | False       # Skip files with 100% coverage
skip_empty = True | False         # Skip files with no executable code
precision = 2                     # Decimal places for percentages

# File filtering
omit = ["pattern1", "pattern2"]   # File patterns to omit from report
include = ["pattern1"]            # File patterns to include in report

# Line exclusion
exclude_lines = [                 # Regex patterns for line exclusion
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError"
]

# Partial exclusion (for branch coverage)
partial_branches = [              # Regex patterns for partial branch exclusion
    "pragma: no branch",
    "if __name__ == .__main__.:"
]

# Sorting
sort = "name" | "stmts" | "miss" | "branch" | "brpart" | "cover"
```

### [html] Section

Controls HTML report generation:

```python { .api }
directory = "htmlcov"             # Output directory for HTML files
title = "Coverage Report"         # Title for HTML pages
show_contexts = True | False      # Include context information
extra_css = "custom.css"          # Additional CSS file
```

### [xml] Section

Controls XML report generation:

```python { .api }
output = "coverage.xml"           # Output file path
```

### [json] Section

Controls JSON report generation:

```python { .api }
output = "coverage.json"          # Output file path
pretty_print = True | False       # Format JSON for readability
show_contexts = True | False      # Include context information
```

### [lcov] Section

Controls LCOV report generation:

```python { .api }
output = "coverage.lcov"          # Output file path
```

### [paths] Section

Configure path mapping for combining data from different environments:

```python { .api }
# Map paths for data combination
source = [
    "/home/user/project/src",     # Local development path
    "/app/src",                   # Docker container path
    "C:\\projects\\myapp\\src"    # Windows path
]
```

## Advanced Configuration

### Dynamic Configuration

Modify configuration at runtime:

```python
import coverage

cov = coverage.Coverage()

# Get current configuration
config = cov.config

# Modify configuration
config.branch = True
config.source = ['src/', 'lib/']
config.omit = ['*/tests/*']

# Update configuration from dictionary
config.from_args(
    branch=True,
    source=['src/'],
    omit=['*/test*']
)
```

### Environment Variables

Some configuration can be controlled via environment variables:

```bash
# Disable coverage measurement
export COVERAGE_PROCESS_START=""

# Set data file location
export COVERAGE_FILE=".coverage.custom"

# Enable debug output
export COVERAGE_DEBUG="trace,config"
```

### Configuration Validation

```python
from coverage.config import CoverageConfig, ConfigError

try:
    config = CoverageConfig()
    config.from_args(
        branch=True,
        source=['nonexistent/'],
        invalid_option='value'  # This will cause an error
    )
except ConfigError as e:
    print(f"Configuration error: {e}")
```

## Plugin Configuration

Configure plugins through dedicated sections:

```toml
[tool.coverage.myPlugin]
option1 = "value1"
option2 = true
list_option = ["item1", "item2"]
```

Plugin receives these options in the `coverage_init` function:

```python
def coverage_init(reg, options):
    # options = {"option1": "value1", "option2": True, "list_option": ["item1", "item2"]}
    plugin = MyPlugin(options)
    reg.add_file_tracer(plugin)
```

## Configuration Priority

Configuration is resolved in this order (highest to lowest priority):

1. Programmatic options (`Coverage(branch=True)`)
2. Command-line options (`coverage run --branch`)
3. Configuration file options
4. Default values

### Configuration File Discovery

Coverage.py searches for configuration in this order:

1. Specified config file (`config_file='path'`)
2. `.coveragerc` in current directory
3. `setup.cfg` with `[coverage:*]` sections
4. `pyproject.toml` with `[tool.coverage.*]` sections
5. `.coveragerc` in user's home directory

## Complete Configuration Example

Here's a comprehensive configuration for a typical Python project:

```toml
# pyproject.toml
[tool.coverage.run]
# Source code to measure
source = ["src", "lib"]

# Enable branch coverage
branch = true

# Files to omit from measurement
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/.tox/*",
    "*/venv/*",
    "*/migrations/*",
    "*/setup.py",
    "*/conftest.py"
]

# Concurrency support
concurrency = ["thread", "multiprocessing"]

# Plugins
plugins = ["coverage_pth"]

[tool.coverage.report]
# Reporting options
show_missing = true
skip_covered = false
skip_empty = true
precision = 2
sort = "cover"

# Exclude lines from coverage
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "def __str__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "@abstract"
]

# Partial branch exclusion
partial_branches = [
    "pragma: no branch",
    "if __name__ == .__main__.:"
]

[tool.coverage.html]
directory = "htmlcov"
title = "My Project Coverage Report"
show_contexts = true

[tool.coverage.xml]
output = "reports/coverage.xml"

[tool.coverage.json]
output = "reports/coverage.json"
pretty_print = true
show_contexts = true

[tool.coverage.lcov]
output = "reports/coverage.lcov"

# Path mapping for different environments
[tool.coverage.paths]
source = [
    "src/",
    "/home/user/project/src/",
    "/app/src/",
    "C:\\Users\\user\\project\\src\\"
]
```

This configuration provides comprehensive coverage measurement with multiple output formats, appropriate exclusions, and support for different deployment environments.