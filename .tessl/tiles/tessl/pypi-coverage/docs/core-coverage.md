# Core Coverage Measurement

Primary coverage measurement and control functionality. The Coverage class serves as the main entry point for all coverage operations, managing measurement lifecycle, configuration, and basic analysis.

## Capabilities

### Coverage Class Constructor

Creates a Coverage instance with comprehensive configuration options for measurement control, data management, and source filtering.

```python { .api }
class Coverage:
    def __init__(
        self,
        data_file=None,                # Path to coverage data file
        data_suffix=None,              # Suffix for data file names
        cover_pylib=None,             # Include Python standard library
        auto_data=False,              # Automatically save data on exit
        timid=None,                   # Use simpler but slower tracing
        branch=None,                  # Enable branch coverage measurement
        config_file=True,             # Path to config file or boolean
        source=None,                  # Source files/directories to measure
        source_pkgs=None,             # Source packages to measure  
        source_dirs=None,             # Source directories to measure
        omit=None,                    # Files/patterns to omit
        include=None,                 # Files/patterns to include
        debug=None,                   # Debug options list
        concurrency=None,             # Concurrency libraries in use
        check_preimported=False,      # Check already imported modules
        context=None,                 # Context label for this run
        messages=False,               # Show messages during execution
        plugins=None                  # Plugin callables list
    ):
        """
        Create a Coverage instance for measurement and reporting.
        
        Parameters:
        - data_file (str | None): Path to the data file. If None, uses default
        - data_suffix (str | bool | None): Suffix for parallel data files
        - cover_pylib (bool | None): Whether to measure standard library
        - auto_data (bool): Automatically save data when program ends
        - timid (bool | None): Use simpler trace function for compatibility
        - branch (bool | None): Measure branch coverage in addition to lines
        - config_file (str | bool): Configuration file path or False to disable
        - source (list[str] | None): Source files or directories to measure
        - source_pkgs (list[str] | None): Source packages to measure
        - source_dirs (list[str] | None): Source directories to measure
        - omit (str | list[str] | None): File patterns to omit from measurement
        - include (str | list[str] | None): File patterns to include in measurement
        - debug (list[str] | None): Debug options ('trace', 'config', 'callers', etc.)
        - concurrency (str | list[str] | None): Concurrency library names
        - check_preimported (bool): Check for modules imported before start
        - context (str | None): Context label for this measurement run
        - messages (bool): Display informational messages during execution
        - plugins (list[callable] | None): Plugin initialization functions
        """
```

Usage example:

```python
import coverage

# Basic usage
cov = coverage.Coverage()

# Advanced configuration
cov = coverage.Coverage(
    data_file='.coverage',
    branch=True,
    source=['src/'],
    omit=['*/tests/*', '*/venv/*'],
    config_file='pyproject.toml'
)
```

### Measurement Control

Start, stop, and manage coverage measurement with context switching support for dynamic analysis.

```python { .api }
def start(self) -> None:
    """
    Start coverage measurement.
    
    Raises:
        CoverageException: If coverage is already started
    """

def stop(self) -> None:
    """
    Stop coverage measurement.
    
    Returns data can be retrieved and reports generated after stopping.
    """

def switch_context(self, new_context: str) -> None:
    """
    Switch to a new dynamic context for measurement.
    
    Parameters:
    - new_context (str): New context label
    """

@contextlib.contextmanager
def collect(self):
    """
    Context manager for temporary coverage collection.
    
    Usage:
        with cov.collect():
            # Code to measure
            pass
    """
```

Usage example:

```python
import coverage

cov = coverage.Coverage()

# Basic measurement
cov.start()
# Your code here
import my_module
result = my_module.function()
cov.stop()

# Context switching
cov.start()
cov.switch_context('test_setup')
setup_code()
cov.switch_context('test_execution')  
test_code()
cov.stop()

# Context manager
with cov.collect():
    code_to_measure()
```

### Data Management

Save, load, and manage coverage data with combining support for parallel execution.

```python { .api }
def save(self) -> None:
    """
    Save collected coverage data to file.
    
    Data is saved to the data file specified in constructor.
    """

def load(self) -> None:
    """
    Load previously collected coverage data from file.
    """

def erase(self) -> None:
    """
    Erase collected coverage data.
    
    Clears both in-memory data and data file.
    """

def get_data(self) -> CoverageData:
    """
    Get the collected coverage data.
    
    Returns:
        CoverageData: Data object for querying and analysis
    """

def combine(
    self,
    data_paths=None,
    strict=False,
    keep=False
) -> None:
    """
    Combine coverage data from multiple files.
    
    Parameters:
    - data_paths (list[str] | None): Paths to data files or directories
    - strict (bool): Raise error if no data files found
    - keep (bool): Keep original data files after combining
    """
```

Usage example:

```python
import coverage

cov = coverage.Coverage()

# Basic data operations
cov.start()
# ... run code ...
cov.stop()
cov.save()

# Later, load and analyze
cov2 = coverage.Coverage()
cov2.load()
data = cov2.get_data()

# Combine parallel data files
cov.combine(['data1/.coverage', 'data2/.coverage'])
```

### Configuration Methods

Get and set configuration options programmatically.

```python { .api }
def get_option(self, option_name: str):
    """
    Get a configuration option value.
    
    Parameters:
    - option_name (str): Name of the configuration option
    
    Returns:
        Configuration option value
    """

def set_option(self, option_name: str, value) -> None:
    """
    Set a configuration option value.
    
    Parameters:
    - option_name (str): Name of the configuration option
    - value: New value for the option
    """
```

Usage example:

```python
import coverage

cov = coverage.Coverage()

# Get current settings
branch_enabled = cov.get_option('run:branch')
source_dirs = cov.get_option('run:source')

# Modify settings
cov.set_option('run:branch', True)
cov.set_option('run:omit', ['*/tests/*'])
```

### Exclusion Management

Manage regex patterns for excluding lines from coverage measurement.

```python { .api }
def exclude(self, regex: str, which: str = 'exclude') -> None:
    """
    Add a regex pattern for excluding lines from coverage.
    
    Parameters:
    - regex (str): Regular expression pattern
    - which (str): Exclusion category ('exclude', 'partial', 'partial-always')
    """

def clear_exclude(self, which: str = 'exclude') -> None:
    """
    Clear all exclusion patterns for a category.
    
    Parameters:
    - which (str): Exclusion category to clear
    """

def get_exclude_list(self, which: str = 'exclude') -> list[str]:
    """
    Get list of exclusion patterns for a category.
    
    Parameters:
    - which (str): Exclusion category
    
    Returns:
        list[str]: List of regex patterns
    """
```

Usage example:

```python
import coverage

cov = coverage.Coverage()

# Add exclusion patterns
cov.exclude(r'pragma: no cover')
cov.exclude(r'def __repr__')
cov.exclude(r'raise NotImplementedError')

# Check current exclusions
patterns = cov.get_exclude_list()
print(f"Exclusion patterns: {patterns}")

# Clear exclusions
cov.clear_exclude()
```

### Analysis Methods

Analyze coverage results for individual modules or files.

```python { .api }
def analysis2(self, morf):
    """
    Analyze coverage for a module or file.
    
    Parameters:
    - morf: Module object or filename string
    
    Returns:
        tuple: (filename, statements, excluded, missing, readable_filename)
    """

def branch_stats(self, morf) -> dict[int, tuple[int, int]]:
    """
    Get branch coverage statistics for a file.
    
    Parameters:
    - morf: Module object or filename string
    
    Returns:
        dict: Mapping of line numbers to (branches_possible, branches_taken)
    """
```

Usage example:

```python
import coverage
import my_module

cov = coverage.Coverage(branch=True)
cov.start()
# ... run code ...
cov.stop()

# Analyze specific module
filename, statements, excluded, missing, readable = cov.analysis2(my_module)
print(f"File: {readable}")
print(f"Statements: {len(statements)}")
print(f"Missing: {len(missing)}")

# Branch statistics
branch_stats = cov.branch_stats(my_module)
for line, (possible, taken) in branch_stats.items():
    print(f"Line {line}: {taken}/{possible} branches taken")
```

### Class Methods

Static methods for accessing the current Coverage instance.

```python { .api }
@classmethod
def current(cls):
    """
    Get the most recently started Coverage instance.
    
    Returns:
        Coverage | None: The current instance or None if none started
    """
```

### System Information

Get debugging and system information.

```python { .api }
def sys_info(self):
    """
    Return system information for debugging.
    
    Returns:
        Iterable[tuple[str, Any]]: System information key-value pairs
    """
```

### Utility Functions

Module-level utility functions for coverage operations.

```python { .api }
def process_startup(*, force: bool = False):
    """
    Initialize coverage measurement in subprocess contexts.
    
    Parameters:
    - force (bool): Force initialization even if already started
    
    Returns:
        Coverage | None: Coverage instance or None if not configured
    """
```

Usage example:

```python
import coverage

# In a subprocess or multiprocessing context
cov = coverage.process_startup()
if cov:
    print("Coverage started in subprocess")
```