# Data Storage and Retrieval

Coverage data storage, querying, and manipulation through the CoverageData class. Handles persistence of execution data, context information, and provides query interfaces for analysis and reporting.

## Capabilities

### CoverageData Class Constructor

Creates a CoverageData instance for managing coverage data storage and retrieval.

```python { .api }
class CoverageData:
    def __init__(
        self,
        basename=None,      # Base name for data file
        suffix=None,        # Suffix for data file name
        warn=None,          # Warning function
        debug=None,         # Debug control function
        no_disk=False       # Operate entirely in memory
    ):
        """
        Create a CoverageData instance for data storage and retrieval.
        
        Parameters:
        - basename (str | None): Base name for the data file, default '.coverage'
        - suffix (str | bool | None): Suffix for parallel data files
        - warn (callable | None): Function to call with warning messages
        - debug (callable | None): Function for debugging output
        - no_disk (bool): If True, operate entirely in memory without file I/O
        """
```

Usage example:

```python
from coverage.data import CoverageData

# Basic usage
data = CoverageData()

# Custom configuration
data = CoverageData(
    basename='my_coverage',
    suffix='worker_1',
    no_disk=False
)

# In-memory only
memory_data = CoverageData(no_disk=True)
```

### Data Access Methods

Query coverage data for files, lines, arcs, and context information.

```python { .api }
def measured_files(self) -> set[str]:
    """
    Get the set of files that have been measured.
    
    Returns:
        set[str]: Set of absolute file paths that have coverage data
    """

def lines(self, filename: str) -> list[int] | None:
    """
    Get executed line numbers for a specific file.
    
    Parameters:
    - filename (str): Path to the file
    
    Returns:
        list[int] | None: List of executed line numbers, or None if file not measured
    """

def arcs(self, filename: str) -> list[tuple[int, int]] | None:
    """
    Get executed arcs (branch pairs) for a specific file.
    
    Parameters:
    - filename (str): Path to the file
    
    Returns:
        list[tuple[int, int]] | None: List of (from_line, to_line) arcs, or None if file not measured
    """

def contexts_by_lineno(self, filename: str) -> dict[int, list[str]]:
    """
    Get context labels for each line number in a file.
    
    Parameters:
    - filename (str): Path to the file
    
    Returns:
        dict[int, list[str]]: Mapping of line numbers to context label lists
    """

def file_tracer(self, filename: str) -> str | None:
    """
    Get the file tracer name used for a specific file.
    
    Parameters:
    - filename (str): Path to the file
    
    Returns:
        str | None: File tracer name, or None if no tracer used
    """

def has_arcs(self) -> bool:
    """
    Check if the data includes branch coverage information.
    
    Returns:
        bool: True if branch coverage data is present
    """
```

Usage example:

```python
from coverage.data import CoverageData

data = CoverageData()
data.read()  # Load from file

# Query measured files
files = data.measured_files()
print(f"Measured {len(files)} files")

# Get line coverage for a file
lines = data.lines('src/my_module.py')
if lines:
    print(f"Executed lines: {sorted(lines)}")

# Check for branch coverage
if data.has_arcs():
    arcs = data.arcs('src/my_module.py')
    print(f"Executed arcs: {arcs}")

# Get context information
contexts = data.contexts_by_lineno('src/my_module.py')
for line, ctx_list in contexts.items():
    print(f"Line {line}: contexts {ctx_list}")
```

### Data Persistence Methods

Read, write, and manage coverage data files.

```python { .api }
def read(self) -> None:
    """
    Read coverage data from the data file.
    
    Loads data from the file specified during construction.
    """

def write(self) -> None:
    """
    Write coverage data to the data file.
    
    Saves current data to the file specified during construction.
    """

def erase(self, parallel: bool = False) -> None:
    """
    Delete the data file.
    
    Parameters:
    - parallel (bool): If True, also erase parallel data files
    """
```

Usage example:

```python
from coverage.data import CoverageData

# Create and populate data
data = CoverageData()

# Load existing data
data.read()

# Make modifications...
# data.update(other_data)

# Save changes
data.write()

# Clean up
data.erase()
```

### Data Modification Methods

Update and modify coverage data programmatically.

```python { .api }
def update(
    self,
    other_data,
    map_path=None
) -> None:
    """
    Merge coverage data from another CoverageData instance.
    
    Parameters:
    - other_data (CoverageData): Another CoverageData instance to merge
    - map_path (callable | None): Function to map file paths during merge
    """

def touch_files(
    self,
    filenames,
    plugin_name: str = ""
) -> None:
    """
    Mark files as measured without adding execution data.
    
    Parameters:
    - filenames (Iterable[str]): Files to mark as measured
    - plugin_name (str): Name of plugin that measured the files
    """
```

Usage example:

```python
from coverage.data import CoverageData

# Combine data from multiple sources
main_data = CoverageData('main.coverage')
main_data.read()

worker_data = CoverageData('worker.coverage')
worker_data.read()

# Merge worker data into main data
main_data.update(worker_data)

# Mark additional files as measured
main_data.touch_files(['src/config.py', 'src/utils.py'])

# Save combined data
main_data.write()
```

### Query Control Methods

Control how data queries are filtered and processed.

```python { .api }
def set_query_contexts(self, contexts: list[str] | None) -> None:
    """
    Set context filter for subsequent data queries.
    
    Parameters:
    - contexts (list[str] | None): Context labels to filter by, or None for no filter
    """
```

Usage example:

```python
from coverage.data import CoverageData

data = CoverageData()
data.read()

# Filter queries to specific contexts
data.set_query_contexts(['test_module_a', 'test_module_b'])

# Now queries only return data from those contexts
lines = data.lines('src/my_module.py')  # Only lines from specified contexts

# Remove filter
data.set_query_contexts(None)
```

### Debugging Methods

Debug and inspect coverage data for troubleshooting.

```python { .api }
@classmethod
def sys_info(cls) -> list[tuple[str, Any]]:
    """
    Return system information about the data storage.
    
    Returns:
        list[tuple[str, Any]]: System information key-value pairs
    """
```

Usage example:

```python
from coverage.data import CoverageData

# Get system information
info = CoverageData.sys_info()
for key, value in info:
    print(f"{key}: {value}")
```

## Utility Functions

Module-level utility functions for working with coverage data.

```python { .api }
def line_counts(data: CoverageData, fullpath: bool = False) -> dict[str, int]:
    """
    Get a summary of line counts from coverage data.
    
    Parameters:
    - data (CoverageData): Coverage data instance
    - fullpath (bool): If True, use full paths as keys; otherwise use basenames
    
    Returns:
        dict[str, int]: Mapping of filenames to executed line counts
    """

def add_data_to_hash(data: CoverageData, filename: str, hasher) -> None:
    """
    Add file coverage data to a hash for fingerprinting.
    
    Parameters:
    - data (CoverageData): Coverage data instance
    - filename (str): File to add to hash
    - hasher: Hash object with update() method
    """

def combinable_files(data_file: str, data_paths=None) -> list[str]:
    """
    List data files that can be combined with a main data file.
    
    Parameters:
    - data_file (str): Path to main data file
    - data_paths (list[str] | None): Additional paths to search
    
    Returns:
        list[str]: List of combinable data file paths
    """

def combine_parallel_data(
    data: CoverageData,
    aliases=None,
    data_paths=None,
    strict: bool = False,
    keep: bool = False,
    message=None
) -> None:
    """
    Combine multiple parallel coverage data files.
    
    Parameters:
    - data (CoverageData): Main data instance to combine into
    - aliases: Path aliases for file mapping
    - data_paths (list[str] | None): Paths to search for data files
    - strict (bool): Raise error if no data files found
    - keep (bool): Keep original data files after combining
    - message (callable | None): Function to display progress messages
    """
```

Usage example:

```python
from coverage.data import CoverageData, line_counts, combine_parallel_data

# Get line count summary
data = CoverageData()
data.read()

counts = line_counts(data, fullpath=True)
for filename, count in counts.items():
    print(f"{filename}: {count} lines executed")

# Combine parallel data files
main_data = CoverageData()
combine_parallel_data(
    main_data,
    data_paths=['worker1/', 'worker2/'],
    strict=True,
    keep=False
)
main_data.write()
```