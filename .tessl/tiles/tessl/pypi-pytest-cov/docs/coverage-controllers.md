# Coverage Controllers

pytest-cov uses different coverage controller classes to handle various testing scenarios including single-process testing, distributed master coordination, and distributed worker execution. Each controller manages coverage measurement, data collection, and reporting appropriate to its context.

## Capabilities

### Base Controller

The foundational controller class that provides common functionality for all coverage scenarios.

```python { .api }
class CovController:
    """
    Base class for coverage controller implementations.
    
    Provides common functionality for coverage measurement including environment
    variable management, directory handling, and report generation.
    """
    
    def __init__(self, options: argparse.Namespace, config: Union[None, object], nodeid: Union[None, str]):
        """
        Initialize coverage controller with configuration.
        
        Args:
            options: Parsed command-line options containing coverage settings
            config: pytest configuration object (may be None)
            nodeid: Node identifier for distributed testing (may be None)
        """
    
    def start(self):
        """
        Start coverage measurement.
        
        Marks the controller as started. Specific implementations override
        this method to initialize their coverage objects and begin measurement.
        """
    
    def finish(self):
        """
        Finish coverage measurement.
        
        Marks the controller as stopped. Specific implementations override
        this method to stop coverage and save data.
        """
    
    def pause(self):
        """
        Pause coverage measurement temporarily.
        
        Stops coverage collection and removes environment variables.
        Used when running code that should not be measured (e.g., tests marked with no_cover).
        """
    
    def resume(self):
        """
        Resume coverage measurement after pausing.
        
        Restarts coverage collection and restores environment variables.
        """
    
    def summary(self, stream) -> Optional[float]:
        """
        Generate coverage reports and return total coverage percentage.
        
        Args:
            stream: Output stream for report text
            
        Returns:
            Optional[float]: Total coverage percentage, or None if reporting disabled
        """
```

### Environment Management

Methods for managing environment variables that enable subprocess coverage.

```python { .api }
def set_env(self):
    """
    Set environment variables for subprocess coverage.
    
    Sets COV_CORE_SOURCE, COV_CORE_CONFIG, COV_CORE_DATAFILE, and 
    COV_CORE_BRANCH environment variables so that forked processes
    and subprocesses can automatically enable coverage measurement.
    """

def unset_env(self):
    """
    Remove coverage-related environment variables.
    
    Cleans up COV_CORE_* environment variables to prevent interference
    with subsequent processes.
    """
```

### Utility Methods

Helper methods for controller operation and display formatting.

```python { .api }
def ensure_topdir(self):
    """
    Context manager ensuring operations run in the correct directory.
    
    Returns:
        Context manager that changes to topdir and restores original directory
    """

def get_node_desc(platform: str, version_info: tuple) -> str:
    """
    Generate description string for a testing node.
    
    Args:
        platform: Platform identifier (e.g., 'linux', 'win32')
        version_info: Python version info tuple
        
    Returns:
        str: Formatted node description
    """

def get_width() -> int:
    """
    Get terminal width for report formatting.
    
    Returns:
        int: Terminal width in characters (minimum 40, default 80)
    """

def sep(self, stream, s: str, txt: str):
    """
    Write separator line to output stream.
    
    Args:
        stream: Output stream
        s: Separator character
        txt: Text to center in separator line
    """
```

### Single-Process Controller

Controller for standard single-process test execution.

```python { .api }
class Central(CovController):
    """
    Coverage controller for single-process (centralized) test execution.
    
    Handles standard pytest runs without distributed testing, managing
    a single coverage instance and combining data from any subprocesses.
    """
    
    def start(self):
        """
        Initialize and start centralized coverage measurement.
        
        Creates coverage.Coverage instance with configured options,
        sets up combining coverage for subprocess data, and begins
        coverage measurement. Warns if dynamic_context=test_function
        is configured (recommends --cov-context instead).
        """
    
    def finish(self):
        """
        Stop coverage and prepare data for reporting.
        
        Stops coverage measurement, saves data, creates combining coverage
        instance, loads and combines all coverage data (including subprocess
        data), and adds node description for reporting.
        """
```

### Distributed Master Controller

Controller for the master process in distributed testing scenarios.

```python { .api }
class DistMaster(CovController):
    """
    Coverage controller for distributed testing master process.
    
    Coordinates coverage measurement across multiple worker processes,
    collecting and combining coverage data from all workers while
    managing distributed testing configuration.
    """
    
    def start(self):
        """
        Initialize master coverage for distributed testing.
        
        Creates master coverage instance, validates configuration
        (raises DistCovError if dynamic_context=test_function with xdist),
        configures path mapping, and starts coverage measurement.
        """
    
    def configure_node(self, node):
        """
        Configure a worker node for distributed testing.
        
        Sends master configuration to worker including hostname,
        directory paths, and rsync roots for proper file path
        mapping and coverage data collection.
        
        Args:
            node: pytest-xdist node object
        """
    
    def testnodedown(self, node, error):
        """
        Handle worker node shutdown and collect coverage data.
        
        Retrieves coverage data from worker node, handles both collocated
        and remote worker scenarios, creates coverage instances for
        remote worker data, and updates path mappings for data combination.
        
        Args:
            node: pytest-xdist node object  
            error: Any error that occurred during shutdown
        """
    
    def finish(self):
        """
        Combine coverage data from all workers.
        
        Stops master coverage, saves data, switches to combining coverage
        instance, loads and combines data from all workers, and saves
        final combined coverage data.
        """
```

### Distributed Worker Controller

Controller for worker processes in distributed testing scenarios.

```python { .api }
class DistWorker(CovController):
    """
    Coverage controller for distributed testing worker processes.
    
    Handles coverage measurement in worker processes, managing path
    mapping for remote workers and sending coverage data back to master.
    """
    
    def start(self):
        """
        Initialize worker coverage for distributed testing.
        
        Determines if worker is collocated with master, adjusts source
        paths and config paths for remote workers, creates worker coverage
        instance with unique data suffix, and starts coverage measurement.
        """
    
    def finish(self):
        """
        Finish worker coverage and send data to master.
        
        Stops coverage measurement and handles data transmission:
        - Collocated workers: Save data file and send node ID to master
        - Remote workers: Combine data, serialize coverage data, and send
          complete data payload to master including path mapping info
        """
    
    def summary(self, stream):
        """
        No-op summary method for workers.
        
        Workers don't generate reports - only the master generates
        coverage summaries after combining all worker data.
        """
```

## Internal Utilities

Helper functions and classes that support controller functionality.

```python { .api }
class BrokenCovConfigError(Exception):
    """Exception raised when coverage configuration is invalid."""

class _NullFile:
    """File-like object that discards all writes."""
    
    @staticmethod
    def write(v):
        """Discard written content."""

def _ensure_topdir(meth):
    """
    Decorator ensuring method runs in the correct directory.
    
    Changes to controller's topdir before method execution and
    restores original directory afterward, handling cases where
    original directory no longer exists.
    """

def _backup(obj, attr):
    """
    Context manager for backing up and restoring object attributes.
    
    Args:
        obj: Object whose attribute will be backed up
        attr: Attribute name to backup
        
    Returns:
        Context manager that creates copy of attribute and restores it
    """

def _data_suffix(name: str) -> str:
    """
    Generate data file suffix for coverage files.
    
    Args:
        name: Suffix name identifier
        
    Returns:
        str: Complete data file suffix including random component
    """
```

## Usage Patterns

### Controller Selection

Controllers are automatically selected based on testing configuration:

```python
# Single-process testing
if not distributed:
    controller = Central(options, config, nodeid)

# Distributed master  
if distributed and master:
    controller = DistMaster(options, config, nodeid)

# Distributed worker
if distributed and worker:
    controller = DistWorker(options, config, nodeid)
```

### Lifecycle Management

All controllers follow the same lifecycle pattern:

```python
controller = ControllerClass(options, config, nodeid)
controller.start()
# ... test execution ...
controller.finish() 
total_coverage = controller.summary(output_stream)
```

### Path Mapping

For distributed testing with remote workers, controllers handle path mapping:

```python
# Master sends configuration
master.configure_node(worker_node)

# Worker adjusts paths if not collocated
if not worker.is_collocated:
    worker.cov_source = adjust_paths(worker.cov_source, master_path, worker_path)
```