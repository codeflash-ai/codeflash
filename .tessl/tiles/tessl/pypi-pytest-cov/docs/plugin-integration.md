# Plugin Integration

pytest-cov integrates deeply with pytest's plugin system through hooks, fixtures, and configuration. The plugin automatically registers when installed and provides seamless coverage measurement throughout the pytest lifecycle.

## Capabilities

### Plugin Registration and Configuration

pytest-cov registers as a pytest11 plugin through setuptools entry points, making it automatically available when installed.

```python { .api }
def pytest_addoption(parser):
    """
    Add coverage-related command-line options to pytest.
    
    Creates a 'cov' option group with all coverage control options including
    --cov, --cov-report, --cov-config, --no-cov, --cov-fail-under, etc.
    
    Args:
        parser: pytest argument parser
    """

def pytest_configure(config):
    """
    Register pytest markers for coverage control.
    
    Adds the 'no_cover' marker that can be used to exclude specific tests
    from coverage measurement.
    
    Args:
        config: pytest configuration object
    """
```

**Usage Examples:**

```python
# The plugin automatically adds these options to pytest
pytest --help  # Shows all --cov-* options

# Marker usage in tests
@pytest.mark.no_cover
def test_excluded_from_coverage():
    pass
```

### Early Initialization

Controls early plugin initialization and handles configuration conflicts.

```python { .api }
def pytest_load_initial_conftests(early_config, parser, args):
    """
    Early plugin initialization hook.
    
    Processes command-line arguments early to detect configuration conflicts
    (like --no-cov combined with --cov options) and initializes the coverage
    plugin if coverage sources are specified.
    
    Args:
        early_config: Early pytest configuration
        parser: Argument parser
        args: Command-line arguments
    """
```

This hook ensures that coverage measurement begins as early as possible in the pytest lifecycle, before most other plugins and test collection occurs.

### Core Plugin Class

The main plugin class that orchestrates coverage measurement across different testing scenarios.

```python { .api }
class CovPlugin:
    """
    Main coverage plugin that delegates to different controllers based on testing mode.
    
    Automatically detects whether running in single-process, distributed master,
    or distributed worker mode and uses the appropriate coverage controller.
    """
    
    def __init__(self, options: argparse.Namespace, pluginmanager, start: bool = True, no_cov_should_warn: bool = False):
        """
        Initialize the coverage plugin.
        
        Args:
            options: Parsed command-line options
            pluginmanager: pytest plugin manager
            start: Whether to start coverage immediately
            no_cov_should_warn: Whether to warn about --no-cov conflicts
        """
    
    def start(self, controller_cls: type, config=None, nodeid: Optional[str] = None):
        """
        Start coverage measurement with specified controller.
        
        Args:
            controller_cls: Controller class (Central, DistMaster, or DistWorker)  
            config: pytest configuration object
            nodeid: Node identifier for distributed testing
        """
```

### Session Lifecycle Hooks

Hooks that integrate with pytest's session lifecycle for comprehensive coverage control.

```python { .api }
def pytest_sessionstart(self, session):
    """
    Initialize coverage at session start.
    
    Determines the appropriate coverage controller based on testing mode:
    - DistWorker for pytest-xdist worker processes
    - Central for single-process testing
    - Uses existing controller if already started
    
    Args:
        session: pytest session object
    """

def pytest_runtestloop(self, session):
    """
    Wrap the main test execution loop.
    
    Configures warning filters for coverage-related warnings and handles
    coverage finalization and reporting after test execution completes.
    Sets up proper warning handling for ResourceWarning, PytestCovWarning,
    and CoverageWarning.
    
    Args:
        session: pytest session object
        
    Returns:
        Test execution result
    """

def pytest_terminal_summary(self, terminalreporter):
    """
    Display coverage summary in terminal output.
    
    Formats and displays coverage reports and threshold checking results.
    Handles warning display for disabled coverage and report generation failures.
    
    Args:
        terminalreporter: pytest terminal reporter
    """
```

### Test Lifecycle Hooks

Hooks that control coverage measurement at the individual test level.

```python { .api }
def pytest_runtest_setup(self, item):
    """
    Handle test setup for coverage measurement.
    
    Initializes subprocess coverage if the test is running in a different
    process than the main session (handles forked test scenarios).
    
    Args:
        item: pytest test item
    """

def pytest_runtest_teardown(self, item):
    """
    Handle test teardown for coverage cleanup.
    
    Cleans up any subprocess coverage data collection initiated during
    test setup.
    
    Args:
        item: pytest test item  
    """

def pytest_runtest_call(self, item):
    """
    Control coverage during test execution.
    
    Pauses coverage measurement for tests marked with 'no_cover' marker
    or tests that use the 'no_cover' fixture, then resumes coverage
    after test completion.
    
    Args:
        item: pytest test item
    """
```

### Distributed Testing Integration

Optional hooks for integration with pytest-xdist distributed testing.

```python { .api }
def pytest_configure_node(self, node):
    """
    Configure a distributed testing node.
    
    Sends configuration information to worker nodes including master host,
    directory paths, and rsync roots for proper coverage data collection
    and file path mapping.
    
    Args:
        node: pytest-xdist node object
        
    Note:
        This hook is marked as optional and only used when pytest-xdist is installed.
    """

def pytest_testnodedown(self, node, error):
    """
    Handle distributed testing node shutdown.
    
    Collects coverage data from worker nodes and integrates it with master
    coverage data. Handles both collocated and remote worker scenarios with
    appropriate data file management and path mapping.
    
    Args:
        node: pytest-xdist node object
        error: Any error that occurred during node shutdown
        
    Note:
        This hook is marked as optional and only used when pytest-xdist is installed.
    """
```

### Context Management Plugin

Specialized plugin for managing coverage contexts during test execution.

```python { .api }
class TestContextPlugin:
    """
    Plugin for managing coverage contexts during test execution.
    
    Provides per-test context switching when --cov-context=test is used,
    enabling detailed tracking of coverage attribution to specific tests.
    """
    
    def __init__(self, cov_controller):
        """
        Initialize context plugin.
        
        Args:
            cov_controller: Coverage controller instance
        """
    
    def pytest_runtest_setup(self, item):
        """Switch coverage context for test setup phase."""
    
    def pytest_runtest_call(self, item):
        """Switch coverage context for test execution phase."""
        
    def pytest_runtest_teardown(self, item):
        """Switch coverage context for test teardown phase."""
    
    def switch_context(self, item, when: str):
        """
        Switch coverage context and update environment.
        
        Args:
            item: pytest test item
            when: Phase of test execution ('setup', 'run', 'teardown')
        """
```

## Internal Helper Functions

Utility functions that support plugin functionality.

```python { .api }
def _is_worker(self, session) -> bool:
    """
    Determine if running in a pytest-xdist worker process.
    
    Args:
        session: pytest session object
        
    Returns:
        bool: True if running in worker process
    """

def _should_report(self) -> bool:
    """
    Determine if coverage reports should be generated.
    
    Returns:
        bool: True if reporting is needed and conditions are met
    """

def write_heading(self, terminalreporter):
    """
    Write coverage section heading to terminal output.
    
    Args:
        terminalreporter: pytest terminal reporter
    """
```