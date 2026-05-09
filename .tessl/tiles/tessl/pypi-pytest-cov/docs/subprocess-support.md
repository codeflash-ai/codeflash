# Subprocess Support

pytest-cov provides automatic coverage measurement for subprocesses and forked processes through environment variable propagation and embedded coverage initialization. This enables comprehensive coverage collection even when tests spawn additional processes.

## Capabilities

### Subprocess Coverage Initialization

Automatic coverage activation in subprocess environments based on environment variable configuration.

```python { .api }
def init():
    """
    Initialize coverage in subprocess if environment variables are set.
    
    Checks for COV_CORE_* environment variables set by parent process
    and automatically starts coverage measurement in subprocess using
    the same configuration. Creates coverage instance with proper
    source filtering, branch coverage, and data file settings.
    
    Returns:
        coverage.Coverage: Active coverage instance, or None if not initialized
        
    Environment Variables:
        COV_CORE_SOURCE: Colon-separated source paths (empty means no filtering)
        COV_CORE_CONFIG: Coverage config file path (colon means use default)
        COV_CORE_DATAFILE: Coverage data file path
        COV_CORE_BRANCH: 'enabled' to enable branch coverage
        COV_CORE_CONTEXT: Coverage context name
    """
```

**Usage Examples:**

```python  
# Automatic usage - no explicit calls needed
# When parent process has coverage enabled:
import subprocess

# This subprocess will automatically have coverage enabled
subprocess.run(['python', 'script.py'])

# Manual initialization in subprocess code
from pytest_cov.embed import init
cov = init()  # Returns coverage object or None
```

### Coverage Cleanup

Proper cleanup of coverage measurement in subprocess environments.

```python { .api }
def cleanup():
    """
    Clean up active coverage measurement in subprocess.
    
    Stops active coverage instance, saves coverage data, disables
    auto-save to prevent double-saving, and unregisters atexit
    handlers. Handles signal-based cleanup gracefully.
    """
```

### Signal-based Cleanup

Automatic coverage cleanup when subprocess receives termination signals.

```python { .api }
def cleanup_on_signal(signum: int):
    """
    Set up signal handler for coverage cleanup.
    
    Installs a signal handler that will clean up coverage data
    when the specified signal is received, ensuring coverage
    data is saved even if subprocess is terminated unexpectedly.
    
    Args:
        signum: Signal number to handle (e.g., signal.SIGTERM, signal.SIGINT)
    """

def cleanup_on_sigterm():
    """
    Set up SIGTERM handler for coverage cleanup.
    
    Convenience function that sets up cleanup for SIGTERM signal,
    the most common signal used to terminate processes gracefully.
    """
```

**Usage Examples:**

```python
from pytest_cov.embed import cleanup_on_sigterm

# Ensure coverage is saved if process receives SIGTERM
cleanup_on_sigterm()

# Handle custom signals
from pytest_cov.embed import cleanup_on_signal
import signal
cleanup_on_signal(signal.SIGUSR1)
```

### Internal Implementation

Low-level functions that support subprocess coverage functionality.

```python { .api }
def _cleanup(cov):
    """
    Internal cleanup implementation for coverage instance.
    
    Args:
        cov: coverage.Coverage instance to clean up
    """

def _signal_cleanup_handler(signum: int, frame):
    """
    Internal signal handler for coverage cleanup.
    
    Handles cleanup during signal processing, manages pending signals
    during cleanup, and properly chains to previous signal handlers.
    
    Args:
        signum: Signal number received
        frame: Signal frame
    """
```

## Environment Variable Protocol

pytest-cov uses specific environment variables to communicate coverage configuration to subprocess environments:

### Coverage Configuration Variables

```python { .api }
# Environment variables set by parent process
COV_CORE_SOURCE: str     # Colon-separated source paths for filtering
COV_CORE_CONFIG: str     # Coverage configuration file path  
COV_CORE_DATAFILE: str   # Coverage data file path
COV_CORE_BRANCH: str     # 'enabled' for branch coverage
COV_CORE_CONTEXT: str    # Coverage context name
```

**Environment Variable Usage:**

```bash
# Example environment variables set by pytest-cov
export COV_CORE_SOURCE="/path/to/src:/path/to/lib"
export COV_CORE_CONFIG="/path/to/.coveragerc"  
export COV_CORE_DATAFILE="/path/to/.coverage"
export COV_CORE_BRANCH="enabled"
export COV_CORE_CONTEXT="test_context"

# These are automatically set - no manual configuration needed
```

### Variable Processing

How environment variables are processed during subprocess initialization:

```python
# Source path handling
if cov_source == os.pathsep:
    cov_source = None  # No source filtering
else:
    cov_source = cov_source.split(os.pathsep)  # Split paths

# Config file handling  
if cov_config == os.pathsep:
    cov_config = True  # Use default config discovery
# Otherwise use specified path

# Branch coverage
cov_branch = True if os.environ.get('COV_CORE_BRANCH') == 'enabled' else None
```

## Integration Patterns

### Automatic Subprocess Coverage

pytest-cov automatically enables subprocess coverage without requiring code changes:

```python
import subprocess
import os

# This subprocess will automatically have coverage
result = subprocess.run(['python', '-c', 'import mymodule; mymodule.function()'])

# Forked processes also get coverage
pid = os.fork()
if pid == 0:
    # Child process automatically has coverage
    import mymodule
    mymodule.child_function()
    os._exit(0)
```

### Manual Control

For advanced scenarios where manual control is needed:

```python
from pytest_cov.embed import init, cleanup, cleanup_on_sigterm

# Manual initialization with signal handling
cleanup_on_sigterm()
cov = init()

try:
    # Subprocess code that should be measured
    import mymodule
    mymodule.do_work()
finally:
    # Manual cleanup (also happens automatically on process exit)
    cleanup()
```

### Testing Subprocess Coverage

Verifying that subprocess coverage is working correctly:

```python
import subprocess
import os

def test_subprocess_coverage():
    """Test that subprocesses get coverage measurement."""
    
    # Run subprocess that imports and uses covered code
    script = '''
import sys
sys.path.insert(0, "src")
import mypackage
mypackage.function_to_cover()
'''
    
    result = subprocess.run([
        'python', '-c', script
    ], capture_output=True)
    
    assert result.returncode == 0
    # Coverage data will automatically include subprocess execution
```

## Signal Handling

pytest-cov implements robust signal handling to ensure coverage data is preserved:

### Signal Handler Management

```python
# Global state for signal handling
_previous_handlers = {}
_pending_signal = None  
_cleanup_in_progress = False
```

### Signal Processing

The signal handling system ensures coverage data is saved even during unexpected process termination:

1. **Handler Installation**: Previous handlers are saved and custom handler is installed
2. **Cleanup Coordination**: Prevents race conditions during cleanup 
3. **Handler Chaining**: Previous handlers are called after coverage cleanup
4. **Graceful Exit**: Processes exit with appropriate codes after cleanup

### Signal Handler Behavior

```python
def _signal_cleanup_handler(signum, frame):
    # If already cleaning up, defer signal
    if _cleanup_in_progress:
        _pending_signal = (signum, frame)
        return
        
    # Clean up coverage
    cleanup()
    
    # Call previous handler if exists
    previous = _previous_handlers.get(signum)
    if previous and previous != current_handler:
        previous(signum, frame)
    elif signum == signal.SIGTERM:
        os._exit(128 + signum)  # Standard exit code
    elif signum == signal.SIGINT:
        raise KeyboardInterrupt
```