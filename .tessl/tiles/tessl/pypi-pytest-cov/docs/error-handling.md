# Error Handling

pytest-cov provides comprehensive error handling with specific exception types and warning categories for different failure scenarios. This enables proper error reporting and allows users to handle coverage-related issues appropriately.

## Capabilities

### Core Exceptions

Custom exception types for coverage-related errors.

```python { .api }
class CoverageError(Exception):
    """
    Indicates that coverage measurement has failed or coverage is too low.
    
    Raised when coverage thresholds are not met or when coverage measurement
    encounters critical errors that prevent proper operation.
    
    Usage:
        Used internally by coverage threshold checking and can be raised
        by user code that implements custom coverage validation.
    """

class DistCovError(Exception):
    """
    Raised when distributed testing configuration conflicts with coverage settings.
    
    Specifically raised when dynamic_context='test_function' is configured
    in coverage settings while using pytest-xdist for distributed testing,
    which is known to cause coverage data corruption.
    
    Reference: https://github.com/pytest-dev/pytest-cov/issues/604
    
    Usage:
        Automatically raised during distributed master initialization
        if incompatible configuration is detected.
    """
```

**Usage Examples:**

```python
from pytest_cov import CoverageError, DistCovError

# CoverageError usage
def validate_coverage_threshold(coverage_percent, threshold):
    if coverage_percent < threshold:
        raise CoverageError(f"Coverage {coverage_percent}% below threshold {threshold}%")

# DistCovError is raised automatically by pytest-cov
# when incompatible configuration is detected:
# pytest --cov=mypackage -n 4  # with dynamic_context=test_function in .coveragerc
```

### Warning System

Hierarchical warning system for non-fatal coverage issues.

```python { .api }
class PytestCovWarning(pytest.PytestWarning):
    """
    Base class for all pytest-cov warnings.
    
    Inherits from pytest.PytestWarning to integrate with pytest's warning
    system. Never raised directly - serves as base for specific warning types.
    
    All pytest-cov warnings inherit from this class, allowing users to
    filter all coverage warnings with a single filter specification.
    """

class CovDisabledWarning(PytestCovWarning):
    """
    Warning when coverage measurement is manually disabled.
    
    Raised when --no-cov flag is used in combination with other coverage
    options, indicating that coverage has been intentionally disabled
    but other coverage-related options were specified.
    """

class CovReportWarning(PytestCovWarning):
    """
    Warning when coverage report generation fails.
    
    Raised when coverage measurement completes successfully but report
    generation encounters errors (e.g., file I/O issues, format errors,
    or coverage.py exceptions during report creation).
    """

class CentralCovContextWarning(PytestCovWarning):
    """
    Warning about suboptimal coverage context configuration.
    
    Raised when dynamic_context='test_function' is detected in coverage
    configuration. Recommends using --cov-context option instead, which
    provides more complete and reliable per-test context tracking.
    """
```

**Usage Examples:**

```python
import warnings
from pytest_cov import PytestCovWarning, CovDisabledWarning

# Filter all pytest-cov warnings
warnings.filterwarnings('ignore', category=PytestCovWarning)

# Filter specific warning types
warnings.filterwarnings('ignore', category=CovDisabledWarning)

# Convert warnings to errors for strict validation
warnings.filterwarnings('error', category=CovReportWarning)
```

### Engine-Level Exceptions

Internal exceptions for coverage engine operation.

```python { .api }
class BrokenCovConfigError(Exception):
    """
    Exception raised when coverage configuration is invalid or corrupted.
    
    Indicates that coverage configuration files cannot be parsed or
    contain invalid settings that prevent coverage measurement from
    starting properly.
    
    Usage:
        Raised internally by coverage controllers when configuration
        validation fails during initialization.
    """
```

## Error Handling Patterns

### Threshold Validation

How coverage threshold failures are handled:

```python
# Internal threshold checking logic
if should_fail_under(total_coverage, threshold, precision):
    message = f'Coverage failure: total of {total_coverage} is less than fail-under={threshold}'
    # Display error and increment test failure count
    terminalreporter.write(f'\nERROR: {message}\n', red=True, bold=True)
    session.testsfailed += 1
```

### Report Generation Errors

Handling failures during report generation:

```python
try:
    total_coverage = self.cov_controller.summary(self.cov_report)
except CoverageException as exc:
    message = f'Failed to generate report: {exc}\n'
    terminalreporter.write(f'\nWARNING: {message}\n', red=True, bold=True)
    warnings.warn(CovReportWarning(message), stacklevel=1)
    total_coverage = 0
```

### Distributed Testing Validation

Validation for distributed testing compatibility:

```python
if self.cov.config.dynamic_context == 'test_function':
    raise DistCovError(
        'Detected dynamic_context=test_function in coverage configuration. '
        'This is known to cause issues when using xdist, see: https://github.com/pytest-dev/pytest-cov/issues/604\n'
        'It is recommended to use --cov-context instead.'
    )
```

### Warning Integration

How warnings are integrated with pytest's warning system:

```python
# Warning filter setup during test execution
warnings.filterwarnings('default', 'unclosed database in <sqlite3.Connection object at', ResourceWarning)
warnings.simplefilter('once', PytestCovWarning)
warnings.simplefilter('once', CoverageWarning)

# Warning emission
warnings.warn(CovDisabledWarning(message), stacklevel=1)
warnings.warn(CovReportWarning(message), stacklevel=1)
warnings.warn(CentralCovContextWarning(message), stacklevel=1)
```

## Configuration Validation

### Command-line Validation

Validation functions for command-line arguments:

```python { .api }
def validate_report(arg: str) -> tuple:
    """
    Validate --cov-report argument format and compatibility.
    
    Args:
        arg: Report specification string
        
    Returns:
        tuple: (report_type, modifier_or_output)
        
    Raises:
        argparse.ArgumentTypeError: If format invalid or unsupported
        
    Validation includes:
    - Report type must be valid (term, html, xml, json, lcov, annotate)
    - LCOV requires coverage.py >= 6.3
    - Modifiers only allowed for appropriate report types
    """

def validate_fail_under(num_str: str) -> Union[int, float]:
    """
    Validate --cov-fail-under threshold value.
    
    Args:
        num_str: Numeric threshold string
        
    Returns:
        Union[int, float]: Validated threshold value
        
    Raises:
        argparse.ArgumentTypeError: If not numeric or > 100
        
    Includes humorous error message for values > 100:
    "Your desire for over-achievement is admirable but misplaced..."
    """

def validate_context(arg: str) -> str:
    """
    Validate --cov-context argument and coverage.py compatibility.
    
    Args:
        arg: Context specification
        
    Returns:
        str: Validated context value
        
    Raises:
        argparse.ArgumentTypeError: If unsupported context or old coverage.py
        
    Requires coverage.py >= 5.0 and currently only supports 'test' context.
    """
```

### Error Recovery

How pytest-cov handles and recovers from various error conditions:

#### Missing Coverage Plugin

```python
# Graceful handling when coverage plugin not available
if request.config.pluginmanager.hasplugin('_cov'):
    plugin = request.config.pluginmanager.getplugin('_cov')
    if plugin.cov_controller:
        return plugin.cov_controller.cov
return None  # Graceful fallback
```

#### Worker Communication Failures

```python
# Handle failed workers in distributed testing
if 'cov_worker_node_id' not in output:
    self.failed_workers.append(node)
    return

# Report failed workers in summary
if self.failed_workers:
    stream.write('The following workers failed to return coverage data...\n')
    for node in self.failed_workers:
        stream.write(f'{node.gateway.id}\n')
```

#### Directory Changes

```python
# Handle cases where working directory disappears
try:
    original_cwd = Path.cwd()
except OSError:
    # Directory gone - can't restore but continue
    original_cwd = None
    
# Robust directory restoration
if original_cwd is not None:
    os.chdir(original_cwd)
```

## User Error Handling

### Common Error Scenarios

Handling common user configuration errors:

```python
# Conflicting options detection
no_cov = False
for arg in args:
    if arg == '--no-cov':
        no_cov = True
    elif arg.startswith('--cov') and no_cov:
        options.no_cov_should_warn = True
        break

# Display warning for conflicting options
if self.options.no_cov_should_warn:
    message = 'Coverage disabled via --no-cov switch!'
    terminalreporter.write(f'WARNING: {message}\n', red=True, bold=True)
    warnings.warn(CovDisabledWarning(message), stacklevel=1)
```

### Helpful Error Messages

Examples of user-friendly error messages:

```python
# Threshold validation with helpful context
if value > 100:
    raise ArgumentTypeError(
        'Your desire for over-achievement is admirable but misplaced. '
        'The maximum value is 100. Perhaps write more integration tests?'
    )

# Clear compatibility requirements
if coverage.version_info <= (5, 0):
    raise ArgumentTypeError('Contexts are only supported with coverage.py >= 5.x')

# Specific guidance for complex issues  
raise DistCovError(
    'Detected dynamic_context=test_function in coverage configuration. '
    'This is known to cause issues when using xdist, see: https://github.com/pytest-dev/pytest-cov/issues/604\n'
    'It is recommended to use --cov-context instead.'
)
```