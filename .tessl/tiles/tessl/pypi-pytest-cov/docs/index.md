# pytest-cov

A comprehensive pytest plugin that provides code coverage measurement with enhanced support for distributed testing, subprocess coverage, and multiple reporting formats. pytest-cov extends the standard coverage.py functionality to seamlessly integrate with the pytest testing framework while handling complex testing scenarios including parallel execution, distributed testing via pytest-xdist, and subprocess coverage collection.

## Package Information

- **Package Name**: pytest-cov
- **Language**: Python
- **Installation**: `pip install pytest-cov`

## Core Imports

```python
import pytest_cov
```

For accessing exceptions and warnings:

```python
from pytest_cov import CoverageError, CovDisabledWarning, CovReportWarning
```

For programmatic access to the plugin:

```python
from pytest_cov.plugin import CovPlugin
```

## Basic Usage

### Command-line Usage

```bash
# Basic coverage for a package
pytest --cov=mypackage tests/

# Multiple packages with HTML report
pytest --cov=package1 --cov=package2 --cov-report=html tests/

# Branch coverage with failure threshold
pytest --cov=mypackage --cov-branch --cov-fail-under=90 tests/

# Multiple report formats
pytest --cov=mypackage --cov-report=term-missing --cov-report=xml tests/
```

### Fixture Usage in Tests

```python
import pytest

def test_with_coverage_access(cov):
    """Test that accesses the coverage object."""
    if cov:
        # Access underlying coverage.py object
        print(f"Coverage data file: {cov.config.data_file}")

@pytest.mark.no_cover
def test_without_coverage():
    """This test will not be included in coverage."""
    pass

def test_with_no_cover_fixture(no_cover):
    """Test using no_cover fixture to disable coverage."""
    pass
```

## Architecture

pytest-cov implements a plugin architecture that adapts to different testing scenarios:

- **Plugin Integration**: Registers as a pytest11 plugin via entry points, automatically available when installed
- **Controller Pattern**: Uses different coverage controllers based on testing mode (Central, DistMaster, DistWorker)
- **Subprocess Support**: Automatically handles coverage for forked processes and subprocesses via environment variable propagation
- **Distributed Testing**: Seamlessly integrates with pytest-xdist for parallel test execution with proper coverage data combination

The plugin hooks into pytest's lifecycle at multiple points to ensure comprehensive coverage collection across all testing scenarios while maintaining compatibility with pytest's extensive ecosystem.

## Capabilities

### Command-line Options

Comprehensive command-line interface providing control over coverage measurement, reporting, and behavior with full integration into pytest's option system.

```python { .api }
# Core coverage options
--cov SOURCE                    # Path or package to measure (multi-allowed)
--cov-reset                     # Reset accumulated sources
--cov-report TYPE               # Report type: term, html, xml, json, lcov, annotate
--cov-config PATH               # Coverage config file (default: .coveragerc)

# Behavior control options  
--no-cov                        # Disable coverage completely
--no-cov-on-fail               # Don't report coverage if tests fail
--cov-append                    # Append to existing coverage data
--cov-branch                   # Enable branch coverage
--cov-fail-under MIN           # Fail if coverage below threshold
--cov-precision N              # Override reporting precision
--cov-context CONTEXT          # Dynamic context (only 'test' supported)
```

[Command-line Interface](./command-line.md)

### Plugin Integration and Hooks

Core pytest plugin functionality providing seamless integration with pytest's lifecycle and hook system for comprehensive coverage measurement.

```python { .api }
def pytest_addoption(parser): ...
def pytest_load_initial_conftests(early_config, parser, args): ...
def pytest_configure(config): ...

class CovPlugin:
    def __init__(self, options, pluginmanager, start=True): ...
    def start(self, controller_cls, config=None, nodeid=None): ...
    def pytest_sessionstart(self, session): ...
    def pytest_configure_node(self, node): ...
    def pytest_testnodedown(self, node, error): ...
    def pytest_runtestloop(self, session): ...
    def pytest_terminal_summary(self, terminalreporter): ...
    def pytest_runtest_setup(self, item): ...
    def pytest_runtest_teardown(self, item): ...
    def pytest_runtest_call(self, item): ...
```

[Plugin Integration](./plugin-integration.md)

### Coverage Controllers

Engine classes that handle different testing scenarios including single-process, distributed master, and distributed worker modes with proper data collection and combination.

```python { .api }
class CovController:
    def __init__(self, options, config, nodeid): ...
    def start(self): ...
    def finish(self): ...
    def pause(self): ...
    def resume(self): ...
    def summary(self, stream): ...

class Central(CovController): ...
class DistMaster(CovController): ...  
class DistWorker(CovController): ...
```

[Coverage Controllers](./coverage-controllers.md)

### Subprocess and Embed Support

Automatic coverage measurement for subprocesses and forked processes through environment variable propagation and signal handling.

```python { .api }
def init(): ...
def cleanup(): ...
def cleanup_on_signal(signum): ...
def cleanup_on_sigterm(): ...
```

[Subprocess Support](./subprocess-support.md)

### Test Fixtures and Markers

Built-in pytest fixtures and markers for controlling coverage behavior in individual tests.

```python { .api }
@pytest.fixture
def no_cover(): ...

@pytest.fixture  
def cov(request): ...

# Marker registration
config.addinivalue_line('markers', 'no_cover: disable coverage for this test.')
```

[Fixtures and Markers](./fixtures-markers.md)

### Exceptions and Warnings

Comprehensive error handling with specific exception types and warning categories for different failure scenarios.

```python { .api }
__version__: str  # Package version string

class CoverageError(Exception): ...
class PytestCovWarning(pytest.PytestWarning): ...
class CovDisabledWarning(PytestCovWarning): ...
class CovReportWarning(PytestCovWarning): ...
class CentralCovContextWarning(PytestCovWarning): ...
class DistCovError(Exception): ...
```

[Error Handling](./error-handling.md)

## Types

```python { .api }
# From typing and other modules used in signatures
from argparse import Namespace
from typing import Union, Optional, List, Dict, Any
from pathlib import Path
from io import StringIO

# Plugin configuration namespace
class Options:
    cov_source: Optional[List[str]]
    cov_report: Dict[str, Optional[str]]  
    cov_config: str
    cov_append: bool
    cov_branch: Optional[bool]
    cov_fail_under: Optional[Union[int, float]]
    cov_precision: Optional[int]
    cov_context: Optional[str]
    no_cov: bool
    no_cov_on_fail: bool
```