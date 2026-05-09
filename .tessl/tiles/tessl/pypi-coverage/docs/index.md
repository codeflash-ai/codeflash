# Coverage.py

Coverage.py is a comprehensive code coverage measurement tool for Python that tracks which lines of code are executed during test runs or program execution. It provides detailed reporting capabilities including HTML, XML, JSON, and LCOV formats, supports branch coverage analysis to track conditional execution paths, and integrates seamlessly with popular testing frameworks like pytest and unittest.

## Package Information

- **Package Name**: coverage
- **Language**: Python
- **Installation**: `pip install coverage`
- **License**: Apache-2.0
- **Documentation**: https://coverage.readthedocs.io/

## Core Imports

```python
import coverage
```

Most common usage patterns:

```python
from coverage import Coverage
from coverage.data import CoverageData
```

## Basic Usage

### Command-line Usage

```bash
# Run your program with coverage measurement
coverage run my_program.py

# Generate a report
coverage report

# Generate HTML report
coverage html

# Combine multiple coverage files
coverage combine
```

### Programmatic Usage

```python
import coverage

# Create a Coverage instance
cov = coverage.Coverage()

# Start measurement
cov.start()

# Run your code here
import my_module
result = my_module.some_function()

# Stop measurement
cov.stop()

# Save data
cov.save()

# Generate report
cov.report()

# Generate HTML report
cov.html_report()
```

## Architecture

Coverage.py uses a layered architecture for flexible code measurement:

- **Coverage Class**: Main control interface managing configuration, measurement lifecycle, and reporting
- **Tracers**: Low-level execution tracking (C extension tracer for performance, Python fallback)
- **Data Storage**: SQLite-based storage with CoverageData interface for persistence and querying
- **Reporters**: Multiple output formats (console, HTML, XML, JSON, LCOV, annotated source)
- **Plugin System**: Extensible architecture for custom file tracers and configurers
- **Configuration**: Flexible configuration via pyproject.toml, .coveragerc, or programmatic setup

This design enables coverage.py to work across different Python implementations (CPython, PyPy), handle complex execution environments (multiprocessing, async code), and integrate with various testing frameworks and CI/CD systems.

## Capabilities

### Core Coverage Measurement

Primary coverage measurement and control functionality including starting/stopping measurement, data persistence, configuration management, and basic analysis. The Coverage class serves as the main entry point for all coverage operations.

```python { .api }
class Coverage:
    def __init__(
        self,
        data_file=None,
        data_suffix=None,
        cover_pylib=None,
        auto_data=False,
        timid=None,
        branch=None,
        config_file=True,
        source=None,
        source_pkgs=None,
        source_dirs=None,
        omit=None,
        include=None,
        debug=None,
        concurrency=None,
        check_preimported=False,
        context=None,
        messages=False,
        plugins=None
    ): ...
    
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def save(self) -> None: ...
    def load(self) -> None: ...
    def erase(self) -> None: ...
    def get_data(self) -> CoverageData: ...
```

[Core Coverage Measurement](./core-coverage.md)

### Data Storage and Retrieval

Coverage data storage, querying, and manipulation through the CoverageData class. Handles persistence of execution data, context information, and provides query interfaces for analysis and reporting.

```python { .api }
class CoverageData:
    def __init__(
        self,
        basename=None,
        suffix=None,
        warn=None,
        debug=None,
        no_disk=False
    ): ...
    
    def measured_files(self) -> set[str]: ...
    def lines(self, filename: str) -> set[int] | None: ...
    def arcs(self, filename: str) -> set[tuple[int, int]] | None: ...
    def has_arcs(self) -> bool: ...
```

[Data Storage and Retrieval](./data-storage.md)

### Report Generation

Multiple output formats for coverage reports including console text, HTML with highlighting, XML (Cobertura), JSON, LCOV, and annotated source files. Each reporter provides different visualization and integration capabilities.

```python { .api }
def report(
    self,
    morfs=None,
    show_missing=None,
    ignore_errors=None,
    file=None,
    omit=None,
    include=None,
    skip_covered=None,
    contexts=None,
    skip_empty=None,
    precision=None,
    sort=None,
    output_format=None
) -> float: ...

def html_report(
    self,
    morfs=None,
    directory=None,
    ignore_errors=None,
    omit=None,
    include=None,
    contexts=None,
    skip_covered=None,
    skip_empty=None,
    show_contexts=None,
    title=None,
    precision=None
) -> float: ...
```

[Report Generation](./reporting.md)

### Plugin System

Extensible plugin architecture for custom file tracers, configurers, and dynamic context switchers. Enables coverage measurement for non-Python files and custom execution environments.

```python { .api }
class CoveragePlugin:
    def file_tracer(self, filename: str) -> FileTracer | None: ...
    def file_reporter(self, filename: str) -> FileReporter | str: ...
    def dynamic_context(self, frame) -> str | None: ...
    def configure(self, config) -> None: ...

class FileTracer:
    def source_filename(self) -> str: ...
    def line_number_range(self, frame) -> tuple[int, int]: ...

class FileReporter:
    def lines(self) -> set[int]: ...
    def arcs(self) -> set[tuple[int, int]]: ...
    def source(self) -> str: ...
```

[Plugin System](./plugins.md)

### Configuration Management

Comprehensive configuration system supporting multiple file formats (pyproject.toml, .coveragerc), command-line options, and programmatic configuration. Manages source inclusion/exclusion, measurement options, and output settings.

```python { .api }
def get_option(self, option_name: str): ...
def set_option(self, option_name: str, value) -> None: ...
```

[Configuration Management](./configuration.md)

### Exception Handling

Complete exception hierarchy for coverage-related errors including data file issues, source code problems, plugin errors, and configuration problems.

```python { .api }
class CoverageException(Exception): ...
class NoDataError(CoverageException): ...
class NoSource(CoverageException): ...
class ConfigError(Exception): ...
class PluginError(CoverageException): ...
```

[Exception Handling](./exceptions.md)

## Types

```python { .api }
import os
import types
from typing import Union, Tuple

# Type aliases for common coverage types
FilePath = Union[str, os.PathLike[str]]
LineNumber = int
Arc = Tuple[LineNumber, LineNumber]
ModuleOrFilename = Union[types.ModuleType, str]

# Version information
version_info: Tuple[int, int, int, str, int]
__version__: str
```