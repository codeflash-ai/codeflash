# Exception Handling

Complete exception hierarchy for coverage-related errors including data file issues, source code problems, plugin errors, and configuration problems.

## Capabilities

### Base Exception Classes

Core exception hierarchy for all coverage.py errors.

```python { .api }
class CoverageException(Exception):
    """
    Base class for all exceptions raised by coverage.py.
    
    This is the parent class for all coverage-specific exceptions.
    Catch this to handle any coverage.py error.
    """

class _BaseCoverageException(Exception):
    """
    Base-base class for all coverage exceptions.
    
    Internal base class - use CoverageException instead.
    """
```

Usage example:

```python
import coverage

try:
    cov = coverage.Coverage()
    cov.start()
    # ... your code ...
    cov.stop()
    cov.report()
except coverage.CoverageException as e:
    print(f"Coverage error: {e}")
    # Handle any coverage-related error
```

### Data-Related Exceptions

Exceptions related to coverage data file operations and data integrity.

```python { .api }
class DataError(CoverageException):
    """
    An error occurred while using a coverage data file.
    
    Raised when there are problems reading, writing, or processing
    coverage data files.
    """

class NoDataError(CoverageException):
    """
    No coverage data was available.
    
    Raised when attempting to generate reports or perform analysis
    without any collected coverage data.
    """
```

Usage example:

```python
import coverage

cov = coverage.Coverage()

try:
    # Try to load data file
    cov.load()
except coverage.NoDataError:
    print("No coverage data found. Run 'coverage run' first.")
except coverage.DataError as e:
    print(f"Data file error: {e}")

try:
    # Try to generate report without data
    cov.report()
except coverage.NoDataError:
    print("No data to report on")
```

### Source Code Exceptions

Exceptions related to finding and processing source code files.

```python { .api }
class NoSource(CoverageException):
    """
    Couldn't find the source code for a module.
    
    Raised when coverage.py cannot locate the source file
    for a module that was imported or executed.
    """

class NoCode(NoSource):
    """
    Couldn't find any executable code.
    
    Raised when a file exists but contains no executable
    Python code (e.g., empty file, comments only).
    """

class NotPython(CoverageException):
    """
    A source file turned out not to be parsable Python.
    
    Raised when coverage.py attempts to parse a file as Python
    but it contains invalid Python syntax.
    """
```

Usage example:

```python
import coverage

cov = coverage.Coverage()
cov.start()

try:
    # Import a module that might not have source
    import compiled_module
except coverage.NoSource as e:
    print(f"Source not found: {e}")
except coverage.NotPython as e:
    print(f"Not valid Python: {e}")

cov.stop()

try:
    # Analyze a problematic file
    analysis = cov.analysis2('problematic_file.py')
except coverage.NoSource:
    print("Cannot find source for analysis")
except coverage.NoCode:
    print("File contains no executable code")
```

### Plugin-Related Exceptions

Exceptions related to plugin operations and misbehavior.

```python { .api }
class PluginError(CoverageException):
    """
    A plugin misbehaved in some way.
    
    Raised when a coverage plugin encounters an error or
    violates the plugin contract.
    """
```

Usage example:

```python
import coverage

try:
    cov = coverage.Coverage(plugins=['my_plugin'])
    cov.start()
    # ... code execution ...
    cov.stop()
except coverage.PluginError as e:
    print(f"Plugin error: {e}")
    # Plugin failed to initialize or caused an error
```

### Configuration Exceptions

Exceptions related to configuration file problems and invalid settings.

```python { .api }
class ConfigError(_BaseCoverageException):
    """
    A problem with a configuration file or configuration value.
    
    Raised when there are syntax errors in configuration files,
    invalid configuration options, or conflicting settings.
    
    Note: Inherits from _BaseCoverageException, not CoverageException.
    """
```

Usage example:

```python
import coverage

try:
    cov = coverage.Coverage(config_file='invalid_config.ini')
except coverage.ConfigError as e:
    print(f"Configuration error: {e}")

# Also can occur with programmatic configuration
try:
    cov = coverage.Coverage()
    cov.set_option('invalid_section:invalid_option', 'value')
except coverage.ConfigError as e:
    print(f"Invalid configuration option: {e}")
```

### Internal Exceptions

Internal exceptions used by coverage.py for error handling during execution.

```python { .api }
class _ExceptionDuringRun(CoverageException):
    """
    An exception happened while running customer code.
    
    This exception is used internally to wrap exceptions that
    occur during code execution under coverage measurement.
    Constructed with three arguments from sys.exc_info().
    """
```

### Warning Class

Warning class for non-fatal coverage issues.

```python { .api }
class CoverageWarning(Warning):
    """
    A warning from coverage.py.
    
    Used for non-fatal issues that users should be aware of,
    such as partial coverage measurement or configuration issues.
    """
```

Usage example:

```python
import coverage
import warnings

# Configure warning handling
warnings.filterwarnings('default', category=coverage.CoverageWarning)

cov = coverage.Coverage()
cov.start()

# This might generate warnings
try:
    import some_problematic_module
except ImportWarning:
    pass

cov.stop()

# Warnings might be issued during reporting
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    cov.report()
    
    for warning in w:
        if issubclass(warning.category, coverage.CoverageWarning):
            print(f"Coverage warning: {warning.message}")
```

## Exception Handling Patterns

### Comprehensive Error Handling

Handle all coverage exceptions in a structured way:

```python
import coverage
import sys

def run_with_coverage():
    cov = coverage.Coverage(branch=True)
    
    try:
        cov.start()
        
        # Your application code here
        import my_app
        my_app.main()
        
    except coverage.CoverageException as e:
        print(f"Coverage measurement error: {e}", file=sys.stderr)
        return 1
    
    finally:
        cov.stop()
    
    try:
        cov.save()
    except coverage.DataError as e:
        print(f"Failed to save coverage data: {e}", file=sys.stderr)
        return 1
    
    try:
        total_coverage = cov.report()
        print(f"Total coverage: {total_coverage:.1f}%")
        
        if total_coverage < 80:
            print("Coverage below threshold!")
            return 1
            
    except coverage.NoDataError:
        print("No coverage data to report", file=sys.stderr)
        return 1
    except coverage.CoverageException as e:
        print(f"Report generation failed: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(run_with_coverage())
```

### Specific Exception Handling

Handle specific types of coverage errors:

```python
import coverage
import os

def analyze_coverage_data(data_file):
    """Analyze coverage data with detailed error handling."""
    
    if not os.path.exists(data_file):
        print(f"Coverage data file not found: {data_file}")
        return None
    
    cov = coverage.Coverage(data_file=data_file)
    
    try:
        cov.load()
    except coverage.DataError as e:
        print(f"Corrupted or invalid data file: {e}")
        return None
    except coverage.NoDataError:
        print("Data file exists but contains no coverage data")
        return None
    
    try:
        data = cov.get_data()
        files = data.measured_files()
        
        if not files:
            print("No files were measured")
            return None
        
        results = {}
        for filename in files:
            try:
                analysis = cov.analysis2(filename)
                results[filename] = {
                    'statements': len(analysis[1]),
                    'missing': len(analysis[3]),
                    'coverage': (len(analysis[1]) - len(analysis[3])) / len(analysis[1]) * 100
                }
            except coverage.NoSource:
                print(f"Warning: Source not found for {filename}")
                results[filename] = {'error': 'source_not_found'}
            except coverage.NotPython:
                print(f"Warning: {filename} is not valid Python")
                results[filename] = {'error': 'not_python'}
            except coverage.NoCode:
                print(f"Warning: {filename} contains no executable code")
                results[filename] = {'error': 'no_code'}
        
        return results
        
    except coverage.CoverageException as e:
        print(f"Unexpected coverage error: {e}")
        return None

# Usage
results = analyze_coverage_data('.coverage')
if results:
    for filename, stats in results.items():
        if 'error' in stats:
            print(f"{filename}: {stats['error']}")
        else:
            print(f"{filename}: {stats['coverage']:.1f}% coverage")
```

### Plugin Error Handling

Handle plugin-related errors gracefully:

```python
import coverage

def create_coverage_with_plugins(plugin_names):
    """Create coverage instance with error handling for plugins."""
    
    successful_plugins = []
    failed_plugins = []
    
    for plugin_name in plugin_names:
        try:
            # Test plugin individually
            test_cov = coverage.Coverage(plugins=[plugin_name])
            test_cov.start()
            test_cov.stop()
            successful_plugins.append(plugin_name)
        except coverage.PluginError as e:
            print(f"Plugin '{plugin_name}' failed: {e}")
            failed_plugins.append(plugin_name)
        except Exception as e:
            print(f"Unexpected error with plugin '{plugin_name}': {e}")
            failed_plugins.append(plugin_name)
    
    if successful_plugins:
        print(f"Using plugins: {successful_plugins}")
        return coverage.Coverage(plugins=successful_plugins)
    else:
        print("No plugins could be loaded, using default coverage")
        return coverage.Coverage()

# Usage
cov = create_coverage_with_plugins(['plugin1', 'plugin2', 'problematic_plugin'])
```

### Configuration Error Handling

Handle configuration errors with fallbacks:

```python
import coverage

def create_coverage_with_config(config_files):
    """Try multiple configuration files until one works."""
    
    for config_file in config_files:
        try:
            cov = coverage.Coverage(config_file=config_file)
            print(f"Using configuration from: {config_file}")
            return cov
        except coverage.ConfigError as e:
            print(f"Configuration error in {config_file}: {e}")
            continue
        except FileNotFoundError:
            print(f"Configuration file not found: {config_file}")
            continue
    
    print("No valid configuration found, using defaults")
    return coverage.Coverage(config_file=False)

# Usage
config_files = ['pyproject.toml', '.coveragerc', 'setup.cfg']
cov = create_coverage_with_config(config_files)
```

This comprehensive exception handling enables robust coverage measurement even in complex environments with multiple potential failure points.