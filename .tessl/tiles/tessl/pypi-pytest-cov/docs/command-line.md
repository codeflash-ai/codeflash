# Command-line Interface

pytest-cov provides comprehensive command-line options that integrate seamlessly with pytest's option system. These options control coverage measurement, reporting formats, behavior, and thresholds.

## Capabilities

### Coverage Source Selection

Control which packages or paths are measured for coverage during test execution.

```python { .api }
--cov SOURCE
```

Specifies a path or package name to measure during execution. Can be used multiple times to measure multiple packages. Use `--cov=` with no argument to record everything without source filtering.

```python { .api }
--cov-reset
```

Resets the accumulated coverage sources from previous `--cov` options, allowing you to start fresh with coverage source selection.

**Usage Examples:**

```bash
# Single package
pytest --cov=mypackage tests/

# Multiple packages  
pytest --cov=package1 --cov=package2 tests/

# Everything (no source filtering)
pytest --cov= tests/

# Reset and specify new sources
pytest --cov=old_pkg --cov-reset --cov=new_pkg tests/
```

### Report Generation

Configure the types and formats of coverage reports generated after test execution.

```python { .api }
--cov-report TYPE
```

Specifies the type of coverage report to generate. Can be used multiple times for multiple report types. Supported types:

- `term`: Terminal output (default)
- `term-missing`: Terminal output with missing line numbers
- `html`: HTML report
- `xml`: XML report (Cobertura format)  
- `json`: JSON report
- `lcov`: LCOV format (requires coverage.py >= 6.3)
- `annotate`: Annotated source code

For file-based reports, append `:DEST` to specify output location. For terminal reports, append `:skip-covered` to hide fully covered files.

**Usage Examples:**

```bash
# Multiple report types
pytest --cov=mypackage --cov-report=html --cov-report=xml tests/

# Custom output locations
pytest --cov=mypackage --cov-report=html:htmlcov --cov-report=xml:coverage.xml tests/

# Terminal with skip-covered
pytest --cov=mypackage --cov-report=term-missing:skip-covered tests/

# No report output
pytest --cov=mypackage --cov-report= tests/
```

### Configuration Control

Specify coverage configuration and control coverage behavior.

```python { .api }
--cov-config PATH
```

Path to the coverage configuration file. Defaults to `.coveragerc` if not specified.

```python { .api }
--cov-append
```

Do not delete existing coverage data; append to current coverage. Default is False (coverage data is reset for each run).

```python { .api }
--cov-branch  
```

Enable branch coverage measurement in addition to statement coverage.

**Usage Examples:**

```bash
# Custom config file
pytest --cov=mypackage --cov-config=custom_coverage.ini tests/

# Append to existing coverage
pytest --cov=mypackage --cov-append tests/

# Enable branch coverage
pytest --cov=mypackage --cov-branch tests/
```

### Reporting Control

Control coverage reporting behavior and failure conditions.

```python { .api }
--cov-fail-under MIN
```

Fail the test run if the total coverage percentage is less than MIN. Accepts integer or float values up to 100.

```python { .api }
--cov-precision N
```

Override the default reporting precision for coverage percentages.

```python { .api }
--no-cov-on-fail
```

Do not report coverage information if the test run fails. Default is False (coverage is reported regardless of test results).

```python { .api }
--no-cov
```

Completely disable coverage reporting. Useful when running tests with debuggers that may conflict with coverage measurement.

**Usage Examples:**

```bash
# Fail if coverage below 90%
pytest --cov=mypackage --cov-fail-under=90 tests/

# Custom precision  
pytest --cov=mypackage --cov-precision=1 tests/

# No coverage on test failure
pytest --cov=mypackage --no-cov-on-fail tests/

# Disable coverage completely
pytest --cov=mypackage --no-cov tests/
```

### Advanced Options

Advanced coverage features for specialized use cases.

```python { .api }
--cov-context CONTEXT
```

Enable dynamic context switching for detailed coverage tracking. Currently only supports `test` as the context value, which provides per-test coverage tracking.

**Usage Examples:**

```bash
# Per-test context tracking  
pytest --cov=mypackage --cov-context=test tests/
```

## Validation Functions

pytest-cov includes validation functions to ensure command-line arguments are properly formatted and supported.

```python { .api }
def validate_report(arg: str) -> tuple:
    """
    Validate --cov-report argument format.
    
    Args:
        arg: Report specification (e.g., 'html', 'xml:output.xml', 'term:skip-covered')
        
    Returns:
        tuple: (report_type, modifier_or_output_path)
        
    Raises:
        argparse.ArgumentTypeError: If format is invalid or unsupported
    """

def validate_fail_under(num_str: str) -> Union[int, float]:
    """
    Validate --cov-fail-under argument.
    
    Args:
        num_str: Numeric string for coverage threshold
        
    Returns:
        Union[int, float]: Validated threshold value
        
    Raises:
        argparse.ArgumentTypeError: If not a valid number or > 100
    """

def validate_context(arg: str) -> str:
    """
    Validate --cov-context argument.
    
    Args:
        arg: Context specification
        
    Returns:
        str: Validated context value
        
    Raises:
        argparse.ArgumentTypeError: If context not supported or coverage.py too old
    """
```

## Option Processing

Internal functions that process and prepare command-line options for use by the coverage engine.

```python { .api }
def _prepare_cov_source(cov_source: List[Union[str, bool]]) -> Optional[List[str]]:
    """
    Process cov_source list to handle special cases.
    
    Makes --cov --cov=foobar equivalent to --cov (no filtering)
    and --cov=foo --cov=bar equivalent to filtering on ['foo', 'bar'].
    
    Args:
        cov_source: List containing source paths and boolean flags
        
    Returns:
        Optional[List[str]]: None for no filtering, list of sources otherwise
    """
```