# Report Generation

Multiple output formats for coverage reports including console text, HTML with highlighting, XML (Cobertura), JSON, LCOV, and annotated source files. Each reporter provides different visualization and integration capabilities.

## Capabilities

### Console Text Reports

Generate formatted console output showing coverage statistics and missing lines.

```python { .api }
def report(
    self,
    morfs=None,                    # Files/modules to report on
    show_missing=None,             # Show line numbers of missing statements
    ignore_errors=None,            # Ignore source file errors
    file=None,                     # Output file object
    omit=None,                     # File patterns to omit
    include=None,                  # File patterns to include
    skip_covered=None,             # Skip files with 100% coverage
    contexts=None,                 # Context labels to filter by
    skip_empty=None,               # Skip files with no executable code
    precision=None,                # Decimal precision for percentages
    sort=None,                     # Sort order for files
    output_format=None             # Output format ('text' or 'total')
) -> float:
    """
    Generate a text coverage report.
    
    Parameters:
    - morfs (list | None): Modules or filenames to report on
    - show_missing (bool | None): Include line numbers of missing statements
    - ignore_errors (bool | None): Continue despite source file errors
    - file (IO | None): File object to write output (default stdout)
    - omit (str | list[str] | None): File patterns to omit from report
    - include (str | list[str] | None): File patterns to include in report
    - skip_covered (bool | None): Don't report files with 100% coverage
    - contexts (list[str] | None): Only include data from these contexts
    - skip_empty (bool | None): Don't report files with no executable code
    - precision (int | None): Number of decimal places for percentages
    - sort (str | None): Sort files by 'name', 'stmts', 'miss', 'branch', 'brpart', 'cover'
    - output_format (str | None): 'text' for full report, 'total' for percentage only
    
    Returns:
        float: Overall coverage percentage
    """
```

Usage example:

```python
import coverage

cov = coverage.Coverage()
cov.start()
# ... run code ...
cov.stop()

# Basic console report
total_coverage = cov.report()
print(f"Total coverage: {total_coverage:.1f}%")

# Detailed report with missing lines
cov.report(
    show_missing=True,
    skip_covered=False,
    precision=1,
    sort='cover'
)

# Report specific files only
cov.report(
    morfs=['src/core.py', 'src/utils.py'],
    show_missing=True
)

# Save report to file
with open('coverage_report.txt', 'w') as f:
    cov.report(file=f, show_missing=True)
```

### HTML Reports

Generate interactive HTML reports with syntax highlighting and detailed coverage visualization.

```python { .api }
def html_report(
    self,
    morfs=None,                    # Files/modules to report on
    directory=None,                # Output directory
    ignore_errors=None,            # Ignore source file errors
    omit=None,                     # File patterns to omit
    include=None,                  # File patterns to include
    contexts=None,                 # Context labels to filter by
    skip_covered=None,             # Skip files with 100% coverage
    skip_empty=None,               # Skip files with no executable code
    show_contexts=None,            # Show context information
    title=None,                    # Title for HTML pages
    precision=None                 # Decimal precision for percentages
) -> float:
    """
    Generate an HTML coverage report.
    
    Parameters:
    - morfs (list | None): Modules or filenames to report on
    - directory (str | None): Directory to write HTML files (default 'htmlcov')
    - ignore_errors (bool | None): Continue despite source file errors
    - omit (str | list[str] | None): File patterns to omit from report
    - include (str | list[str] | None): File patterns to include in report
    - contexts (list[str] | None): Only include data from these contexts
    - skip_covered (bool | None): Don't report files with 100% coverage
    - skip_empty (bool | None): Don't report files with no executable code
    - show_contexts (bool | None): Include context information in HTML
    - title (str | None): Title for the HTML report pages
    - precision (int | None): Number of decimal places for percentages
    
    Returns:
        float: Overall coverage percentage
    """
```

Usage example:

```python
import coverage

cov = coverage.Coverage()
cov.start()
# ... run code ...
cov.stop()

# Basic HTML report
total_coverage = cov.html_report()
print(f"HTML report generated, total coverage: {total_coverage:.1f}%")

# Customized HTML report
cov.html_report(
    directory='custom_htmlcov',
    title='My Project Coverage Report',
    show_contexts=True,
    skip_covered=True,
    precision=2
)

# Report specific modules
cov.html_report(
    morfs=['src/'],
    directory='src_coverage',
    omit=['*/tests/*']
)
```

### XML Reports

Generate XML reports in Cobertura format for integration with CI/CD systems and IDEs.

```python { .api }
def xml_report(
    self,
    morfs=None,                    # Files/modules to report on
    outfile=None,                  # Output file path or object
    ignore_errors=None,            # Ignore source file errors
    omit=None,                     # File patterns to omit
    include=None,                  # File patterns to include
    contexts=None,                 # Context labels to filter by
    skip_empty=None,               # Skip files with no executable code
    precision=None                 # Decimal precision for percentages
) -> float:
    """
    Generate an XML coverage report in Cobertura format.
    
    Parameters:
    - morfs (list | None): Modules or filenames to report on
    - outfile (str | IO | None): File path or object to write XML (default 'coverage.xml')
    - ignore_errors (bool | None): Continue despite source file errors
    - omit (str | list[str] | None): File patterns to omit from report
    - include (str | list[str] | None): File patterns to include in report
    - contexts (list[str] | None): Only include data from these contexts
    - skip_empty (bool | None): Don't report files with no executable code
    - precision (int | None): Number of decimal places for percentages
    
    Returns:
        float: Overall coverage percentage
    """
```

Usage example:

```python
import coverage

cov = coverage.Coverage()
cov.start()
# ... run code ...
cov.stop()

# Basic XML report
total_coverage = cov.xml_report()
print(f"XML report written to coverage.xml, total: {total_coverage:.1f}%")

# Custom XML output file
cov.xml_report(outfile='reports/cobertura.xml')

# XML report with file object
with open('custom_coverage.xml', 'w') as f:
    cov.xml_report(outfile=f, precision=2)
```

### JSON Reports

Generate JSON format reports for programmatic processing and integration.

```python { .api }
def json_report(
    self,
    morfs=None,                    # Files/modules to report on
    outfile=None,                  # Output file path or object
    ignore_errors=None,            # Ignore source file errors
    omit=None,                     # File patterns to omit
    include=None,                  # File patterns to include
    contexts=None,                 # Context labels to filter by
    skip_empty=None,               # Skip files with no executable code
    precision=None,                # Decimal precision for percentages
    pretty_print=None,             # Format JSON for readability
    show_contexts=None             # Include context information
) -> float:
    """
    Generate a JSON coverage report.
    
    Parameters:
    - morfs (list | None): Modules or filenames to report on
    - outfile (str | IO | None): File path or object to write JSON (default 'coverage.json')
    - ignore_errors (bool | None): Continue despite source file errors
    - omit (str | list[str] | None): File patterns to omit from report
    - include (str | list[str] | None): File patterns to include in report
    - contexts (list[str] | None): Only include data from these contexts
    - skip_empty (bool | None): Don't report files with no executable code
    - precision (int | None): Number of decimal places for percentages
    - pretty_print (bool | None): Format JSON for human readability
    - show_contexts (bool | None): Include context information in output
    
    Returns:
        float: Overall coverage percentage
    """
```

Usage example:

```python
import coverage

cov = coverage.Coverage()
cov.start()
# ... run code ...
cov.stop()

# Basic JSON report
total_coverage = cov.json_report()
print(f"JSON report written to coverage.json, total: {total_coverage:.1f}%")

# Pretty-printed JSON with contexts
cov.json_report(
    outfile='reports/coverage.json',
    pretty_print=True,
    show_contexts=True,
    precision=2
)

# Process JSON programmatically
import json
import io

json_buffer = io.StringIO()
cov.json_report(outfile=json_buffer)
json_data = json.loads(json_buffer.getvalue())
print(f"Files: {len(json_data['files'])}")
```

### LCOV Reports

Generate LCOV format reports for integration with LCOV tools and web interfaces.

```python { .api }
def lcov_report(
    self,
    morfs=None,                    # Files/modules to report on
    outfile=None,                  # Output file path or object
    ignore_errors=None,            # Ignore source file errors
    omit=None,                     # File patterns to omit
    include=None,                  # File patterns to include
    contexts=None,                 # Context labels to filter by
    skip_empty=None,               # Skip files with no executable code
    precision=None                 # Decimal precision for percentages
) -> float:
    """
    Generate an LCOV coverage report.
    
    Parameters:
    - morfs (list | None): Modules or filenames to report on
    - outfile (str | IO | None): File path or object to write LCOV (default 'coverage.lcov')
    - ignore_errors (bool | None): Continue despite source file errors
    - omit (str | list[str] | None): File patterns to omit from report
    - include (str | list[str] | None): File patterns to include in report
    - contexts (list[str] | None): Only include data from these contexts
    - skip_empty (bool | None): Don't report files with no executable code
    - precision (int | None): Number of decimal places for percentages
    
    Returns:
        float: Overall coverage percentage
    """
```

Usage example:

```python
import coverage

cov = coverage.Coverage()
cov.start()
# ... run code ...
cov.stop()

# Basic LCOV report
total_coverage = cov.lcov_report()
print(f"LCOV report written to coverage.lcov, total: {total_coverage:.1f}%")

# Custom LCOV output
cov.lcov_report(
    outfile='reports/lcov.info',
    omit=['*/test_*']
)
```

### Annotated Source Files

Generate annotated source files showing coverage line-by-line.

```python { .api }
def annotate(
    self,
    morfs=None,                    # Files/modules to annotate
    directory=None,                # Output directory
    ignore_errors=None,            # Ignore source file errors
    omit=None,                     # File patterns to omit
    include=None,                  # File patterns to include
    contexts=None                  # Context labels to filter by
) -> None:
    """
    Generate annotated source files showing coverage.
    
    Parameters:
    - morfs (list | None): Modules or filenames to annotate
    - directory (str | None): Directory to write annotated files (default '.')
    - ignore_errors (bool | None): Continue despite source file errors
    - omit (str | list[str] | None): File patterns to omit from annotation
    - include (str | list[str] | None): File patterns to include in annotation
    - contexts (list[str] | None): Only include data from these contexts
    """
```

Usage example:

```python
import coverage

cov = coverage.Coverage()
cov.start()
# ... run code ...
cov.stop()

# Generate annotated source files
cov.annotate(directory='annotated_source')

# Annotate specific files
cov.annotate(
    morfs=['src/core.py', 'src/utils.py'],
    directory='annotations'
)
```

## Report Configuration

All reporting methods support common filtering and formatting options:

### File Selection

- **`morfs`**: Specify particular modules or files to include
- **`omit`**: Exclude files matching glob patterns
- **`include`**: Only include files matching glob patterns
- **`skip_covered`**: Exclude files with 100% coverage
- **`skip_empty`**: Exclude files with no executable statements

### Data Filtering

- **`contexts`**: Filter data by context labels (for dynamic context switching)
- **`ignore_errors`**: Continue processing despite source file errors

### Output Formatting

- **`precision`**: Number of decimal places for coverage percentages
- **`show_contexts`**: Include context information in output (HTML/JSON)
- **`pretty_print`**: Format JSON output for readability

### Example: Comprehensive Reporting Workflow

```python
import coverage

# Set up coverage with branch measurement
cov = coverage.Coverage(
    branch=True,
    source=['src/'],
    omit=['*/tests/*', '*/migrations/*']
)

cov.start()
# ... run your application/tests ...
cov.stop()
cov.save()

# Generate all report formats
print("Generating coverage reports...")

# Console report
total_coverage = cov.report(
    show_missing=True,
    skip_covered=True,
    precision=1
)

# HTML report for browsing
cov.html_report(
    directory='htmlcov',
    title='My Project Coverage Report',
    show_contexts=True
)

# XML report for CI/CD
cov.xml_report(outfile='reports/coverage.xml')

# JSON report for programmatic use
cov.json_report(
    outfile='reports/coverage.json',
    pretty_print=True,
    show_contexts=True
)

# LCOV report for integration tools
cov.lcov_report(outfile='reports/coverage.lcov')

print(f"Total coverage: {total_coverage:.1f}%")
print("All reports generated successfully!")
```