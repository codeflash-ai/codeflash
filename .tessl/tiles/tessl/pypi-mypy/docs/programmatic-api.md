# Programmatic API

Simple API functions for integrating mypy into Python applications. These functions provide programmatic access to mypy's type checking functionality without requiring subprocess calls.

## Capabilities

### Basic Type Checking

Run mypy programmatically with the same interface as the command line tool.

```python { .api }
def run(args: list[str]) -> tuple[str, str, int]:
    """
    Run mypy programmatically with command line arguments.
    
    Parameters:
    - args: list[str] - Command line arguments (same as mypy CLI)
    
    Returns:
    - tuple[str, str, int]: (stdout_output, stderr_output, exit_code)
        - stdout_output: Normal type checking output and reports
        - stderr_output: Error messages and warnings
        - exit_code: 0 for success, non-zero for errors
    
    Usage:
    result = api.run(['--strict', 'myfile.py'])
    """
```

#### Usage Example

```python
from mypy import api

# Type check a single file
result = api.run(['myfile.py'])
stdout, stderr, exit_code = result

if exit_code == 0:
    print("Type checking passed!")
    if stdout:
        print("Output:", stdout)
else:
    print("Type checking failed!")
    if stderr:
        print("Errors:", stderr)

# Type check with options
result = api.run([
    '--strict',
    '--show-error-codes', 
    '--python-version', '3.11',
    'src/'
])

# Type check multiple files
result = api.run(['file1.py', 'file2.py', 'package/'])
```

### Daemon Mode Type Checking

Run the dmypy daemon client programmatically for faster incremental type checking.

```python { .api }
def run_dmypy(args: list[str]) -> tuple[str, str, int]:
    """
    Run dmypy daemon client programmatically.
    
    Parameters:
    - args: list[str] - Command line arguments for dmypy
    
    Returns:
    - tuple[str, str, int]: (stdout_output, stderr_output, exit_code)
    
    Note: 
    - Not thread-safe, modifies sys.stdout and sys.stderr during execution
    - Requires dmypy daemon to be running (start with: dmypy daemon)
    
    Usage:
    result = api.run_dmypy(['check', 'myfile.py'])
    """
```

#### Usage Example

```python
from mypy import api

# Start daemon first (usually done separately):
# subprocess.run(['dmypy', 'daemon'])

# Use daemon for faster checking
result = api.run_dmypy(['check', 'myfile.py'])
stdout, stderr, exit_code = result

# Daemon supports incremental checking
result = api.run_dmypy(['check', '--verbose', 'src/'])

# Stop daemon when done
result = api.run_dmypy(['stop'])
```

## Integration Patterns

### CI/CD Integration

```python
from mypy import api
import sys

def type_check_project():
    """Type check project in CI/CD pipeline."""
    result = api.run([
        '--strict',
        '--show-error-codes',
        '--junit-xml', 'mypy-results.xml',
        'src/'
    ])
    
    stdout, stderr, exit_code = result
    
    if exit_code != 0:
        print("Type checking failed:")
        print(stderr)
        sys.exit(1)
    
    print("Type checking passed!")
    return True

if __name__ == "__main__":
    type_check_project()
```

### Development Tools Integration

```python
from mypy import api
import os

class TypeChecker:
    """Wrapper for mypy integration in development tools."""
    
    def __init__(self, strict=True, python_version="3.11"):
        self.strict = strict
        self.python_version = python_version
    
    def check_file(self, filepath):
        """Check a single file."""
        args = []
        
        if self.strict:
            args.append('--strict')
        
        args.extend(['--python-version', self.python_version])
        args.append(filepath)
        
        return api.run(args)
    
    def check_files(self, filepaths):
        """Check multiple files."""
        args = []
        
        if self.strict:
            args.append('--strict')
        
        args.extend(['--python-version', self.python_version])
        args.extend(filepaths)
        
        return api.run(args)
    
    def has_errors(self, result):
        """Check if result has type errors."""
        stdout, stderr, exit_code = result
        return exit_code != 0

# Usage
checker = TypeChecker(strict=True)
result = checker.check_file('mymodule.py')

if checker.has_errors(result):
    print("Type errors found!")
```

### IDE/Editor Integration

```python
from mypy import api
import tempfile
import os

def check_code_snippet(code, filename="<string>"):
    """Type check a code snippet from an editor."""
    # Write code to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name
    
    try:
        # Run mypy on the temporary file
        result = api.run(['--show-error-codes', temp_path])
        stdout, stderr, exit_code = result
        
        # Parse errors and convert file paths back to original filename
        if stderr:
            stderr = stderr.replace(temp_path, filename)
        
        return stdout, stderr, exit_code
    finally:
        # Clean up temporary file
        os.unlink(temp_path)

# Usage in editor plugin
code = '''
def greet(name: str) -> str:
    return f"Hello, {name}!"

# This will cause a type error
result = greet(42)
'''

stdout, stderr, exit_code = check_code_snippet(code, "editor_buffer.py")
if exit_code != 0:
    print("Type errors in code:")
    print(stderr)
```

## Error Handling

### Common Return Codes

- **0**: No errors found
- **1**: Type errors found  
- **2**: Mypy crashed or invalid arguments

### Parsing Output

```python
from mypy import api
import re

def parse_mypy_output(result):
    """Parse mypy output into structured format."""
    stdout, stderr, exit_code = result
    
    errors = []
    if stderr:
        # Parse error lines: filename:line:column: error: message [error-code]
        error_pattern = r'([^:]+):(\d+):(\d+): (error|warning|note): (.+?)(?:\s+\[([^\]]+)\])?$'
        
        for line in stderr.strip().split('\n'):
            match = re.match(error_pattern, line)
            if match:
                filename, line_num, col_num, level, message, error_code = match.groups()
                errors.append({
                    'file': filename,
                    'line': int(line_num),
                    'column': int(col_num),
                    'level': level,
                    'message': message,
                    'error_code': error_code
                })
    
    return {
        'exit_code': exit_code,
        'stdout': stdout,
        'stderr': stderr,
        'errors': errors
    }

# Usage
result = api.run(['myfile.py'])
parsed = parse_mypy_output(result)

for error in parsed['errors']:
    print(f"{error['file']}:{error['line']}: {error['message']}")
```

## Performance Considerations

### Single vs Multiple Files

```python
# Less efficient - multiple mypy invocations
for file in files:
    result = api.run([file])

# More efficient - single mypy invocation
result = api.run(files)
```

### Daemon Mode for Repeated Checks

```python
# For repeated type checking, use daemon mode
result = api.run_dmypy(['check'] + files)  # Much faster for large projects
```

### Configuration via Files

```python
# Use mypy.ini or pyproject.toml for configuration instead of command line args
result = api.run(['src/'])  # Configuration loaded from files
```