# Command Line Tools

Collection of command-line utilities provided by mypy for type checking, stub generation, stub testing, daemon mode, and Python-to-C compilation. These tools form the core interface for mypy's functionality.

## Capabilities

### Main Type Checker (mypy)

The primary static type checker command that analyzes Python code for type errors.

```python { .api }
# Console entry point: mypy.__main__:console_entry
# Usage: mypy [options] files/directories
```

#### Common Usage Patterns

```bash
# Basic type checking
mypy myfile.py
mypy src/

# Strict mode checking
mypy --strict myfile.py

# Check with specific Python version
mypy --python-version 3.11 src/

# Show error codes
mypy --show-error-codes myfile.py

# Generate coverage report
mypy --html-report mypy-report src/

# Configuration via file
mypy src/  # Uses mypy.ini or pyproject.toml
```

#### Key Options

- `--strict`: Enable all strict mode flags
- `--python-version X.Y`: Target Python version
- `--platform PLATFORM`: Target platform (linux, win32, darwin)
- `--show-error-codes`: Display error codes with messages
- `--ignore-missing-imports`: Ignore errors from missing imports
- `--follow-imports MODE`: How to handle imports (normal, silent, skip, error)
- `--disallow-untyped-defs`: Require type annotations for all functions
- `--warn-unused-ignores`: Warn about unused `# type: ignore` comments
- `--html-report DIR`: Generate HTML coverage report
- `--junit-xml FILE`: Generate JUnit XML test results

### Stub Generator (stubgen)

Generates Python stub files (.pyi) from Python modules for type checking.

```python { .api }
# Console entry point: mypy.stubgen:main
# Usage: stubgen [options] modules/packages
```

#### Usage Examples

```bash
# Generate stubs for a module
stubgen mymodule

# Generate stubs for multiple modules
stubgen module1 module2 package1

# Generate stubs with output directory
stubgen -o stubs/ mymodule

# Generate stubs for packages with submodules
stubgen -p mypackage

# Include private definitions
stubgen --include-private mymodule

# Generate stubs from installed packages
stubgen -m requests numpy
```

#### Key Options

- `-o DIR`: Output directory for generated stubs
- `-p PACKAGE`: Generate stubs for package and subpackages
- `-m MODULE`: Generate stubs for installed module
- `--include-private`: Include private definitions in stubs
- `--export-less`: Don't export names imported from other modules
- `--ignore-errors`: Continue despite import errors
- `--no-import`: Don't import modules, analyze source only
- `--parse-only`: Parse AST without semantic analysis

### Stub Tester (stubtest)

Tests stub files for accuracy against the runtime modules they describe.

```python { .api }
# Console entry point: mypy.stubtest:main  
# Usage: stubtest [options] modules
```

#### Usage Examples

```bash
# Test stubs against runtime module
stubtest mymodule

# Test multiple modules
stubtest module1 module2

# Test with custom stub search path
stubtest --mypy-config-file mypy.ini mymodule

# Allow missing stubs for some items
stubtest --allowlist stubtest-allowlist.txt mymodule

# Generate allowlist of differences
stubtest --generate-allowlist mymodule > allowlist.txt
```

#### Key Options

- `--mypy-config-file FILE`: Use specific mypy configuration
- `--allowlist FILE`: File listing acceptable differences
- `--generate-allowlist`: Generate allowlist for current differences  
- `--ignore-missing-stub`: Don't error on missing stub files
- `--ignore-positional-only`: Allow positional-only differences
- `--concise`: Show fewer details in error messages

### Daemon Client (dmypy)

Client for mypy daemon mode, providing faster incremental type checking.

```python { .api }
# Console entry point: mypy.dmypy.client:console_entry
# Usage: dmypy [command] [options]
```

#### Daemon Commands

```bash
# Start daemon
dmypy daemon

# Check files using daemon  
dmypy check myfile.py
dmypy check src/

# Get daemon status
dmypy status

# Stop daemon
dmypy stop

# Restart daemon
dmypy restart

# Kill daemon (force)
dmypy kill
```

#### Usage Examples

```bash
# Start daemon with specific options
dmypy daemon --log-file daemon.log

# Incremental checking (much faster)
dmypy check --follow-imports=error src/

# Check with custom configuration
dmypy check --config-file mypy.ini src/

# Verbose output
dmypy check --verbose myfile.py

# Use specific daemon
dmypy --status-file custom.status check myfile.py
```

#### Key Options

- `--status-file FILE`: Custom daemon status file location
- `--timeout SECONDS`: Timeout for daemon operations
- `--log-file FILE`: Daemon log file location
- `--verbose`: Enable verbose output
- Standard mypy options apply to `check` command

### Python-to-C Compiler (mypyc)

Compiles Python modules with type annotations to efficient C extensions.

```python { .api }
# Console entry point: mypyc.__main__:main
# Usage: mypyc [options] files/modules
```

#### Usage Examples

```bash
# Compile a module
mypyc mymodule.py

# Compile multiple modules
mypyc module1.py module2.py

# Compile with optimizations
mypyc --opt-level 3 mymodule.py

# Multi-file compilation
mypyc --multi-file package/

# Debug compilation
mypyc --debug-level 2 mymodule.py

# Show generated C code
mypyc --show-c mymodule.py
```

#### Key Options

- `--opt-level LEVEL`: Optimization level (0-3)
- `--debug-level LEVEL`: Debug information level (0-3)  
- `--multi-file`: Generate multiple C files
- `--show-c`: Display generated C code
- `--verbose`: Enable verbose compilation output
- `--separate`: Compile modules separately
- `--no-compile`: Generate C code without compiling

## Integration Examples

### Continuous Integration

```yaml
# GitHub Actions example
- name: Type check with mypy
  run: |
    mypy --strict --show-error-codes src/
    
- name: Generate and test stubs
  run: |
    stubgen -o stubs/ src/
    stubtest --mypy-config-file mypy.ini src/
```

### Development Workflow

```bash
#!/bin/bash
# development type checking script

echo "Starting mypy daemon..."
dmypy daemon --log-file .mypy-daemon.log

echo "Type checking project..."
dmypy check --follow-imports=error src/

if [ $? -eq 0 ]; then
    echo "Type checking passed!"
    
    echo "Generating stubs..."
    stubgen -o typestubs/ -p myproject
    
    echo "Testing stubs..."
    stubtest --allowlist stubtest.allowlist myproject
    
    if [ $? -eq 0 ]; then
        echo "All checks passed!"
    else
        echo "Stub tests failed!"
        exit 1
    fi
else
    echo "Type checking failed!"
    exit 1
fi

echo "Stopping daemon..."
dmypy stop
```

### Build System Integration

```python
# setup.py with mypyc integration
from setuptools import setup
from mypyc.build import mypycify

ext_modules = mypycify([
    "mypackage/core.py",
    "mypackage/utils.py",
], opt_level="3")

setup(
    name="mypackage",
    ext_modules=ext_modules,
    # ... other setup parameters
)
```

### Editor Integration

```python
# Example editor plugin integration
import subprocess
from typing import List, Tuple

def run_mypy(files: List[str]) -> Tuple[bool, str]:
    """Run mypy on files and return success status and output."""
    try:
        result = subprocess.run(
            ['mypy', '--show-error-codes'] + files,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        return result.returncode == 0, result.stdout + result.stderr
        
    except subprocess.TimeoutExpired:
        return False, "Mypy timed out"
    except FileNotFoundError:
        return False, "Mypy not found - please install mypy"

def run_stubgen(module: str, output_dir: str) -> bool:
    """Generate stubs for module."""
    try:
        result = subprocess.run(
            ['stubgen', '-o', output_dir, module],
            capture_output=True,
            timeout=30
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

# Usage in editor
success, output = run_mypy(['myfile.py'])
if not success:
    # Show errors in editor
    display_errors(output)
```

## Configuration Files

### mypy.ini Configuration

```ini
[mypy]
python_version = 3.11
strict_mode = true
show_error_codes = true
warn_unused_ignores = true
warn_redundant_casts = true

[mypy-tests.*]
ignore_errors = true

[mypy-thirdparty.*]
ignore_missing_imports = true
```

### pyproject.toml Configuration

```toml
[tool.mypy]
python_version = "3.11"
strict_mode = true
show_error_codes = true
warn_unused_ignores = true
warn_redundant_casts = true

[[tool.mypy.overrides]]
module = "tests.*"
ignore_errors = true

[[tool.mypy.overrides]]
module = "thirdparty.*"
ignore_missing_imports = true
```

## Performance Tips

### Daemon Mode Benefits

- **First run**: Same speed as regular mypy
- **Subsequent runs**: 10-50x faster for large codebases
- **Best for**: Development environments with frequent checking

### Incremental Checking

```bash
# Enable incremental mode for faster repeated checks
mypy --incremental src/

# Use cache directory
mypy --cache-dir .mypy_cache src/
```

### Stub Generation Optimization

```bash
# Generate stubs for commonly used libraries once
stubgen -m requests numpy pandas -o stubs/

# Use generated stubs in mypy configuration
# mypy.ini: [mypy] | mypy_path = stubs/
```