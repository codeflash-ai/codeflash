# Environment Management

Python environment detection and management supporting virtualenvs, system environments, and custom Python installations. Provides environment-aware analysis and completion with proper isolation and version-specific behavior.

## Capabilities

### Environment Discovery

Discover available Python environments including virtual environments and system installations.

```python { .api }
def find_virtualenvs(paths=None, *, safe=True, use_environment_vars=True):
    """
    Find virtual environments.
    
    Parameters:
    - paths (list, optional): Paths to search for virtual environments.
    - safe (bool): Only return safe/valid environments. Default True.
    - use_environment_vars (bool): Use environment variables in search. Default True.
    
    Returns:
    Generator of Environment objects.
    """

def find_system_environments(*, env_vars=None):
    """
    Find system Python environments.
    
    Parameters:
    - env_vars (dict, optional): Environment variables to use.
    
    Returns:
    Generator of Environment objects.
    """
```

**Usage Example:**
```python
import jedi
from jedi.api.environment import find_virtualenvs, find_system_environments

# Find all virtual environments
print("Virtual environments:")
for env in find_virtualenvs():
    print(f"  {env.path} - Python {env.version_info}")
    print(f"  Executable: {env.executable}")

# Find system Python installations
print("\nSystem environments:")
for env in find_system_environments():
    print(f"  {env.path} - Python {env.version_info}")
    print(f"  Executable: {env.executable}")

# Search specific paths for virtual environments
custom_paths = ["/home/user/.virtualenvs", "/opt/venvs"]
for env in find_virtualenvs(paths=custom_paths):
    print(f"Custom env: {env.path}")
```

### Environment Creation and Selection

Create and configure environments for jedi analysis.

```python { .api }
def create_environment(path, *, safe=True, env_vars=None):
    """
    Create an environment from a path.
    
    Parameters:
    - path (str or Path): Path to Python executable or environment.
    - safe (bool): Validate environment safety. Default True.
    - env_vars (dict, optional): Environment variables.
    
    Returns:
    Environment object.
    """

def get_default_environment():
    """
    Get the default Python environment.
    
    Returns:
    Environment object for current Python.
    """

def get_system_environment(version, *, env_vars=None):
    """
    Get a specific system environment by version.
    
    Parameters:
    - version (str): Python version (e.g., '3.9', '3.10').
    - env_vars (dict, optional): Environment variables.
    
    Returns:
    Environment object or None if not found.
    """
```

**Usage Example:**
```python
import jedi
from jedi.api.environment import (
    create_environment, get_default_environment, get_system_environment
)

# Get default environment
default_env = get_default_environment()
print(f"Default: Python {default_env.version_info} at {default_env.executable}")

# Create environment from specific path
venv_path = "/path/to/venv/bin/python"
try:
    custom_env = create_environment(venv_path)
    print(f"Custom env: Python {custom_env.version_info}")
except jedi.InvalidPythonEnvironment:
    print("Invalid environment path")

# Get specific Python version
py39_env = get_system_environment("3.9")
if py39_env:
    print(f"Python 3.9: {py39_env.executable}")
else:
    print("Python 3.9 not found")

# Use environment with Script
script = jedi.Script(
    code="import sys; sys.version",
    environment=custom_env
)
```

### Environment Properties and Information

Access environment properties and system information.

```python { .api }
class Environment:
    executable: Path  # Python executable path
    path: Path  # Environment path (sys.prefix)
    version_info: tuple  # Python version tuple (major, minor, micro)
    
    def get_sys_path(self):
        """Get sys.path for this environment."""
    
    def get_grammar(self):
        """
        Get parso grammar for this Python version.
        
        Returns:
        Grammar: Parso grammar object for parsing Python code.
        """
```

**Usage Example:**
```python
import jedi
from jedi.api.environment import find_virtualenvs

for env in find_virtualenvs():
    print(f"Environment: {env.path}")
    print(f"Executable: {env.executable}")
    print(f"Python version: {env.version_info}")
    
    # Get environment-specific sys.path
    sys_path = env.get_sys_path()
    print(f"Sys path entries: {len(sys_path)}")
    for path in sys_path[:3]:  # Show first 3 entries
        print(f"  {path}")
    
    # Get grammar for this Python version
    grammar = env.get_grammar()
    print(f"Grammar version: {grammar.version}")
    print("---")
```

### Environment-Aware Analysis

Use specific environments for version-appropriate analysis and completions.

**Usage Example:**
```python
import jedi
from jedi.api.environment import get_system_environment

# Code that uses Python 3.8+ features
code = '''
# Python 3.8+ walrus operator
if (n := len([1, 2, 3])) > 2:
    print(f"Length is {n}")

# Type hints
from typing import List, Dict
def process_data(items: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in items}
'''

# Analyze with Python 3.8
py38_env = get_system_environment("3.8")
if py38_env:
    script = jedi.Script(code=code, environment=py38_env)
    errors = script.get_syntax_errors()
    print(f"Python 3.8 syntax errors: {len(errors)}")

# Analyze with Python 3.7 (should have syntax errors)
py37_env = get_system_environment("3.7")
if py37_env:
    script = jedi.Script(code=code, environment=py37_env)
    errors = script.get_syntax_errors()
    print(f"Python 3.7 syntax errors: {len(errors)}")
    for error in errors:
        print(f"  Line {error.line}: {error.get_message()}")
```

### Virtual Environment Integration

Integrate with virtual environments for isolated package analysis.

**Usage Example:**
```python
import jedi
from jedi.api.environment import find_virtualenvs

# Find virtual environment with specific packages
def find_env_with_package(package_name):
    for env in find_virtualenvs():
        # Test if package is available in environment
        test_code = f"import {package_name}"
        script = jedi.Script(code=test_code, environment=env)
        
        # If no import errors, package is likely available
        errors = script.get_syntax_errors()
        if not errors:
            # Try to get completions to verify package availability
            completions = script.complete(line=1, column=len(f"import {package_name}."))
            if completions:
                return env
    return None

# Find environment with Django
django_env = find_env_with_package("django")
if django_env:
    print(f"Found Django in: {django_env.path}")
    
    # Analyze Django-specific code
    django_code = '''
from django.http import HttpResponse
from django.shortcuts import render

def my_view(request):
    return HttpResponse("Hello")
'''
    
    script = jedi.Script(code=django_code, environment=django_env)
    completions = script.complete(line=4, column=20)  # At HttpResponse
    print(f"Django completions: {len(completions)}")
```

### Environment-Specific Completions

Get environment-appropriate completions and type information.

**Usage Example:**
```python
import jedi
from jedi.api.environment import get_default_environment, find_virtualenvs

code = '''
import sys
sys.'''

# Compare completions across environments
environments = [get_default_environment()]
environments.extend(list(find_virtualenvs())[:2])  # Add first 2 venvs

for i, env in enumerate(environments):
    if env:
        script = jedi.Script(code=code, environment=env)
        completions = script.complete(line=2, column=4)
        
        print(f"Environment {i+1} (Python {env.version_info}):")
        print(f"  Path: {env.path}")
        print(f"  Completions: {len(completions)}")
        
        # Show version-specific attributes
        version_attrs = [c for c in completions if 'version' in c.name.lower()]
        for attr in version_attrs[:3]:
            print(f"    {attr.name}: {attr.description}")
        print()
```

### Error Handling

Handle environment-related errors and validation.

```python { .api }
class InvalidPythonEnvironment(Exception):
    """Raised when Python environment is invalid or inaccessible."""
```

**Usage Example:**
```python
import jedi
from jedi.api.environment import create_environment, InvalidPythonEnvironment

# Attempt to create environments with error handling
test_paths = [
    "/usr/bin/python3",
    "/path/to/nonexistent/python",
    "/bin/ls",  # Not a Python executable
    "/path/to/broken/venv/bin/python"
]

for path in test_paths:
    try:
        env = create_environment(path)
        print(f"✓ Valid environment: {path}")
        print(f"  Python {env.version_info}")
    except InvalidPythonEnvironment as e:
        print(f"✗ Invalid environment: {path}")
        print(f"  Error: {e}")
    except FileNotFoundError:
        print(f"✗ Not found: {path}")
    print()

# Safe environment creation
def safe_create_environment(path):
    try:
        return create_environment(path, safe=True)
    except (InvalidPythonEnvironment, FileNotFoundError):
        return None

env = safe_create_environment("/usr/bin/python3")
if env:
    print(f"Successfully created environment: {env.executable}")
```

## Environment Types

### SameEnvironment

```python { .api }
class SameEnvironment(Environment):
    """Environment representing the current Python process."""
```

### InterpreterEnvironment

```python { .api }
class InterpreterEnvironment(SameEnvironment):
    """Environment optimized for interpreter usage."""
```

**Usage Example:**
```python
import jedi
from jedi.api.environment import InterpreterEnvironment

# Create interpreter-specific environment
env = InterpreterEnvironment()

# Use with Interpreter class
namespaces = [globals(), locals()]
interpreter = jedi.Interpreter('import os; os.', namespaces, environment=env)
completions = interpreter.complete()

print(f"Interpreter environment completions: {len(completions)}")
```