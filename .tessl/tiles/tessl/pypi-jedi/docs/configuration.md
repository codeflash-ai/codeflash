# Configuration

Global configuration settings and debugging utilities for controlling jedi behavior, performance tuning, and troubleshooting analysis issues. Provides comprehensive control over completion behavior, caching, dynamic analysis, and debugging output.

## Capabilities

### Debug Configuration

Configure debug output and logging for troubleshooting jedi analysis issues.

```python { .api }
def set_debug_function(func_cb=debug.print_to_stdout, warnings=True, 
                       notices=True, speed=True):
    """
    Configure debug output function and levels.
    
    Parameters:
    - func_cb (callable): Debug callback function. Default prints to stdout.
    - warnings (bool): Enable warning messages. Default True.
    - notices (bool): Enable notice messages. Default True.
    - speed (bool): Enable speed/performance messages. Default True.
    """
```

**Usage Example:**
```python
import jedi

# Basic debug setup (print to stdout)
jedi.set_debug_function()

# Custom debug function
def custom_debug(level, message):
    print(f"[JEDI-{level}] {message}")

jedi.set_debug_function(custom_debug, warnings=True, notices=False, speed=True)

# File-based logging
import logging
logging.basicConfig(filename='jedi_debug.log', level=logging.DEBUG)

def log_debug(level, message):
    logging.debug(f"{level}: {message}")

jedi.set_debug_function(log_debug)

# Analyze code with debug output
code = '''
import json
json.loads("test").'''

script = jedi.Script(code=code)
completions = script.complete(line=2, column=20)  # Debug info will be output
```

### Module Preloading

Preload modules for improved performance in repeated analysis operations.

```python { .api }
def preload_module(*modules):
    """
    Preload modules for better performance.
    
    Parameters:
    - modules: Module names to preload.
    """
```

**Usage Example:**
```python
import jedi

# Preload commonly used modules
jedi.preload_module('json', 'os', 'sys', 'collections', 'itertools')

# Preload third-party modules
jedi.preload_module('numpy', 'pandas', 'requests', 'django')

# Preload project-specific modules
jedi.preload_module('my_project.utils', 'my_project.models')

# Now analysis will be faster for these modules
code = '''
import json
import numpy as np
from my_project.utils import helper_function

json.'''

script = jedi.Script(code=code)
completions = script.complete(line=4, column=5)  # Faster due to preloading
```

### Completion Behavior Settings

Control completion output behavior and formatting.

```python { .api }
# jedi.settings module
case_insensitive_completion: bool = True  # Case insensitive completions
add_bracket_after_function: bool = False  # Add brackets after function completions
```

**Usage Example:**
```python
import jedi

# Configure case sensitivity
jedi.settings.case_insensitive_completion = False  # Exact case matching

code = '''
class MyClass:
    def MyMethod(self):
        pass

obj = MyClass()
obj.my'''  # Won't match MyMethod with case_insensitive_completion=False

script = jedi.Script(code=code)
completions = script.complete(line=6, column=6)
print(f"Completions: {[c.name for c in completions]}")

# Enable automatic bracket addition
jedi.settings.add_bracket_after_function = True

function_code = '''
def my_function():
    pass

my_func'''

script = jedi.Script(code=function_code)
completions = script.complete(line=4, column=7)
for comp in completions:
    if comp.name == 'my_function':
        print(f"Completion: {comp.name}, Complete: {comp.complete}")
        # Will show '(' when add_bracket_after_function=True
```

### Cache Configuration

Configure filesystem caching and performance settings.

```python { .api }
# jedi.settings module
cache_directory: str  # Cache storage directory path
call_signatures_validity: float = 3.0  # Function call cache duration in seconds
_cropped_file_size: int = 10000000  # Max file size for analysis (10MB)
```

**Usage Example:**
```python
import jedi
import os

# Check current cache directory
print(f"Current cache directory: {jedi.settings.cache_directory}")

# Set custom cache directory
custom_cache = "/tmp/my_jedi_cache"
os.makedirs(custom_cache, exist_ok=True)
jedi.settings.cache_directory = custom_cache

# Configure call signature caching
jedi.settings.call_signatures_validity = 10.0  # Cache for 10 seconds

# Test signature caching
code = '''
def my_function(param1, param2, param3):
    return param1 + param2 + param3

my_function('''

script = jedi.Script(code=code)

# First call - will be cached
signatures1 = script.get_signatures(line=4, column=12)
print("First signature call")

# Second call within cache validity - will use cache
signatures2 = script.get_signatures(line=4, column=12)
print("Second signature call (cached)")
```

### Parser Configuration

Control parser behavior and performance optimization.

```python { .api }
# jedi.settings module
fast_parser: bool = True  # Use Parso's diff parser for performance
```

**Usage Example:**
```python
import jedi

# Disable fast parser for debugging or thread safety
jedi.settings.fast_parser = False

# Warning: This makes jedi thread-safe but slower
# Useful when using multiple Script instances simultaneously

import threading

def analyze_code(code, results, index):
    script = jedi.Script(code=code)
    completions = script.complete()
    results[index] = len(completions)

# Multiple threads can safely use jedi with fast_parser=False
results = {}
threads = []

codes = [
    "import json; json.",
    "import os; os.",
    "import sys; sys."
]

for i, code in enumerate(codes):
    thread = threading.Thread(target=analyze_code, args=(code, results, i))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

print(f"Thread results: {results}")

# Re-enable for better performance in single-threaded usage
jedi.settings.fast_parser = True
```

### Dynamic Analysis Configuration

Control dynamic analysis features and behavior.

```python { .api }
# jedi.settings module
dynamic_array_additions: bool = True      # Analyze array.append() etc.
dynamic_params: bool = True               # Dynamic parameter completion
dynamic_params_for_other_modules: bool = True  # Dynamic params for other modules
dynamic_flow_information: bool = True     # Use isinstance() for type inference
auto_import_modules: list = ['gi']        # Modules to import rather than analyze
allow_unsafe_interpreter_executions: bool = True  # Allow descriptor evaluation
```

**Usage Example:**
```python
import jedi

# Configure dynamic analysis
jedi.settings.dynamic_array_additions = True
jedi.settings.dynamic_params = True
jedi.settings.dynamic_flow_information = True

# Test dynamic array analysis
array_code = '''
my_list = []
my_list.append("item1")
my_list.append("item2")
my_list.'''  # Should know it contains strings

script = jedi.Script(code=array_code)
completions = script.complete(line=4, column=8)
print("Array completions:")
for comp in completions[:5]:
    print(f"  {comp.name}: {comp.description}")

# Test isinstance flow analysis
isinstance_code = '''
def process_value(value):
    if isinstance(value, str):
        return value.  # Should show string methods
    elif isinstance(value, list):
        return value.  # Should show list methods
    return value
'''

script = jedi.Script(code=isinstance_code)

# String context completions
str_completions = script.complete(line=3, column=21)
print("\nString context completions:")
for comp in str_completions[:3]:
    print(f"  {comp.name}")

# List context completions
list_completions = script.complete(line=5, column=21)
print("\nList context completions:")
for comp in list_completions[:3]:
    print(f"  {comp.name}")

# Configure auto-import modules (modules that should be imported, not analyzed)
jedi.settings.auto_import_modules = ['gi', 'tensorflow', 'torch']

# Disable unsafe executions for security
jedi.settings.allow_unsafe_interpreter_executions = False
```

### Performance Tuning

Optimize jedi performance for different use cases.

**Usage Example:**
```python
import jedi

# High-performance configuration for IDE usage
def configure_for_ide():
    jedi.settings.fast_parser = True
    jedi.settings.call_signatures_validity = 5.0
    jedi.settings.dynamic_params = True
    jedi.settings.dynamic_array_additions = True
    
    # Preload common modules
    jedi.preload_module(
        'os', 'sys', 'json', 'collections', 'itertools',
        'functools', 'operator', 'typing'
    )

# Conservative configuration for reliability
def configure_for_reliability():
    jedi.settings.fast_parser = False  # Thread-safe
    jedi.settings.dynamic_params = False  # More predictable
    jedi.settings.allow_unsafe_interpreter_executions = False  # Safer

# Minimal configuration for basic completions
def configure_minimal():
    jedi.settings.dynamic_array_additions = False
    jedi.settings.dynamic_params = False
    jedi.settings.dynamic_params_for_other_modules = False
    jedi.settings.dynamic_flow_information = False

# Apply configuration based on use case
configure_for_ide()

# Test performance
import time

code = '''
import collections
from typing import List, Dict

def process_data(items: List[str]) -> Dict[str, int]:
    counter = collections.Counter(items)
    return dict(counter)

counter.'''

script = jedi.Script(code=code)

start_time = time.time()
completions = script.complete(line=8, column=8)
end_time = time.time()

print(f"Completion time: {end_time - start_time:.4f} seconds")
print(f"Number of completions: {len(completions)}")
```

### Environment-Specific Configuration

Configure jedi for different environments and contexts.

**Usage Example:**
```python
import jedi
import platform

# Configure based on operating system
if platform.system() == "Windows":
    # Windows-specific cache directory
    import os
    jedi.settings.cache_directory = os.path.join(
        os.getenv('LOCALAPPDATA', os.path.expanduser('~')),
        'Jedi', 'Cache'
    )
elif platform.system() == "Darwin":
    # macOS cache directory
    jedi.settings.cache_directory = os.path.expanduser('~/Library/Caches/Jedi')
else:
    # Linux cache directory
    import os
    cache_home = os.getenv('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
    jedi.settings.cache_directory = os.path.join(cache_home, 'jedi')

# Configure for different Python versions
import sys

if sys.version_info >= (3, 9):
    # Enable advanced features for newer Python
    jedi.settings.dynamic_flow_information = True
    jedi.settings.dynamic_params_for_other_modules = True
else:
    # Conservative settings for older Python
    jedi.settings.dynamic_flow_information = False
    jedi.settings.dynamic_params_for_other_modules = False

# Configure for different usage patterns
def configure_for_repl():
    """Configuration optimized for REPL usage."""
    jedi.settings.call_signatures_validity = 1.0  # Shorter cache
    jedi.settings.allow_unsafe_interpreter_executions = True
    jedi.settings.dynamic_array_additions = True

def configure_for_static_analysis():
    """Configuration optimized for static analysis."""
    jedi.settings.allow_unsafe_interpreter_executions = False
    jedi.settings.fast_parser = False  # More thorough analysis
    jedi.settings.dynamic_params = False  # Deterministic results

# Apply REPL configuration
configure_for_repl()
```

## Configuration Settings Reference

### Completion Settings
- `case_insensitive_completion`: Boolean, default `True`
- `add_bracket_after_function`: Boolean, default `False`

### Cache Settings  
- `cache_directory`: String, platform-specific default
- `call_signatures_validity`: Float, default `3.0` seconds

### Parser Settings
- `fast_parser`: Boolean, default `True`

### Dynamic Analysis Settings
- `dynamic_array_additions`: Boolean, default `True`  
- `dynamic_params`: Boolean, default `True`
- `dynamic_params_for_other_modules`: Boolean, default `True`
- `dynamic_flow_information`: Boolean, default `True`
- `auto_import_modules`: List, default `['gi']`

### Interpreter Settings
- `allow_unsafe_interpreter_executions`: Boolean, default `True`