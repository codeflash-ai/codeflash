# Script Analysis

Core functionality for analyzing Python source code, providing completions, type inference, definition lookup, and navigation. The Script class is the main entry point for static code analysis with full project context support.

## Capabilities

### Script Creation

Initialize script analysis for source code with optional file path, environment, and project configuration.

```python { .api }
class Script:
    def __init__(self, code=None, *, path=None, environment=None, project=None):
        """
        Create a Script for code analysis.
        
        Parameters:
        - code (str, optional): Source code string. If None, reads from path.
        - path (str or Path, optional): File path for the code.
        - environment (Environment, optional): Python environment to use.
        - project (Project, optional): Project configuration.
        """
```

**Usage Example:**
```python
import jedi

# Analyze code string
script = jedi.Script(code="import json\njson.loads", path="example.py")

# Analyze existing file
script = jedi.Script(path="/path/to/file.py")

# With custom environment and project
from jedi import Project
from jedi.api.environment import get_default_environment

project = Project("/path/to/project")
env = get_default_environment()
script = jedi.Script(code="import requests", project=project, environment=env)
```

### Code Completion

Get intelligent completions at a specific position in the code, supporting both normal and fuzzy matching.

```python { .api }
def complete(self, line=None, column=None, *, fuzzy=False):
    """
    Get completions at cursor position.
    
    Parameters:
    - line (int, optional): Line number (1-based). Defaults to end of file.
    - column (int, optional): Column number (0-based). Defaults to end of line.
    - fuzzy (bool): Enable fuzzy matching. Default False.
    
    Returns:
    List of Completion objects sorted by name.
    """
```

**Usage Example:**
```python
code = '''
import json
json.lo'''

script = jedi.Script(code=code, path='example.py')
completions = script.complete(line=3, column=7)  # At 'json.lo'

for completion in completions:
    print(f"Name: {completion.name}")
    print(f"Complete: {completion.complete}")
    print(f"Type: {completion.type}")
    print(f"Description: {completion.description}")
    print("---")

# Fuzzy completion
completions = script.complete(line=3, column=7, fuzzy=True)
```

### Type Inference

Infer the type of expressions at a specific position, following complex paths through imports and statements.

```python { .api }
def infer(self, line=None, column=None, *, only_stubs=False, prefer_stubs=False):
    """
    Infer types at cursor position.
    
    Parameters:
    - line (int, optional): Line number (1-based).
    - column (int, optional): Column number (0-based).
    - only_stubs (bool): Only return stub definitions. Default False.
    - prefer_stubs (bool): Prefer stubs over Python objects. Default False.
    
    Returns:
    List of Name objects representing inferred types.
    """
```

**Usage Example:**
```python
code = '''
import json
data = json.loads('{"key": "value"}')
data.'''

script = jedi.Script(code=code, path='example.py')
definitions = script.infer(line=3, column=5)  # At 'data.'

for definition in definitions:
    print(f"Type: {definition.type}")
    print(f"Full name: {definition.full_name}")
    print(f"Module: {definition.module_name}")
    print(f"Description: {definition.description}")
```

### Go to Definition

Navigate to the definition of symbols, with options for following imports and handling stubs.

```python { .api }
def goto(self, line=None, column=None, *, follow_imports=False, 
         follow_builtin_imports=False, only_stubs=False, prefer_stubs=False):
    """
    Go to definition of symbol at cursor.
    
    Parameters:
    - line (int, optional): Line number (1-based).
    - column (int, optional): Column number (0-based).
    - follow_imports (bool): Follow import statements. Default False.
    - follow_builtin_imports (bool): Follow builtin imports if follow_imports=True.
    - only_stubs (bool): Only return stub definitions.
    - prefer_stubs (bool): Prefer stubs over Python objects.
    
    Returns:
    List of Name objects representing definitions.
    """
```

**Usage Example:**
```python
code = '''
from collections import defaultdict
my_dict = defaultdict(list)
my_dict.append'''

script = jedi.Script(code=code, path='example.py')

# Basic goto
definitions = script.goto(line=3, column=8)  # At 'my_dict'

# Follow imports to get actual implementation
definitions = script.goto(line=2, column=20, follow_imports=True)  # At 'defaultdict'

for definition in definitions:
    print(f"Name: {definition.name}")
    print(f"File: {definition.module_path}")
    print(f"Line: {definition.line}")
    print(f"Type: {definition.type}")
```

### Function Signatures

Get function signatures showing parameters and their types for function calls.

```python { .api }
def get_signatures(self, line=None, column=None):
    """
    Get function signatures at cursor position.
    
    Parameters:
    - line (int, optional): Line number (1-based).
    - column (int, optional): Column number (0-based).
    
    Returns:
    List of Signature objects.
    """
```

**Usage Example:**
```python
code = '''
import json
json.loads('''

script = jedi.Script(code=code, path='example.py')
signatures = script.get_signatures(line=2, column=11)  # Inside 'json.loads('

for signature in signatures:
    print(f"Function: {signature.name}")
    print(f"Current param index: {signature.index}")
    print("Parameters:")
    for param in signature.params:
        print(f"  {param.name}: {param.description}")
        if param.infer_default():
            defaults = param.infer_default()
            print(f"    Default: {defaults[0].name if defaults else 'None'}")
```

### Syntax Error Detection

Detect and report syntax errors in the analyzed code.

```python { .api }
def get_syntax_errors(self):
    """
    Get syntax errors in the current code.
    
    Returns:
    List of SyntaxError objects.
    """
```

**Usage Example:**
```python
code = '''
def broken_function(
    print("missing closing parenthesis")
'''

script = jedi.Script(code=code, path='example.py')
errors = script.get_syntax_errors()

for error in errors:
    print(f"Error at line {error.line}, column {error.column}")
    print(f"Message: {error.get_message()}")
    if error.until_line:
        print(f"Error spans to line {error.until_line}, column {error.until_column}")
```

### Name Extraction

Get all names defined in the current file with filtering options.

```python { .api }
def get_names(self, *, all_scopes=False, definitions=True, references=False):
    """
    Get names defined in the current file.
    
    Parameters:
    - all_scopes (bool): Include names from all scopes, not just module level. Default False.
    - definitions (bool): Include definition names (class, function, variable assignments). Default True.
    - references (bool): Include reference names (variable usage). Default False.
    
    Returns:
    List of Name objects.
    """
```

**Usage Example:**
```python
code = '''
class MyClass:
    def method(self, param):
        local_var = param
        return local_var

def function():
    instance = MyClass()
    return instance.method("test")
'''

script = jedi.Script(code=code, path='example.py')

# Get all definitions
definitions = script.get_names(definitions=True, references=False)
for name in definitions:
    print(f"{name.type}: {name.name} at line {name.line}")

# Get all names including references
all_names = script.get_names(all_scopes=True, definitions=True, references=True)
```

### File-Level Search

Search for symbols within the current file being analyzed.

```python { .api }
def search(self, string, *, all_scopes=False):
    """
    Search for names in the current file.
    
    Parameters:
    - string (str): Search string pattern.
    - all_scopes (bool): Search in all scopes, not just module level. Default False.
    
    Returns:
    Generator of Name objects matching the search.
    """

def complete_search(self, string, *, all_scopes=False, fuzzy=False):
    """
    Search with completion-style matching.
    
    Parameters:
    - string (str): Search string pattern.
    - all_scopes (bool): Search in all scopes. Default False.
    - fuzzy (bool): Enable fuzzy matching. Default False.
    
    Returns:
    Generator of Completion objects.
    """
```

**Usage Example:**
```python
code = '''
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total

def calculate_average(numbers):
    return calculate_sum(numbers) / len(numbers)
'''

script = jedi.Script(code=code, path='example.py')

# Search for names containing 'calculate'
results = list(script.search('calculate'))
for result in results:
    print(f"Found: {result.name} at line {result.line}")

# Fuzzy search
fuzzy_results = list(script.complete_search('calc', fuzzy=True))
for result in fuzzy_results:
    print(f"Fuzzy match: {result.name}")
```

## Types

### Completion

```python { .api }
class Completion(BaseName):
    complete: str  # Rest of word to complete (None for fuzzy)
    name_with_symbols: str  # Name including symbols like 'param='
    
    def get_completion_prefix_length(self):
        """Get length of prefix being completed."""
```

### SyntaxError

```python { .api }
class SyntaxError:
    line: int  # Error start line (1-based)
    column: int  # Error start column (0-based)
    until_line: int  # Error end line
    until_column: int  # Error end column
    
    def get_message(self):
        """Get error message string."""
```