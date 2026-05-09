# Code Navigation

Advanced code navigation including reference finding, symbol search, context analysis, and help lookup. Provides IDE-level navigation capabilities across entire projects with sophisticated scope analysis and cross-reference tracking.

## Capabilities

### Reference Finding

Find all references to a symbol across the project or within a file, with options for controlling scope and builtin inclusion.

```python { .api }
def get_references(self, line=None, column=None, *, include_builtins=True, scope='project'):
    """
    Find all references to the symbol at cursor position.
    
    Parameters:
    - line (int, optional): Line number (1-based).
    - column (int, optional): Column number (0-based).
    - include_builtins (bool): Include builtin references. Default True.
    - scope (str): Search scope - 'project' or 'file'. Default 'project'.
    
    Returns:
    List of Name objects representing references.
    """
```

**Usage Example:**
```python
import jedi

code = '''
class Calculator:
    def add(self, a, b):
        return a + b

calc = Calculator()
result = calc.add(1, 2)
print(calc.add)
'''

script = jedi.Script(code=code, path='example.py')

# Find all references to 'add' method
references = script.get_references(line=2, column=8)  # At 'add' definition

for ref in references:
    print(f"Reference at line {ref.line}, column {ref.column}")
    print(f"Type: {ref.type}")
    print(f"Context: {ref.get_line_code()}")
    print(f"Is definition: {ref.is_definition()}")
    print("---")

# Find references in current file only
file_refs = script.get_references(line=2, column=8, scope='file')

# Exclude builtin references
no_builtins = script.get_references(line=2, column=8, include_builtins=False)
```

### Symbol Search

Search for symbols by name within the current file, supporting both exact and fuzzy matching.

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
```

**Usage Example:**
```python
import jedi

code = '''
class DataProcessor:
    def process_data(self, data):
        processed_result = self.transform_data(data)
        return processed_result
    
    def transform_data(self, data):
        return data.upper()

def process_file(filename):
    processor = DataProcessor()
    return processor.process_data(filename)
'''

script = jedi.Script(code=code, path='example.py')

# Search for names containing 'process'
results = list(script.search('process'))
for result in results:
    print(f"Found: {result.name} at line {result.line}")
    print(f"Type: {result.type}")
    print(f"Full name: {result.full_name}")

# Search in all scopes (including function/class internals)
all_results = list(script.search('data', all_scopes=True))
for result in all_results:
    print(f"Found: {result.name} in {result.parent().name if result.parent() else 'global'}")
```

### Completion Search

Search with completion-style fuzzy matching for symbol discovery.

```python { .api }
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
import jedi

code = '''
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

def calculate_median(numbers):
    sorted_nums = sorted(numbers)
    return sorted_nums[len(sorted_nums) // 2]

def calc_standard_deviation(numbers):
    avg = calculate_average(numbers)
    return (sum((x - avg) ** 2 for x in numbers) / len(numbers)) ** 0.5
'''

script = jedi.Script(code=code, path='example.py')

# Fuzzy search for calculation functions
completions = list(script.complete_search('calc', fuzzy=True))
for comp in completions:
    print(f"Match: {comp.name}")
    print(f"Complete: {comp.complete}")
    print(f"Type: {comp.type}")

# Search for all functions
functions = list(script.complete_search('', all_scopes=True))
functions = [f for f in functions if f.type == 'function']
```

### Context Analysis

Get the current scope context at a specific position, determining the enclosing function, class, or module.

```python { .api }
def get_context(self, line=None, column=None):
    """
    Get the scope context at cursor position.
    
    Parameters:
    - line (int, optional): Line number (1-based).
    - column (int, optional): Column number (0-based).
    
    Returns:
    Name object representing the current scope context.
    """
```

**Usage Example:**
```python
import jedi

code = '''
class WebServer:
    def __init__(self, port):
        self.port = port
    
    def handle_request(self, request):
        if request.method == 'GET':
            return self.handle_get(request)
        elif request.method == 'POST':
            # cursor position here
            return self.handle_post(request)
    
    def handle_get(self, request):
        return "GET response"
'''

script = jedi.Script(code=code, path='example.py')

# Get context at different positions
contexts = [
    script.get_context(line=3, column=20),  # Inside __init__
    script.get_context(line=8, column=12),  # Inside handle_request
    script.get_context(line=12, column=8),  # Inside handle_get
    script.get_context(line=1, column=0),   # Module level
]

for i, context in enumerate(contexts):
    print(f"Context {i+1}: {context.name} (type: {context.type})")
    # Get parent context
    parent = context.parent() if context.type != 'module' else None
    if parent:
        print(f"  Parent: {parent.name} (type: {parent.type})")
```

### Help and Documentation

Get help information for symbols, including keyword and operator help.

```python { .api }
def help(self, line=None, column=None):
    """
    Get help information for the symbol at cursor position.
    
    Parameters:
    - line (int, optional): Line number (1-based).
    - column (int, optional): Column number (0-based).
    
    Returns:
    List of Name objects with help information.
    """
```

**Usage Example:**
```python
import jedi

code = '''
import json
json.loads("test")
for item in range(10):
    if item > 5:
        break
'''

script = jedi.Script(code=code, path='example.py')

# Get help for function
help_info = script.help(line=2, column=5)  # At 'loads'
for info in help_info:
    print(f"Help for: {info.name}")
    print(f"Type: {info.type}")
    print(f"Docstring: {info.docstring()[:100]}...")

# Get help for keywords
keyword_help = script.help(line=3, column=0)  # At 'for'
for info in keyword_help:
    if info.is_keyword:
        print(f"Keyword: {info.name}")
        print(f"Help: {info.docstring()}")

# Get help for operators
break_help = script.help(line=5, column=8)  # At 'break'
```

### Cross-Reference Navigation

Navigate between definitions and references with detailed relationship information.

**Usage Example:**
```python
import jedi

code = '''
class DatabaseConnection:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.connection = None
    
    def connect(self):
        self.connection = f"Connected to {self.host}:{self.port}"
        return self.connection
    
    def disconnect(self):
        if self.connection:
            self.connection = None

# Usage
db = DatabaseConnection("localhost", 5432)
conn = db.connect()
print(conn)
db.disconnect()
'''

script = jedi.Script(code=code, path='example.py')

# Navigate from usage to definition
usage_pos = (17, 5)  # At 'db' in db.connect()
definitions = script.goto(*usage_pos)
for definition in definitions:
    print(f"Definition: {definition.name} at line {definition.line}")

# Find all references from definition
definition_pos = (16, 0)  # At 'db = DatabaseConnection...'
references = script.get_references(*definition_pos)
for ref in references:
    print(f"Reference: line {ref.line}, column {ref.column}")
    print(f"  Code: {ref.get_line_code().strip()}")
    print(f"  Is definition: {ref.is_definition()}")

# Get related definitions (methods of the class)
class_methods = script.goto(line=1, column=6)  # At 'DatabaseConnection'
if class_methods:
    class_def = class_methods[0]
    methods = class_def.defined_names()
    print("Class methods:")
    for method in methods:
        if method.type == 'function':
            print(f"  {method.name} at line {method.line}")
```

### Advanced Navigation Patterns

Complex navigation scenarios for IDE-level functionality.

**Multi-file Navigation:**
```python
import jedi
from jedi import Project

# Set up project for cross-file navigation
project = Project("/path/to/project")

# File 1: models.py
models_code = '''
class User:
    def __init__(self, name, email):
        self.name = name
        self.email = email
'''

# File 2: main.py  
main_code = '''
from models import User

user = User("John", "john@example.com")
print(user.name)
'''

# Analyze with project context
script = jedi.Script(code=main_code, path="main.py", project=project)

# Navigate from usage to definition across files
definitions = script.goto(line=3, column=7, follow_imports=True)  # At 'User'
for definition in definitions:
    print(f"Found definition in: {definition.module_path}")
    print(f"At line: {definition.line}")
```

**Inheritance Navigation:**
```python
code = '''
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

dog = Dog()
sound = dog.speak()  # Navigate to Dog.speak, not Animal.speak
'''

script = jedi.Script(code=code, path='example.py')

# Navigate to the actual implementation, not the base class
definitions = script.goto(line=13, column=12)  # At 'speak'
for definition in definitions:
    print(f"Actual implementation: {definition.parent().name}.{definition.name}")
    print(f"At line: {definition.line}")
```