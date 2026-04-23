# Interpreter Integration

REPL and interactive environment support with runtime namespace integration. The Interpreter class extends Script functionality to work with actual runtime objects and namespaces, enabling completions and analysis in interactive Python sessions.

## Capabilities

### Interpreter Creation

Initialize interpreter analysis with runtime namespaces for REPL environments.

```python { .api }
class Interpreter(Script):
    def __init__(self, code, namespaces, *, project=None, **kwds):
        """
        Create an Interpreter for REPL analysis.
        
        Parameters:
        - code (str): Code to analyze.
        - namespaces (list): List of namespace dictionaries (globals(), locals(), etc.).
        - project (Project, optional): Project configuration.
        - **kwds: Additional keyword arguments passed to Script.
        """
```

**Usage Example:**
```python
import jedi

# Basic interpreter usage
namespace = {'x': 42, 'y': [1, 2, 3]}
interpreter = jedi.Interpreter('x.', [namespace])
completions = interpreter.complete()

# With multiple namespaces
global_ns = globals()
local_ns = {'custom_var': 'hello'}
interpreter = jedi.Interpreter('custom_var.', [global_ns, local_ns])

# In an IPython/Jupyter context
def get_completions(code, cursor_pos):
    namespaces = [get_ipython().user_ns, get_ipython().user_global_ns]
    interpreter = jedi.Interpreter(code, namespaces)
    return interpreter.complete(cursor_pos)
```

### Runtime Object Analysis

Analyze objects that exist in the runtime environment, providing accurate type information and completions based on actual object state.

**Usage Example:**
```python
import jedi
import pandas as pd

# Create runtime objects
df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
namespace = {'df': df, 'pd': pd}

# Get completions for actual DataFrame object
interpreter = jedi.Interpreter('df.', [namespace])
completions = interpreter.complete()

for completion in completions:
    print(f"{completion.name}: {completion.type}")

# Analyze method signatures with actual object
interpreter = jedi.Interpreter('df.groupby(', [namespace])
signatures = interpreter.get_signatures()
for sig in signatures:
    print(f"Method: {sig.name}")
    for param in sig.params:
        print(f"  {param.name}: {param.description}")
```

### Interactive Session Support

Support for interactive Python sessions, REPLs, and notebook environments with dynamic namespace tracking.

**Usage Example:**
```python
# Simulating an interactive session
import jedi

class InteractiveSession:
    def __init__(self):
        self.namespace = {}
    
    def execute(self, code):
        exec(code, self.namespace)
    
    def get_completions(self, code):
        interpreter = jedi.Interpreter(code, [self.namespace])
        return interpreter.complete()

# Usage
session = InteractiveSession()
session.execute("import json")
session.execute("data = {'key': 'value'}")

# Get completions with current session state
completions = session.get_completions("json.")
for c in completions:
    print(c.name)

completions = session.get_completions("data.")
for c in completions:
    print(c.name)
```

### REPL Integration Patterns

Common patterns for integrating jedi with different REPL environments.

**IPython Integration:**
```python
from IPython import get_ipython
import jedi

def jedi_completions(text, line, cursor_pos):
    """Custom IPython completer using jedi."""
    ip = get_ipython()
    namespaces = [ip.user_ns, ip.user_global_ns]
    
    interpreter = jedi.Interpreter(text, namespaces)
    completions = interpreter.complete()
    
    return [c.name for c in completions]

# Register with IPython
get_ipython().set_custom_completer(jedi_completions)
```

**Code.InteractiveConsole Integration:**
```python
import code
import jedi

class JediConsole(code.InteractiveConsole):
    def __init__(self, locals=None):
        super().__init__(locals)
        self.jedi_locals = locals or {}
    
    def get_completions(self, text):
        namespaces = [self.jedi_locals, __builtins__]
        interpreter = jedi.Interpreter(text, namespaces)
        return [c.name for c in interpreter.complete()]

# Usage
console = JediConsole({'myvar': 42})
console.interact()
```

### Namespace Management

Handle multiple namespaces and namespace precedence for accurate analysis.

```python { .api }
class Interpreter:
    namespaces: list  # List of namespace dictionaries
```

**Usage Example:**
```python
import jedi

# Multiple namespace levels
builtins_ns = __builtins__
global_ns = globals()
local_ns = {'local_var': 'local_value'}
custom_ns = {'custom_func': lambda x: x * 2}

# Namespace precedence: later namespaces override earlier ones
namespaces = [builtins_ns, global_ns, local_ns, custom_ns]

interpreter = jedi.Interpreter('local_var.', namespaces)
completions = interpreter.complete()

# Check what namespace a completion comes from
for completion in completions:
    definitions = completion.goto()
    for definition in definitions:
        print(f"{completion.name} defined in: {definition.module_path}")
```

### Dynamic Execution Analysis

Analyze code with dynamic execution capabilities for descriptor evaluation and property access.

**Configuration:**
```python
import jedi

# Control dynamic execution behavior
jedi.settings.allow_unsafe_interpreter_executions = True  # Default: True

# This allows jedi to evaluate descriptors and properties
class MyClass:
    @property
    def dynamic_prop(self):
        return "computed value"

obj = MyClass()
namespace = {'obj': obj}

interpreter = jedi.Interpreter('obj.dynamic_prop.', [namespace])
completions = interpreter.complete()  # Can access string methods
```

## Interpreter-Specific Considerations

### Safety and Security

The Interpreter can execute code through descriptor evaluation. Control this behavior through settings:

```python
import jedi

# Disable potentially unsafe executions
jedi.settings.allow_unsafe_interpreter_executions = False

# This affects property and descriptor evaluation
class UnsafeClass:
    @property
    def dangerous_prop(self):
        # This won't be evaluated if unsafe executions are disabled
        return os.system("rm -rf /")

obj = UnsafeClass()
interpreter = jedi.Interpreter('obj.dangerous_prop.', [{'obj': obj}])
```

### Performance Considerations

Interpreter analysis may be slower than Script analysis due to runtime object inspection:

```python
import jedi

# For better performance in REPL environments
jedi.settings.call_signatures_validity = 3.0  # Cache signatures for 3 seconds

# Preload commonly used modules
jedi.preload_module('json', 'os', 'sys', 'collections')

# Use environment-specific settings
interpreter = jedi.Interpreter(
    code, 
    namespaces,
    environment=jedi.api.environment.InterpreterEnvironment()
)
```