# Symbolic Execution Engine

CrossHair's symbolic execution engine provides the core functionality for creating and manipulating symbolic values, managing execution state, and performing symbolic reasoning. This system enables the tool to explore execution paths and find counterexamples automatically.

## Capabilities

### Value Realization

Convert symbolic values to concrete representations for inspection and debugging.

```python { .api }
def realize(value):
    """
    Convert symbolic values to concrete values.
    
    Parameters:
    - value: Any symbolic or concrete value
    
    Returns:
    Concrete representation of the value
    """

def deep_realize(value, memo=None):
    """
    Deeply convert symbolic values using copy mechanism.
    
    Parameters:
    - value: Value to realize
    - memo: Optional memoization dictionary to handle circular references
    
    Returns:
    Deeply realized value with all nested symbolic values converted
    """
```

**Usage Example:**

```python
from crosshair import realize, deep_realize

# Realize a simple symbolic value
concrete_val = realize(symbolic_int)

# Deep realization for complex nested structures
complex_data = {"items": [symbolic_list], "metadata": symbolic_dict}
concrete_data = deep_realize(complex_data)
```

### Symbolic Value Creation

Create symbolic values for different Python types to enable symbolic execution.

```python { .api }
class SymbolicFactory:
    """
    Factory for creating symbolic values of various types.
    
    The SymbolicFactory is the primary interface for creating symbolic values
    during symbolic execution. It maintains the connection to the StateSpace
    and handles type-specific symbolic value creation.
    """
    
    def __init__(self, space, pytype, varname):
        """
        Initialize symbolic factory.
        
        Parameters:
        - space: StateSpace instance managing solver state
        - pytype: Python type for the factory
        - varname: Base variable name for created symbols
        """
    
    def __call__(self, typ, suffix="", allow_subtypes=True):
        """
        Create symbolic value of given type.
        
        Parameters:
        - typ: Type to create symbolic value for
        - suffix: Optional suffix for variable name uniqueness
        - allow_subtypes: Whether to allow subtypes of the specified type
        
        Returns:
        Symbolic value of the specified type
        """
    
    def get_suffixed_varname(self, suffix):
        """
        Get unique variable name with suffix.
        
        Parameters:
        - suffix: Suffix to append
        
        Returns:
        Unique variable name string
        """
```

**Usage Example:**

```python
from crosshair import SymbolicFactory, StateSpace

# Create a symbolic factory
space = StateSpace(deadline, timeout, root)  
factory = SymbolicFactory(space, int, "x")

# Create symbolic integers
sym_int1 = factory(int)
sym_int2 = factory(int, suffix="alt")

# Create symbolic values for custom types
sym_custom = factory(MyCustomClass, allow_subtypes=False)
```

### Execution State Management

Manage SMT solver state and execution branching during symbolic execution.

```python { .api }
class StateSpace:
    """
    Holds SMT solver state and execution information.
    
    StateSpace manages the Z3 SMT solver instance, tracks execution branches,
    handles constraint solving, and provides utilities for symbolic execution.
    """
    
    def __init__(self, execution_deadline, model_check_timeout, search_root):
        """
        Initialize state space.
        
        Parameters:
        - execution_deadline: Maximum execution time
        - model_check_timeout: Timeout for SMT solver queries
        - search_root: Root node for execution tree
        """
    
    def add(self, expr):
        """
        Add constraint to solver.
        
        Parameters:
        - expr: Z3 expression to add as constraint
        """
    
    def fork_parallel(self, false_probability, desc=""):
        """
        Create execution fork with specified probability.
        
        Parameters:
        - false_probability: Probability of taking false branch (0.0-1.0)
        - desc: Optional description for debugging
        
        Returns:
        Boolean indicating which branch was taken
        """
    
    def is_possible(self, expr):
        """
        Check if expression could be true given current constraints.
        
        Parameters:
        - expr: Z3 expression to check
        
        Returns:
        True if expression is satisfiable, False otherwise
        """
    
    def choose_possible(self, expr_choices):
        """
        Choose from possible expressions based on satisfiability.
        
        Parameters:
        - expr_choices: List of Z3 expressions to choose from
        
        Returns:
        First satisfiable expression from the list
        """
    
    def find_model_value(self, expr):
        """
        Find model value for SMT expression.
        
        Parameters:
        - expr: Z3 expression to find value for
        
        Returns:
        Concrete value satisfying the expression in current model
        """
    
    def smt_fork(self, expr):
        """
        Fork execution based on SMT expression.
        
        Parameters:
        - expr: Z3 boolean expression to fork on
        
        Returns:
        Boolean indicating which branch (true/false) was taken
        """
    
    def defer_assumption(self, description, checker):
        """
        Defer assumption checking until later in execution.
        
        Parameters:
        - description: Human-readable description of assumption
        - checker: Callable that returns boolean for assumption validity
        """
    
    def rand(self):
        """
        Get random number generator.
        
        Returns:
        Random.Random instance for this execution path
        """
    
    def extra(self, typ):
        """
        Get extra data of specified type.
        
        Parameters:
        - typ: Type of extra data to retrieve
        
        Returns:
        Extra data instance of specified type
        """
    
    def uniq(self):
        """
        Get unique identifier.
        
        Returns:
        Unique integer identifier for this state space
        """
```

**Usage Example:**

```python
from crosshair import StateSpace
import z3

# Create state space
space = StateSpace(deadline=30.0, timeout=5.0, root=None)

# Add constraints
x = z3.Int('x')
space.add(x > 0)
space.add(x < 100)

# Check possibilities
if space.is_possible(x == 50):
    print("x could be 50")

# Fork execution
if space.smt_fork(x > 50):
    # Handle x > 50 case
    print("Exploring x > 50")
else:
    # Handle x <= 50 case  
    print("Exploring x <= 50")
```

### Function Patching and Registration

Register patches and custom symbolic types for extending symbolic execution capabilities.

```python { .api }
def register_patch(entity, patch_value):
    """
    Register a patch for a callable.
    
    Parameters:
    - entity: Callable to patch
    - patch_value: Replacement callable to use during symbolic execution
    """

def register_type(typ, creator):
    """
    Register custom symbolic value creator for a type.
    
    Parameters:
    - typ: Type to register creator for  
    - creator: Callback function that creates symbolic values of this type
    """

def with_realized_args(fn, deep=False):
    """
    Decorator that realizes function arguments before calling.
    
    Parameters:
    - fn: Function to wrap
    - deep: Whether to perform deep realization of arguments
    
    Returns:
    Wrapped function that realizes arguments before execution
    """

class patch_to_return:
    """
    Context manager for patching functions to return specific values.
    
    Allows temporary patching of functions during symbolic execution
    to control their return values for testing purposes.
    """
    
    def __init__(self, return_values):
        """
        Initialize patch context.
        
        Parameters:
        - return_values: Dict mapping callables to lists of return values
        """
    
    def __enter__(self):
        """Enter patch context."""
        
    def __exit__(self, *args):
        """Exit patch context and restore original functions."""
```

**Usage Example:**

```python
from crosshair import register_patch, register_type, with_realized_args, patch_to_return

# Register a patch for a function
def mock_database_call():
    return "mocked_result"

register_patch(real_database_call, mock_database_call)

# Register custom type creator
def create_symbolic_custom_type(space, typ, varname):
    # Return symbolic instance of custom type
    return SymbolicCustomType(varname)

register_type(MyCustomType, create_symbolic_custom_type)

# Use argument realization decorator
@with_realized_args
def debug_function(x, y):
    print(f"Called with concrete values: {x}, {y}")
    return x + y

# Temporary function patching
with patch_to_return({expensive_function: [42, 100, -1]}):
    # expensive_function will return 42, then 100, then -1 on successive calls
    result1 = expensive_function()  # Returns 42
    result2 = expensive_function()  # Returns 100
```

### Tracing Control

Control symbolic execution tracing for performance and debugging.

```python { .api }
def NoTracing():
    """
    Context manager to disable tracing.
    
    Returns:
    Context manager that disables CrossHair's execution tracing
    """

def ResumedTracing():
    """
    Context manager to resume tracing.
    
    Returns:
    Context manager that re-enables CrossHair's execution tracing
    """

def is_tracing():
    """
    Check if currently tracing.
    
    Returns:
    True if tracing is currently enabled, False otherwise
    """
```

**Usage Example:**

```python
from crosshair import NoTracing, ResumedTracing, is_tracing

# Disable tracing for performance-critical sections
with NoTracing():
    # This code runs without CrossHair tracing overhead
    result = expensive_computation()

# Resume tracing when needed
with ResumedTracing():
    # This code runs with full tracing enabled
    analyzed_result = analyze_result(result)

# Check tracing status
if is_tracing():
    print("Tracing is active")
```

### Utility Functions

Additional utilities for type handling and debugging during symbolic execution.

```python { .api }
def python_type(obj):
    """
    Get the Python type of an object, handling symbolic objects.
    
    Parameters:
    - obj: Object to get type of
    
    Returns:
    Python type, properly handling symbolic value types
    
    For symbolic objects with __ch_pytype__ method, returns the actual
    Python type being symbolically represented rather than the symbolic
    wrapper type. Raises CrossHairInternal if called while tracing.
    """

def normalize_pytype(typ):
    """
    Normalize type annotations for symbolic execution.
    
    Parameters:
    - typ: Type to normalize
    
    Returns:
    Normalized type suitable for symbolic execution
    
    Handles TypeVar bounds (converting to bound type or object), 
    converts Any to object, and handles other type system complexities 
    to produce types suitable for symbolic value creation.
    """

def debug(*args):
    """
    Print debugging information in CrossHair's nested log output.
    
    Parameters:
    - *args: Arguments to print for debugging
    """
```

**Usage Example:**

```python
from crosshair import python_type, normalize_pytype, debug

# Get type of symbolic values
sym_val = create_symbolic_int()
actual_type = python_type(sym_val)  # Returns int, not symbolic wrapper type

# Normalize complex type annotations  
from typing import List, Optional
normalized = normalize_pytype(Optional[List[int]])

# Debug symbolic execution
debug("Current symbolic value:", sym_val, "Type:", actual_type)
```