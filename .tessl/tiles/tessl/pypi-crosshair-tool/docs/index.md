# CrossHair

CrossHair is a Python analysis tool that uses symbolic execution and SMT solving to automatically find counterexamples for functions with type annotations and contracts. It blurs the line between testing and type systems by exploring viable execution paths through symbolic reasoning, making it a comprehensive solution for code correctness verification.

## Package Information

- **Package Name**: crosshair-tool
- **Language**: Python
- **Installation**: `pip install crosshair-tool`
- **License**: MIT
- **Console Commands**: `crosshair`, `mypycrosshair`

## Core Imports

```python
import crosshair
```

For symbolic execution and core functionality:

```python
from crosshair import (
    realize,
    deep_realize,
    register_type,
    register_patch,
    with_realized_args,
    SymbolicFactory,
    StateSpace,
    IgnoreAttempt
)
```

For tracing control:

```python
from crosshair import NoTracing, ResumedTracing
```

For debugging:

```python
from crosshair import debug
```

## Basic Usage

### Command Line Analysis

```bash
# Check a single function
crosshair check mymodule.py::my_function

# Check all functions in a module
crosshair check mymodule.py

# Watch for changes and auto-recheck
crosshair watch mymodule.py

# Generate unit tests
crosshair cover mymodule.py::my_function --coverage_type=pytest
```

### Programmatic Usage

```python
import crosshair
from crosshair import realize, SymbolicFactory, StateSpace

# Basic symbolic execution
def analyze_function(x: int) -> int:
    """
    pre: x > 0
    post: __return__ > x
    """
    return x + 1

# Using CrossHair's symbolic execution
from crosshair.core_and_libs import run_checkables
from crosshair.options import AnalysisOptions

options = AnalysisOptions()
# Analysis will find counterexamples if contracts can be violated
```

### Contract Verification

```python
from crosshair.register_contract import register_contract

def my_precondition(x):
    return x > 0

def my_postcondition(x, result):
    return result > x

# Register contract for function
register_contract(my_function, pre=my_precondition, post=my_postcondition)
```

## Architecture

CrossHair's symbolic execution engine consists of several key components:

- **StateSpace**: Manages SMT solver state and execution branches
- **SymbolicFactory**: Creates symbolic values for different types
- **Core Engine**: Handles symbolic execution, patching, and realization
- **Contract System**: Validates preconditions and postconditions
- **CLI Interface**: Provides command-line tools for analysis
- **LSP Server**: Enables IDE integration

The tool can analyze built-in types, user-defined classes, and much of the Python standard library through symbolic reasoning, making it comprehensive for code correctness verification.

## Capabilities

### Symbolic Execution Engine

Core symbolic execution functionality for creating and manipulating symbolic values, managing execution state, and converting between symbolic and concrete representations.

```python { .api }
def realize(value):
    """Convert symbolic values to concrete values."""

def deep_realize(value, memo=None):
    """Deeply convert symbolic values using copy mechanism."""

class SymbolicFactory:
    """Factory for creating symbolic values."""
    def __init__(self, space, pytype, varname): ...
    def __call__(self, typ, suffix="", allow_subtypes=True): ...

class StateSpace:
    """Holds SMT solver state and execution information."""
    def __init__(self, execution_deadline, model_check_timeout, search_root): ...
    def add(self, expr): ...
    def fork_parallel(self, false_probability, desc=""): ...
    def is_possible(self, expr): ...
```

[Symbolic Execution](./symbolic-execution.md)

### Command Line Interface

Comprehensive CLI tools for analyzing Python code, including checking contracts, watching for changes, generating tests, comparing function behaviors, and running an LSP server for IDE integration.

```python { .api }
def main(cmd_args=None):
    """Main CLI entry point."""

def check(args, options, stdout, stderr):
    """Check command implementation."""

def watch(args, options, max_watch_iterations=sys.maxsize):
    """Watch command implementation."""

def cover(args, options, stdout, stderr):
    """Cover command implementation."""

def diffbehavior(args, options, stdout, stderr):
    """Diff behavior command."""

def server(args, options, stdout, stderr):
    """LSP server implementation."""
```

[CLI Tools](./cli-tools.md)

### Contract Registration and Enforcement

System for registering and enforcing preconditions and postconditions on functions, enabling contract-based programming and verification.

```python { .api }
def register_contract(fn, *, pre=None, post=None, sig=None, skip_body=True):
    """Register contract for function."""

def clear_contract_registrations():
    """Clear all registered contracts."""

def get_contract(fn):
    """Get contract for function."""

class PreconditionFailed(BaseException):
    """Exception for precondition failures."""

class PostconditionFailed(BaseException):
    """Exception for postcondition failures."""

def WithEnforcement(fn):
    """Ensure conditions are enforced on callable."""
```

[Contract System](./contract-system.md)

### Behavioral Analysis and Comparison

Tools for analyzing and comparing the behavior of different function implementations, finding behavioral differences, and generating comprehensive behavior descriptions.

```python { .api }
def diff_behavior(*args):
    """Compare behaviors of functions."""

def describe_behavior(*args):
    """Describe function behavior."""

class BehaviorDiff:
    """Difference between function behaviors."""

class ExceptionEquivalenceType(enum.Enum):
    """Types of exception equivalence."""
    SAME_TYPE = ...
    SAME_TYPE_AND_MESSAGE = ...
    EXACT = ...
```

[Behavioral Analysis](./behavioral-analysis.md)

## Types

```python { .api }
class IgnoreAttempt(Exception):
    """Exception to ignore analysis attempts."""

class CrossHairInternal(Exception):
    """Internal CrossHair exception."""

class UnexploredPath(Exception):
    """Exception for unexplored execution paths."""

class NotDeterministic(Exception):
    """Exception for non-deterministic behavior."""

class PathTimeout(Exception):
    """Exception for path execution timeouts."""

class AnalysisKind(enum.Enum):
    """Types of analysis."""
    PEP316 = ...
    icontract = ...
    deal = ...
    hypothesis = ...
    asserts = ...

class AnalysisOptions:
    """Configuration options for analysis."""

class MessageType(enum.Enum):
    """Types of analysis messages."""
    CONFIRMED = ...
    CANNOT_CONFIRM = ...
    PRE_UNSAT = ...
    POST_FAIL = ...
    POST_ERR = ...
    EXEC_ERR = ...
    SYNTAX_ERR = ...
    IMPORT_ERR = ...
```