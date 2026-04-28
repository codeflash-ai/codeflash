# Contract System

CrossHair's contract system enables contract-based programming and verification through preconditions and postconditions. It supports multiple contract formats and provides enforcement mechanisms for runtime validation.

## Capabilities

### Contract Registration

Register contracts for functions to enable verification and enforcement.

```python { .api }
def register_contract(fn, *, pre=None, post=None, sig=None, skip_body=True):
    """
    Register contract for function.
    
    Parameters:
    - fn: Function to register contract for
    - pre: Optional precondition function that takes same args as fn
    - post: Optional postcondition function that takes fn args plus result
    - sig: Optional function signature or list of signatures
    - skip_body: Whether to skip function body during analysis (default True)
    
    Returns:
    True if registration successful, False otherwise
    
    Registers preconditions and postconditions for a function, enabling
    CrossHair to verify the contracts during symbolic execution.
    """

def clear_contract_registrations():
    """
    Clear all registered contracts.
    
    Removes all previously registered contracts, useful for testing
    or when dynamically changing contract configurations.
    """

def get_contract(fn):
    """
    Get contract for function.
    
    Parameters:
    - fn: Function to get contract for
    
    Returns:
    ContractOverride instance if contract exists, None otherwise
    
    Retrieves the registered contract information for a function,
    including preconditions, postconditions, and signature overrides.
    """

def register_modules(*modules):
    """
    Register contracts from modules.
    
    Parameters:
    - *modules: Module objects to scan for contract annotations
    
    Automatically discovers and registers contracts from modules that
    use supported contract annotation formats (PEP316, icontract, etc.).
    """
```

**Usage Examples:**

```python
from crosshair.register_contract import register_contract, clear_contract_registrations

def sqrt_precondition(x):
    """Precondition: input must be non-negative"""
    return x >= 0

def sqrt_postcondition(x, result):
    """Postcondition: result squared should approximately equal input"""
    return abs(result * result - x) < 1e-10

import math

# Register contract for math.sqrt
register_contract(
    math.sqrt,
    pre=sqrt_precondition,
    post=sqrt_postcondition
)

# Clear all contracts when done
clear_contract_registrations()
```

### Contract Enforcement

Runtime enforcement of contracts with specialized exception types.

```python { .api }
class PreconditionFailed(BaseException):
    """
    Exception for precondition failures.
    
    Raised when a function is called with arguments that violate
    the registered precondition. Inherits from BaseException to
    avoid being caught by general exception handlers.
    """

class PostconditionFailed(BaseException):
    """
    Exception for postcondition failures.
    
    Raised when a function returns a value that violates the
    registered postcondition. Inherits from BaseException to
    ensure contract violations are not silently ignored.
    """

class EnforcedConditions:
    """
    Module for enforcing conditions during execution.
    
    Provides tracing and enforcement mechanisms to ensure contracts
    are checked at runtime, with proper exception handling and
    reporting of contract violations.
    """

def WithEnforcement(fn):
    """
    Ensure conditions are enforced on callable.
    
    Parameters:
    - fn: Function to enforce conditions on
    
    Returns:
    Wrapped function that checks contracts on every call
    
    Decorator that enables runtime contract checking for a function,
    raising PreconditionFailed or PostconditionFailed when contracts
    are violated.
    """
```

**Usage Examples:**

```python
from crosshair.enforce import WithEnforcement, PreconditionFailed, PostconditionFailed

@WithEnforcement  
def divide(a: float, b: float) -> float:
    """
    pre: b != 0
    post: __return__ == a / b
    """
    return a / b

try:
    result = divide(10.0, 0.0)  # Will raise PreconditionFailed
except PreconditionFailed as e:
    print(f"Precondition violation: {e}")

try:
    # Hypothetical function with buggy implementation
    result = buggy_divide(10.0, 2.0)  # Might raise PostconditionFailed
except PostconditionFailed as e:
    print(f"Postcondition violation: {e}")
```

### Contract Annotation Formats

Support for multiple contract specification formats used in the Python ecosystem.

**PEP 316 Style (Docstring Contracts):**

```python
def factorial(n: int) -> int:
    """
    Calculate factorial of n.
    
    pre: n >= 0
    post: __return__ >= 1
    post: n == 0 or __return__ == n * factorial(n - 1)
    """
    if n == 0:
        return 1
    return n * factorial(n - 1)
```

**icontract Library Integration:**

```python
import icontract

@icontract.require(lambda x: x >= 0)
@icontract.ensure(lambda result: result >= 1)
@icontract.ensure(lambda x, result: x == 0 or result == x * factorial(x - 1))
def factorial(x: int) -> int:
    if x == 0:
        return 1
    return x * factorial(x - 1)
```

**deal Library Integration:**

```python
import deal

@deal.pre(lambda x: x >= 0)
@deal.post(lambda result: result >= 1)
def factorial(x: int) -> int:
    if x == 0:
        return 1
    return x * factorial(x - 1)
```

### Contract Verification Workflow

Complete workflow for contract verification and analysis.

```python { .api }
class ContractOverride:
    """
    Container for contract override information.
    
    Stores preconditions, postconditions, signature information,
    and other metadata needed for contract verification during
    symbolic execution.
    """

class AnalysisMessage:
    """
    Message reporting analysis results.
    
    Contains information about contract verification results,
    including counterexamples, error details, and execution
    context for failed contracts.
    """

def run_checkables(fn_info, options):
    """
    Run contract checking on function.
    
    Parameters:
    - fn_info: FunctionInfo containing function metadata
    - options: AnalysisOptions configuration
    
    Returns:
    Generator of AnalysisMessage results
    
    Performs symbolic execution to verify contracts, yielding
    messages for each contract verification attempt.
    """
```

**Advanced Contract Registration Example:**

```python
from crosshair.register_contract import register_contract
from inspect import Signature, Parameter

def complex_precondition(data, threshold, options=None):
    """Complex precondition with multiple parameters"""
    if not isinstance(data, list):
        return False
    if len(data) == 0:
        return False
    if threshold <= 0:
        return False
    return True

def complex_postcondition(data, threshold, options, result):
    """Complex postcondition checking result properties"""
    if result is None:
        return False
    if not isinstance(result, dict):
        return False
    return 'processed_count' in result

# Create custom signature for overloaded function
custom_sig = Signature([
    Parameter('data', Parameter.POSITIONAL_OR_KEYWORD, annotation=list),
    Parameter('threshold', Parameter.POSITIONAL_OR_KEYWORD, annotation=float),
    Parameter('options', Parameter.KEYWORD_ONLY, default=None, annotation=dict)
])

register_contract(
    complex_data_processor,
    pre=complex_precondition,
    post=complex_postcondition,
    sig=custom_sig,
    skip_body=False  # Analyze function body as well
)
```

### Contract Analysis Results

Understanding and interpreting contract verification results.

```python { .api }
class MessageType(enum.Enum):
    """
    Types of analysis messages from contract verification.
    
    - CONFIRMED: Contract violation confirmed with counterexample
    - CANNOT_CONFIRM: Unable to find violation within time limits
    - PRE_UNSAT: Precondition is unsatisfiable (always false)
    - POST_FAIL: Postcondition fails with satisfiable precondition
    - POST_ERR: Error occurred while checking postcondition
    - EXEC_ERR: Error occurred during function execution
    - SYNTAX_ERR: Syntax error in contract specification
    - IMPORT_ERR: Error importing required modules
    """
    CONFIRMED = "confirmed"
    CANNOT_CONFIRM = "cannot_confirm"  
    PRE_UNSAT = "pre_unsat"
    POST_FAIL = "post_fail"
    POST_ERR = "post_err"
    EXEC_ERR = "exec_err"
    SYNTAX_ERR = "syntax_err"
    IMPORT_ERR = "import_err"
```

**Interpreting Analysis Results:**

```python
from crosshair.core_and_libs import run_checkables
from crosshair.options import AnalysisOptions

options = AnalysisOptions()

for message in run_checkables(function_info, options):
    if message.message_type == MessageType.CONFIRMED:
        print(f"Contract violation found: {message.message}")
        print(f"Counterexample: {message.test_fn}")
    elif message.message_type == MessageType.CANNOT_CONFIRM:
        print(f"No violation found within time limit")
    elif message.message_type == MessageType.PRE_UNSAT:
        print(f"Precondition is unsatisfiable: {message.message}")
```

### Contract Testing Integration

Integration with testing frameworks for contract-based testing.

**Generating Contract Tests:**

```python
# CrossHair can generate tests that verify contracts
def test_contract_compliance():
    """Generated test ensuring contract compliance"""
    # Test case that satisfies preconditions
    result = sqrt(4.0)
    # Verify postcondition manually
    assert abs(result * result - 4.0) < 1e-10
```

**Property-Based Testing Integration:**

```python
from hypothesis import given, strategies as st

@given(st.floats(min_value=0.0, max_value=1000.0))
def test_sqrt_contract(x):
    """Property-based test for sqrt contract"""
    try:
        result = sqrt(x)
        # Verify postcondition holds
        assert abs(result * result - x) < 1e-10
    except PreconditionFailed:
        # This should not happen with valid inputs
        assert False, f"Precondition failed for valid input {x}"
```