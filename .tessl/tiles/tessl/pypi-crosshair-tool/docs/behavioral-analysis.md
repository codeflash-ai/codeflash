# Behavioral Analysis and Comparison

CrossHair provides sophisticated tools for analyzing and comparing the behavior of different function implementations. These capabilities enable developers to find behavioral differences, validate refactoring efforts, and generate comprehensive behavior descriptions.

## Capabilities

### Function Behavior Comparison

Compare the behavior of two function implementations to identify differences in outputs, exceptions, or side effects.

```python { .api }
def diff_behavior(fn1, fn2, *, sig=None, options=None):
    """
    Compare behaviors of two functions.
    
    Parameters:
    - fn1: First function to compare
    - fn2: Second function to compare  
    - sig: Optional signature specification for comparison
    - options: Optional AnalysisOptions for controlling comparison
    
    Returns:
    Generator yielding BehaviorDiff instances describing differences
    
    Uses symbolic execution to explore input spaces and find cases
    where the two functions exhibit different behavior. Compares
    return values, exception types, and execution characteristics.
    """

def diff_behavior_with_signature(fn1, fn2, sig, options):
    """
    Compare behaviors with specific signature constraints.
    
    Parameters:
    - fn1: First function to compare
    - fn2: Second function to compare
    - sig: Signature object defining input constraints
    - options: AnalysisOptions configuration
    
    Returns:
    Generator yielding detailed behavioral differences
    
    Performs targeted comparison using a specific function signature,
    enabling more focused analysis of particular input patterns.
    """
```

**Usage Examples:**

```python
from crosshair.diff_behavior import diff_behavior

def old_sort(items, reverse=False):
    """Original implementation"""
    result = list(items)
    result.sort(reverse=reverse)
    return result

def new_sort(items, reverse=False):
    """New implementation with bug"""
    result = list(items)
    result.sort()  # Bug: ignores reverse parameter
    return result

# Find behavioral differences
for diff in diff_behavior(old_sort, new_sort):
    print(f"Difference found:")
    print(f"  Input: {diff.args}")
    print(f"  old_sort: {diff.result1}")
    print(f"  new_sort: {diff.result2}")
    print(f"  Reason: {diff.message}")
```

### Behavior Description and Analysis

Generate comprehensive descriptions of function behavior for documentation and verification.

```python { .api }
def describe_behavior(fn, *, sig=None, options=None):
    """
    Describe function behavior comprehensively.
    
    Parameters:
    - fn: Function to analyze
    - sig: Optional signature specification
    - options: Optional AnalysisOptions configuration
    
    Returns:
    Detailed behavior description including:
    - Input/output relationships
    - Exception conditions  
    - Edge case handling
    - Performance characteristics
    
    Performs exhaustive symbolic execution to characterize
    the complete behavior space of a function.
    """
```

**Usage Example:**

```python
from crosshair.diff_behavior import describe_behavior

def calculate_discount(price, discount_percent, customer_type="standard"):
    """
    Calculate discount for a purchase.
    
    pre: price >= 0
    pre: 0 <= discount_percent <= 100
    post: 0 <= __return__ <= price
    """
    if customer_type == "premium":
        discount_percent *= 1.2
    
    discount_amount = price * (discount_percent / 100)
    return min(discount_amount, price)

# Generate behavior description
behavior = describe_behavior(calculate_discount)
print(behavior.summary)
print("Edge cases found:", behavior.edge_cases)
print("Exception conditions:", behavior.exception_conditions)
```

### Behavioral Difference Analysis

Detailed analysis of differences found between function implementations.

```python { .api }
class BehaviorDiff:
    """
    Represents a difference between function behaviors.
    
    Contains detailed information about inputs that cause different
    behaviors, including return values, exceptions, and execution
    characteristics.
    """
    
    def __init__(self, args, result1, result2, message, exception_diff=None):
        """
        Initialize behavior difference.
        
        Parameters:
        - args: Input arguments that cause the difference
        - result1: Result from first function
        - result2: Result from second function  
        - message: Human-readable description of difference
        - exception_diff: Optional exception difference details
        """
    
    @property
    def args(self):
        """Input arguments that trigger the behavioral difference."""
        
    @property
    def result1(self):
        """Result from the first function."""
        
    @property
    def result2(self):
        """Result from the second function."""
        
    @property
    def message(self):
        """Human-readable description of the difference."""
        
    @property
    def exception_diff(self):
        """Details about exception-related differences."""

class Result:
    """
    Result of behavior analysis containing outcomes and metadata.
    
    Encapsulates the results of function execution during behavioral
    analysis, including return values, exceptions, and execution context.
    """
    
    def __init__(self, return_value=None, exception=None, execution_log=None):
        """
        Initialize analysis result.
        
        Parameters:
        - return_value: Value returned by function (if any)
        - exception: Exception raised by function (if any)
        - execution_log: Optional log of execution steps
        """
```

**Advanced Comparison Example:**

```python
from crosshair.diff_behavior import diff_behavior, BehaviorDiff

def safe_divide(a, b):
    """Safe division with error handling"""
    if b == 0:
        return float('inf') if a > 0 else float('-inf') if a < 0 else float('nan')
    return a / b

def unsafe_divide(a, b):
    """Unsafe division that raises exceptions"""
    return a / b

# Compare with exception analysis
for diff in diff_behavior(safe_divide, unsafe_divide):
    if diff.exception_diff:
        print(f"Exception handling difference:")
        print(f"  Input: a={diff.args[0]}, b={diff.args[1]}")
        print(f"  safe_divide: returns {diff.result1}")
        print(f"  unsafe_divide: raises {diff.result2}")
```

### Exception Equivalence Configuration

Configure how exceptions are compared during behavioral analysis.

```python { .api }
class ExceptionEquivalenceType(enum.Enum):
    """
    Types of exception equivalence for behavioral comparison.
    
    Controls how strictly exceptions are compared when analyzing
    behavioral differences between functions.
    """
    
    SAME_TYPE = "same_type"
    """Consider exceptions equivalent if they have the same type."""
    
    SAME_TYPE_AND_MESSAGE = "same_type_and_message"  
    """Consider exceptions equivalent if type and message match."""
    
    EXACT = "exact"
    """Consider exceptions equivalent only if completely identical."""
```

**Usage with Different Equivalence Types:**

```python
from crosshair.diff_behavior import diff_behavior, ExceptionEquivalenceType
from crosshair.options import AnalysisOptions

def func1(x):
    if x < 0:
        raise ValueError("Negative input")
    return x * 2

def func2(x):
    if x < 0:
        raise ValueError("Input cannot be negative")  # Different message
    return x * 2

# Compare with different exception equivalence settings
options = AnalysisOptions()

# Strict comparison (different messages = different behavior)
options.exception_equivalence = ExceptionEquivalenceType.EXACT
strict_diffs = list(diff_behavior(func1, func2, options=options))

# Lenient comparison (same type = equivalent)  
options.exception_equivalence = ExceptionEquivalenceType.SAME_TYPE
lenient_diffs = list(diff_behavior(func1, func2, options=options))

print(f"Strict comparison found {len(strict_diffs)} differences")
print(f"Lenient comparison found {len(lenient_diffs)} differences")
```

### Performance and Behavior Profiling

Analyze performance characteristics alongside behavioral properties.

```python { .api }
class ExecutionProfile:
    """
    Profile of function execution including performance metrics.
    
    Contains timing information, resource usage, and execution
    path statistics gathered during symbolic execution.
    """
    
    def __init__(self, execution_time=None, path_count=None, solver_calls=None):
        """
        Initialize execution profile.
        
        Parameters:
        - execution_time: Time spent in symbolic execution
        - path_count: Number of execution paths explored
        - solver_calls: Number of SMT solver queries made
        """
```

**Performance-Aware Comparison:**

```python
def optimized_fibonacci(n, memo={}):
    """Optimized fibonacci with memoization"""
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = optimized_fibonacci(n-1, memo) + optimized_fibonacci(n-2, memo)
    return memo[n]

def naive_fibonacci(n):
    """Naive recursive fibonacci"""
    if n <= 1:
        return n
    return naive_fibonacci(n-1) + naive_fibonacci(n-2)

# Compare behavior with performance implications
from crosshair.options import AnalysisOptions

options = AnalysisOptions(per_path_timeout=1.0)  # Short timeout

# The analysis might find that both produce same results
# but optimized version completes within timeout while naive doesn't
for diff in diff_behavior(optimized_fibonacci, naive_fibonacci, options=options):
    print(f"Performance difference: {diff.message}")
```

### Integration with Testing Frameworks

Generate behavioral tests based on discovered differences and edge cases.

**Generating Regression Tests:**

```python
from crosshair.diff_behavior import diff_behavior

def generate_regression_tests(old_func, new_func, test_name_prefix="test"):
    """Generate regression tests from behavioral differences"""
    test_cases = []
    
    for i, diff in enumerate(diff_behavior(old_func, new_func)):
        test_code = f"""
def {test_name_prefix}_difference_{i}():
    '''Test case for behavioral difference {i}'''
    args = {diff.args}
    
    # Expected behavior from old function
    expected = {repr(diff.result1)}
    
    # Actual behavior from new function  
    actual = new_func(*args)
    
    assert actual == expected, f"Behavioral difference: {{actual}} != {{expected}}"
"""
        test_cases.append(test_code)
    
    return test_cases

# Generate tests for refactored function
test_cases = generate_regression_tests(original_implementation, refactored_implementation)
for test in test_cases:
    print(test)
```

### Behavioral Invariant Discovery

Discover and verify behavioral invariants across function implementations.

```python
def discover_invariants(functions, input_generator):
    """
    Discover behavioral invariants across multiple function implementations.
    
    Parameters:
    - functions: List of function implementations to compare
    - input_generator: Generator providing test inputs
    
    Returns:
    List of discovered invariants that hold across all implementations
    """
    invariants = []
    
    for inputs in input_generator:
        results = [f(*inputs) for f in functions]
        
        # Check if all functions produce same result
        if all(r == results[0] for r in results):
            invariant = f"For input {inputs}, all functions return {results[0]}"
            invariants.append(invariant)
    
    return invariants
```

**Example Invariant Discovery:**

```python
def math_sqrt_wrapper(x):
    """Wrapper for math.sqrt"""
    import math
    return math.sqrt(x)

def newton_sqrt(x, precision=1e-10):
    """Newton's method square root"""
    if x < 0:
        raise ValueError("Cannot compute square root of negative number")
    if x == 0:
        return 0
    
    guess = x / 2
    while abs(guess * guess - x) > precision:
        guess = (guess + x / guess) / 2
    return guess

# Discover invariants between implementations
def input_gen():
    for x in [0, 1, 4, 9, 16, 25, 100]:
        yield (x,)

invariants = discover_invariants(
    [math_sqrt_wrapper, newton_sqrt], 
    input_gen()
)

for invariant in invariants:
    print(invariant)
```