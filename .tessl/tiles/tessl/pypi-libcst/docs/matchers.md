# Pattern Matching

LibCST's matcher system provides declarative pattern matching for finding, extracting, and replacing specific code patterns in concrete syntax trees. The system supports complex matching logic, data extraction, and integration with the visitor framework.

## Capabilities

### Core Matching Functions

Find and analyze code patterns declaratively without writing custom visitors.

```python { .api }
def matches(node: CSTNode, matcher: Union[CSTNode, MatcherPattern]) -> bool:
    """
    Check if a node matches a given pattern.
    
    Parameters:
    - node: CST node to test
    - matcher: Pattern to match against
    
    Returns:
    bool: True if node matches the pattern
    """

def findall(tree: CSTNode, matcher: Union[CSTNode, MatcherPattern]) -> Sequence[CSTNode]:
    """
    Find all nodes in tree that match the pattern.
    
    Parameters:
    - tree: CST tree to search
    - matcher: Pattern to match
    
    Returns:
    Sequence[CSTNode]: All matching nodes
    """

def extract(node: CSTNode, matcher: Union[CSTNode, MatcherPattern]) -> Dict[str, Union[CSTNode, Sequence[CSTNode]]]:
    """
    Extract captured groups from a matching node.
    
    Parameters:
    - node: CST node that matches the pattern
    - matcher: Pattern with capture groups
    
    Returns:
    Dict[str, Union[CSTNode, Sequence[CSTNode]]]: Captured groups by name
    """

def extractall(tree: CSTNode, matcher: Union[CSTNode, MatcherPattern]) -> Sequence[Dict[str, Union[CSTNode, Sequence[CSTNode]]]]:
    """
    Extract captured groups from all matching nodes in tree.
    
    Parameters:
    - tree: CST tree to search
    - matcher: Pattern with capture groups
    
    Returns:
    Sequence[Dict]: List of captured groups from each match
    """

def replace(tree: CSTNode, matcher: Union[CSTNode, MatcherPattern], replacement: Union[CSTNode, Callable]) -> CSTNode:
    """
    Replace all matching nodes with replacement.
    
    Parameters:
    - tree: CST tree to transform
    - matcher: Pattern to match
    - replacement: Replacement node or callable
    
    Returns:
    CSTNode: Transformed tree
    """
```

### Matcher Combinators

Compose complex matching logic from simple patterns.

```python { .api }
class AllOf:
    """Match if all sub-matchers match."""
    def __init__(self, *matchers: Union[CSTNode, MatcherPattern]) -> None: ...

class OneOf:
    """Match if any sub-matcher matches."""
    def __init__(self, *matchers: Union[CSTNode, MatcherPattern]) -> None: ...

class AtLeastN:
    """Match if at least N sub-matchers match."""
    def __init__(self, n: int, matcher: Union[CSTNode, MatcherPattern]) -> None: ...

class AtMostN:
    """Match if at most N sub-matchers match."""
    def __init__(self, n: int, matcher: Union[CSTNode, MatcherPattern]) -> None: ...

class ZeroOrMore:
    """Match zero or more occurrences."""
    def __init__(self, matcher: Union[CSTNode, MatcherPattern]) -> None: ...

class ZeroOrOne:
    """Match zero or one occurrence."""
    def __init__(self, matcher: Union[CSTNode, MatcherPattern]) -> None: ...

class DoesNotMatch:
    """Inverse matcher - match if sub-matcher does not match."""
    def __init__(self, matcher: Union[CSTNode, MatcherPattern]) -> None: ...

class DoNotCare:
    """Wildcard matcher - matches any node."""
```

### Special Matchers

Advanced matching capabilities for specific use cases.

```python { .api }
class MatchIfTrue:
    """Match based on predicate function."""
    def __init__(self, func: Callable[[CSTNode], bool]) -> None: ...

class MatchRegex:
    """Match string content with regular expressions."""
    def __init__(self, pattern: str, flags: int = 0) -> None: ...

class MatchMetadata:
    """Match based on metadata values."""
    def __init__(self, provider: Type[BaseMetadataProvider], metadata: Any) -> None: ...

class MatchMetadataIfTrue:
    """Match based on metadata predicate."""
    def __init__(self, provider: Type[BaseMetadataProvider], func: Callable[[Any], bool]) -> None: ...

class SaveMatchedNode:
    """Capture matched nodes with a name."""
    def __init__(self, name: str, matcher: Union[CSTNode, MatcherPattern] = DoNotCare()) -> None: ...

class TypeOf:
    """Match nodes of specific type."""
    def __init__(self, node_type: Type[CSTNode]) -> None: ...
```

### Visitor Integration

Integrate matchers with the visitor framework for enhanced traversal control.

```python { .api }
def call_if_inside(matcher: Union[CSTNode, MatcherPattern]) -> Callable:
    """
    Decorator to call visitor method only if inside matching context.
    
    Parameters:
    - matcher: Pattern that must match ancestor node
    
    Returns:
    Callable: Decorator function
    """

def call_if_not_inside(matcher: Union[CSTNode, MatcherPattern]) -> Callable:
    """
    Decorator to call visitor method only if not inside matching context.
    
    Parameters:
    - matcher: Pattern that must not match ancestor node
    
    Returns:
    Callable: Decorator function
    """

def visit(*matchers: Union[CSTNode, MatcherPattern]) -> Callable:
    """
    Decorator to call visitor method only for matching nodes.
    
    Parameters:
    - matchers: Patterns that must match the current node
    
    Returns:
    Callable: Decorator function
    """

def leave(*matchers: Union[CSTNode, MatcherPattern]) -> Callable:
    """
    Decorator to call leave method only for matching nodes.
    
    Parameters:
    - matchers: Patterns that must match the current node
    
    Returns:
    Callable: Decorator function
    """
```

### Matcher-Enabled Visitors

Base classes that provide matcher decorator support.

```python { .api }
class MatcherDecoratableVisitor(CSTVisitor):
    """Base visitor with matcher decorator support."""

class MatcherDecoratableTransformer(CSTTransformer):
    """Base transformer with matcher decorator support."""

class MatchDecoratorMismatch(Exception):
    """Raised when matcher decorators are used incorrectly."""
```

## Usage Examples

### Basic Pattern Matching

```python
import libcst as cst
from libcst import matchers as m

source = '''
def foo():
    x = 42
    y = "hello"
    z = [1, 2, 3]
'''

module = cst.parse_module(source)

# Find all assignment statements
assignments = m.findall(module, m.Assign())
print(f"Found {len(assignments)} assignments")

# Check if specific pattern exists
has_string_assignment = m.matches(
    module, 
    m.Module(body=[
        m.AtLeastN(1, m.SimpleStatementLine(body=[
            m.Assign(value=m.SimpleString())
        ]))
    ])
)
print(f"Has string assignment: {has_string_assignment}")
```

### Data Extraction

```python
import libcst as cst
from libcst import matchers as m

source = '''
def calculate(x, y):
    return x + y

def process(data):
    return data * 2
'''

module = cst.parse_module(source)

# Extract function names and parameter counts
function_pattern = m.FunctionDef(
    name=m.SaveMatchedNode("name"),
    params=m.SaveMatchedNode("params")
)

matches = m.extractall(module, function_pattern)
for match in matches:
    name = match["name"].value
    param_count = len(match["params"].params)
    print(f"Function {name} has {param_count} parameters")
```

### Pattern Replacement

```python
import libcst as cst
from libcst import matchers as m

source = '''
def foo():
    print("debug: starting")
    result = calculate()
    print("debug: finished")
    return result
'''

module = cst.parse_module(source)

# Replace debug print statements with logging calls
debug_print = m.Call(
    func=m.Name("print"),
    args=[m.Arg(value=m.SimpleString(value=m.MatchRegex(r'"debug:.*"')))]
)

def make_logging_call(node):
    # Extract the debug message
    message = node.args[0].value.value
    return cst.Call(
        func=cst.Attribute(value=cst.Name("logging"), attr=cst.Name("debug")),
        args=[cst.Arg(value=cst.SimpleString(message))]
    )

new_module = m.replace(module, debug_print, make_logging_call)
print(new_module.code)
```

### Visitor with Matcher Decorators

```python
import libcst as cst
from libcst import matchers as m

class SecurityAnalyzer(m.MatcherDecoratableVisitor):
    def __init__(self):
        self.security_issues = []
    
    @m.visit(m.Call(func=m.Name("eval")))
    def visit_eval_call(self, node):
        self.security_issues.append("Dangerous eval() call found")
    
    @m.visit(m.Call(func=m.Name("exec")))
    def visit_exec_call(self, node):
        self.security_issues.append("Dangerous exec() call found")
    
    @m.call_if_inside(m.FunctionDef(name=m.Name("__init__")))
    @m.visit(m.Assign())
    def visit_init_assignment(self, node):
        # Only called for assignments inside __init__ methods
        pass

# Usage
source = '''
class MyClass:
    def __init__(self):
        self.x = eval("42")
    
    def process(self):
        exec("print('hello')")
'''

module = cst.parse_module(source)
analyzer = SecurityAnalyzer()
module.visit(analyzer)
print(analyzer.security_issues)
```

### Complex Pattern Matching

```python
import libcst as cst
from libcst import matchers as m

# Find all function calls inside try blocks
complex_pattern = m.Try(
    body=m.SimpleStatementSuite(body=[
        m.ZeroOrMore(m.SimpleStatementLine(body=[
            m.OneOf(
                m.Expr(value=m.Call()),  # Expression statements with calls
                m.Assign(value=m.Call())  # Assignments with call values
            )
        ]))
    ])
)

# Find functions that take at least 3 parameters
many_param_functions = m.FunctionDef(
    params=m.Parameters(params=m.AtLeastN(3, m.Param()))
)

# Match string literals containing SQL keywords
sql_strings = m.SimpleString(
    value=m.MatchRegex(r'".*\b(SELECT|INSERT|UPDATE|DELETE)\b.*"', re.IGNORECASE)
)
```

## Types

```python { .api }
# Core matcher types
MatcherPattern = Union[
    CSTNode,
    "AllOf",
    "OneOf", 
    "AtLeastN",
    "AtMostN",
    "ZeroOrMore",
    "ZeroOrOne",
    "DoesNotMatch",
    "DoNotCare",
    "MatchIfTrue",
    "MatchRegex",
    "MatchMetadata",
    "MatchMetadataIfTrue",
    "SaveMatchedNode",
    "TypeOf"
]

# Visitor decorator types
VisitorMethod = Callable[[CSTVisitor, CSTNode], None]
TransformerMethod = Callable[[CSTTransformer, CSTNode, CSTNode], Union[CSTNode, RemovalSentinel]]

# Exception types
class MatchDecoratorMismatch(Exception):
    """Raised when matcher decorators are misused."""

# Sentinel for wildcards
class DoNotCareSentinel:
    """Wildcard matcher that matches anything."""

DoNotCare: DoNotCareSentinel
```