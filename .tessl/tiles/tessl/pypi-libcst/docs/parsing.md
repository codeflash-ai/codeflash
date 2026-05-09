# Core Parsing Functions

LibCST provides parsing functions to convert Python source code into concrete syntax trees that preserve all formatting details including comments, whitespace, and parentheses. These functions serve as the primary entry points for code analysis and transformation workflows.

## Capabilities

### Module Parsing

Parse complete Python modules including all statements, comments, and formatting.

```python { .api }
def parse_module(
    source: Union[str, bytes],
    config: Optional[PartialParserConfig] = None
) -> Module:
    """
    Parse Python source code into a Module CST node.
    
    Parameters:
    - source: Python source code as string or bytes
    - config: Parser configuration options
    
    Returns:
    Module: Root CST node representing the entire module
    
    Raises:
    ParserSyntaxError: If the code contains syntax errors
    """
```

### Statement Parsing

Parse individual Python statements for focused analysis or transformation.

```python { .api }
def parse_statement(
    source: str,
    config: Optional[PartialParserConfig] = None
) -> Union[SimpleStatementLine, BaseCompoundStatement]:
    """
    Parse a single Python statement into CST.
    
    Parameters:
    - source: Python statement source code
    - config: Parser configuration options
    
    Returns:
    Union[SimpleStatementLine, BaseCompoundStatement]: Parsed statement node
    
    Raises:
    ParserSyntaxError: If the statement contains syntax errors
    """
```

### Expression Parsing

Parse Python expressions for analysis of individual expressions within larger contexts.

```python { .api }
def parse_expression(
    source: str,
    config: Optional[PartialParserConfig] = None
) -> BaseExpression:
    """
    Parse a single Python expression into CST.
    
    Parameters:
    - source: Python expression source code
    - config: Parser configuration options
    
    Returns:
    BaseExpression: Parsed expression node
    
    Raises:
    ParserSyntaxError: If the expression contains syntax errors
    """
```

### Parser Configuration

Configure parser behavior for specific Python versions and features.

```python { .api }
class PartialParserConfig:
    """Configuration for parser behavior."""
    
    def __init__(
        self,
        *,
        python_version: Union[str, AutoConfig] = AutoConfig.token,
        encoding: Union[str, AutoConfig] = AutoConfig.token,
        future_imports: Union[FrozenSet[str], AutoConfig] = AutoConfig.token,
        default_indent: Union[str, AutoConfig] = AutoConfig.token,
        default_newline: Union[str, AutoConfig] = AutoConfig.token
    ) -> None:
        """
        Initialize parser configuration.
        
        Parameters:
        - python_version: Python version for syntax compatibility (e.g., "3.8", "3.7.1")
        - encoding: File encoding format (e.g., "utf-8", "latin-1") 
        - future_imports: Set of detected __future__ import names
        - default_indent: Indentation style (spaces/tabs, e.g., "    ", "\t")
        - default_newline: Newline style ("\n", "\r\n", or "\r")
        
        All parameters default to auto-detection from source code.
        """

class AutoConfig:
    """Marker for auto-configuration of parser settings."""
    token: ClassVar[AutoConfig]
```

### Utilities

Helper functions for common parsing operations and version management.

```python { .api }
# Version information
LIBCST_VERSION: str
KNOWN_PYTHON_VERSION_STRINGS: List[str]

def ensure_type(node: CSTNode, nodetype: Type[CSTNodeT]) -> CSTNodeT:
    """
    Type refinement utility for CST nodes (deprecated).
    
    Parameters:
    - node: CST node to check
    - nodetype: Expected node type
    
    Returns:
    CSTNodeT: Node cast to expected type
    
    Raises:
    TypeError: If node is not of expected type
    """
```

## Usage Examples

### Basic Module Parsing

```python
import libcst as cst

source = '''
def factorial(n):
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''

module = cst.parse_module(source)
print(type(module))  # <class 'libcst._nodes.module.Module'>
print(module.code)   # Original source with exact formatting
```

### Statement Parsing

```python
import libcst as cst

# Parse different statement types
stmt1 = cst.parse_statement("x = 42")
stmt2 = cst.parse_statement("for i in range(10): pass")
stmt3 = cst.parse_statement("class MyClass: pass")

print(type(stmt1))  # <class 'libcst._nodes.statement.SimpleStatementLine'>
print(type(stmt2))  # <class 'libcst._nodes.statement.For'>
print(type(stmt3))  # <class 'libcst._nodes.statement.ClassDef'>
```

### Expression Parsing

```python
import libcst as cst

# Parse various expression types
expr1 = cst.parse_expression("x + y * z")
expr2 = cst.parse_expression("[1, 2, 3]")
expr3 = cst.parse_expression("lambda x: x**2")

print(type(expr1))  # <class 'libcst._nodes.expression.BinaryOperation'>
print(type(expr2))  # <class 'libcst._nodes.expression.List'>
print(type(expr3))  # <class 'libcst._nodes.expression.Lambda'>
```

### Error Handling

```python
import libcst as cst

try:
    # This will raise a ParserSyntaxError
    module = cst.parse_module("def invalid syntax")
except cst.ParserSyntaxError as e:
    print(f"Parse error: {e}")
    print(f"Line: {e.line}, Column: {e.column}")
```

## Types

```python { .api }
# Core module type
class Module(CSTNode):
    """Root node representing a complete Python module."""
    body: Sequence[Union[SimpleStatementLine, BaseCompoundStatement]]
    header: Sequence[Union[SimpleStatementLine, BaseCompoundStatement]]
    footer: Sequence[Union[SimpleStatementLine, BaseCompoundStatement]]
    encoding: str
    default_indent: str
    default_newline: str
    has_trailing_newline: bool
    
    def code(self) -> str:
        """Generate Python source code from the CST."""

# Parser configuration
class PartialParserConfig:
    """Configuration options for the parser."""
    python_version: str
    future_annotations: bool
    strict: bool

# Exception types
class ParserSyntaxError(Exception):
    """Raised when parser encounters syntax errors."""
    message: str
    line: int
    column: int
    
class CSTValidationError(Exception):
    """Raised when CST node validation fails."""
    
class CSTLogicError(Exception):  
    """Raised for internal logic errors in CST operations."""
```