# AST and Node System

Abstract syntax tree representation and node types used internally by mypy for code analysis. This system provides structured access to Python code for type checking and static analysis.

## Capabilities

### Base Node Classes

Foundation classes for the AST hierarchy providing location information and visitor pattern support.

```python { .api }
class Context:
    """
    Base class providing source location information for error reporting.
    
    All AST nodes inherit location tracking for precise error messages
    and diagnostic information.
    
    Attributes:
    - line: int - Line number in source file (1-based)
    - column: int - Column number in source line (0-based)
    """

class Node(Context):
    """
    Abstract base class for all AST nodes.
    
    Root of the AST hierarchy implementing the visitor pattern
    for traversal and transformation operations.
    
    Methods:
    - accept(visitor: NodeVisitor[T]) -> T
    """

class Statement(Node):
    """
    Base class for statement AST nodes.
    
    Represents executable statements like assignments, function calls,
    control flow statements, and declarations.
    """

class Expression(Node):  
    """
    Base class for expression AST nodes.
    
    Represents expressions that evaluate to values, including
    literals, names, operations, and function calls.
    """
```

### File and Module Nodes

Top-level nodes representing entire source files and modules.

```python { .api }
class MypyFile(Node):
    """
    Represents an entire source file in mypy's AST.
    
    Top-level container for all definitions, imports, and statements
    in a Python source file.
    
    Attributes:
    - defs: list[Statement] - Top-level definitions and statements
    - names: SymbolTable - Module-level symbol table
    - imports: list[ImportBase] - Import statements
    - is_bom: bool - Whether file starts with BOM
    - path: str - File path
    - module: str - Module name
    """
```

### Class and Function Definition Nodes

Core definition nodes for classes, functions, and methods.

```python { .api }
class ClassDef(Statement):
    """
    Represents class definitions.
    
    Contains class body, inheritance information, decorators,
    and metadata for type checking.
    
    Attributes:
    - name: str - Class name
    - defs: Block - Class body statements
    - base_type_exprs: list[Expression] - Base class expressions
    - decorators: list[Expression] - Class decorators
    - info: TypeInfo - Semantic information about the class
    - type_vars: list[TypeVarLikeType] - Generic type parameters
    """

class FuncDef(Statement):
    """
    Represents function and method definitions.
    
    Contains function signature, body, decorators, and type information
    for both regular functions and methods.
    
    Attributes:
    - name: str - Function name
    - arguments: Arguments - Parameter information
    - body: Block - Function body statements
    - type: Type | None - Function type annotation
    - info: TypeInfo | None - Class info (for methods)
    - is_property: bool - Whether this is a property
    - is_class_method: bool - Whether this is a classmethod
    - is_static_method: bool - Whether this is a staticmethod
    """

class Arguments(Node):
    """
    Represents function parameter list.
    
    Contains detailed information about function parameters including
    types, defaults, and parameter kinds.
    
    Attributes:
    - arguments: list[Argument] - Individual parameter definitions
    - arg_names: list[str] - Parameter names
    - arg_kinds: list[int] - Parameter kinds (positional, keyword, etc.)
    - initializers: list[Expression | None] - Default value expressions
    """

class Argument(Node):
    """
    Represents a single function parameter.
    
    Attributes:
    - variable: Var - Variable representing the parameter
    - type_annotation: Expression | None - Type annotation
    - initializer: Expression | None - Default value
    - kind: int - Parameter kind (ARG_POS, ARG_OPT, ARG_STAR, ARG_STAR2)
    """
```

### Symbol Table and Type Information

Core semantic analysis nodes for name resolution and type information.

```python { .api }
class TypeInfo(Node):
    """
    Contains semantic information about classes.
    
    Central repository for class metadata including inheritance hierarchy,
    method resolution order, and member information.
    
    Attributes:
    - names: SymbolTable - Class member symbol table
    - mro: list[TypeInfo] - Method resolution order
    - bases: list[Instance] - Direct base classes
    - abstract_attributes: list[str] - Abstract method/property names
    - is_abstract: bool - Whether class is abstract
    - is_protocol: bool - Whether class is a Protocol
    - fallback_to_any: bool - Whether to fallback to Any
    """

class SymbolTable(dict[str, SymbolTableNode]):
    """
    Maps names to their definitions within a scope.
    
    Used for name resolution during semantic analysis and type checking.
    Inherits from dict for convenient name lookup operations.
    """

class SymbolTableNode:
    """
    Individual entries in symbol tables.
    
    Links names to their AST nodes, types, and import information.
    
    Attributes:
    - kind: int - Symbol kind (LDEF, GDEF, MDEF, etc.)
    - node: Node | None - AST node for the symbol
    - type_override: Type | None - Explicit type override
    - module_public: bool - Whether symbol is publicly exported
    - implicit: bool - Whether symbol was implicitly created
    """

class Var(Node):
    """
    Represents variable declarations and references.
    
    Used for local variables, instance variables, class variables,
    and global variables.
    
    Attributes:
    - name: str - Variable name
    - type: Type | None - Variable type
    - is_self: bool - Whether this is 'self' parameter
    - is_cls: bool - Whether this is 'cls' parameter
    - is_ready: bool - Whether variable is fully analyzed
    """
```

### Expression Nodes

Nodes representing different types of expressions and operations.

```python { .api }
class NameExpr(Expression):
    """
    Represents name references (variable access).
    
    Used when accessing variables, functions, classes, or other names.
    
    Attributes:
    - name: str - Name being referenced
    - kind: int - Name kind (LDEF, GDEF, MDEF, etc.)
    - node: Node | None - Definition node for the name
    - fullname: str | None - Fully qualified name
    """

class MemberExpr(Expression):
    """
    Represents attribute access (obj.attr).
    
    Used for accessing attributes, methods, and properties of objects.
    
    Attributes:
    - expr: Expression - Object being accessed
    - name: str - Attribute name
    - kind: int | None - Member kind
    - fullname: str | None - Fully qualified attribute name
    """

class CallExpr(Expression):
    """
    Represents function and method calls.
    
    Contains call target, arguments, and metadata for type checking
    function calls and method invocations.
    
    Attributes:
    - callee: Expression - Function/method being called
    - args: list[Expression] - Positional and keyword arguments
    - arg_names: list[str | None] - Keyword argument names
    - arg_kinds: list[int] - Argument kinds
    """

class IndexExpr(Expression):
    """
    Represents indexing operations (obj[key]).
    
    Used for list/dict access, generic type instantiation,
    and other subscript operations.
    
    Attributes:
    - base: Expression - Object being indexed
    - index: Expression - Index/key expression
    - method: str | None - Special method name (__getitem__, etc.)
    """

class SliceExpr(Expression):
    """
    Represents slicing operations (obj[start:end:step]).
    
    Attributes:
    - begin_index: Expression | None - Start index
    - end_index: Expression | None - End index  
    - stride: Expression | None - Step value
    """

class OpExpr(Expression):
    """
    Represents binary operations (x + y, x == y, etc.).
    
    Attributes:
    - op: str - Operator string ('+', '==', 'and', etc.)
    - left: Expression - Left operand
    - right: Expression - Right operand
    - method: str | None - Special method name (__add__, __eq__, etc.)
    """

class UnaryExpr(Expression):
    """
    Represents unary operations (-x, not x, etc.).
    
    Attributes:
    - op: str - Operator string ('-', 'not', '~', etc.)
    - expr: Expression - Operand expression
    - method: str | None - Special method name (__neg__, etc.)
    """
```

### Literal and Collection Expressions

Nodes for literal values and collection constructors.

```python { .api }
class IntExpr(Expression):
    """
    Represents integer literals.
    
    Attributes:
    - value: int - Integer value
    """

class StrExpr(Expression):
    """
    Represents string literals.
    
    Attributes:
    - value: str - String value
    """

class FloatExpr(Expression):
    """
    Represents floating-point literals.
    
    Attributes:
    - value: float - Float value
    """

class ListExpr(Expression):
    """
    Represents list literals [1, 2, 3].
    
    Attributes:
    - items: list[Expression] - List elements
    """

class DictExpr(Expression):
    """
    Represents dictionary literals {'a': 1, 'b': 2}.
    
    Attributes:
    - items: list[tuple[Expression | None, Expression]] - Key-value pairs
    """

class SetExpr(Expression):
    """
    Represents set literals {1, 2, 3}.
    
    Attributes:
    - items: list[Expression] - Set elements
    """

class TupleExpr(Expression):
    """
    Represents tuple literals (1, 2, 3).
    
    Attributes:
    - items: list[Expression] - Tuple elements
    """
```

### Statement Nodes

Nodes representing different types of statements and control flow.

```python { .api }
class AssignmentStmt(Statement):
    """
    Represents assignment statements (x = y).
    
    Handles both simple assignments and complex assignment patterns
    including tuple unpacking and multiple targets.
    
    Attributes:
    - lvalues: list[Expression] - Left-hand side expressions
    - rvalue: Expression - Right-hand side value
    - type: Type | None - Optional type annotation
    """

class IfStmt(Statement):
    """
    Represents if statements and elif chains.
    
    Attributes:
    - expr: list[Expression] - Condition expressions for if/elif
    - body: list[Block] - Statement blocks for each condition
    - else_body: Block | None - Else block if present
    """

class ForStmt(Statement):
    """
    Represents for loops.
    
    Attributes:
    - index: Expression - Loop variable(s)
    - expr: Expression - Iterable expression
    - body: Block - Loop body
    - else_body: Block | None - Else block if present
    """

class WhileStmt(Statement):
    """
    Represents while loops.
    
    Attributes:
    - expr: Expression - Loop condition
    - body: Block - Loop body
    - else_body: Block | None - Else block if present
    """

class ReturnStmt(Statement):
    """
    Represents return statements.
    
    Attributes:
    - expr: Expression | None - Return value expression
    """

class RaiseStmt(Statement):
    """
    Represents raise statements.
    
    Attributes:
    - expr: Expression | None - Exception expression
    - from_expr: Expression | None - Chained exception
    """

class TryStmt(Statement):
    """
    Represents try/except/finally statements.
    
    Attributes:
    - body: Block - Try block
    - handlers: list[ExceptHandler] - Exception handlers
    - orelse: Block | None - Else block
    - finally_body: Block | None - Finally block
    """

class ExceptHandler(Node):
    """
    Represents individual except clauses.
    
    Attributes:
    - type: Expression | None - Exception type
    - name: str | None - Exception variable name
    - body: Block - Handler body
    """
```

### Import Statements

Nodes for import and from-import statements.

```python { .api }
class ImportStmt(Statement):
    """
    Represents import statements (import x, import y as z).
    
    Attributes:
    - ids: list[str] - Module names being imported
    - names: list[str | None] - Alias names (None if no alias)
    """

class ImportFromStmt(Statement):
    """
    Represents from-import statements (from x import y).
    
    Attributes:
    - module: str | None - Module name (None for relative imports)
    - names: list[tuple[str, str | None]] - (name, alias) pairs
    - relative: int - Relative import level (number of dots)
    """

class ImportAllStmt(Statement):
    """
    Represents star imports (from x import *).
    
    Attributes:
    - module: str - Module name
    - relative: int - Relative import level
    """
```

## AST Traversal and Manipulation

### Visitor Pattern

```python
from mypy.visitor import NodeVisitor

class ASTAnalyzer(NodeVisitor[None]):
    """Example AST visitor for analyzing code patterns."""
    
    def __init__(self):
        self.function_count = 0
        self.class_count = 0
        self.import_count = 0
    
    def visit_func_def(self, node: FuncDef) -> None:
        """Visit function definitions."""
        self.function_count += 1
        print(f"Found function: {node.name}")
        
        # Visit function body
        super().visit_func_def(node)
    
    def visit_class_def(self, node: ClassDef) -> None:
        """Visit class definitions."""
        self.class_count += 1
        print(f"Found class: {node.name}")
        
        # Analyze base classes
        for base in node.base_type_exprs:
            print(f"  Base class: {base}")
        
        # Visit class body
        super().visit_class_def(node)
    
    def visit_import_stmt(self, node: ImportStmt) -> None:
        """Visit import statements."""
        self.import_count += 1
        for module, alias in zip(node.ids, node.names):
            if alias:
                print(f"Import: {module} as {alias}")
            else:
                print(f"Import: {module}")

# Usage
analyzer = ASTAnalyzer()
mypy_file.accept(analyzer)

print(f"Summary: {analyzer.function_count} functions, "
      f"{analyzer.class_count} classes, {analyzer.import_count} imports")
```

### AST Construction

```python
from mypy.nodes import (
    FuncDef, Arguments, Argument, Var, Block, ReturnStmt,
    NameExpr, StrExpr, CallExpr
)

def create_function_node(name: str, return_type_name: str) -> FuncDef:
    """Create a function AST node programmatically."""
    
    # Create function arguments
    args = Arguments(
        arguments=[],
        arg_names=[],
        arg_kinds=[],
        initializers=[]
    )
    
    # Create return statement  
    return_stmt = ReturnStmt(StrExpr("Hello, World!"))
    
    # Create function body
    body = Block([return_stmt])
    
    # Create function definition
    func_def = FuncDef(
        name=name,
        arguments=args,
        body=body,
        type=None,  # Type will be inferred
        type_annotation=NameExpr(return_type_name)
    )
    
    return func_def

# Usage
hello_func = create_function_node("hello", "str")
```

### AST Transformation

```python
from mypy.visitor import NodeTransformer
from mypy.nodes import Expression, StrExpr, CallExpr

class StringLiteralTransformer(NodeTransformer):
    """Transform string literals to function calls."""
    
    def visit_str_expr(self, node: StrExpr) -> Expression:
        """Transform string literals to function calls."""
        if node.value.startswith("LOG:"):
            # Transform "LOG: message" to log_function("message")
            message = node.value[4:].strip()
            return CallExpr(
                callee=NameExpr("log_function"),
                args=[StrExpr(message)],
                arg_names=[None],
                arg_kinds=[ARG_POS]
            )
        
        return node

# Usage
transformer = StringLiteralTransformer()
transformed_ast = mypy_file.accept(transformer)
```

## Integration with Type Checking

### Semantic Analysis Integration

```python
from mypy.semanal import SemanticAnalyzer
from mypy.nodes import MypyFile

def analyze_file_semantics(mypy_file: MypyFile) -> None:
    """Perform semantic analysis on AST."""
    
    # Create semantic analyzer
    analyzer = SemanticAnalyzer(
        modules={},
        missing_modules=set(),
        incomplete_type_vars=set(),
        options=Options()
    )
    
    # Analyze file
    analyzer.visit_mypy_file(mypy_file)
    
    # Access symbol tables
    for name, node in mypy_file.names.items():
        print(f"Symbol: {name} -> {node}")
```

### Type Checker Integration

```python
from mypy.checker import TypeChecker
from mypy.nodes import Expression

def check_expression_type(expr: Expression, type_checker: TypeChecker) -> Type:
    """Get the type of an expression using the type checker."""
    
    # Type check the expression
    expr_type = expr.accept(type_checker)
    
    return expr_type

def validate_function_call(call_expr: CallExpr, type_checker: TypeChecker) -> bool:
    """Validate a function call using type information."""
    
    # Check callee type
    callee_type = call_expr.callee.accept(type_checker)
    
    # Check argument types
    arg_types = [arg.accept(type_checker) for arg in call_expr.args]
    
    # Validate call compatibility
    if isinstance(callee_type, CallableType):
        # Check argument count and types
        return len(arg_types) == len(callee_type.arg_types)
    
    return False
```

## AST Node Constants

### Node Kinds and Types

```python
# Symbol table node kinds
LDEF = 0  # Local definition
GDEF = 1  # Global definition  
MDEF = 2  # Class member definition
UNBOUND_IMPORTED = 3  # Unbound imported name

# Argument kinds
ARG_POS = 0       # Positional argument
ARG_OPT = 1       # Optional argument (with default)
ARG_STAR = 2      # *args
ARG_STAR2 = 3     # **kwargs

# Operator mappings
op_methods = {
    '+': '__add__',
    '-': '__sub__',  
    '*': '__mul__',
    '/': '__truediv__',
    '==': '__eq__',
    '!=': '__ne__',
    '<': '__lt__',
    '>': '__gt__',
    # ... additional operators
}
```

### AST Utilities

```python
def get_qualified_name(node: Node) -> str | None:
    """Get fully qualified name for a node."""
    if isinstance(node, (FuncDef, ClassDef)):
        return node.fullname
    elif isinstance(node, NameExpr):
        return node.fullname
    return None

def is_method(func_def: FuncDef) -> bool:
    """Check if function definition is a method."""
    return func_def.info is not None

def get_method_type(func_def: FuncDef) -> str:
    """Get method type (instance, class, static)."""
    if func_def.is_class_method:
        return "classmethod"
    elif func_def.is_static_method:
        return "staticmethod"
    elif func_def.is_property:
        return "property"
    else:
        return "method"
```