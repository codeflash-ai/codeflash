# CST Node Types

LibCST provides a comprehensive set of immutable dataclasses representing every element of Python syntax. These nodes preserve all formatting details including comments, whitespace, and punctuation, enabling lossless round-trip transformations.

## Capabilities

### Base Node Types

Foundation classes for all CST nodes.

```python { .api }
class CSTNode:
    """Base class for all CST nodes."""
    
    def visit(self, visitor: CSTVisitor) -> CSTNode:
        """
        Visit this node with a visitor.
        
        Parameters:
        - visitor: Visitor to apply
        
        Returns:
        CSTNode: Potentially transformed node
        """
    
    def with_changes(self, **changes: Any) -> CSTNode:
        """
        Create a copy of this node with specified changes.
        
        Parameters:
        - changes: Field values to change
        
        Returns:
        CSTNode: New node with changes applied
        """
    
    @property
    def code(self) -> str:
        """Generate source code from this node."""

class BaseExpression(CSTNode):
    """Base class for all expression nodes."""

class BaseStatement(CSTNode):
    """Base class for all statement nodes."""

class BaseCompoundStatement(BaseStatement):
    """Base class for compound statements (if, for, while, etc.)."""

class BaseSmallStatement(BaseStatement):
    """Base class for simple statements (assign, return, etc.)."""
```

### Module Structure

Root node and organizational elements.

```python { .api }
class Module(CSTNode):
    """Root node representing a complete Python module."""
    body: Sequence[Union[SimpleStatementLine, BaseCompoundStatement]]
    header: Sequence[Union[SimpleStatementLine, BaseCompoundStatement]]
    footer: Sequence[Union[SimpleStatementLine, BaseCompoundStatement]]
    encoding: str
    default_indent: str
    default_newline: str
    has_trailing_newline: bool

class SimpleStatementLine(BaseStatement):
    """Line containing one or more simple statements."""
    body: Sequence[BaseSmallStatement]
    leading_lines: Sequence[EmptyLine]
    trailing_whitespace: TrailingWhitespace

class SimpleStatementSuite(BaseSuite):
    """Suite of simple statements."""
    body: Sequence[SimpleStatementLine]

class IndentedBlock(BaseSuite):
    """Indented block of statements."""
    body: Sequence[Union[SimpleStatementLine, BaseCompoundStatement]]
    indent: str
    whitespace_before_colon: SimpleWhitespace
```

### Expression Nodes

Nodes representing Python expressions.

```python { .api }
# Literals
class Name(BaseExpression):
    """Variable names and identifiers."""
    value: str
    lpar: Sequence[LeftParen]
    rpar: Sequence[RightParen]

class Integer(BaseNumber):
    """Integer literals."""
    value: str

class Float(BaseNumber):
    """Float literals."""
    value: str

class Imaginary(BaseNumber):
    """Complex number literals."""
    value: str

class SimpleString(BaseString):
    """String literals."""
    value: str
    quote: str

class ConcatenatedString(BaseString):
    """Adjacent string concatenation."""
    left: BaseString
    right: BaseString

class FormattedString(BaseString):
    """f-string literals."""
    parts: Sequence[BaseFormattedStringContent]
    start: str
    end: str

# Collections
class List(BaseList):
    """List literals [1, 2, 3]."""
    elements: Sequence[Element]
    lbracket: LeftSquareBracket
    rbracket: RightSquareBracket

class Tuple(BaseExpression):
    """Tuple literals (1, 2, 3)."""
    elements: Sequence[Element]
    lpar: Sequence[LeftParen]
    rpar: Sequence[RightParen]

class Set(BaseSet):
    """Set literals {1, 2, 3}."""
    elements: Sequence[Element]
    lbrace: LeftCurlyBrace
    rbrace: RightCurlyBrace

class Dict(BaseDict):
    """Dictionary literals {key: value}."""
    elements: Sequence[BaseDictElement]
    lbrace: LeftCurlyBrace
    rbrace: RightCurlyBrace

# Operations
class BinaryOperation(BaseExpression):
    """Binary operations (+, -, *, etc.)."""
    left: BaseExpression
    operator: BaseBinaryOp
    right: BaseExpression

class UnaryOperation(BaseExpression):
    """Unary operations (-, +, ~, not)."""
    operator: BaseUnaryOp
    expression: BaseExpression

class BooleanOperation(BaseExpression):
    """Boolean operations (and, or)."""
    left: BaseExpression
    operator: BaseBooleanOp
    right: BaseExpression

class Comparison(BaseExpression):
    """Comparison operations."""
    left: BaseExpression
    comparisons: Sequence[ComparisonTarget]

# Function calls and access
class Call(BaseExpression):
    """Function calls."""
    func: BaseExpression
    args: Sequence[Arg]
    lpar: Sequence[LeftParen]
    rpar: Sequence[RightParen]

class Attribute(BaseExpression):
    """Attribute access obj.attr."""
    value: BaseExpression
    attr: Name
    dot: Dot

class Subscript(BaseExpression):
    """Subscription operations obj[key]."""
    value: BaseExpression
    slice: Sequence[SubscriptElement]
    lbracket: LeftSquareBracket
    rbracket: RightSquareBracket

# Comprehensions
class ListComp(BaseExpression):
    """List comprehensions [x for x in iterable]."""
    elt: BaseExpression
    for_in: CompFor

class SetComp(BaseExpression):
    """Set comprehensions {x for x in iterable}."""
    elt: BaseExpression
    for_in: CompFor

class DictComp(BaseExpression):
    """Dict comprehensions {k: v for k, v in iterable}."""
    key: BaseExpression
    value: BaseExpression 
    for_in: CompFor

class GeneratorExp(BaseExpression):
    """Generator expressions (x for x in iterable)."""
    elt: BaseExpression
    for_in: CompFor

# Advanced expressions
class IfExp(BaseExpression):
    """Conditional expressions (x if test else y)."""
    test: BaseExpression
    body: BaseExpression
    orelse: BaseExpression

class NamedExpr(BaseExpression):
    """Named expressions/walrus operator (x := y)."""
    target: Name
    value: BaseExpression

class Yield(BaseExpression):
    """Yield expressions."""
    value: Optional[BaseExpression]

class YieldFrom(BaseExpression):
    """Yield from expressions."""
    value: BaseExpression

class Await(BaseExpression):
    """Await expressions."""
    expression: BaseExpression

class Lambda(BaseExpression):
    """Lambda expressions."""
    params: Parameters
    body: BaseExpression

# Literals and constants
class Ellipsis(BaseExpression):
    """Ellipsis literal (...)."""
```

### Statement Nodes

Nodes representing Python statements.

```python { .api }
# Assignments
class Assign(BaseSmallStatement):
    """Assignment statements (x = y)."""
    targets: Sequence[AssignTarget]
    value: BaseExpression

class AnnAssign(BaseSmallStatement):
    """Annotated assignments (x: int = y)."""
    target: BaseAssignTargetExpression
    annotation: Annotation
    value: Optional[BaseExpression]

class AugAssign(BaseSmallStatement):
    """Augmented assignments (x += y)."""
    target: BaseAssignTargetExpression
    operator: BaseAugOp
    value: BaseExpression

# Function and class definitions
class FunctionDef(BaseCompoundStatement):
    """Function definitions."""
    name: Name
    params: Parameters
    body: BaseSuite
    decorators: Sequence[Decorator]
    returns: Optional[Annotation]
    asynchronous: Optional[Asynchronous]

class ClassDef(BaseCompoundStatement):
    """Class definitions."""
    name: Name
    args: Sequence[Arg]
    body: BaseSuite
    decorators: Sequence[Decorator]
    bases: Sequence[Arg]
    keywords: Sequence[Arg]

# Control flow
class If(BaseCompoundStatement):
    """If statements."""
    test: BaseExpression
    body: BaseSuite
    orelse: Optional[Union[If, Else]]

class For(BaseCompoundStatement):
    """For loops."""
    target: BaseAssignTargetExpression
    iter: BaseExpression
    body: BaseSuite
    orelse: Optional[Else]
    asynchronous: Optional[Asynchronous]

class While(BaseCompoundStatement):
    """While loops."""
    test: BaseExpression
    body: BaseSuite
    orelse: Optional[Else]

class Try(BaseCompoundStatement):
    """Try statements."""
    body: BaseSuite
    handlers: Sequence[ExceptHandler]
    orelse: Optional[Else]
    finalbody: Optional[Finally]

# Exception handling
class ExceptHandler(CSTNode):
    """Except clauses."""
    type: Optional[BaseExpression]
    name: Optional[AsName]
    body: BaseSuite

class Raise(BaseSmallStatement):
    """Raise statements."""
    exc: Optional[BaseExpression]
    cause: Optional[BaseExpression]

# Imports
class Import(BaseSmallStatement):
    """Import statements."""
    names: Sequence[ImportAlias]

class ImportFrom(BaseSmallStatement):
    """From-import statements."""
    module: Optional[Union[Attribute, Name]]
    names: Union[ImportStar, Sequence[ImportAlias]]
    relative: Sequence[Dot]

class ImportAlias(CSTNode):
    """Import aliases (as clauses)."""
    name: Union[Attribute, Name]
    asname: Optional[AsName]

# Pattern matching (Python 3.10+)
class Match(BaseCompoundStatement):
    """Match statements."""
    subject: BaseExpression
    cases: Sequence[MatchCase]

class MatchCase(CSTNode):
    """Match case clauses."""
    pattern: MatchPattern
    guard: Optional[BaseExpression]
    body: BaseSuite

class MatchAs(MatchPattern):
    """As patterns in match statements."""
    pattern: Optional[MatchPattern]
    name: Optional[Name]

class MatchClass(MatchPattern):
    """Class patterns in match statements."""
    cls: BaseExpression
    patterns: Sequence[MatchPattern]
    kwds: Sequence[MatchKeywordElement]

class MatchKeywordElement(CSTNode):
    """Keyword elements in class patterns."""
    key: Name
    pattern: MatchPattern

class MatchList(MatchPattern):
    """List patterns in match statements."""
    patterns: Sequence[MatchSequenceElement]

class MatchMapping(MatchPattern):
    """Mapping patterns in match statements."""
    elements: Sequence[MatchMappingElement]
    rest: Optional[Name]

class MatchMappingElement(CSTNode):
    """Mapping elements in match patterns."""
    key: BaseExpression
    pattern: MatchPattern

class MatchOr(MatchPattern):
    """Or patterns in match statements."""
    patterns: Sequence[MatchOrElement]

class MatchOrElement(CSTNode):
    """Elements in or patterns."""
    pattern: MatchPattern

class MatchSequence(MatchPattern):
    """Sequence patterns in match statements."""
    patterns: Sequence[MatchSequenceElement]

class MatchSequenceElement(CSTNode):
    """Elements in sequence patterns."""
    pattern: MatchPattern

class MatchSingleton(MatchPattern):
    """Singleton patterns (None, True, False)."""
    value: BaseExpression

class MatchStar(MatchPattern):
    """Star patterns in match statements."""
    name: Optional[Name]

class MatchTuple(MatchPattern):
    """Tuple patterns in match statements."""
    patterns: Sequence[MatchSequenceElement]

class MatchValue(MatchPattern):
    """Value patterns in match statements."""
    value: BaseExpression

class MatchPattern(CSTNode):
    """Base class for match patterns."""
```

### Operator Nodes

Nodes representing Python operators.

```python { .api }
# Arithmetic operators
class Add(BaseBinaryOp):
    """+ operator."""

class Subtract(BaseBinaryOp):
    """- operator."""

class Multiply(BaseBinaryOp):
    """* operator."""

class Divide(BaseBinaryOp):
    """/ operator."""

class FloorDivide(BaseBinaryOp):
    """// operator."""

class Modulo(BaseBinaryOp):
    """% operator."""

class Power(BaseBinaryOp):
    """** operator."""

class MatrixMultiply(BaseBinaryOp):
    """@ operator."""

# Augmented assignment operators
class AddAssign(BaseAugOp):
    """+= operator."""

class SubtractAssign(BaseAugOp):
    """-= operator."""

class MultiplyAssign(BaseAugOp):
    """*= operator."""

class DivideAssign(BaseAugOp):
    """/= operator."""

class FloorDivideAssign(BaseAugOp):
    """//= operator."""

class ModuloAssign(BaseAugOp):
    """%= operator."""

class PowerAssign(BaseAugOp):
    """**= operator."""

class MatrixMultiplyAssign(BaseAugOp):
    """@= operator."""

class LeftShiftAssign(BaseAugOp):
    """<<= operator."""

class RightShiftAssign(BaseAugOp):
    """>>= operator."""

class BitAndAssign(BaseAugOp):
    """&= operator."""

class BitOrAssign(BaseAugOp):
    """|= operator."""

class BitXorAssign(BaseAugOp):
    """^= operator."""

# Comparison operators  
class Equal(BaseCompOp):
    """== operator."""

class NotEqual(BaseCompOp):
    """!= operator."""

class LessThan(BaseCompOp):
    """< operator."""

class LessThanEqual(BaseCompOp):
    """<= operator."""

class GreaterThan(BaseCompOp):
    """> operator."""

class GreaterThanEqual(BaseCompOp):
    """>= operator."""

class Is(BaseCompOp):
    """is operator."""

class IsNot(BaseCompOp):
    """is not operator."""

class In(BaseCompOp):
    """in operator."""

class NotIn(BaseCompOp):
    """not in operator."""

# Boolean operators
class And(BaseBooleanOp):
    """and operator."""

class Or(BaseBooleanOp):  
    """or operator."""

class Not(BaseUnaryOp):
    """not operator."""

# Bitwise operators
class BitAnd(BaseBinaryOp):
    """& operator."""

class BitOr(BaseBinaryOp):
    """| operator."""

class BitXor(BaseBinaryOp):
    """^ operator."""

class LeftShift(BaseBinaryOp):
    """<< operator."""

class RightShift(BaseBinaryOp):
    """>> operator."""

class BitInvert(BaseUnaryOp):
    """~ operator."""
```

### Whitespace and Formatting

Nodes preserving formatting details.

```python { .api }
class SimpleWhitespace(CSTNode):
    """Spaces and tabs."""
    value: str

class Comment(CSTNode):
    """Code comments."""
    value: str

class Newline(CSTNode):
    """Line breaks."""
    indent: Optional[str]
    whitespace_before_newline: SimpleWhitespace
    comment: Optional[Comment]

class EmptyLine(CSTNode):
    """Empty lines with whitespace."""
    indent: Optional[str]
    whitespace: SimpleWhitespace
    comment: Optional[Comment]

class TrailingWhitespace(CSTNode):
    """Whitespace at end of lines."""
    whitespace: SimpleWhitespace
    comment: Optional[Comment]

class ParenthesizedWhitespace(CSTNode):
    """Whitespace in parentheses."""
    first_line: TrailingWhitespace
    empty_lines: Sequence[EmptyLine]
    indent: Optional[str]
    whitespace_before_newline: SimpleWhitespace
    comment: Optional[Comment]
```

## Usage Examples

### Node Creation

```python
import libcst as cst

# Create nodes programmatically
name_node = cst.Name("variable")
integer_node = cst.Integer("42")
string_node = cst.SimpleString('"hello"')

# Create complex expressions
binary_op = cst.BinaryOperation(
    left=cst.Name("x"),
    operator=cst.Add(),
    right=cst.Integer("1")
)

# Create function call
call_node = cst.Call(
    func=cst.Name("print"),
    args=[cst.Arg(value=cst.SimpleString('"hello"'))]
)
```

### Node Inspection

```python
import libcst as cst

source = '''
def example(x, y=10):
    """Example function."""
    return x + y
'''

module = cst.parse_module(source)
func_def = module.body[0].body[0]  # Get the function definition

print(f"Function name: {func_def.name.value}")
print(f"Parameter count: {len(func_def.params.params)}")
print(f"Has docstring: {isinstance(func_def.body.body[0].body[0], cst.Expr)}")
print(f"Return statement: {isinstance(func_def.body.body[1].body[0], cst.Return)}")
```

### Node Modification

```python
import libcst as cst

# Parse original code
module = cst.parse_module("x = 42")
assign = module.body[0].body[0]

# Modify the assignment value
new_assign = assign.with_changes(value=cst.Integer("100"))

# Create new module with modified assignment
new_module = module.with_changes(
    body=[cst.SimpleStatementLine(body=[new_assign])]
)

print(new_module.code)  # "x = 100"
```

### Pattern Matching with Nodes

```python
import libcst as cst
import libcst.matchers as m

source = '''
def process():
    x = [1, 2, 3]
    y = {"a": 1, "b": 2}
    return x, y
'''

module = cst.parse_module(source)

# Find all list literals
lists = m.findall(module, cst.List())
print(f"Found {len(lists)} list literals")

# Find all dictionary literals  
dicts = m.findall(module, cst.Dict())
print(f"Found {len(dicts)} dictionary literals")

# Find assignments to list literals
list_assignments = m.findall(module, cst.Assign(value=cst.List()))
print(f"Found {len(list_assignments)} assignments to lists")
```

## Types

```python { .api }
# Base node hierarchy
class CSTNode:
    """Base class for all CST nodes."""

class BaseExpression(CSTNode):
    """Base for expression nodes."""

class BaseStatement(CSTNode):
    """Base for statement nodes."""

class BaseCompoundStatement(BaseStatement):
    """Base for compound statements."""

class BaseSmallStatement(BaseStatement):
    """Base for simple statements."""

# Expression base classes
class BaseAssignTargetExpression(BaseExpression):
    """Base for assignment target expressions."""

class BaseDelTargetExpression(BaseExpression):
    """Base for deletion target expressions."""

class BaseNumber(BaseExpression):
    """Base for numeric literals."""

class BaseString(BaseExpression):
    """Base for string literals."""

class BaseDict(BaseExpression):
    """Base for dict-like containers."""

class BaseList(BaseExpression):
    """Base for list-like containers."""

class BaseSet(BaseExpression):
    """Base for set-like containers."""

# Operator base classes
class BaseBinaryOp(CSTNode):
    """Base for binary operators."""

class BaseUnaryOp(CSTNode):
    """Base for unary operators."""

class BaseBooleanOp(CSTNode):
    """Base for boolean operators."""

class BaseCompOp(CSTNode):
    """Base for comparison operators."""

class BaseAugOp(CSTNode):  
    """Base for augmented assignment operators."""

# Suite types
class BaseSuite(CSTNode):
    """Base for statement suites."""

# Exception types
class CSTValidationError(Exception):
    """Raised when CST node validation fails."""

# Type variables
CSTNodeT = TypeVar("CSTNodeT", bound=CSTNode)
```