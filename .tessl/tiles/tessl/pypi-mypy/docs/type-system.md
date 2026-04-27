# Type System

Comprehensive type representations used by mypy's static analysis engine. This system implements Python's complete type system including generics, unions, protocols, and advanced typing features.

## Capabilities

### Base Type Classes

Foundation classes that form the hierarchy for all type representations.

```python { .api }
class Type:
    """
    Abstract base class for all type representations in mypy.
    
    All types inherit from this class and implement the visitor pattern
    for type operations and transformations.
    """

class ProperType(Type):
    """
    Base class for concrete type representations.
    
    Inherits from Type and represents fully resolved types that are
    not type aliases or other indirection types.
    """
```

### Concrete Type Classes

Core type representations for Python's type system.

```python { .api }
class AnyType(ProperType):
    """
    Represents the Any type - the top type in Python's type system.
    
    Any is compatible with all other types in both directions.
    Used when type information is unknown or when opting out of type checking.
    
    Attributes:
    - type_of_any: int - Reason for Any (missing annotation, error, etc.)
    """

class NoneType(ProperType):
    """
    Represents the type of None value.
    
    In Python's type system, None has its own special type.
    Often used in Optional[T] which is equivalent to Union[T, None].
    """

class UninhabitedType(ProperType):
    """
    Represents the bottom type (Never) - a type with no possible values.
    
    Used for functions that never return (always raise exceptions)
    and for unreachable code paths.
    """

class Instance(ProperType):
    """
    Represents instances of classes and built-in types.
    
    This is the most common type representation, covering class instances,
    built-in types like int/str/list, and generic instantiations like List[int].
    
    Attributes:
    - type: TypeInfo - Information about the class
    - args: list[Type] - Type arguments for generic types
    - erased: bool - Whether type arguments were erased
    """

class CallableType(ProperType):
    """
    Represents function and method types with complete call signatures.
    
    Includes parameter types, return type, and calling conventions.
    Supports overloads, keyword-only parameters, and varargs.
    
    Attributes:
    - arg_types: list[Type] - Parameter types
    - return_type: Type - Return type
    - arg_names: list[str | None] - Parameter names
    - arg_kinds: list[int] - Parameter kinds (positional, keyword, etc.)
    - variables: list[TypeVarLikeType] - Type variables in scope
    - is_ellipsis: bool - Whether this is Callable[..., RetType]
    """

class TupleType(ProperType):
    """
    Represents tuple types with known element types.
    
    Supports both fixed-length tuples like Tuple[int, str] and
    variable-length tuples like Tuple[int, ...].
    
    Attributes:
    - items: list[Type] - Element types
    - partial_fallback: Instance - Fallback to tuple[Any, ...]
    """

class UnionType(ProperType):
    """
    Represents union types (X | Y or Union[X, Y]).
    
    A type that can be one of several alternatives.
    Mypy automatically simplifies unions and removes duplicates.
    
    Attributes:
    - items: list[Type] - Union member types
    """

class TypedDictType(ProperType):
    """
    Represents TypedDict types - structured dictionary types.
    
    Defines dictionaries with specific required and optional keys
    and their corresponding value types.
    
    Attributes:
    - items: dict[str, Type] - Required key-value types
    - required_keys: set[str] - Required keys
    - fallback: Instance - Fallback to dict type
    """

class LiteralType(ProperType):
    """
    Represents literal value types (Literal['value']).
    
    Types restricted to specific literal values, enabling
    fine-grained type checking based on actual values.
    
    Attributes:
    - value: Any - The literal value
    - fallback: Instance - Fallback type for the value
    """

class TypeType(ProperType):
    """
    Represents Type[X] types - the type of type objects.
    
    Used for metaclass typing and when working with class objects
    rather than instances of classes.
    
    Attributes:
    - item: Type - The type being referenced
    """
```

### Type Variable Classes

Support for generic programming with type variables.

```python { .api }
class TypeVarType(Type):
    """
    Represents type variables (T, K, V, etc.) for generic programming.
    
    Type variables are placeholders that can be bound to specific types
    when generic functions or classes are instantiated.
    
    Attributes:
    - name: str - Type variable name
    - id: TypeVarId - Unique identifier
    - values: list[Type] - Allowed values (for constrained type vars)
    - upper_bound: Type - Upper bound constraint
    - variance: int - Variance (covariant, contravariant, invariant)
    """

class ParamSpecType(Type):
    """
    Represents parameter specification variables (ParamSpec).
    
    Used to preserve callable signatures in higher-order functions
    and generic callable types.
    
    Attributes:
    - name: str - ParamSpec name
    - id: TypeVarId - Unique identifier
    - upper_bound: Type - Upper bound (usually Callable)
    """

class TypeVarTupleType(Type):
    """
    Represents variadic type variables (TypeVarTuple).
    
    Used for variable-length generic type parameters,
    enabling generic types with arbitrary numbers of type arguments.
    
    Attributes:
    - name: str - TypeVarTuple name
    - id: TypeVarId - Unique identifier
    - upper_bound: Type - Upper bound constraint
    """
```

### Intermediate and Special Types

Types used during analysis and for special purposes.

```python { .api }
class TypeAliasType(Type):
    """
    Type alias to another type - supports recursive type aliases.
    
    Represents user-defined type aliases that can reference themselves
    for recursive data structures.
    
    Attributes:
    - alias: TypeAlias - The type alias definition
    - args: list[Type] - Type arguments for generic aliases
    """

class UnboundType(ProperType):
    """
    Instance type that has not been bound during semantic analysis.
    
    Used during the early phases of type checking before names
    are resolved to their actual type definitions.
    
    Attributes:
    - name: str - The unresolved type name
    - args: list[Type] - Type arguments if present
    """

class UnpackType(ProperType):
    """
    Type operator Unpack from PEP646 for TypeVarTuple unpacking.
    
    Used for unpacking TypeVarTuple in generic types, enabling
    variable-length generic parameter lists.
    
    Attributes:
    - type: Type - The type being unpacked
    """

class PartialType(ProperType):
    """
    Type with unknown type arguments during inference.
    
    Used for types like List[?] during multiphase initialization
    where type arguments are inferred later.
    
    Attributes:
    - type: Type | None - Partial type information
    - var: Var | None - Associated variable
    """

class ErasedType(ProperType):
    """
    Placeholder for an erased type during type inference.
    
    Used internally when type information has been erased
    during generic type processing.
    """

class DeletedType(ProperType):
    """
    Type of deleted variables.
    
    Variables with this type can be used as lvalues (assignment targets)
    but not as rvalues (in expressions).
    """

class EllipsisType(ProperType):
    """
    The type of ... (ellipsis) literal.
    
    Used in Callable[..., ReturnType] and other contexts where
    ellipsis has special meaning.
    """

class PlaceholderType(ProperType):
    """
    Temporary, yet-unknown type during semantic analysis.
    
    Used when there's a forward reference to a type before
    the actual type definition is encountered.
    
    Attributes:
    - fullname: str - Full name of the referenced type
    - args: list[Type] - Type arguments if present
    """

class RawExpressionType(ProperType):
    """
    Synthetic type representing arbitrary expressions.
    
    Used for expressions that don't cleanly translate into
    a specific type representation.
    
    Attributes:
    - literal_value: Any - The raw expression value
    """
```

### Advanced Type Features

Types supporting modern Python typing features.

```python { .api }
class RequiredType(Type):
    """
    Required[T] or NotRequired[T] for TypedDict fields.
    
    Used to mark TypedDict fields as required or not required,
    overriding the default requirement status.
    
    Attributes:
    - item: Type - The field type
    - required: bool - Whether the field is required
    """

class ReadOnlyType(Type):
    """
    ReadOnly[T] for TypedDict fields.
    
    Marks TypedDict fields as read-only, preventing modification
    after initialization.
    
    Attributes:
    - item: Type - The field type
    """

class TypeGuardedType(Type):
    """
    Type used internally for isinstance checks and type guards.
    
    Used by mypy's type narrowing system to track types that have
    been confirmed through runtime type checks.
    
    Attributes:
    - type_guard: Type - The guarded type
    """
```

## Type Operations

### Type Construction

```python
from mypy.types import Instance, CallableType, UnionType, TypeVarType
from mypy.nodes import TypeInfo

# Create instance type
int_type = Instance(int_typeinfo, [], line=-1)
str_type = Instance(str_typeinfo, [], line=-1)

# Create callable type
def_type = CallableType(
    arg_types=[str_type],  # Parameter types
    arg_kinds=[ARG_POS],   # Parameter kinds
    arg_names=['name'],    # Parameter names
    return_type=str_type,  # Return type
    fallback=function_type # Fallback type
)

# Create union type
optional_str = UnionType([str_type, none_type])

# Create generic type with type variables
T = TypeVarType('T', 'T', -1, [], object_type)
generic_list = Instance(list_typeinfo, [T], line=-1)
```

### Type Checking Operations

```python
from mypy.subtypes import is_subtype
from mypy.typeops import make_simplified_union
from mypy.meet import meet_types
from mypy.join import join_types

# Subtype checking
if is_subtype(child_type, parent_type):
    print("child_type is a subtype of parent_type")

# Type joins (least upper bound)
common_type = join_types(type1, type2)

# Type meets (greatest lower bound)  
narrow_type = meet_types(type1, type2)

# Union simplification
simplified = make_simplified_union([int_type, str_type, int_type])
```

### Type Visitor Pattern

```python
from mypy.visitor import TypeVisitor

class TypeCollector(TypeVisitor[list[Type]]):
    """Collect all types in a type expression."""
    
    def visit_instance(self, t: Instance) -> list[Type]:
        result = [t]
        for arg in t.args:
            result.extend(arg.accept(self))
        return result
    
    def visit_callable_type(self, t: CallableType) -> list[Type]:
        result = [t]
        for arg_type in t.arg_types:
            result.extend(arg_type.accept(self))
        result.extend(t.return_type.accept(self))
        return result
    
    def visit_union_type(self, t: UnionType) -> list[Type]:
        result = [t]
        for item in t.items:
            result.extend(item.accept(self))
        return result

# Usage
collector = TypeCollector()
all_types = some_type.accept(collector)
```

## Advanced Type System Features

### Generic Types

```python
from mypy.types import Instance, TypeVarType

# Type variable definition
T = TypeVarType('T', 'T', -1, [], object_type)
K = TypeVarType('K', 'K', -1, [], object_type)
V = TypeVarType('V', 'V', -1, [], object_type)

# Generic class instantiation
list_of_int = Instance(list_typeinfo, [int_type], line=-1)
dict_str_int = Instance(dict_typeinfo, [str_type, int_type], line=-1)

# Constrained type variables
AnyStr = TypeVarType(
    'AnyStr', 'AnyStr', -1, 
    values=[str_type, bytes_type],  # Constrained to str or bytes
    upper_bound=object_type
)
```

### Protocol Types

```python
from mypy.types import Instance
from mypy.nodes import TypeInfo

# Protocol definition (structural typing)
class ProtocolType(Instance):
    """
    Represents protocol types for structural subtyping.
    
    Protocols define interfaces based on structure rather than
    explicit inheritance, enabling duck typing with static checking.
    """
    
    def __init__(self, typeinfo: TypeInfo, args: list[Type]):
        super().__init__(typeinfo, args, line=-1)
        self.protocol = True

# Usage example
iterable_protocol = ProtocolType(iterable_typeinfo, [T])
```

### Callable Overloads

```python
from mypy.types import CallableType, Overloaded

# Multiple callable signatures
overload1 = CallableType([int_type], str_type, ...)
overload2 = CallableType([str_type], str_type, ...)

# Overloaded function type
overloaded_func = Overloaded([overload1, overload2])
```

## Type Analysis Integration

### Working with AST Nodes

```python
from mypy.nodes import FuncDef, ClassDef
from mypy.types import CallableType, Instance

def analyze_function(funcdef: FuncDef) -> CallableType:
    """Analyze function definition and create callable type."""
    arg_types = []
    arg_names = []
    arg_kinds = []
    
    for arg in funcdef.arguments:
        if arg.type_annotation:
            arg_types.append(analyze_type(arg.type_annotation))
        else:
            arg_types.append(AnyType(TypeOfAny.unannotated))
        
        arg_names.append(arg.variable.name)
        arg_kinds.append(arg.kind)
    
    return_type = analyze_type(funcdef.type) if funcdef.type else AnyType(TypeOfAny.unannotated)
    
    return CallableType(
        arg_types=arg_types,
        arg_kinds=arg_kinds,
        arg_names=arg_names,
        return_type=return_type,
        fallback=function_type
    )
```

### Type Inference

```python
from mypy.constraints import infer_constraints
from mypy.solve import solve_constraints

def infer_type_arguments(callable: CallableType, arg_types: list[Type]) -> list[Type]:
    """Infer type arguments for generic function call."""
    constraints = []
    
    # Collect constraints from arguments
    for formal, actual in zip(callable.arg_types, arg_types):
        constraints.extend(infer_constraints(formal, actual, SUPERTYPE_OF))
    
    # Solve constraints to get type variable bindings
    solution = solve_constraints(callable.variables, constraints)
    
    return [solution.get(tv.id, AnyType(TypeOfAny.from_error)) 
            for tv in callable.variables]
```

## Type System Constants

### Type Kinds and Flags

```python
# Type variable variance
COVARIANT = 1
CONTRAVARIANT = -1
INVARIANT = 0

# Argument kinds
ARG_POS = 0          # Positional argument
ARG_OPT = 1          # Optional argument (with default)
ARG_STAR = 2         # *args
ARG_STAR2 = 3        # **kwargs

# Type sources
TypeOfAny.from_error    # Any from type error
TypeOfAny.unannotated   # Missing annotation
TypeOfAny.explicit      # Explicit Any annotation
TypeOfAny.from_omitted_generics  # Unparameterized generic
```

### Built-in Type Names

```python
# Common type name constants
TUPLE_NAMES = ('builtins.tuple', 'tuple')
LIST_NAMES = ('builtins.list', 'list')  
DICT_NAMES = ('builtins.dict', 'dict')
SET_NAMES = ('builtins.set', 'set')
CALLABLE_NAMES = ('builtins.function', 'typing.Callable')
PROTOCOL_NAMES = ('typing.Protocol', 'typing_extensions.Protocol')
```