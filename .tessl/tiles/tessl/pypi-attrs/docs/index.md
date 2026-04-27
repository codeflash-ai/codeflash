# attrs

Classes Without Boilerplate - A Python library that brings back the joy of writing classes by relieving you from the drudgery of implementing object protocols (dunder methods). It provides declarative attribute definitions that automatically generate concise and correct code including `__repr__`, equality-checking methods, initializers, and much more without writing boilerplate code.

## Package Information

- **Package Name**: attrs
- **Language**: Python
- **Installation**: `pip install attrs`
- **License**: MIT
- **Documentation**: https://www.attrs.org/

## Core Imports

Modern API (recommended):
```python
import attrs
```

Legacy API (also available):
```python
import attr
```

Common patterns:
```python  
from attrs import define, field, validators, converters, filters
from functools import partial
```

## Basic Usage

```python
import attrs

@attrs.define
class Person:
    name: str
    age: int = attrs.field(validator=attrs.validators.instance_of(int))
    email: str = ""

# Automatically gets __init__, __repr__, __eq__, etc.
person = Person("Alice", 30, "alice@example.com")
print(person)  # Person(name='Alice', age=30, email='alice@example.com')

# Create modified copy
older_person = attrs.evolve(person, age=31)
print(older_person)  # Person(name='Alice', age=31, email='alice@example.com')

# Convert to dict
person_dict = attrs.asdict(person)
print(person_dict)  # {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}
```

## Architecture

attrs provides both modern (`attrs`) and legacy (`attr`) APIs with the same underlying functionality:

- **Class Decorators**: Transform regular classes into feature-rich data classes
- **Field Definitions**: Declarative attribute configuration with validation, conversion, and metadata
- **Automatic Methods**: Generate `__init__`, `__repr__`, `__eq__`, `__hash__`, `__lt__`, etc.
- **Extensibility**: Custom validators, converters, and setters for complex behaviors
- **Introspection**: Runtime access to class metadata and field information

The library is designed for maximum reusability, supports both modern type-annotated and classic APIs, and maintains high performance without runtime penalties.

## Capabilities

### Class Definition and Decoration

Core decorators for creating attrs classes with automatic method generation, including modern type-annotated and legacy approaches.

```python { .api }
def define(
    maybe_cls=None,
    *,
    these=None,
    repr=None,
    unsafe_hash=None,
    hash=None,
    init=None,
    slots=True,
    frozen=False,
    weakref_slot=True,
    str=False,
    auto_attribs=None,
    kw_only=False,
    cache_hash=False,
    auto_exc=True,
    eq=None,
    order=False,
    auto_detect=True,
    getstate_setstate=None,
    on_setattr=None,
    field_transformer=None,
    match_args=True,
): ...

# frozen is a partial function of define with frozen=True, on_setattr=None
frozen = partial(define, frozen=True, on_setattr=None)

def attrs(
    maybe_cls=None,
    these=None,
    repr_ns=None,
    repr=None,
    cmp=None,
    hash=None,
    init=None,
    slots=False,
    frozen=False,
    weakref_slot=True,
    str=False,
    auto_attribs=False,
    kw_only=False,
    cache_hash=False,
    auto_exc=False,
    eq=None,
    order=None,
    auto_detect=False,
    collect_by_mro=True,
    getstate_setstate=None,
    on_setattr=None,
    field_transformer=None,
    match_args=True,
    unsafe_hash=None,
): ...
```

[Class Definition](./class-definition.md)

### Field Definition and Configuration

Define attributes with validation, conversion, default values, and metadata using both modern and legacy APIs.

```python { .api }
def field(
    *,
    default=NOTHING,
    validator=None,
    repr=True,
    hash=None,
    init=True,
    metadata=None,
    type=None,
    converter=None,
    factory=None,
    kw_only=False,
    eq=None,
    order=None,
    on_setattr=None,
    alias=None,
): ...

def attrib(
    default=NOTHING,
    validator=None,
    repr=True,
    cmp=None,
    hash=None,
    init=True,
    metadata=None,
    type=None,
    converter=None,
    factory=None,
    kw_only=False,
    eq=None,
    order=None,
    on_setattr=None,
    alias=None,
    *,
    # Legacy parameters
    convert=None,
): ...
```

[Field Configuration](./field-configuration.md)

### Validation and Conversion

Comprehensive validation and conversion system with built-in validators, converters, and combinators for complex data processing.

```python { .api }
# Validators
def instance_of(type): ...
def optional(validator): ...
def in_(options): ...

# Converters  
def optional(converter): ...
def default_if_none(default): ...
def pipe(*converters): ...
```

[Validation and Conversion](./validation-conversion.md)

### Utilities and Introspection

Functions for working with attrs classes including serialization, introspection, and instance manipulation.

```python { .api }
def asdict(inst, *, recurse=True, filter=None, **kwargs): ...
def astuple(inst, *, recurse=True, filter=None): ...
def fields(cls): ...
def fields_dict(cls): ...
def has(cls): ...
def evolve(*args, **changes): ...
def assoc(inst, **changes): ...  # Deprecated, use evolve instead
def make_class(name, attrs, bases=(), **attrs_kwargs): ...
def resolve_types(cls, globalns=None, localns=None, attribs=None, include_extras=True): ...
def validate(inst): ...
```

[Utilities](./utilities.md)

### Filters

Functions for creating filters used with `asdict()` and `astuple()` to control which attributes are included or excluded during serialization.

```python { .api }
def include(*what): ...  # Create filter that only allows specified types/names/attributes
def exclude(*what): ...  # Create filter that excludes specified types/names/attributes
```

### Configuration and Error Handling

Configuration options and comprehensive exception hierarchy for error handling in attrs applications.

```python { .api }
# Exceptions
class NotAnAttrsClassError(ValueError): ...
class FrozenInstanceError(AttributeError): ...
class AttrsAttributeNotFoundError(ValueError): ...

# Configuration  
def set_run_validators(run: bool): ...  # Deprecated
def get_run_validators(): ...  # Deprecated
```

[Configuration and Errors](./configuration-errors.md)

## Constants and Types

```python { .api }
# Constants
NOTHING: NothingType  # Sentinel for missing values

# Core Classes
class Attribute:
    name: str
    default: Any
    validator: Optional[Callable]
    converter: Optional[Callable]
    # ... additional properties

class Factory:
    factory: Callable
    takes_self: bool = False

class Converter:
    converter: Callable
    takes_self: bool = False
    takes_field: bool = False

# Type Definitions
AttrsInstance = Protocol  # Protocol for attrs instances
NothingType = Literal[NOTHING]  # Type for NOTHING sentinel
```