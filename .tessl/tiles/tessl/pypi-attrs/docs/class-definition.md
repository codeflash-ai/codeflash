# Class Definition and Decoration

Core decorators for creating attrs classes with automatic method generation. attrs provides both modern type-annotated and legacy APIs for maximum flexibility and compatibility.

## Capabilities

### Modern Class Decorators

#### Define Classes
Create classes with automatic method generation using type annotations and modern defaults.

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
):
    """
    Create a class with attrs features using modern defaults.

    Parameters:
    - these (dict, optional): Dictionary of attributes to add
    - repr (bool, optional): Generate __repr__ method
    - hash (bool, optional): Generate __hash__ method
    - init (bool, optional): Generate __init__ method
    - slots (bool): Use __slots__ for memory efficiency (default: True)
    - frozen (bool): Make instances immutable (default: False)
    - auto_attribs (bool, optional): Automatically detect attributes from type annotations
    - kw_only (bool): Make all attributes keyword-only
    - eq (bool, optional): Generate __eq__ and __ne__ methods
    - order (bool): Generate ordering methods (__lt__, __le__, __gt__, __ge__)
    - auto_detect (bool): Automatically determine method generation based on existing methods
    - on_setattr (callable or list, optional): Hook(s) to run on attribute change

    Returns:
    Class or decorator function
    """
```

Usage example:
```python
@attrs.define
class Point:
    x: float
    y: float = 0.0
    
    def distance_from_origin(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5
```

#### Frozen Classes
Create immutable classes with automatic method generation.

```python { .api }
def frozen(
    maybe_cls=None,
    *,
    these=None,
    repr=None,
    unsafe_hash=None,
    hash=None,
    init=None,
    slots=True,
    frozen=True,
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
):
    """
    Create an immutable class with attrs features.

    Similar to define() but with frozen=True by default and on_setattr=None.
    Instances cannot be modified after creation.

    Parameters: Same as define() but frozen=True by default

    Returns:
    Immutable class or decorator function
    """
```

Usage example:
```python
@attrs.frozen
class ImmutablePoint:
    x: float
    y: float
    
# point.x = 5  # Would raise FrozenInstanceError
```

#### Mutable Classes
Alias for define() for explicit clarity when working with both mutable and frozen classes.

```python { .api }
mutable = define  # Alias for explicit clarity
```

### Legacy Class Decorators

#### attrs Decorator
Legacy class decorator with traditional defaults for backward compatibility.

```python { .api }
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
    cache_hash=False,
    auto_attribs=False,
    kw_only=False,
    auto_exc=False,
    eq=None,
    order=None,
    auto_detect=False,
    getstate_setstate=None,
    on_setattr=None,
    field_transformer=None,
    match_args=True,
):
    """
    Legacy class decorator with traditional defaults.

    Parameters: Similar to define() but with different defaults:
    - slots=False (vs True in define)
    - auto_attribs=False (vs None/True in define)
    - auto_detect=False (vs True in define)

    Returns:
    Class or decorator function
    """
```

Usage example:
```python
@attr.attrs
class LegacyPoint:
    x = attr.attrib()
    y = attr.attrib(default=0.0)
```

### Dynamic Class Creation

#### Make Class
Dynamically create attrs classes at runtime.

```python { .api }
def make_class(
    name,
    attrs,
    bases=(object,),
    **attributes_arguments
):
    """
    Dynamically create a new attrs class.

    Parameters:
    - name (str): Name of the new class
    - attrs (dict or list): Attributes to add to the class
    - bases (tuple): Base classes
    - **attributes_arguments: Arguments passed to attrs decorator

    Returns:
    New attrs class
    """
```

Usage example:
```python
Point = attrs.make_class("Point", ["x", "y"])
point = Point(1, 2)

# With more configuration
Person = attrs.make_class(
    "Person", 
    {
        "name": attrs.field(),
        "age": attrs.field(validator=attrs.validators.instance_of(int))
    },
    frozen=True
)
```

### Aliases and Convenience Functions

Legacy aliases for backward compatibility:

```python { .api }
# In attr module
s = attributes = attrs  # Class decorator aliases
dataclass = functools.partial(attrs, auto_attribs=True)  # Dataclass-like interface
```

Usage example:
```python
# Using aliases
@attr.s
class OldStyle:
    x = attr.ib()

@attr.dataclass  
class DataclassStyle:
    x: int
    y: str = "default"
```

## Common Patterns

### Type Annotations with Define
```python
@attrs.define
class User:
    name: str
    age: int
    email: str = ""
    is_active: bool = True
```

### Legacy Style with Attrib
```python  
@attr.attrs
class User:
    name = attr.attrib()
    age = attr.attrib()
    email = attr.attrib(default="")
    is_active = attr.attrib(default=True)
```

### Mixed Approach
```python
@attrs.define
class Config:
    debug: bool = False
    timeout: int = attrs.field(default=30, validator=attrs.validators.instance_of(int))
```

### Inheritance
```python
@attrs.define
class Animal:
    name: str
    species: str

@attrs.define  
class Dog(Animal):
    breed: str
    species: str = "Canis lupus"  # Override default
```