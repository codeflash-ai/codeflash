# Field Configuration

Define attributes with validation, conversion, default values, and metadata using both modern and legacy APIs. Fields provide fine-grained control over attribute behavior, serialization, and validation.

## Capabilities

### Modern Field Definition

#### Field Function
Define attributes with comprehensive configuration options for modern attrs classes.

```python { .api }
def field(
    *,
    default=NOTHING,
    factory=None,
    validator=None,
    repr=True,
    hash=None,
    init=True,
    metadata=None,
    converter=None,
    kw_only=False,
    eq=True,
    order=None,
    on_setattr=None,
    alias=None,
    type=None,
):
    """
    Define a field for attrs classes with comprehensive configuration.

    Parameters:
    - default (Any): Default value for the attribute
    - factory (callable, optional): Function to generate default values
    - validator (callable or list, optional): Validation function(s)
    - repr (bool): Include in __repr__ output (default: True)
    - hash (bool, optional): Include in __hash__ calculation
    - init (bool): Include in __init__ method (default: True)
    - metadata (dict, optional): Arbitrary metadata for the field
    - converter (callable or list, optional): Value conversion function(s)
    - kw_only (bool): Make this field keyword-only (default: False)
    - eq (bool): Include in equality comparison (default: True)
    - order (bool, optional): Include in ordering comparison
    - on_setattr (callable or list, optional): Hooks for attribute setting
    - alias (str, optional): Alternative name for the field in __init__
    - type (type, optional): Type annotation for the field

    Returns:
    Field descriptor for use in class definitions
    """
```

Usage examples:
```python
@attrs.define
class Person:
    # Simple field with default
    name: str = attrs.field()
    
    # Field with validation
    age: int = attrs.field(validator=attrs.validators.instance_of(int))
    
    # Field with factory function
    created_at: datetime = attrs.field(factory=datetime.now)
    
    # Field excluded from repr
    password: str = attrs.field(repr=False)
    
    # Keyword-only field
    debug: bool = attrs.field(default=False, kw_only=True)
    
    # Field with converter
    tags: list = attrs.field(factory=list, converter=list)
    
    # Field with metadata
    score: float = attrs.field(
        default=0.0,
        metadata={"unit": "points", "min": 0.0, "max": 100.0}
    )
```

### Legacy Field Definition

#### Attrib Function
Define attributes using the legacy API with comprehensive configuration options.

```python { .api }
def attrib(
    default=NOTHING,
    validator=None,
    repr=True,
    cmp=None,
    hash=None,
    init=True,
    metadata=None,
    converter=None,
    factory=None,
    kw_only=False,
    eq=None,
    order=None,
    on_setattr=None,
    alias=None,
    type=None,
):
    """
    Legacy field definition with comprehensive configuration.

    Parameters: Similar to field() but uses cmp instead of eq/order
    - cmp (bool, optional): Include in comparison methods (deprecated, use eq/order)
    - Other parameters same as field()

    Returns:
    Attribute descriptor for use in class definitions
    """
```

Usage examples:
```python
@attr.attrs
class Person:
    name = attr.attrib()
    age = attr.attrib(validator=attr.validators.instance_of(int))
    email = attr.attrib(default="")
    created_at = attr.attrib(factory=datetime.now)
```

### Field Aliases

Legacy aliases for backward compatibility:

```python { .api }
# In attr module
ib = attr = attrib  # Field definition aliases
```

### Factory Functions

#### Factory Class
Wrapper for factory functions that generate default values.

```python { .api }
class Factory:
    """
    Wrapper for factory functions used as default values.
    
    Attributes:
    - factory (callable): Function to generate default values
    - takes_self (bool): Whether factory function receives instance as first argument
    """
    
    def __init__(self, factory, takes_self=False):
        """
        Create a factory wrapper.
        
        Parameters:
        - factory (callable): Function to call for default values
        - takes_self (bool): Pass instance as first argument (default: False)
        """
```

Usage examples:
```python
@attrs.define
class Record:
    # Simple factory
    created_at: datetime = attrs.field(factory=datetime.now)
    
    # Factory that takes self
    id: str = attrs.field(factory=Factory(lambda self: f"{self.name}_{uuid.uuid4()}", takes_self=True))
    name: str = ""
    
    # Using Factory class directly
    data: dict = attrs.field(factory=Factory(dict))
```

### Converters

#### Converter Class
Wrapper for converter functions with additional context.

```python { .api }
class Converter:
    """
    Wrapper for converter functions with context information.
    
    Attributes:
    - converter (callable): Function to convert values
    - takes_self (bool): Whether converter receives instance as argument
    - takes_field (bool): Whether converter receives field info as argument
    """
    
    def __init__(self, converter, *, takes_self=False, takes_field=False):
        """
        Create a converter wrapper.
        
        Parameters:
        - converter (callable): Function to convert values
        - takes_self (bool): Pass instance as argument (default: False)
        - takes_field (bool): Pass field info as argument (default: False)
        """
```

Usage examples:
```python
def normalize_email(value):
    return value.lower().strip()

def context_converter(value, instance, field):
    return f"{field.name}: {value} (from {instance.__class__.__name__})"

@attrs.define
class User:
    email: str = attrs.field(converter=normalize_email)
    name: str = attrs.field(converter=Converter(context_converter, takes_self=True, takes_field=True))
```

### Special Values

#### NOTHING Constant
Sentinel value indicating the absence of a value when None is ambiguous.

```python { .api }
NOTHING: NothingType  # Sentinel for missing values

class NothingType:
    """Type for NOTHING literal value."""
    
def __bool__(self) -> False:
    """NOTHING is always falsy."""
```

Usage example:
```python
@attrs.define
class Config:
    # Distinguish between None and not set
    timeout: Optional[int] = attrs.field(default=NOTHING)
    
    def get_timeout(self) -> int:
        if self.timeout is NOTHING:
            return 30  # Default timeout
        return self.timeout
```

## Common Patterns

### Validation with Type Hints
```python
@attrs.define
class Point:
    x: float = attrs.field(validator=attrs.validators.instance_of(float))
    y: float = attrs.field(validator=attrs.validators.instance_of(float))
```

### Complex Default Values
```python
@attrs.define  
class Configuration:
    # Mutable defaults using factory
    features: list = attrs.field(factory=list)
    settings: dict = attrs.field(factory=dict)
    
    # Computed defaults
    created_at: datetime = attrs.field(factory=datetime.now)
    id: str = attrs.field(factory=lambda: str(uuid.uuid4()))
```

### Field Exclusion and Control
```python
@attrs.define
class User:
    username: str
    password: str = attrs.field(repr=False)  # Hide from repr
    internal_id: int = attrs.field(init=False, eq=False)  # Not in init or comparison
    
    def __attrs_post_init__(self):
        self.internal_id = hash(self.username)
```

### Metadata and Introspection
```python
@attrs.define
class APIField:
    value: str = attrs.field(metadata={"api_name": "fieldValue", "required": True})
    optional: str = attrs.field(default="", metadata={"api_name": "optionalField", "required": False})

# Access metadata
field_info = attrs.fields(APIField)
for field in field_info:
    if field.metadata.get("required"):
        print(f"Required field: {field.name}")
```

### Conversion Chains
```python
@attrs.define
class Document:
    # Convert string to Path object
    path: Path = attrs.field(converter=Path)
    
    # Chain converters: strip whitespace, then convert to int
    priority: int = attrs.field(
        converter=attrs.converters.pipe(str.strip, int),
        default="0"
    )
```