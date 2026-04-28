# Utilities and Introspection

Functions for working with attrs classes including serialization, introspection, instance manipulation, and runtime type resolution. These utilities enable dynamic interaction with attrs classes and instances.

## Capabilities

### Serialization

#### Dictionary Conversion

Convert attrs instances to dictionaries with comprehensive customization options.

```python { .api }
def asdict(
    inst,
    recurse=True,
    filter=None,
    dict_factory=dict,
    retain_collection_types=False,
    value_serializer=None,
):
    """
    Convert attrs instance to dictionary.
    
    Parameters:
    - inst: Attrs instance to convert
    - recurse (bool): Recursively convert nested attrs instances (default: True)
    - filter (callable, optional): Function to filter attributes (attr, value) -> bool
    - dict_factory (callable): Factory for creating dictionaries (default: dict)
    - retain_collection_types (bool): Keep original collection types vs converting to list
    - value_serializer (callable, optional): Hook to serialize individual values
    
    Returns:
    Dictionary representation of the instance
    
    Raises:
    NotAnAttrsClassError: If inst is not an attrs instance
    """
```

Usage examples:
```python
@attrs.define
class Person:
    name: str
    age: int
    email: str = ""

person = Person("Alice", 30, "alice@example.com")

# Basic conversion
data = attrs.asdict(person)
# {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}

# With filter to exclude empty strings
data = attrs.asdict(person, filter=lambda attr, value: value != "")
# {'name': 'Alice', 'age': 30}

# Using OrderedDict
from collections import OrderedDict
data = attrs.asdict(person, dict_factory=OrderedDict)

# Custom value serialization
def serialize_dates(inst, field, value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value

data = attrs.asdict(person, value_serializer=serialize_dates)
```

#### Tuple Conversion

Convert attrs instances to tuples.

```python { .api }
def astuple(
    inst,
    recurse=True,
    filter=None,
):
    """
    Convert attrs instance to tuple.
    
    Parameters:
    - inst: Attrs instance to convert
    - recurse (bool): Recursively convert nested attrs instances (default: True)
    - filter (callable, optional): Function to filter attributes (attr, value) -> bool
    
    Returns:
    Tuple representation of the instance in field definition order
    
    Raises:
    NotAnAttrsClassError: If inst is not an attrs instance
    """
```

Usage example:
```python
person = Person("Alice", 30, "alice@example.com")
data = attrs.astuple(person)
# ('Alice', 30, 'alice@example.com')

# With filter
data = attrs.astuple(person, filter=lambda attr, value: attr.name != 'email')
# ('Alice', 30)
```

### Introspection

#### Class Inspection

Examine attrs classes and their field definitions.

```python { .api }
def has(cls):
    """
    Check if class is an attrs class.
    
    Parameters:
    - cls: Class to check
    
    Returns:
    bool: True if class was decorated with attrs, False otherwise
    """

def fields(cls):
    """
    Get field information for attrs class.
    
    Parameters:
    - cls: Attrs class to inspect
    
    Returns:
    tuple[Attribute, ...]: Tuple of Attribute objects for the class
    
    Raises:
    NotAnAttrsClassError: If cls is not an attrs class
    """

def fields_dict(cls):
    """
    Get field information as ordered dictionary.
    
    Parameters:
    - cls: Attrs class to inspect
    
    Returns:
    dict[str, Attribute]: Ordered dict mapping field names to Attribute objects
    
    Raises:
    NotAnAttrsClassError: If cls is not an attrs class
    """
```

Usage examples:
```python
@attrs.define
class Person:
    name: str
    age: int = 0

# Check if attrs class
print(attrs.has(Person))  # True
print(attrs.has(dict))    # False

# Get fields
field_tuple = attrs.fields(Person)
print(field_tuple[0].name)  # 'name'
print(field_tuple[1].default)  # 0

# Get fields as dict
field_dict = attrs.fields_dict(Person)
print(field_dict['name'].type)  # <class 'str'>
print(field_dict['age'].default)  # 0
```

#### Attribute Information

The `Attribute` class provides detailed information about each field.

```python { .api }
class Attribute:
    """
    Immutable representation of an attrs attribute.
    
    Read-only properties:
    - name (str): Name of the attribute
    - default: Default value (NOTHING if no default)
    - factory: Factory function for default values
    - validator: Validation function or list of functions
    - converter: Conversion function or list of functions
    - type: Type annotation
    - kw_only (bool): Whether attribute is keyword-only
    - eq (bool): Whether included in equality comparison
    - order (bool): Whether included in ordering comparison
    - hash (bool): Whether included in hash calculation
    - init (bool): Whether included in __init__
    - repr (bool): Whether included in __repr__
    - metadata (dict): Arbitrary metadata dictionary
    - on_setattr: Setter functions for attribute changes
    - alias (str): Alternative name for __init__ parameter
    """
```

Usage example:
```python
@attrs.define
class Config:
    name: str = attrs.field(metadata={"required": True})
    debug: bool = attrs.field(default=False, repr=False)

for field in attrs.fields(Config):
    print(f"Field: {field.name}")
    print(f"  Type: {field.type}")
    print(f"  Default: {field.default}")
    print(f"  Metadata: {field.metadata}")
    print(f"  In repr: {field.repr}")
```

### Instance Manipulation

#### Instance Evolution

Create modified copies of attrs instances.

```python { .api }
def evolve(*args, **changes):
    """
    Create new instance with specified changes.
    
    Parameters:
    - *args: Instance to evolve (can be positional or keyword)
    - **changes: Field values to change
    
    Returns:
    New instance of same type with specified changes
    
    Raises:
    AttrsAttributeNotFoundError: If change refers to non-existent field
    TypeError: If instance is not an attrs instance
    """
```

Usage examples:
```python
@attrs.define
class Person:
    name: str
    age: int
    email: str = ""

person = Person("Alice", 30, "alice@example.com")

# Create older version
older_person = attrs.evolve(person, age=31)
print(older_person)  # Person(name='Alice', age=31, email='alice@example.com')

# Multiple changes
updated_person = attrs.evolve(person, age=25, email="alice.new@example.com")
```

#### Legacy Association (Deprecated)

```python { .api }
def assoc(inst, **changes):
    """
    Create new instance with changes (deprecated - use evolve instead).
    
    Parameters:
    - inst: Instance to modify
    - **changes: Field values to change
    
    Returns:
    New instance with specified changes
    """
```

### Validation and Type Resolution

#### Instance Validation

Validate all attributes on an instance.

```python { .api }
def validate(inst):
    """
    Validate all attributes on an attrs instance.
    
    Runs all validators defined on the instance's fields.
    
    Parameters:
    - inst: Attrs instance to validate
    
    Raises:
    Various validation errors depending on validators
    NotAnAttrsClassError: If inst is not an attrs instance
    """
```

Usage example:
```python
@attrs.define
class Person:
    name: str = attrs.field(validator=attrs.validators.instance_of(str))
    age: int = attrs.field(validator=attrs.validators.instance_of(int))

person = Person("Alice", 30)
attrs.validate(person)  # Passes

# This would raise validation error:
# person.age = "thirty"
# attrs.validate(person)
```

#### Type Resolution

Resolve string type annotations to actual types.

```python { .api }
def resolve_types(cls, globalns=None, localns=None, attribs=None, include_extras=True):
    """
    Resolve string type annotations to actual types.
    
    Parameters:
    - cls: Attrs class with string type annotations
    - globalns (dict, optional): Global namespace for type resolution
    - localns (dict, optional): Local namespace for type resolution  
    - attribs (list, optional): Specific attributes to resolve
    - include_extras (bool): Include typing_extensions types (default: True)
    
    Returns:
    Attrs class with resolved type annotations
    """
```

Usage example:
```python
# Forward reference scenario
@attrs.define
class Node:
    value: int
    parent: "Optional[Node]" = None  # Forward reference
    children: "List[Node]" = attrs.field(factory=list)

# Resolve forward references
Node = attrs.resolve_types(Node, globalns=globals())
```

### Filtering

Create filters for use with `asdict` and `astuple`.

```python { .api }
def include(*what):
    """
    Create filter that includes only specified items.
    
    Parameters:
    - *what: Types, attribute names, or Attribute objects to include
    
    Returns:
    Filter function for use with asdict/astuple
    """

def exclude(*what):
    """
    Create filter that excludes specified items.
    
    Parameters:
    - *what: Types, attribute names, or Attribute objects to exclude
    
    Returns:
    Filter function for use with asdict/astuple
    """
```

Usage examples:
```python
@attrs.define
class Person:
    name: str
    age: int
    email: str = ""
    password: str = attrs.field(repr=False)

person = Person("Alice", 30, "alice@example.com", "secret123")

# Include only specific fields
data = attrs.asdict(person, filter=attrs.filters.include("name", "age"))
# {'name': 'Alice', 'age': 30}

# Exclude sensitive fields
data = attrs.asdict(person, filter=attrs.filters.exclude("password"))
# {'name': 'Alice', 'age': 30, 'email': 'alice@example.com'}

# Exclude by attribute object
password_field = attrs.fields_dict(Person)["password"]
data = attrs.asdict(person, filter=attrs.filters.exclude(password_field))
```

### Comparison Utilities

#### Custom Comparison

Create classes with custom comparison behavior.

```python { .api }
def cmp_using(
    eq=None,
    lt=None,
    le=None,
    gt=None,
    ge=None,
    require_same_type=True,
    class_name="Comparable",
):
    """
    Create class with custom comparison methods.
    
    Parameters:
    - eq (callable, optional): Equality comparison function
    - lt (callable, optional): Less-than comparison function  
    - le (callable, optional): Less-equal comparison function
    - gt (callable, optional): Greater-than comparison function
    - ge (callable, optional): Greater-equal comparison function
    - require_same_type (bool): Require same type for comparisons
    - class_name (str): Name for the generated class
    
    Returns:
    Class with custom comparison behavior
    """
```

Usage example:
```python
@attrs.define
class Version:
    major: int
    minor: int
    patch: int
    
    # Custom comparison based on semantic versioning
    def _cmp_key(self):
        return (self.major, self.minor, self.patch)

# Create comparable version class
ComparableVersion = attrs.cmp_using(
    eq=lambda self, other: self._cmp_key() == other._cmp_key(),
    lt=lambda self, other: self._cmp_key() < other._cmp_key(),
    class_name="ComparableVersion"
)

# Use in attrs class
@attrs.define
class Package(ComparableVersion):
    name: str
    version: Version
```

## Common Patterns

### Serialization with Custom Logic
```python
@attrs.define
class TimestampedData:
    data: dict
    created_at: datetime = attrs.field(factory=datetime.now)

def serialize_timestamps(inst, field, value):
    if isinstance(value, datetime):
        return value.isoformat()
    return value

# Serialize with timestamp formatting
data = attrs.asdict(
    instance,
    value_serializer=serialize_timestamps
)
```

### Dynamic Field Access
```python
def get_field_info(cls, field_name):
    """Get information about a specific field."""
    if not attrs.has(cls):
        raise ValueError("Not an attrs class")
    
    field_dict = attrs.fields_dict(cls)
    if field_name not in field_dict:
        raise ValueError(f"Field {field_name} not found")
    
    field = field_dict[field_name]
    return {
        "name": field.name,
        "type": field.type,
        "default": field.default,
        "has_validator": field.validator is not None,
        "has_converter": field.converter is not None,
    }
```

### Conditional Serialization
```python
def serialize_for_api(instance, include_private=False):
    """Serialize instance for API with privacy controls."""
    def api_filter(attr, value):
        # Exclude private fields unless requested
        if not include_private and attr.name.startswith('_'):
            return False
        # Exclude None values
        if value is None:
            return False
        return True
    
    return attrs.asdict(instance, filter=api_filter)
```

### Bulk Operations with Evolution
```python
def update_all_ages(people, age_increment):
    """Update ages for multiple people efficiently."""
    return [
        attrs.evolve(person, age=person.age + age_increment)
        for person in people
    ]

def merge_configs(base_config, updates):
    """Merge configuration updates into base config."""
    return attrs.evolve(base_config, **updates)
```