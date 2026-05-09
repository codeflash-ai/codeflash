# Dataclasses and Type Adapters

Integration with Python dataclasses and standalone type validation without model inheritance.

## Capabilities

### Pydantic Dataclasses

Enhanced dataclasses with pydantic validation, providing dataclass syntax with pydantic's validation capabilities.

```python { .api }
def dataclass(*, init=True, repr=True, eq=True, order=False, unsafe_hash=False,
              frozen=False, config=None, validate_on_init=None, use_enum_values=None,
              kw_only=False, slots=False):
    """
    Decorator to create pydantic dataclass with validation.
    
    Args:
        init (bool): Generate __init__ method
        repr (bool): Generate __repr__ method
        eq (bool): Generate __eq__ method
        order (bool): Generate comparison methods
        unsafe_hash (bool): Generate __hash__ method
        frozen (bool): Make instances immutable
        config: Pydantic configuration
        validate_on_init (bool): Validate during initialization
        use_enum_values (bool): Use enum values instead of instances
        kw_only (bool): Make all fields keyword-only
        slots (bool): Generate __slots__
        
    Returns:
        Decorated dataclass with pydantic validation
    """

class Field:
    """
    Field definition for pydantic dataclasses.
    
    Similar to pydantic.Field but for dataclass fields.
    """
    
    def __init__(self, default=dataclasses.MISSING, *, default_factory=dataclasses.MISSING,
                 init=True, repr=True, hash=None, compare=True, metadata=None, **kwargs):
        """
        Initialize dataclass field.
        
        Args:
            default: Default value
            default_factory: Factory for default values
            init (bool): Include in __init__
            repr (bool): Include in __repr__
            hash (bool): Include in __hash__
            compare (bool): Include in comparison methods
            metadata (dict): Field metadata
            **kwargs: Additional pydantic field options
        """
```

### Type Adapters

Standalone validation for any type without requiring model inheritance, useful for validating individual values or complex types.

```python { .api }
class TypeAdapter(Generic[T]):
    """
    Type adapter for validating and serializing any type.
    
    Provides pydantic validation for types without BaseModel inheritance.
    """
    
    def __init__(self, type_: type, *, config=None, _root=True):
        """
        Initialize type adapter.
        
        Args:
            type_: Type to adapt
            config: Validation configuration
            _root (bool): Whether this is a root type adapter
        """
    
    def validate_python(self, obj, /, *, strict=None, from_attributes=None, context=None):
        """
        Validate Python object against the type.
        
        Args:
            obj: Object to validate
            strict (bool): Enable strict validation
            from_attributes (bool): Extract data from object attributes
            context (dict): Validation context
            
        Returns:
            Validated object of the specified type
            
        Raises:
            ValidationError: If validation fails
        """
    
    def validate_json(self, json_data, /, *, strict=None, context=None):
        """
        Validate JSON string against the type.
        
        Args:
            json_data (str | bytes): JSON data to validate
            strict (bool): Enable strict validation
            context (dict): Validation context
            
        Returns:
            Validated object of the specified type
            
        Raises:
            ValidationError: If validation fails
        """
    
    def validate_strings(self, obj, /, *, strict=None, context=None):
        """
        Validate with string inputs against the type.
        
        Args:
            obj: Object to validate
            strict (bool): Enable strict validation
            context (dict): Validation context
            
        Returns:
            Validated object of the specified type
            
        Raises:
            ValidationError: If validation fails
        """
    
    def dump_python(self, instance, /, *, mode='python', include=None, exclude=None,
                    context=None, by_alias=False, exclude_unset=False, exclude_defaults=False,
                    exclude_none=False, round_trip=False, warnings=True, serialize_as_any=False):
        """
        Serialize instance to Python object.
        
        Args:
            instance: Instance to serialize
            mode (str): Serialization mode
            include: Fields to include
            exclude: Fields to exclude
            context (dict): Serialization context
            by_alias (bool): Use field aliases
            exclude_unset (bool): Exclude unset fields
            exclude_defaults (bool): Exclude default values
            exclude_none (bool): Exclude None values
            round_trip (bool): Enable round-trip serialization
            warnings (bool): Show serialization warnings
            serialize_as_any (bool): Serialize using Any serializer
            
        Returns:
            Serialized Python object
        """
    
    def dump_json(self, instance, /, *, indent=None, include=None, exclude=None,
                  context=None, by_alias=False, exclude_unset=False, exclude_defaults=False,
                  exclude_none=False, round_trip=False, warnings=True, serialize_as_any=False):
        """
        Serialize instance to JSON string.
        
        Args:
            instance: Instance to serialize
            indent (int): JSON indentation
            include: Fields to include
            exclude: Fields to exclude
            context (dict): Serialization context
            by_alias (bool): Use field aliases
            exclude_unset (bool): Exclude unset fields
            exclude_defaults (bool): Exclude default values
            exclude_none (bool): Exclude None values
            round_trip (bool): Enable round-trip serialization
            warnings (bool): Show serialization warnings
            serialize_as_any (bool): Serialize using Any serializer
            
        Returns:
            str: JSON string
        """
    
    def json_schema(self, *, by_alias=True, ref_template='#/$defs/{model}'):
        """
        Generate JSON schema for the type.
        
        Args:
            by_alias (bool): Use field aliases in schema
            ref_template (str): Template for schema references
            
        Returns:
            dict: JSON schema
        """
    
    @property
    def core_schema(self):
        """dict: Core schema for the type"""

    @property
    def validator(self):
        """Validator: Core validator instance"""

    @property
    def serializer(self):
        """Serializer: Core serializer instance"""
```

### Legacy Functions

Legacy functions for backward compatibility with pydantic v1.

```python { .api }
def parse_obj_as(type_, obj):
    """
    Parse object as specified type (legacy function).
    
    Args:
        type_: Type to parse as
        obj: Object to parse
        
    Returns:
        Parsed object
        
    Note:
        Deprecated: Use TypeAdapter.validate_python() instead
    """

def schema_of(type_, *, title='Generated schema'):
    """
    Generate schema for type (legacy function).
    
    Args:
        type_: Type to generate schema for
        title (str): Schema title
        
    Returns:
        dict: Type schema
        
    Note:
        Deprecated: Use TypeAdapter.json_schema() instead
    """
```

## Usage Examples

### Pydantic Dataclasses

```python
from pydantic.dataclasses import dataclass, Field
from typing import Optional
from datetime import datetime

@dataclass
class User:
    id: int
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: Optional[int] = Field(None, ge=0, le=150)
    created_at: datetime = Field(default_factory=datetime.now)
    active: bool = True

# Usage like regular dataclass with validation
user = User(
    id=123,
    name="John Doe",
    email="john@example.com",
    age=30
)

print(user.name)     # "John Doe"
print(user.active)   # True

# Validation errors are raised for invalid data
try:
    invalid_user = User(id=123, name="", email="invalid")
except ValueError as e:
    print(f"Validation error: {e}")
```

### Dataclass Configuration

```python
from pydantic.dataclasses import dataclass
from pydantic import ConfigDict

@dataclass(config=ConfigDict(str_strip_whitespace=True, frozen=True))
class ImmutableUser:
    name: str
    email: str

# Whitespace is stripped, object is immutable
user = ImmutableUser(name="  John  ", email="john@example.com")
print(user.name)  # "John"

# This would raise an error - object is frozen
# user.name = "Jane"
```

### TypeAdapter for Simple Types

```python
from pydantic import TypeAdapter, ValidationError
from typing import List

# Create adapter for list of integers
list_adapter = TypeAdapter(List[int])

# Validate Python objects
valid_list = list_adapter.validate_python([1, 2, 3, "4"])  # "4" converted to 4
print(valid_list)  # [1, 2, 3, 4]

# Validate JSON
json_list = list_adapter.validate_json('[1, 2, 3, 4]')
print(json_list)  # [1, 2, 3, 4]

# Handle validation errors
try:
    invalid_list = list_adapter.validate_python([1, 2, "invalid"])
except ValidationError as e:
    print(f"Validation failed: {e}")

# Serialize back to JSON
json_output = list_adapter.dump_json([1, 2, 3, 4])
print(json_output)  # '[1,2,3,4]'
```

### TypeAdapter for Complex Types

```python
from pydantic import TypeAdapter
from typing import Dict, List, Optional
from datetime import datetime

# Complex nested type
ComplexType = Dict[str, List[Dict[str, Optional[datetime]]]]

adapter = TypeAdapter(ComplexType)

# Validate complex data structure
data = {
    "events": [
        {"timestamp": "2023-12-25T10:30:00", "name": None},
        {"timestamp": "2023-12-26T15:45:00", "name": "2023-12-26T16:00:00"}
    ],
    "logs": [
        {"created": "2023-12-25T08:00:00", "level": None}
    ]
}

validated_data = adapter.validate_python(data)
print(type(validated_data["events"][0]["timestamp"]))  # <class 'datetime.datetime'>

# Generate JSON schema
schema = adapter.json_schema()
print(schema)  # Complete JSON schema for the complex type
```

### TypeAdapter with Custom Types

```python
from pydantic import TypeAdapter, field_validator
from typing import Annotated

def validate_positive(v):
    if v <= 0:
        raise ValueError("Must be positive")
    return v

# Create adapter for annotated type
PositiveInt = Annotated[int, field_validator(validate_positive)]
adapter = TypeAdapter(PositiveInt)

# Validate with custom logic
valid_value = adapter.validate_python(42)  # OK
print(valid_value)  # 42

try:
    invalid_value = adapter.validate_python(-5)  # Raises ValidationError
except ValidationError as e:
    print(f"Custom validation failed: {e}")
```

### Integration with Existing Classes

```python
from pydantic import TypeAdapter
from dataclasses import dataclass as stdlib_dataclass
from typing import List

# Regular Python dataclass (not pydantic)
@stdlib_dataclass
class Point:
    x: float
    y: float

# Use TypeAdapter to add validation
PointList = List[Point]
adapter = TypeAdapter(PointList)

# Validate list of points
points_data = [
    {"x": 1.0, "y": 2.0},
    {"x": 3.5, "y": 4.2}
]

validated_points = adapter.validate_python(points_data)
print(validated_points)  # [Point(x=1.0, y=2.0), Point(x=3.5, y=4.2)]
print(type(validated_points[0]))  # <class '__main__.Point'>
```

### Legacy Functions

```python
from pydantic import parse_obj_as, schema_of, ValidationError
from typing import List, Dict

# Legacy parsing (use TypeAdapter instead in new code)
data = [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}]
UserDict = Dict[str, str | int]

try:
    parsed = parse_obj_as(List[UserDict], data)
    print(parsed)  # [{'name': 'John', 'age': 30}, {'name': 'Jane', 'age': 25}]
except ValidationError as e:
    print(f"Parsing failed: {e}")

# Legacy schema generation (use TypeAdapter instead in new code)
schema = schema_of(List[UserDict], title="User List Schema")
print(schema)  # JSON schema for List[Dict[str, str | int]]
```

### Dataclass with Validation

```python
from pydantic.dataclasses import dataclass
from pydantic import field_validator, model_validator
from typing import Optional

@dataclass
class Rectangle:
    width: float
    height: float
    name: Optional[str] = None
    
    @field_validator('width', 'height')
    @classmethod
    def validate_dimensions(cls, v):
        if v <= 0:
            raise ValueError('Dimensions must be positive')
        return v
    
    @model_validator(mode='after')
    def validate_aspect_ratio(self):
        if self.width / self.height > 10 or self.height / self.width > 10:
            raise ValueError('Aspect ratio too extreme')
        return self
    
    @property
    def area(self):
        return self.width * self.height

# Usage with validation
rect = Rectangle(width=5.0, height=3.0, name="My Rectangle")
print(f"Area: {rect.area}")  # Area: 15.0

# Validation error for invalid dimensions
try:
    invalid_rect = Rectangle(width=-1.0, height=3.0)
except ValueError as e:
    print(f"Validation error: {e}")
```