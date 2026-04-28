# Serialization and Configuration

Computed fields, serialization customization, and model configuration options for controlling validation and serialization behavior.

## Capabilities

### Computed Fields

Create fields that are computed from other fields or model state, useful for derived values and dynamic properties.

```python { .api }
def computed_field(*, alias=None, alias_priority=None, title=None, description=None,
                   examples=None, exclude=None, discriminator=None, json_schema_extra=None,
                   frozen=None, validate_default=None, repr=True, return_type=PydanticUndefined):
    """
    Decorator for computed field properties.
    
    Args:
        alias: Alias for the field name
        alias_priority (int): Priority for alias resolution
        title (str): Human-readable title
        description (str): Field description
        examples: Example values
        exclude: Whether to exclude from serialization
        discriminator: Discriminator for union types
        json_schema_extra: Extra JSON schema properties
        frozen (bool): Whether field is frozen after initialization
        validate_default (bool): Validate default values
        repr (bool): Include in repr output
        return_type: Return type annotation
        
    Returns:
        Decorator function
    """

@computed_field
@property
def computed_property(self) -> ReturnType:
    """
    Template for computed field property.
    
    Returns:
        Computed value
    """
```

### Model Configuration

Configuration options that control model behavior, validation, and serialization.

```python { .api }
class ConfigDict(TypedDict, total=False):
    """
    Configuration dictionary for pydantic models.
    
    Can be used as model_config class attribute.
    """
    
    # Validation configuration
    strict: bool  # Enable strict validation mode
    extra: str  # Handle extra fields ('ignore', 'allow', 'forbid')
    frozen: bool  # Make model immutable after creation
    populate_by_name: bool  # Allow field population by field name and alias
    use_enum_values: bool  # Use enum values instead of enum instances
    validate_assignment: bool  # Validate field assignments after creation
    arbitrary_types_allowed: bool  # Allow arbitrary types in fields
    from_attributes: bool  # Allow model creation from object attributes
    
    # Serialization configuration
    ser_json_timedelta: str  # How to serialize timedelta ('iso8601', 'float')
    ser_json_bytes: str  # How to serialize bytes ('utf8', 'base64')
    ser_json_inf_nan: str  # How to serialize inf/nan ('null', 'constants')
    hide_input_in_errors: bool  # Hide input data in validation errors
    
    # String handling
    str_to_lower: bool  # Convert strings to lowercase
    str_to_upper: bool  # Convert strings to uppercase  
    str_strip_whitespace: bool  # Strip whitespace from strings
    
    # JSON schema configuration
    title: str  # Schema title
    json_schema_extra: dict  # Extra JSON schema properties
    json_encoders: dict  # Custom JSON encoders (deprecated)
    
    # Deprecated/legacy options
    validate_default: bool  # Validate default values
    defer_build: bool  # Defer model building
```

### Serialization Decorators

Decorators for customizing field and model serialization behavior.

```python { .api }
def field_serializer(*fields, mode='wrap', when_used='json-unless-none', check_fields=None):
    """
    Decorator for custom field serialization.
    
    Args:
        *fields (str): Field names to apply serializer to
        mode (str): Serialization mode ('wrap', 'plain', 'before', 'after')
        when_used (str): When to use serializer ('json', 'json-unless-none', 'always')
        check_fields (bool): Whether to check if fields exist
        
    Returns:
        Decorator function
    """

def model_serializer(mode='wrap', when_used='json-unless-none'):
    """
    Decorator for custom model serialization.
    
    Args:
        mode (str): Serialization mode ('wrap', 'plain')
        when_used (str): When to use serializer ('json', 'json-unless-none', 'always')
        
    Returns:
        Decorator function
    """

class PlainSerializer:
    """
    Serializer that completely replaces default serialization.
    """
    
    def __init__(self, func, *, return_type=PydanticUndefined, when_used='json-unless-none'):
        """
        Initialize serializer.
        
        Args:
            func: Serialization function
            return_type: Return type annotation
            when_used (str): When to use serializer
        """

class WrapSerializer:
    """
    Serializer that wraps default serialization.
    """
    
    def __init__(self, func, *, return_type=PydanticUndefined, when_used='json-unless-none'):
        """
        Initialize serializer.
        
        Args:
            func: Serialization function
            return_type: Return type annotation
            when_used (str): When to use serializer
        """

class BeforeSerializer:
    """
    Serializer that runs before default serialization.
    """
    
    def __init__(self, func, *, when_used='json-unless-none'):
        """
        Initialize serializer.
        
        Args:
            func: Serialization function
            when_used (str): When to use serializer
        """

class AfterSerializer:
    """
    Serializer that runs after default serialization.
    """
    
    def __init__(self, func, *, return_type=PydanticUndefined, when_used='json-unless-none'):
        """
        Initialize serializer.
        
        Args:
            func: Serialization function
            return_type: Return type annotation
            when_used (str): When to use serializer
        """
```

### Field and Alias Configuration

Advanced field configuration including aliases and serialization control.

```python { .api }
class AliasGenerator:
    """Base class for alias generators."""
    
    def generate_alias(self, field_name: str) -> str:
        """
        Generate alias for field name.
        
        Args:
            field_name (str): Original field name
            
        Returns:
            str: Generated alias
        """

def alias_generator(func):
    """
    Create alias generator from function.
    
    Args:
        func: Function that takes field name and returns alias
        
    Returns:
        AliasGenerator instance
    """

class AliasChoices:
    """
    Multiple alias choices for a field.
    """
    
    def __init__(self, *choices):
        """
        Initialize with alias choices.
        
        Args:
            *choices: Alias options
        """

class AliasPath:
    """
    Path-based alias for nested data extraction.
    """
    
    def __init__(self, *path):
        """
        Initialize with path components.
        
        Args:
            *path: Path components for nested access
        """

### Alias Generator Functions

Built-in functions for common alias generation patterns.

```python { .api }
def to_pascal(snake_str):
    """
    Convert snake_case string to PascalCase.
    
    Args:
        snake_str (str): String in snake_case format
        
    Returns:
        str: String in PascalCase format
        
    Example:
        to_pascal('user_name') -> 'UserName'
    """

def to_camel(snake_str):
    """
    Convert snake_case string to camelCase.
    
    Args:
        snake_str (str): String in snake_case format
        
    Returns:
        str: String in camelCase format
        
    Example:
        to_camel('user_name') -> 'userName'
    """

def to_snake(camel_str):
    """
    Convert PascalCase or camelCase string to snake_case.
    
    Args:
        camel_str (str): String in PascalCase or camelCase format
        
    Returns:
        str: String in snake_case format
        
    Example:
        to_snake('UserName') -> 'user_name'
        to_snake('userName') -> 'user_name'
    """
```

### Core Schema Classes

Core classes from pydantic-core that are part of the serialization API.

```python { .api }
class SerializationInfo:
    """
    Information available during serialization.
    """
    
    @property
    def include(self):
        """set | dict | None: Fields to include"""
    
    @property
    def exclude(self):
        """set | dict | None: Fields to exclude"""
    
    @property
    def context(self):
        """dict | None: Serialization context"""
    
    @property
    def mode(self):
        """str: Serialization mode"""
    
    @property
    def by_alias(self):
        """bool: Whether to use aliases"""

class FieldSerializationInfo:
    """
    Information available during field serialization.
    """
    
    @property
    def field_name(self):
        """str: Field name"""
    
    @property
    def by_alias(self):
        """bool: Whether to use aliases"""

class SerializerFunctionWrapHandler:
    """
    Handler for wrap serializers.
    """
    
    def __call__(self, value):
        """
        Call the wrapped serializer.
        
        Args:
            value: Value to serialize
            
        Returns:
            Serialized value
        """
```

## Usage Examples

### Computed Fields

```python
from pydantic import BaseModel, computed_field
from typing import Optional

class Person(BaseModel):
    first_name: str
    last_name: str
    birth_year: int
    
    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
    
    @computed_field
    @property
    def age(self) -> int:
        from datetime import date
        return date.today().year - self.birth_year

# Usage
person = Person(
    first_name="John",
    last_name="Doe", 
    birth_year=1990
)

print(person.full_name)  # "John Doe"
print(person.age)        # Current age
print(person.model_dump())  # Includes computed fields
```

### Model Configuration

```python
from pydantic import BaseModel, ConfigDict

class StrictModel(BaseModel):
    model_config = ConfigDict(
        strict=True,           # Strict validation
        extra='forbid',        # Forbid extra fields
        frozen=True,           # Immutable after creation
        validate_assignment=True,  # Validate assignments
        str_strip_whitespace=True  # Strip string whitespace
    )
    
    name: str
    value: int

# Usage
model = StrictModel(name="  test  ", value=42)
print(model.name)  # "test" (whitespace stripped)

# This would raise ValidationError due to strict mode
# model = StrictModel(name="test", value="42")  # string instead of int

# This would raise ValidationError due to extra='forbid'
# model = StrictModel(name="test", value=42, extra_field="not allowed")
```

### Field Serializers

```python
from pydantic import BaseModel, field_serializer
from datetime import datetime
from typing import Optional

class Event(BaseModel):
    name: str
    timestamp: datetime
    metadata: Optional[dict] = None
    
    @field_serializer('timestamp')
    def serialize_timestamp(self, value: datetime) -> str:
        return value.strftime('%Y-%m-%d %H:%M:%S UTC')
    
    @field_serializer('metadata', when_used='json')
    def serialize_metadata(self, value: Optional[dict]) -> Optional[str]:
        if value is None:
            return None
        import json
        return json.dumps(value, sort_keys=True)

# Usage
event = Event(
    name="Conference",
    timestamp=datetime(2023, 12, 25, 10, 30),
    metadata={"location": "New York", "capacity": 100}
)

print(event.model_dump_json())
# Custom serialization applied to timestamp and metadata
```

### Model Serializers

```python
from pydantic import BaseModel, model_serializer

class APIResponse(BaseModel):
    success: bool
    data: dict
    message: str
    
    @model_serializer
    def serialize_model(self):
        # Custom serialization logic
        result = {
            'status': 'ok' if self.success else 'error',
            'payload': self.data,
            'info': self.message
        }
        
        # Add timestamp
        from datetime import datetime
        result['timestamp'] = datetime.utcnow().isoformat()
        
        return result

# Usage
response = APIResponse(
    success=True,
    data={'user_id': 123, 'name': 'John'},
    message='User retrieved successfully'
)

print(response.model_dump())
# Uses custom serialization format
```

### Alias Configuration

```python
from pydantic import BaseModel, Field, AliasPath, AliasChoices

class UserData(BaseModel):
    user_id: int = Field(alias='id')
    full_name: str = Field(alias=AliasChoices('fullName', 'full_name', 'name'))
    address: str = Field(alias=AliasPath('location', 'address'))
    
    class Config:
        populate_by_name = True  # Allow both field name and alias

# Usage with different alias formats
data1 = {'id': 123, 'fullName': 'John Doe', 'location': {'address': '123 Main St'}}
data2 = {'user_id': 456, 'full_name': 'Jane Smith', 'location': {'address': '456 Oak Ave'}}

user1 = UserData(**data1)
user2 = UserData(**data2)

print(user1.model_dump(by_alias=True))  # Uses aliases in output
```

### Functional Serializers with Annotated

```python
from pydantic import BaseModel, PlainSerializer, field_serializer
from typing import Annotated
from decimal import Decimal

def money_serializer(value: Decimal) -> str:
    """Serialize decimal as currency string."""
    return f"${value:.2f}"

class Invoice(BaseModel):
    amount: Annotated[Decimal, PlainSerializer(money_serializer, when_used='json')]
    tax: Decimal
    
    @field_serializer('tax', when_used='json')
    def serialize_tax(self, value: Decimal) -> str:
        return f"${value:.2f}"

# Usage
invoice = Invoice(amount=Decimal('100.50'), tax=Decimal('8.25'))
print(invoice.model_dump_json())
# {"amount": "$100.50", "tax": "$8.25"}
```

### Configuration Inheritance

```python
from pydantic import BaseModel, ConfigDict

class BaseConfig(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        frozen=True
    )

class APIModel(BaseConfig):
    model_config = ConfigDict(
        # Inherits from BaseConfig and adds:
        extra='forbid',
        alias_generator=lambda field_name: field_name.replace('_', '-')
    )
    
    user_id: int
    user_name: str

# Usage
model = APIModel(user_id=123, **{'user-name': '  John  '})
print(model.user_name)  # "John" (whitespace stripped)
print(model.model_dump(by_alias=True))  # {'user-id': 123, 'user-name': 'John'}
```

### Alias Generator Functions Usage

```python
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel, to_pascal, to_snake

class APIModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel)
    
    user_id: int
    user_name: str
    created_at: str

# Field names are automatically converted to camelCase
data = {'userId': 123, 'userName': 'John', 'createdAt': '2023-12-25T10:30:00Z'}
model = APIModel(**data)

print(model.model_dump(by_alias=True))
# {'userId': 123, 'userName': 'John', 'createdAt': '2023-12-25T10:30:00Z'}

# Manual alias generation
class CustomModel(BaseModel):
    snake_case_field: str = Field(alias=to_pascal('snake_case_field'))  # 'SnakeCaseField'
    another_field: int = Field(alias=to_camel('another_field'))         # 'anotherField'

# Convert between naming conventions
original_name = 'user_profile_data'
camel_name = to_camel(original_name)      # 'userProfileData'
pascal_name = to_pascal(original_name)    # 'UserProfileData'
back_to_snake = to_snake(pascal_name)     # 'user_profile_data'

print(f"Original: {original_name}")
print(f"Camel: {camel_name}")
print(f"Pascal: {pascal_name}")
print(f"Back to snake: {back_to_snake}")
```