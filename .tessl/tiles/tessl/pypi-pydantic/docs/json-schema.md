# JSON Schema Generation

JSON schema generation capabilities for creating OpenAPI-compatible schemas from pydantic models and types, enabling automatic API documentation and validation.

## Capabilities

### Schema Generation Functions

Functions for generating JSON schemas from models and types.

```python { .api }
def model_json_schema(
    cls, by_alias=True, ref_template='#/$defs/{model}', schema_generator=None
):
    """
    Generate JSON schema for a model class.
    
    Args:
        by_alias (bool): Use field aliases in schema
        ref_template (str): Template for schema references
        schema_generator: Custom schema generator class
        
    Returns:
        dict: JSON schema dictionary
    """

def models_json_schema(models, *, by_alias=True, title='Generated schema', 
                      description=None, ref_template='#/$defs/{model}'):
    """
    Generate JSON schema for multiple model classes.
    
    Args:
        models: Sequence of model classes or tuples of (model, mode)
        by_alias (bool): Use field aliases in schema
        title (str): Schema title
        description (str): Schema description  
        ref_template (str): Template for schema references
        
    Returns:
        dict: Combined JSON schema dictionary
    """
```

### Schema Generator Class

Core class for customizing JSON schema generation behavior.

```python { .api }
class GenerateJsonSchema:
    """
    JSON schema generator with customizable behavior.
    
    Can be subclassed to customize schema generation for specific needs.
    """
    
    def __init__(self, by_alias=True, ref_template='#/$defs/{model}'):
        """
        Initialize schema generator.
        
        Args:
            by_alias (bool): Use field aliases in schema
            ref_template (str): Template for schema references
        """
    
    def generate_schema(self, schema):
        """
        Generate JSON schema from core schema.
        
        Args:
            schema: Core schema to convert
            
        Returns:
            dict: JSON schema dictionary
        """
    
    def generate_field_schema(self, schema, validation_alias, serialization_alias):
        """
        Generate JSON schema for a field.
        
        Args:
            schema: Field core schema
            validation_alias: Field validation alias
            serialization_alias: Field serialization alias
            
        Returns:
            dict: Field JSON schema
        """
    
    def generate_definitions(self, definitions):
        """
        Generate schema definitions.
        
        Args:
            definitions: Dictionary of definitions
            
        Returns:
            dict: Generated definitions
        """
```

### Schema Annotations

Classes for customizing JSON schema generation with annotations.

```python { .api }
class WithJsonSchema:
    """
    Annotation for providing custom JSON schema for a type.
    
    Used with Annotated to override default schema generation.
    """
    
    def __init__(self, json_schema, *, mode='validation'):
        """
        Initialize with custom JSON schema.
        
        Args:
            json_schema: Custom JSON schema (dict or callable)
            mode (str): When to apply ('validation', 'serialization', 'both')
        """

def SkipJsonSchema(inner_type):
    """
    Skip JSON schema generation for a type.
    
    Args:
        inner_type: Type to skip schema generation for
        
    Returns:
        Annotated type that skips JSON schema generation
    """
```

### Schema Utilities

Utility functions and classes for JSON schema operations.

```python { .api }
class JsonSchemaValue:
    """
    Represents a JSON schema value with mode information.
    """
    
    def __init__(self, value, *, mode='both'):
        """
        Initialize JSON schema value.
        
        Args:
            value: Schema value
            mode (str): Application mode ('validation', 'serialization', 'both')
        """

def field_json_schema(field_info, *, by_alias=True, validation_alias=None, 
                     serialization_alias=None, schema_generator=None):
    """
    Generate JSON schema for a field.
    
    Args:
        field_info: Field information object
        by_alias (bool): Use field aliases
        validation_alias: Validation alias override
        serialization_alias: Serialization alias override
        schema_generator: Custom schema generator
        
    Returns:
        dict: Field JSON schema
    """

class PydanticJsonSchemaWarning(UserWarning):
    """
    Warning raised during JSON schema generation.
    """
```

### Type Adapters Schema Generation

JSON schema generation for TypeAdapters.

```python { .api }
class TypeAdapter:
    def json_schema(self, *, by_alias=True, ref_template='#/$defs/{model}'):
        """
        Generate JSON schema for the adapted type.
        
        Args:
            by_alias (bool): Use field aliases in schema
            ref_template (str): Template for schema references
            
        Returns:
            dict: JSON schema for the type
        """
```

## Usage Examples

### Basic Model Schema Generation

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class User(BaseModel):
    id: int
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: Optional[int] = Field(None, ge=0, le=150)
    tags: List[str] = Field(default_factory=list)

# Generate JSON schema
schema = User.model_json_schema()
print(schema)

# Output includes:
# {
#   "type": "object",
#   "properties": {
#     "id": {"type": "integer"},
#     "name": {"type": "string", "minLength": 1, "maxLength": 100},
#     "email": {"type": "string", "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"},
#     "age": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 150}, {"type": "null"}]},
#     "tags": {"type": "array", "items": {"type": "string"}, "default": []}
#   },
#   "required": ["id", "name", "email"]
# }
```

### Multiple Models Schema

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    email: str

class Post(BaseModel):
    title: str
    content: str
    author: User

# Generate combined schema
from pydantic import models_json_schema

schema = models_json_schema(
    [User, Post],
    title="Blog API Schema",
    description="Schema for blog users and posts"
)

print(schema)
# Includes both User and Post definitions with proper references
```

### Custom Schema Generation

```python
from pydantic import BaseModel, GenerateJsonSchema

class CustomSchemaGenerator(GenerateJsonSchema):
    def generate_field_schema(self, schema, validation_alias, serialization_alias):
        field_schema = super().generate_field_schema(schema, validation_alias, serialization_alias)
        
        # Add custom properties
        if schema.get('type') == 'string':
            field_schema['x-custom-string'] = True
            
        return field_schema

class User(BaseModel):
    name: str
    email: str

# Use custom generator
schema = User.model_json_schema(schema_generator=CustomSchemaGenerator)
print(schema)
# String fields will have 'x-custom-string': True
```

### Schema Annotations

```python
from pydantic import BaseModel, Field
from pydantic.json_schema import WithJsonSchema
from typing import Annotated

def custom_password_schema(schema, model_type):
    """Custom schema for password fields."""
    schema.update({
        'type': 'string',
        'format': 'password',
        'writeOnly': True,
        'minLength': 8
    })
    return schema

class User(BaseModel):
    username: str
    password: Annotated[str, WithJsonSchema(custom_password_schema)]
    
    class Config:
        json_schema_extra = {
            'examples': [
                {
                    'username': 'johndoe',
                    'password': 'secretpassword'
                }
            ]
        }

schema = User.model_json_schema()
print(schema['properties']['password'])
# {'type': 'string', 'format': 'password', 'writeOnly': True, 'minLength': 8}
```

### TypeAdapter Schema Generation

```python
from pydantic import TypeAdapter
from typing import Dict, List, Optional

# Complex nested type
UserData = Dict[str, List[Optional[int]]]

adapter = TypeAdapter(UserData)
schema = adapter.json_schema()

print(schema)
# {
#   "type": "object",
#   "additionalProperties": {
#     "type": "array", 
#     "items": {"anyOf": [{"type": "integer"}, {"type": "null"}]}
#   }
# }
```

### OpenAPI Integration

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class UserCreate(BaseModel):
    """Schema for creating a new user."""
    
    name: str = Field(..., description="User's full name", example="John Doe")
    email: str = Field(..., description="User's email address", example="john@example.com")
    age: Optional[int] = Field(None, description="User's age", ge=0, le=150, example=30)

class UserResponse(BaseModel):
    """Schema for user response."""
    
    id: int = Field(..., description="Unique user identifier", example=123)
    name: str = Field(..., description="User's full name", example="John Doe")  
    email: str = Field(..., description="User's email address", example="john@example.com")
    created_at: str = Field(..., description="Creation timestamp", example="2023-12-25T10:30:00Z")

# Generate schemas for OpenAPI
create_schema = UserCreate.model_json_schema()
response_schema = UserResponse.model_json_schema()

# These can be used directly in OpenAPI/FastAPI documentation
openapi_schemas = {
    'UserCreate': create_schema,
    'UserResponse': response_schema
}
```

### Schema Customization with Field Info

```python
from pydantic import BaseModel, Field
import json

class Product(BaseModel):
    name: str = Field(
        ..., 
        title="Product Name",
        description="The name of the product",
        examples=["Laptop", "Phone", "Tablet"]
    )
    price: float = Field(
        ...,
        title="Price",
        description="Product price in USD",
        gt=0,
        examples=[999.99, 1299.00]
    )
    category: str = Field(
        ...,
        title="Category", 
        description="Product category",
        examples=["Electronics", "Clothing", "Books"]
    )

schema = Product.model_json_schema()
print(json.dumps(schema, indent=2))

# Schema includes titles, descriptions, and examples for rich documentation
```