# Pydantic

A comprehensive Python data validation library that leverages Python type hints to provide fast and extensible data validation. Pydantic enables developers to define data schemas using pure, canonical Python syntax and validates data against these schemas with high performance through its Rust-based pydantic-core backend.

## Package Information

- **Package Name**: pydantic
- **Language**: Python
- **Installation**: `pip install pydantic`
- **Documentation**: https://docs.pydantic.dev/

## Core Imports

```python
import pydantic
```

Common for working with models:

```python
from pydantic import BaseModel, Field, ConfigDict
```

Specific imports for advanced features:

```python
from pydantic import (
    ValidationError, TypeAdapter, field_validator, model_validator,
    computed_field, create_model, validate_call, WithJsonSchema
)
```

JSON schema and plugin imports:

```python
from pydantic.json_schema import GenerateJsonSchema, model_json_schema
from pydantic.plugin import PydanticPluginProtocol
from pydantic.alias_generators import to_camel, to_pascal, to_snake
```

## Basic Usage

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class User(BaseModel):
    id: int
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: Optional[int] = Field(None, ge=0, le=150)
    created_at: datetime = Field(default_factory=datetime.now)

# Validation from dictionary
user_data = {
    'id': 123,
    'name': 'John Doe',
    'email': 'john@example.com',
    'age': 30
}

user = User(**user_data)
print(user.model_dump())  # Convert to dict
print(user.model_dump_json())  # Convert to JSON string

# Validation from JSON
json_str = '{"id": 456, "name": "Jane Smith", "email": "jane@example.com"}'
user2 = User.model_validate_json(json_str)

# Handle validation errors
try:
    invalid_user = User(id="not-an-int", name="", email="invalid-email")
except ValidationError as e:
    print(e.errors())
```

## Architecture

Pydantic v2 is built on a robust architecture that separates concerns:

- **BaseModel**: Core model class providing validation, serialization, and schema generation
- **Field System**: Flexible field definitions with validation constraints and metadata
- **Type System**: Rich type support including custom types, constraints, and generic types
- **Validation Engine**: High-performance Rust-based validation via pydantic-core
- **Serialization System**: Configurable serialization with custom serializers
- **Configuration System**: Model-level and field-level configuration options

This design enables pydantic to serve as the foundation for API frameworks (FastAPI), configuration management (pydantic-settings), and data processing pipelines across the Python ecosystem.

## Capabilities

### Core Models and Validation

Base model classes, field definitions, and core validation functionality that forms the foundation of pydantic's data validation capabilities.

```python { .api }
class BaseModel:
    def __init__(self, **data): ...
    @classmethod
    def model_validate(cls, obj): ...
    @classmethod 
    def model_validate_json(cls, json_str): ...
    def model_dump(self, **kwargs): ...
    def model_dump_json(self, **kwargs): ...

def Field(default=..., **kwargs): ...
def create_model(model_name: str, **field_definitions): ...
```

[Core Models](./core-models.md)

### Validation System

Decorators and functions for custom validation logic, including field validators, model validators, and functional validation utilities.

```python { .api }
def field_validator(*fields, **kwargs): ...
def model_validator(*, mode: str): ...
def validate_call(func): ...
```

[Validation System](./validation-system.md)

### Type System and Constraints

Specialized types for common data patterns including network addresses, file paths, dates, colors, and constrained types with built-in validation.

```python { .api }
class EmailStr(str): ...
class HttpUrl(str): ...
class UUID4(UUID): ...
class PositiveInt(int): ...
class constr(str): ...
class conint(int): ...
```

[Type System](./type-system.md)

### Serialization and Configuration

Computed fields, serialization customization, and model configuration options for controlling validation and serialization behavior.

```python { .api }
def computed_field(**kwargs): ...
class ConfigDict(TypedDict): ...
def field_serializer(*fields, **kwargs): ...
def model_serializer(**kwargs): ...
```

[Serialization and Configuration](./serialization-config.md)

### Dataclasses and Type Adapters

Integration with Python dataclasses and standalone type validation without model inheritance.

```python { .api }
def dataclass(**kwargs): ...
class TypeAdapter:
    def __init__(self, type_): ...
    def validate_python(self, obj): ...
    def validate_json(self, json_str): ...
```

[Dataclasses and Type Adapters](./dataclasses-adapters.md)

### JSON Schema Generation

JSON schema generation capabilities for creating OpenAPI-compatible schemas from pydantic models and types.

```python { .api }
def model_json_schema(cls, by_alias=True, ref_template='#/$defs/{model}'): ...
class GenerateJsonSchema: ...
class WithJsonSchema: ...
```

[JSON Schema](./json-schema.md)

### Plugin System

Advanced plugin system for extending pydantic's validation and schema generation capabilities.

```python { .api }
class PydanticPluginProtocol: ...
class BaseValidateHandlerProtocol: ...
def register_plugin(plugin): ...
```

[Plugins](./plugins.md)

### Error Handling and Utilities

Exception classes, warning system, and utility functions for advanced pydantic usage patterns.

```python { .api }
class ValidationError(ValueError): ...
class PydanticUserError(TypeError): ...
def parse_obj_as(type_, obj): ...
def schema_of(type_): ...
```

[Error Handling](./error-handling.md)