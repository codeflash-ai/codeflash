# Error Handling and Utilities

Exception classes, warning system, and utility functions for advanced pydantic usage patterns.

## Capabilities

### Exception Classes

Pydantic-specific exception classes for handling validation and configuration errors.

```python { .api }
class ValidationError(ValueError):
    """
    Raised when validation fails.
    
    Contains detailed information about validation failures.
    """
    
    def __init__(self, errors, model=PydanticUndefined):
        """
        Initialize validation error.
        
        Args:
            errors: List of error dictionaries
            model: Model class that failed validation
        """
    
    def errors(self, *, include_url=True, include_context=True):
        """
        Get list of error dictionaries.
        
        Args:
            include_url (bool): Include error documentation URLs
            include_context (bool): Include error context
            
        Returns:
            list: List of error dictionaries
        """
    
    def error_count(self):
        """
        Get total number of errors.
        
        Returns:
            int: Number of validation errors
        """
    
    @property
    def title(self):
        """str: Error title based on model name"""

class PydanticUserError(TypeError):
    """
    Raised when user makes an error in pydantic usage.
    
    Indicates incorrect usage of pydantic APIs or configuration.
    """

class PydanticUndefinedAnnotation(AttributeError):
    """
    Raised when a field annotation is undefined or cannot be resolved.
    """

class PydanticSchemaGenerationError(TypeError):
    """
    Raised when JSON schema generation fails.
    """

class PydanticImportError(ImportError):
    """
    Raised when required imports are missing for optional features.
    """

class PydanticInvalidForJsonSchema(TypeError):
    """
    Raised when a type cannot be represented in JSON schema.
    """
```

### Warning Classes

Warning classes for deprecated features and potential issues.

```python { .api }
class PydanticDeprecatedSince20(UserWarning):
    """
    Warning for features deprecated since pydantic v2.0.
    """

class PydanticExperimentalWarning(UserWarning):
    """
    Warning for experimental features that may change.
    """
```

### Utility Functions

Utility functions for working with pydantic models and types.

```python { .api }
def __version__():
    """
    Get pydantic version string.
    
    Returns:
        str: Version string (e.g., "2.11.7")
    """

def compiled():
    """
    Check if pydantic is running with compiled extensions.
    
    Returns:
        bool: True if compiled extensions are available
    """

class PydanticUndefined:
    """
    Sentinel value for undefined/missing values.
    
    Used internally to distinguish between None and undefined.
    """

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

def schema_json_of(type_, *, title='Generated schema', indent=2):
    """
    Generate JSON schema string for type (legacy function).
    
    Args:
        type_: Type to generate schema for
        title (str): Schema title
        indent (int): JSON indentation
        
    Returns:
        str: JSON schema string
        
    Note:
        Deprecated: Use TypeAdapter.json_schema() instead
    """
```

### Error Context and Formatting

Advanced error handling utilities for better error reporting.

```python { .api }
class ErrorWrapper:
    """
    Wrapper for validation errors with additional context.
    """
    
    def __init__(self, exc, loc):
        """
        Initialize error wrapper.
        
        Args:
            exc: Exception to wrap
            loc: Location tuple for the error
        """

def format_errors(errors, *, model_name=None):
    """
    Format validation errors for display.
    
    Args:
        errors: List of error dictionaries
        model_name (str): Model name for context
        
    Returns:
        str: Formatted error message
    """
```

## Usage Examples

### Handling ValidationError

```python
from pydantic import BaseModel, ValidationError, Field
from typing import List

class User(BaseModel):
    id: int = Field(..., gt=0)
    name: str = Field(..., min_length=1, max_length=50)
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: int = Field(..., ge=0, le=150)

# Handle validation errors
try:
    invalid_user = User(
        id=-1,           # Invalid: must be > 0
        name="",         # Invalid: too short
        email="invalid", # Invalid: bad format
        age=200          # Invalid: too high
    )
except ValidationError as e:
    print(f"Validation failed with {e.error_count()} errors:")
    
    for error in e.errors():
        field = " -> ".join(str(loc) for loc in error['loc'])
        message = error['msg']
        value = error.get('input', 'N/A')
        print(f"  {field}: {message} (got: {value})")
    
    # Output:
    # Validation failed with 4 errors:
    #   id: Input should be greater than 0 (got: -1)
    #   name: String should have at least 1 character (got: )
    #   email: String should match pattern '^[\w\.-]+@[\w\.-]+\.\w+$' (got: invalid)
    #   age: Input should be less than or equal to 150 (got: 200)
```

### Custom Error Messages

```python
from pydantic import BaseModel, Field, field_validator, ValidationError

class Product(BaseModel):
    name: str = Field(..., min_length=1, description="Product name")
    price: float = Field(..., gt=0, description="Product price in USD")
    category: str
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        allowed = ['electronics', 'clothing', 'books', 'home']
        if v.lower() not in allowed:
            raise ValueError(f'Category must be one of: {", ".join(allowed)}')
        return v.lower()

try:
    product = Product(
        name="",
        price=-10,
        category="invalid_category"
    )
except ValidationError as e:
    # Print detailed error information
    for error in e.errors():
        print(f"Field: {error['loc']}")
        print(f"Error: {error['msg']}")
        print(f"Type: {error['type']}")
        if 'ctx' in error:
            print(f"Context: {error['ctx']}")
        print("---")
```

### Nested Validation Errors

```python
from pydantic import BaseModel, ValidationError
from typing import List

class Address(BaseModel):
    street: str
    city: str
    zip_code: str = Field(..., regex=r'^\d{5}(-\d{4})?$')

class User(BaseModel):
    name: str
    addresses: List[Address]

try:
    user_data = {
        'name': 'John',
        'addresses': [
            {'street': '123 Main St', 'city': 'Anytown', 'zip_code': 'invalid'},
            {'street': '', 'city': 'Other City', 'zip_code': '12345'}
        ]
    }
    user = User(**user_data)
except ValidationError as e:
    for error in e.errors():
        # Error location shows nested path
        location = ' -> '.join(str(loc) for loc in error['loc'])
        print(f"{location}: {error['msg']}")
    
    # Output:
    # addresses -> 0 -> zip_code: String should match pattern '^\d{5}(-\d{4})?$'
    # addresses -> 1 -> street: String should have at least 1 character
```

### Error Context and URLs

```python
from pydantic import BaseModel, ValidationError, Field

class Config(BaseModel):
    timeout: int = Field(..., ge=1, le=3600)
    max_connections: int = Field(..., ge=1, le=1000)

try:
    config = Config(timeout=0, max_connections=2000)
except ValidationError as e:
    # Get errors with full context and URLs
    for error in e.errors(include_url=True, include_context=True):
        print(f"Field: {error['loc']}")
        print(f"Error: {error['msg']}")
        print(f"Input: {error['input']}")
        
        # Show constraint context if available
        if 'ctx' in error:
            print(f"Constraints: {error['ctx']}")
        
        # Show documentation URL if available
        if 'url' in error:
            print(f"Help: {error['url']}")
        print("---")
```

### Using PydanticUserError

```python
from pydantic import BaseModel, Field, PydanticUserError

class MyModel(BaseModel):
    value: int
    
    @classmethod
    def create_with_validation(cls, **kwargs):
        """Factory method with additional validation."""
        
        # Check for deprecated usage patterns
        if 'old_field' in kwargs:
            raise PydanticUserError(
                "The 'old_field' parameter is no longer supported. "
                "Use 'value' instead."
            )
        
        return cls(**kwargs)

# This would raise PydanticUserError
try:
    model = MyModel.create_with_validation(old_field=42)
except PydanticUserError as e:
    print(f"Usage error: {e}")
```

### Checking for Compiled Extensions

```python
from pydantic import compiled

if compiled:
    print("Pydantic is running with compiled Rust extensions for better performance")
else:
    print("Pydantic is running in pure Python mode")
    print("Consider installing with: pip install pydantic[email]")
```

### Custom Error Formatting

```python
from pydantic import BaseModel, ValidationError, Field
import json

class ErrorFormatter:
    @staticmethod
    def format_validation_error(error: ValidationError) -> dict:
        """Format ValidationError for API responses."""
        formatted_errors = []
        
        for err in error.errors():
            formatted_errors.append({
                'field': '.'.join(str(loc) for loc in err['loc']),
                'message': err['msg'],
                'type': err['type'],
                'value': err.get('input')
            })
        
        return {
            'error': 'validation_failed',
            'message': f'Validation failed with {error.error_count()} errors',
            'details': formatted_errors
        }

class User(BaseModel):
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    age: int = Field(..., ge=0, le=150)

try:
    user = User(email='invalid-email', age=-5)
except ValidationError as e:
    error_response = ErrorFormatter.format_validation_error(e)
    print(json.dumps(error_response, indent=2))
```

### Legacy Function Migration

```python
from pydantic import parse_obj_as, schema_of, TypeAdapter, ValidationError
from typing import List, Dict

# Old way (deprecated)
try:
    result = parse_obj_as(List[int], ['1', '2', '3'])
    schema = schema_of(List[int])
except Exception as e:
    print(f"Legacy function error: {e}")

# New way (recommended)
adapter = TypeAdapter(List[int])
try:
    result = adapter.validate_python(['1', '2', '3'])
    schema = adapter.json_schema()
    print(f"Result: {result}")
    print(f"Schema keys: {list(schema.keys())}")
except ValidationError as e:
    print(f"Validation error: {e}")
```