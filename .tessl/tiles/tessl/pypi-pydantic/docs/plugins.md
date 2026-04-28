# Plugin System

Advanced plugin system for extending pydantic's validation and schema generation capabilities, allowing custom validation logic and integration with external libraries.

## Capabilities

### Plugin Protocol

Base protocol for creating pydantic plugins that can hook into validation and schema generation.

```python { .api }
class PydanticPluginProtocol:
    """
    Protocol for pydantic plugins.
    
    Plugins can modify validation behavior and schema generation.
    """
    
    def new_schema_validator(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings):
        """
        Modify or replace schema validation.
        
        Args:
            schema: The schema being processed
            schema_type: Type associated with the schema
            schema_type_path: Path to the schema type
            schema_kind: Kind of schema ('BaseModel', 'TypeAdapter', etc.)
            config: Validation configuration
            plugin_settings: Plugin-specific settings
            
        Returns:
            New schema validator or None to use default
        """
    
    def new_schema_serializer(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings):
        """
        Modify or replace schema serialization.
        
        Args:
            schema: The schema being processed
            schema_type: Type associated with the schema
            schema_type_path: Path to the schema type
            schema_kind: Kind of schema
            config: Validation configuration
            plugin_settings: Plugin-specific settings
            
        Returns:
            New schema serializer or None to use default
        """
```

### Validation Handler Protocols

Protocols for creating validation handlers that process specific validation events.

```python { .api }
class BaseValidateHandlerProtocol:
    """
    Base protocol for validation handlers.
    """
    
    def __call__(self, source_type, field_name=None, field_value=None, **kwargs):
        """
        Handle validation event.
        
        Args:
            source_type: Type of the source object
            field_name (str): Name of the field being validated
            field_value: Value of the field being validated
            **kwargs: Additional validation context
            
        Returns:
            Validation result
        """

class ValidatePythonHandlerProtocol(BaseValidateHandlerProtocol):
    """
    Protocol for Python object validation handlers.
    """
    
    def __call__(self, source_type, input_value, **kwargs):
        """
        Handle Python object validation.
        
        Args:
            source_type: Expected type
            input_value: Value to validate
            **kwargs: Validation context
            
        Returns:
            Validated value
        """

class ValidateJsonHandlerProtocol(BaseValidateHandlerProtocol):
    """
    Protocol for JSON validation handlers.
    """
    
    def __call__(self, source_type, input_value, **kwargs):
        """
        Handle JSON validation.
        
        Args:
            source_type: Expected type
            input_value: JSON string or bytes to validate
            **kwargs: Validation context
            
        Returns:
            Validated value
        """

class ValidateStringsHandlerProtocol(BaseValidateHandlerProtocol):
    """
    Protocol for string validation handlers.
    """
    
    def __call__(self, source_type, input_value, **kwargs):
        """
        Handle string-based validation.
        
        Args:
            source_type: Expected type
            input_value: String value to validate
            **kwargs: Validation context
            
        Returns:
            Validated value
        """
```

### Schema Type Utilities

Utility types and functions for working with schema types in plugins.

```python { .api }
class SchemaTypePath:
    """Named tuple representing path to a schema type."""
    
    def __init__(self, module, qualname):
        """
        Initialize schema type path.
        
        Args:
            module (str): Module name
            qualname (str): Qualified name within module
        """
    
    @property
    def module(self):
        """str: Module name"""
    
    @property  
    def qualname(self):
        """str: Qualified name"""

SchemaKind = str  # Type alias for schema kinds
NewSchemaReturns = dict  # Type alias for new schema return values
```

### Plugin Registration

Functions for registering and managing plugins.

```python { .api }
def register_plugin(plugin):
    """
    Register a pydantic plugin.
    
    Args:
        plugin: Plugin instance implementing PydanticPluginProtocol
    """

def get_plugins():
    """
    Get list of registered plugins.
    
    Returns:
        list: List of registered plugin instances
    """
```

## Usage Examples

### Basic Plugin Creation

```python
from pydantic.plugin import PydanticPluginProtocol
from pydantic import BaseModel
from typing import Any, Dict, Optional

class LoggingPlugin(PydanticPluginProtocol):
    """Plugin that logs validation events."""
    
    def __init__(self, log_file="validation.log"):
        self.log_file = log_file
    
    def new_schema_validator(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings):
        """Add logging to validation."""
        original_validator = None
        
        def logging_validator(input_value, **kwargs):
            with open(self.log_file, 'a') as f:
                f.write(f"Validating {schema_type} with value: {input_value}\n")
            
            if original_validator:
                return original_validator(input_value, **kwargs)
            return input_value
        
        return logging_validator

# Register the plugin
from pydantic.plugin import register_plugin
register_plugin(LoggingPlugin())

# Now all pydantic validations will be logged
class User(BaseModel):
    name: str
    age: int

user = User(name="John", age=30)  # This will log validation
```

### Custom Validation Plugin

```python
from pydantic.plugin import PydanticPluginProtocol, ValidatePythonHandlerProtocol
from pydantic import BaseModel
import re

class EmailValidationPlugin(PydanticPluginProtocol):
    """Plugin that provides enhanced email validation."""
    
    def __init__(self, strict_domains=None):
        self.strict_domains = strict_domains or []
    
    def new_schema_validator(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings):
        """Enhance email validation."""
        
        if schema_type_path and 'email' in schema_type_path.qualname.lower():
            def enhanced_email_validator(input_value, **kwargs):
                # Basic email validation
                if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', input_value):
                    raise ValueError("Invalid email format")
                
                # Check domain restrictions
                if self.strict_domains:
                    domain = input_value.split('@')[1]
                    if domain not in self.strict_domains:
                        raise ValueError(f"Email domain must be one of: {self.strict_domains}")
                
                return input_value.lower()  # Normalize to lowercase
            
            return enhanced_email_validator
        
        return None  # Use default validation

# Register with domain restrictions
email_plugin = EmailValidationPlugin(strict_domains=['company.com', 'organization.org'])
register_plugin(email_plugin)

class Employee(BaseModel):
    name: str
    work_email: str  # Will use enhanced validation

# This will use the enhanced email validation
employee = Employee(name="John", work_email="john@company.com")
```

### Schema Modification Plugin

```python
from pydantic.plugin import PydanticPluginProtocol
from pydantic import BaseModel, Field
from typing import Any

class DefaultsPlugin(PydanticPluginProtocol):
    """Plugin that adds default values based on field names."""
    
    DEFAULT_VALUES = {
        'created_at': '2023-01-01T00:00:00Z',
        'updated_at': '2023-01-01T00:00:00Z',
        'version': 1,
        'active': True
    }
    
    def new_schema_validator(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings):
        """Add automatic defaults for common field names."""
        
        if hasattr(schema_type, '__fields__'):
            for field_name, field_info in schema_type.__fields__.items():
                if field_name in self.DEFAULT_VALUES and field_info.default is None:
                    field_info.default = self.DEFAULT_VALUES[field_name]
        
        return None  # Use default validation

register_plugin(DefaultsPlugin())

class Record(BaseModel):
    id: int
    name: str
    created_at: str = None  # Will get automatic default
    active: bool = None     # Will get automatic default

record = Record(id=1, name="Test")
print(record.created_at)  # "2023-01-01T00:00:00Z"
print(record.active)      # True
```

### Third-Party Integration Plugin

```python
from pydantic.plugin import PydanticPluginProtocol
from pydantic import BaseModel
import json

class JSONSchemaEnhancerPlugin(PydanticPluginProtocol):
    """Plugin that enhances JSON schema generation."""
    
    def new_schema_serializer(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings):
        """Add enhanced metadata to JSON schemas."""
        
        original_serializer = None
        
        def enhanced_serializer(value, **kwargs):
            # Get original serialized value
            if original_serializer:
                result = original_serializer(value, **kwargs)
            else:
                result = value
            
            # Add metadata if this is schema generation
            if isinstance(result, dict) and 'type' in result:
                result['x-generated-by'] = 'pydantic-enhanced'
                result['x-timestamp'] = '2023-12-25T10:30:00Z'
                
                # Add examples based on field type
                if result['type'] == 'string' and 'examples' not in result:
                    result['examples'] = ['example string']
                elif result['type'] == 'integer' and 'examples' not in result:
                    result['examples'] = [42]
            
            return result
        
        return enhanced_serializer

register_plugin(JSONSchemaEnhancerPlugin())

class Product(BaseModel):
    name: str
    price: int

# JSON schema will include enhanced metadata
schema = Product.model_json_schema()
print(schema)  # Will include x-generated-by and examples
```

### Validation Handler Plugin

```python
from pydantic.plugin import ValidatePythonHandlerProtocol
from pydantic import BaseModel
from typing import Any

class TypeCoercionHandler(ValidatePythonHandlerProtocol):
    """Handler that provides aggressive type coercion."""
    
    def __call__(self, source_type, input_value, **kwargs):
        """Coerce types more aggressively."""
        
        # String to number coercion
        if source_type == int and isinstance(input_value, str):
            try:
                return int(float(input_value))  # Handle "42.0" -> 42
            except ValueError:
                pass
        
        # List to string coercion
        if source_type == str and isinstance(input_value, list):
            return ', '.join(str(item) for item in input_value)
        
        # Default behavior
        return input_value

class CoercionPlugin(PydanticPluginProtocol):
    """Plugin that enables aggressive type coercion."""
    
    def __init__(self):
        self.handler = TypeCoercionHandler()
    
    def new_schema_validator(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings):
        """Apply type coercion validation."""
        
        def coercing_validator(input_value, **kwargs):
            # Apply coercion
            coerced_value = self.handler(schema_type, input_value, **kwargs)
            return coerced_value
        
        return coercing_validator

register_plugin(CoercionPlugin())

class Data(BaseModel):
    count: int
    tags: str

# These will work with aggressive coercion
data1 = Data(count="42.5", tags=["python", "pydantic"])  
print(data1.count)  # 42 (from "42.5")
print(data1.tags)   # "python, pydantic" (from list)
```

### Plugin with Settings

```python
from pydantic.plugin import PydanticPluginProtocol
from pydantic import BaseModel
from typing import Dict, Any

class CachingPlugin(PydanticPluginProtocol):
    """Plugin that caches validation results."""
    
    def __init__(self, cache_size=1000):
        self.cache = {}
        self.cache_size = cache_size
    
    def new_schema_validator(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings):
        """Add caching to validation."""
        
        # Get cache settings from plugin_settings
        enabled = plugin_settings.get('cache_enabled', True) if plugin_settings else True
        
        if not enabled:
            return None
        
        def caching_validator(input_value, **kwargs):
            # Create cache key
            cache_key = f"{schema_type}:{hash(str(input_value))}"
            
            # Check cache
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # Validate and cache result
            # In real implementation, call original validator
            result = input_value  # Simplified
            
            # Manage cache size
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry (simplified)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = result
            return result
        
        return caching_validator

# Register with settings
caching_plugin = CachingPlugin(cache_size=500)
register_plugin(caching_plugin)

class CachedModel(BaseModel):
    value: str
    
    class Config:
        # Plugin-specific settings
        plugin_settings = {
            'cache_enabled': True
        }

# Repeated validations will be cached
model1 = CachedModel(value="test")
model2 = CachedModel(value="test")  # Retrieved from cache
```