# Core Models and Validation

Base model classes, field definitions, and core validation functionality that forms the foundation of pydantic's data validation capabilities.

## Capabilities

### BaseModel

The primary base class for creating pydantic models with comprehensive validation, serialization, and schema generation capabilities.

```python { .api }
class BaseModel(metaclass=ModelMetaclass):
    """
    Base class for creating pydantic models.
    
    Provides validation, serialization, and schema generation functionality.
    """
    
    def __init__(self, **data):
        """
        Initialize model with validation.
        
        Args:
            **data: Field values as keyword arguments
            
        Raises:
            ValidationError: If validation fails
        """
    
    @classmethod
    def model_validate(cls, obj, /, *, strict=None, from_attributes=None, context=None):
        """
        Validate data and create model instance.
        
        Args:
            obj: Data to validate (dict, model instance, etc.)
            strict (bool, optional): Enable strict validation mode
            from_attributes (bool, optional): Extract data from object attributes
            context (dict, optional): Additional context for validation
            
        Returns:
            Instance of the model
            
        Raises:
            ValidationError: If validation fails
        """
    
    @classmethod
    def model_validate_json(cls, json_data, /, *, strict=None, context=None):
        """
        Validate JSON string and create model instance.
        
        Args:
            json_data (str | bytes): JSON string to validate
            strict (bool, optional): Enable strict validation mode
            context (dict, optional): Additional context for validation
            
        Returns:
            Instance of the model
            
        Raises:
            ValidationError: If validation fails
        """
    
    @classmethod
    def model_validate_strings(cls, obj, /, *, strict=None, context=None):
        """
        Validate data with string inputs and create model instance.
        
        Args:
            obj: Data to validate
            strict (bool, optional): Enable strict validation mode
            context (dict, optional): Additional context for validation
            
        Returns:
            Instance of the model
            
        Raises:
            ValidationError: If validation fails
        """
    
    def model_dump(self, *, include=None, exclude=None, context=None, by_alias=False, 
                   exclude_unset=False, exclude_defaults=False, exclude_none=False,
                   round_trip=False, warnings=True, serialize_as_any=False):
        """
        Convert model to dictionary.
        
        Args:
            include: Fields to include
            exclude: Fields to exclude
            context (dict, optional): Serialization context
            by_alias (bool): Use field aliases in output
            exclude_unset (bool): Exclude fields that weren't set
            exclude_defaults (bool): Exclude fields with default values
            exclude_none (bool): Exclude fields with None values
            round_trip (bool): Enable round-trip serialization
            warnings (bool): Show serialization warnings
            serialize_as_any (bool): Serialize using Any serializer
            
        Returns:
            dict: Model data as dictionary
        """
    
    def model_dump_json(self, *, include=None, exclude=None, context=None, by_alias=False,
                        exclude_unset=False, exclude_defaults=False, exclude_none=False,
                        round_trip=False, warnings=True, serialize_as_any=False):
        """
        Convert model to JSON string.
        
        Args:
            include: Fields to include
            exclude: Fields to exclude  
            context (dict, optional): Serialization context
            by_alias (bool): Use field aliases in output
            exclude_unset (bool): Exclude fields that weren't set
            exclude_defaults (bool): Exclude fields with default values
            exclude_none (bool): Exclude fields with None values
            round_trip (bool): Enable round-trip serialization
            warnings (bool): Show serialization warnings
            serialize_as_any (bool): Serialize using Any serializer
            
        Returns:
            str: Model data as JSON string
        """
    
    def model_copy(self, *, update=None, deep=False):
        """
        Create a copy of the model.
        
        Args:
            update (dict, optional): Fields to update in the copy
            deep (bool): Create deep copy
            
        Returns:
            New instance of the model
        """
    
    @classmethod
    def model_construct(cls, _fields_set=None, **values):
        """
        Create model instance without validation.
        
        Args:
            _fields_set (set, optional): Set of field names that were explicitly set
            **values: Field values
            
        Returns:
            Instance of the model
        """
    
    @classmethod
    def model_json_schema(cls, by_alias=True, ref_template='#/$defs/{model}'):
        """
        Generate JSON schema for the model.
        
        Args:
            by_alias (bool): Use field aliases in schema
            ref_template (str): Template for schema references
            
        Returns:
            dict: JSON schema
        """
    
    @classmethod
    def model_rebuild(cls, *, force=False, raise_errors=True, _parent_namespace_depth=2,
                      _types_namespace=None):
        """
        Rebuild model schema and validators.
        
        Args:
            force (bool): Force rebuild even if not needed
            raise_errors (bool): Raise errors during rebuild
            _parent_namespace_depth (int): Depth for namespace resolution
            _types_namespace (dict, optional): Types namespace
        """
    
    @property
    def model_extra(self):
        """dict: Extra fields not defined in the model"""
    
    @property
    def model_fields_set(self):
        """set: Set of field names that were set during initialization"""
    
    @classmethod
    @property
    def model_fields(cls):
        """dict: Dictionary of field name to FieldInfo"""
    
    @classmethod
    @property
    def model_computed_fields(cls):
        """dict: Dictionary of computed field name to ComputedFieldInfo"""
```

### RootModel

Generic base class for models where the entire model is a single value, useful for wrapping primitive types or creating type aliases with validation.

```python { .api }
class RootModel(BaseModel, Generic[RootType]):
    """
    Base class for models where the root value is the entire model.
    
    Useful for creating validated type aliases or wrapping primitive types.
    """
    
    root: RootType
    
    def __init__(self, root: RootType = PydanticUndefined, **data):
        """
        Initialize with root value.
        
        Args:
            root: The root value for the model
            **data: Additional data (typically empty for RootModel)
        """
    
    @classmethod
    def model_construct(cls, root, _fields_set=None):
        """
        Create RootModel instance without validation.
        
        Args:
            root: Root value
            _fields_set (set, optional): Set of fields that were set
            
        Returns:
            RootModel instance
        """
```

### Field Function

Create field definitions with validation constraints, metadata, and configuration options.

```python { .api }
def Field(default=PydanticUndefined, *, default_factory=None, alias=None, 
          alias_priority=None, validation_alias=None, serialization_alias=None,
          title=None, field_title_generator=None, description=None, examples=None, 
          exclude=None, discriminator=None, deprecated=None, json_schema_extra=None, 
          frozen=None, validate_default=None, repr=True, init=None, init_var=None, 
          kw_only=None, pattern=None, strict=None, gt=None, ge=None, lt=None, le=None, 
          multiple_of=None, allow_inf_nan=None, max_digits=None, decimal_places=None, 
          min_length=None, max_length=None, **kwargs):
    """
    Create a field definition with validation and metadata.
    
    Args:
        default: Default value for the field
        default_factory: Factory function for default values
        alias: Alias for the field name
        alias_priority (int): Priority for alias resolution
        validation_alias: Alias used during validation
        serialization_alias: Alias used during serialization
        title (str): Human-readable title
        field_title_generator: Function to generate field title
        description (str): Field description
        examples: Example values
        exclude: Whether to exclude from serialization
        discriminator: Discriminator for union types
        deprecated: Deprecation information
        json_schema_extra: Extra JSON schema properties
        frozen (bool): Whether field is frozen after initialization
        validate_default (bool): Validate default values
        repr (bool): Include in repr output
        init (bool): Include in __init__ method
        init_var (bool): Mark as init-only variable
        kw_only (bool): Keyword-only parameter
        pattern (str): Regex pattern for string validation
        strict (bool): Enable strict validation
        gt: Greater than constraint
        ge: Greater than or equal constraint
        lt: Less than constraint
        le: Less than or equal constraint
        multiple_of: Multiple of constraint
        allow_inf_nan (bool): Allow infinity and NaN values
        max_digits (int): Maximum number of digits
        decimal_places (int): Maximum decimal places
        min_length (int): Minimum length constraint
        max_length (int): Maximum length constraint
        
    Returns:
        FieldInfo: Field definition object
    """
```

### Dynamic Model Creation

Create pydantic model classes dynamically at runtime.

```python { .api }
def create_model(__model_name, *, __config__=None, __base__=None, __module__=None,
                 __validators__=None, __cls_kwargs__=None, **field_definitions):
    """
    Dynamically create a pydantic model class.
    
    Args:
        __model_name (str): Name of the model class
        __config__: Model configuration
        __base__: Base class (defaults to BaseModel)
        __module__ (str): Module name for the class
        __validators__: Dictionary of validators
        __cls_kwargs__: Additional class keyword arguments
        **field_definitions: Field definitions as name=(type, field_info) pairs
        
    Returns:
        type: Dynamically created model class
    """
```

## Usage Examples

### Basic Model Definition

```python
from pydantic import BaseModel, Field
from typing import Optional

class Product(BaseModel):
    id: int
    name: str = Field(..., min_length=1, max_length=100)
    price: float = Field(..., gt=0)
    description: Optional[str] = None
    in_stock: bool = True

# Create and validate
product = Product(
    id=1,
    name="Laptop",
    price=999.99,
    description="High-performance laptop"
)
```

### RootModel Usage

```python
from pydantic import RootModel
from typing import List

class UserIds(RootModel[List[int]]):
    root: List[int]

# Validate list of integers
user_ids = UserIds([1, 2, 3, 4])
print(user_ids.root)  # [1, 2, 3, 4]
```

### Dynamic Model Creation

```python
from pydantic import create_model, Field

# Create model dynamically
DynamicModel = create_model(
    'DynamicModel',
    name=(str, Field(..., min_length=1)),
    age=(int, Field(..., ge=0, le=150))
)

# Use the dynamic model
instance = DynamicModel(name="Alice", age=30)
```