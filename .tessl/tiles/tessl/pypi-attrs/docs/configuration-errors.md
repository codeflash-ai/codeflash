# Configuration and Error Handling

Configuration options and comprehensive exception hierarchy for error handling in attrs applications. attrs provides global configuration settings and specific exceptions for different error conditions.

## Capabilities

### Configuration

#### Validator Configuration

Global control over validator execution.

```python { .api }
def set_run_validators(run):
    """
    Globally enable or disable validator execution (deprecated).
    
    This function is deprecated. Use attrs.validators.set_disabled() instead.
    
    Parameters:
    - run (bool): True to enable validators, False to disable
    """

def get_run_validators():
    """
    Check if validators are globally enabled (deprecated).
    
    This function is deprecated. Use attrs.validators.get_disabled() instead.
    
    Returns:
    bool: True if validators are enabled, False if disabled
    """
```

Modern validator control (preferred):
```python
# Using validators module
attrs.validators.set_disabled(False)  # Enable validators
is_disabled = attrs.validators.get_disabled()

# Temporarily disable validators
with attrs.validators.disabled():
    # Create instances without validation
    user = User(name="", age=-5)  # Would normally fail validation
```

### Exception Hierarchy

attrs provides a comprehensive exception hierarchy for different error conditions.

#### Base Frozen Errors

Exceptions related to immutable instance or attribute modification.

```python { .api }
class FrozenError(AttributeError):
    """
    Base exception for frozen/immutable modification attempts.
    
    Mirrors namedtuple behavior by subclassing AttributeError.
    
    Attributes:
    - msg (str): Error message "can't set attribute"
    """

class FrozenInstanceError(FrozenError):
    """
    Raised when attempting to modify a frozen attrs instance.
    
    Occurs when trying to set attributes on classes decorated with frozen=True
    or @attrs.frozen.
    """

class FrozenAttributeError(FrozenError):
    """
    Raised when attempting to modify a frozen attribute.
    
    Occurs when trying to set attributes with on_setattr=attrs.setters.frozen.
    """
```

Usage examples:
```python
@attrs.frozen
class ImmutablePoint:
    x: float
    y: float

point = ImmutablePoint(1.0, 2.0)
try:
    point.x = 5.0  # Raises FrozenInstanceError
except attrs.FrozenInstanceError as e:
    print(f"Cannot modify frozen instance: {e}")

@attrs.define
class PartiallyImmutable:
    mutable_field: str
    immutable_field: str = attrs.field(on_setattr=attrs.setters.frozen)

obj = PartiallyImmutable("can change", "cannot change")
obj.mutable_field = "changed"  # OK
try:
    obj.immutable_field = "new value"  # Raises FrozenAttributeError  
except attrs.FrozenAttributeError as e:
    print(f"Cannot modify frozen attribute: {e}")
```

#### Attrs Lookup Errors

Exceptions for attrs-specific operations and lookups.

```python { .api }
class AttrsAttributeNotFoundError(ValueError):
    """
    Raised when an attrs function can't find a requested attribute.
    
    Occurs in functions like evolve() when specifying non-existent field names.
    """

class NotAnAttrsClassError(ValueError):
    """
    Raised when a non-attrs class is passed to an attrs function.
    
    Occurs when calling attrs functions like fields(), asdict(), etc. 
    on regular classes.
    """
```

Usage examples:
```python
@attrs.define
class Person:
    name: str
    age: int

person = Person("Alice", 30)

try:
    # Typo in field name
    updated = attrs.evolve(person, namee="Bob")
except attrs.AttrsAttributeNotFoundError as e:
    print(f"Field not found: {e}")

class RegularClass:
    def __init__(self, value):
        self.value = value

regular = RegularClass(42)
try:
    attrs.fields(RegularClass)  # Raises NotAnAttrsClassError
except attrs.NotAnAttrsClassError as e:
    print(f"Not an attrs class: {e}")
```

#### Definition Errors

Exceptions related to incorrect attrs class or field definitions.

```python { .api }
class DefaultAlreadySetError(RuntimeError):
    """
    Raised when attempting to set a default value multiple times.
    
    Occurs when both default and factory are specified for the same field,
    or when default is set in conflicting ways.
    """

class UnannotatedAttributeError(RuntimeError):
    """
    Raised when type annotation is missing with auto_attribs=True.
    
    Occurs when using @attrs.define or auto_attribs=True without proper
    type annotations on all fields.
    """

class PythonTooOldError(RuntimeError):
    """
    Raised when a feature requires a newer Python version.
    
    Occurs when using attrs features that require Python versions
    newer than the current runtime.
    """
```

Usage examples:
```python
try:
    @attrs.define
    class BadClass:
        # Missing type annotation with auto_attribs (implicit with @define)
        name = "default"  # Raises UnannotatedAttributeError
except attrs.UnannotatedAttributeError as e:
    print(f"Missing type annotation: {e}")

try:
    @attrs.define
    class ConflictingDefaults:
        # Cannot specify both default and factory
        value: int = attrs.field(default=0, factory=int)  # Raises DefaultAlreadySetError
except attrs.DefaultAlreadySetError as e:
    print(f"Conflicting defaults: {e}")
```

#### Validation Errors

Exceptions related to validation failures.

```python { .api }
class NotCallableError(TypeError):
    """
    Raised when a non-callable is passed where callable is required.
    
    Occurs when using attrs.validators.is_callable() validator
    or when passing non-callable validators or converters.
    """
```

Usage example:
```python
@attrs.define
class Config:
    callback: callable = attrs.field(validator=attrs.validators.is_callable())

try:
    config = Config(callback="not a function")  # Raises NotCallableError
except attrs.NotCallableError as e:
    print(f"Not callable: {e}")
```

### Error Handling Patterns

#### Graceful Validation Handling

Handle validation errors gracefully in applications.

```python
def create_user_safely(name, age, email):
    """Create user with error handling."""
    try:
        return User(name=name, age=age, email=email)
    except (TypeError, ValueError) as e:
        # Handle validation errors
        print(f"Invalid user data: {e}")
        return None

def validate_data_batch(data_list):
    """Validate batch of data with individual error handling."""
    results = []
    errors = []
    
    for i, data in enumerate(data_list):
        try:
            user = User(**data)
            attrs.validate(user)  # Explicit validation
            results.append(user)
        except Exception as e:
            errors.append((i, str(e)))
    
    return results, errors
```

#### Configuration Validation

Validate attrs configuration at class definition time.

```python
def validate_attrs_class(cls):
    """Validate attrs class configuration."""
    if not attrs.has(cls):
        raise attrs.NotAnAttrsClassError(f"{cls} is not an attrs class")
    
    field_names = set()
    for field in attrs.fields(cls):
        if field.name in field_names:
            raise ValueError(f"Duplicate field name: {field.name}")
        field_names.add(field.name)
        
        # Check for conflicting configurations
        if field.default is not attrs.NOTHING and field.factory is not None:
            raise attrs.DefaultAlreadySetError(f"Field {field.name} has both default and factory")
    
    return True
```

#### Migration and Compatibility

Handle attrs version compatibility and migration issues.

```python
def safe_evolve(instance, **changes):
    """Safely evolve instance with error handling."""
    try:
        return attrs.evolve(instance, **changes)
    except attrs.AttrsAttributeNotFoundError as e:
        # Handle field name changes or typos
        available_fields = [f.name for f in attrs.fields(instance.__class__)]
        print(f"Available fields: {available_fields}")
        raise ValueError(f"Cannot evolve: {e}") from e

def check_attrs_compatibility(cls):
    """Check if class is compatible with current attrs version."""
    try:
        # Try to access modern features
        attrs.fields_dict(cls)
        return True
    except AttributeError:
        # Older attrs version
        return False
```

## Common Patterns

### Defensive Programming
```python
def safe_attrs_operation(obj, operation):
    """Safely perform attrs operations with comprehensive error handling."""
    if not attrs.has(obj.__class__):
        raise ValueError("Object is not an attrs instance")
    
    try:
        return operation(obj)
    except attrs.FrozenInstanceError:
        print("Cannot modify frozen instance")
        return obj  # Return unchanged
    except attrs.AttrsAttributeNotFoundError as e:
        print(f"Attribute not found: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise
```

### Configuration Management
```python
class AttrsConfig:
    """Centralized attrs configuration management."""
    
    @staticmethod
    def disable_validators():
        """Disable validators for performance."""
        attrs.validators.set_disabled(True)
    
    @staticmethod  
    def enable_validators():
        """Enable validators for safety."""
        attrs.validators.set_disabled(False)
    
    @contextmanager
    def temporary_config(self, disable_validators=False):
        """Temporarily change attrs configuration."""
        old_disabled = attrs.validators.get_disabled()
        
        try:
            if disable_validators:
                attrs.validators.set_disabled(True)
            yield
        finally:
            attrs.validators.set_disabled(old_disabled)
```

### Error Context Enhancement
```python
def enhanced_evolve(instance, **changes):
    """Evolve with enhanced error messages."""
    try:
        return attrs.evolve(instance, **changes)
    except attrs.AttrsAttributeNotFoundError as e:
        # Provide helpful suggestions
        available = [f.name for f in attrs.fields(instance.__class__)]
        suggestions = []
        
        for change_name in changes:
            # Simple fuzzy matching for typos
            for field_name in available:
                if abs(len(change_name) - len(field_name)) <= 2:
                    # Simple edit distance check
                    if sum(c1 != c2 for c1, c2 in zip(change_name, field_name)) <= 2:
                        suggestions.append(field_name)
        
        error_msg = str(e)
        if suggestions:
            error_msg += f". Did you mean: {', '.join(suggestions)}?"
        
        raise attrs.AttrsAttributeNotFoundError(error_msg) from e
```

### Testing Utilities
```python
def assert_attrs_equal(obj1, obj2, ignore_fields=None):
    """Assert two attrs objects are equal, optionally ignoring fields."""
    if not attrs.has(obj1.__class__) or not attrs.has(obj2.__class__):
        raise ValueError("Both objects must be attrs instances")
    
    if obj1.__class__ != obj2.__class__:
        raise ValueError("Objects must be of the same class")
    
    ignore_fields = ignore_fields or []
    
    for field in attrs.fields(obj1.__class__):
        if field.name in ignore_fields:
            continue
            
        val1 = getattr(obj1, field.name)
        val2 = getattr(obj2, field.name)
        
        if val1 != val2:
            raise AssertionError(f"Field {field.name} differs: {val1} != {val2}")

def create_test_instance(cls, **overrides):
    """Create test instance with safe defaults."""
    try:
        # Try to create with minimal required fields
        required_fields = {}
        for field in attrs.fields(cls):
            if field.default is attrs.NOTHING and field.factory is None:
                # Required field, provide a test default
                if field.type == str:
                    required_fields[field.name] = "test"
                elif field.type == int:
                    required_fields[field.name] = 0
                # Add more type defaults as needed
        
        required_fields.update(overrides)
        return cls(**required_fields)
    
    except Exception as e:
        raise ValueError(f"Could not create test instance of {cls}: {e}")
```