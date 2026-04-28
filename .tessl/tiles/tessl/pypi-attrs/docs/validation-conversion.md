# Validation and Conversion

Comprehensive validation and conversion system with built-in validators, converters, and combinators for complex data processing. attrs provides both simple validation/conversion functions and powerful combinators for complex scenarios.

## Capabilities

### Validators

#### Type Validation

Validate that attribute values are instances of specific types.

```python { .api }
def instance_of(*types):
    """
    Validate that value is an instance of any of the given types.
    
    Parameters:
    - *types: One or more types to check against
    
    Returns:
    Validator function that raises TypeError if value is not an instance
    """

def is_callable():
    """
    Validate that value is callable.
    
    Returns:
    Validator function that raises NotCallableError if value is not callable
    """
```

Usage examples:
```python
@attrs.define
class Person:
    name: str = attrs.field(validator=attrs.validators.instance_of(str))
    age: int = attrs.field(validator=attrs.validators.instance_of(int))
    callback: callable = attrs.field(validator=attrs.validators.is_callable())
```

#### Value Validation

Validate attribute values against specific criteria.

```python { .api }
def in_(collection):
    """
    Validate that value is in the given collection.
    
    Parameters:
    - collection: Collection to check membership in
    
    Returns:
    Validator function that raises ValueError if value not in collection
    """

def matches_re(regex, flags=0, func=None):
    """
    Validate that string value matches regular expression pattern.
    
    Parameters:
    - regex (str or Pattern): Regular expression pattern
    - flags (int): Regex flags (default: 0)  
    - func (callable, optional): Function to extract string from value
    
    Returns:
    Validator function that raises ValueError if value doesn't match
    """
```

Usage examples:
```python
@attrs.define
class Config:
    level: str = attrs.field(validator=attrs.validators.in_(["debug", "info", "warning", "error"]))
    email: str = attrs.field(validator=attrs.validators.matches_re(r'^[^@]+@[^@]+\.[^@]+$'))
```

#### Numeric Validation

Validate numeric values against comparison criteria.

```python { .api }
def lt(val):
    """Validate that value is less than val."""

def le(val):
    """Validate that value is less than or equal to val."""

def ge(val):
    """Validate that value is greater than or equal to val."""

def gt(val):
    """Validate that value is greater than val."""

def max_len(length):
    """
    Validate that value has maximum length.
    
    Parameters:
    - length (int): Maximum allowed length
    """

def min_len(length):
    """
    Validate that value has minimum length.
    
    Parameters:
    - length (int): Minimum required length
    """
```

Usage examples:
```python
@attrs.define
class Product:
    price: float = attrs.field(validator=[attrs.validators.ge(0), attrs.validators.lt(10000)])
    name: str = attrs.field(validator=[attrs.validators.min_len(1), attrs.validators.max_len(100)])
    rating: int = attrs.field(validator=[attrs.validators.ge(1), attrs.validators.le(5)])
```

#### Deep Validation

Validate nested data structures recursively.

```python { .api }
def deep_iterable(member_validator, iterable_validator=None):
    """
    Validate iterable and its members.
    
    Parameters:
    - member_validator: Validator for each member
    - iterable_validator (optional): Validator for the iterable itself
    
    Returns:
    Validator function for iterables with validated members
    """

def deep_mapping(key_validator, value_validator, mapping_validator=None):
    """
    Validate mapping and its keys/values.
    
    Parameters:
    - key_validator: Validator for each key
    - value_validator: Validator for each value
    - mapping_validator (optional): Validator for the mapping itself
    
    Returns:
    Validator function for mappings with validated keys and values
    """
```

Usage examples:
```python
@attrs.define
class Database:
    # List of strings
    tables: list = attrs.field(
        validator=attrs.validators.deep_iterable(
            member_validator=attrs.validators.instance_of(str),
            iterable_validator=attrs.validators.instance_of(list)
        )
    )
    
    # Dict with string keys and int values
    counts: dict = attrs.field(
        validator=attrs.validators.deep_mapping(
            key_validator=attrs.validators.instance_of(str),
            value_validator=attrs.validators.instance_of(int)
        )
    )
```

### Validator Combinators

#### Optional Validation

Make validators optional (allow None values).

```python { .api }
def optional(validator):
    """
    Make a validator optional by allowing None values.
    
    Parameters:
    - validator: Validator to make optional
    
    Returns:
    Validator that passes None values and validates others
    """
```

Usage example:
```python
@attrs.define
class User:
    name: str = attrs.field(validator=attrs.validators.instance_of(str))
    nickname: str = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.instance_of(str))
    )
```

#### Logical Combinators

Combine validators with logical operations.

```python { .api }
def and_(*validators):
    """
    Combine validators with AND logic - all must pass.
    
    Parameters:
    - *validators: Validators to combine
    
    Returns:
    Validator that requires all validators to pass
    """

def or_(*validators):
    """
    Combine validators with OR logic - at least one must pass.
    
    Parameters:
    - *validators: Validators to combine
    
    Returns:
    Validator that requires at least one validator to pass
    """

def not_(validator, *, msg=None, exc_types=(ValueError, TypeError)):
    """
    Negate a validator - pass if validator fails.
    
    Parameters:
    - validator: Validator to negate
    - msg (str, optional): Custom error message
    - exc_types (tuple): Exception types to catch and negate
    
    Returns:
    Validator that passes when the given validator fails
    """
```

Usage examples:
```python
@attrs.define
class Configuration:
    # Must be string and not empty
    name: str = attrs.field(
        validator=attrs.validators.and_(
            attrs.validators.instance_of(str),
            attrs.validators.min_len(1)
        )
    )
    
    # Must be int or float
    value: Union[int, float] = attrs.field(
        validator=attrs.validators.or_(
            attrs.validators.instance_of(int),
            attrs.validators.instance_of(float)
        )
    )
```

### Validator Control

#### Global Validator Control

Control validator execution globally.

```python { .api }
def set_disabled(disabled):
    """
    Globally disable or enable validator execution.
    
    Parameters:
    - disabled (bool): True to disable validators, False to enable
    """

def get_disabled():
    """
    Check if validators are globally disabled.
    
    Returns:
    bool: True if validators are disabled, False if enabled
    """

@contextmanager
def disabled():
    """
    Context manager to temporarily disable validators.
    
    Returns:
    Context manager that disables validators within its scope
    """
```

Usage examples:
```python
# Globally disable validators (not recommended for production)
attrs.validators.set_disabled(True)

# Temporarily disable for bulk operations
with attrs.validators.disabled():
    # Create objects without validation
    users = [User(name=name, age=age) for name, age in raw_data]
```

### Converters

#### Basic Converters

Convert values to different types or formats.

```python { .api }
def optional(converter):
    """
    Make a converter optional by allowing None values to pass through.
    
    Parameters:
    - converter: Converter to make optional
    
    Returns:
    Converter that passes None values and converts others
    """

def default_if_none(default=NOTHING, factory=None):
    """
    Replace None values with default or factory result.
    
    Parameters:
    - default: Default value to use (mutually exclusive with factory)
    - factory (callable, optional): Function to generate default value
    
    Returns:
    Converter that replaces None with default/factory result
    """

def to_bool(val):
    """
    Convert value to boolean using intelligent rules.
    
    Parameters:
    - val: Value to convert ("true", "false", 1, 0, etc.)
    
    Returns:
    bool: Converted boolean value
    """
```

Usage examples:
```python
@attrs.define
class Settings:
    # Convert string to int, but allow None
    timeout: Optional[int] = attrs.field(
        default=None,
        converter=attrs.converters.optional(int)
    )
    
    # Replace None with empty list
    features: list = attrs.field(
        converter=attrs.converters.default_if_none(factory=list)
    )
    
    # Convert various formats to boolean
    debug: bool = attrs.field(
        default="false",
        converter=attrs.converters.to_bool
    )
```

#### Converter Combinators

Chain multiple converters together.

```python { .api }
def pipe(*converters):
    """
    Chain converters - output of one becomes input of next.
    
    Parameters:
    - *converters: Converters to chain in order
    
    Returns:
    Converter that applies all converters in sequence
    """
```

Usage example:
```python
@attrs.define
class Document:
    # Strip whitespace, then convert to Path
    filename: Path = attrs.field(
        converter=attrs.converters.pipe(str.strip, Path)
    )
    
    # Convert to string, strip, then to uppercase
    category: str = attrs.field(
        converter=attrs.converters.pipe(str, str.strip, str.upper)
    )
```

### Setters

Control how attributes are set after initialization.

```python { .api }
def pipe(*setters):
    """
    Chain multiple setters together.
    
    Parameters:
    - *setters: Setters to chain in order
    
    Returns:
    Setter that applies all setters in sequence
    """

def frozen(instance, attribute, new_value):
    """
    Prevent attribute modification by raising FrozenAttributeError.
    
    Use this to make specific attributes immutable even in mutable classes.
    """

def validate(instance, attribute, new_value):
    """
    Run attribute's validator on new value during setattr.
    
    Parameters:
    - instance: Object instance
    - attribute: Attribute descriptor
    - new_value: Value being set
    
    Returns:
    Validated value (or raises validation error)
    """

def convert(instance, attribute, new_value):
    """
    Run attribute's converter on new value during setattr.
    
    Parameters:
    - instance: Object instance  
    - attribute: Attribute descriptor
    - new_value: Value being set
    
    Returns:
    Converted value
    """

NO_OP: object  # Sentinel to disable on_setattr for specific attributes
```

Usage examples:
```python
@attrs.define(on_setattr=attrs.setters.validate)
class ValidatedClass:
    name: str = attrs.field(validator=attrs.validators.instance_of(str))
    # Changes to name after initialization will be validated

@attrs.define
class MixedClass:
    # Always validate and convert on changes
    value: int = attrs.field(
        converter=int,
        validator=attrs.validators.instance_of(int),
        on_setattr=attrs.setters.pipe(attrs.setters.convert, attrs.setters.validate)
    )
    
    # Frozen after initialization  
    id: str = attrs.field(on_setattr=attrs.setters.frozen)
    
    # No special setattr behavior
    temp: str = attrs.field(on_setattr=attrs.setters.NO_OP)
```

## Common Patterns

### Comprehensive Validation
```python
@attrs.define
class User:
    username: str = attrs.field(
        validator=[
            attrs.validators.instance_of(str),
            attrs.validators.min_len(3),
            attrs.validators.max_len(20),
            attrs.validators.matches_re(r'^[a-zA-Z0-9_]+$')
        ]
    )
    
    age: int = attrs.field(
        validator=[
            attrs.validators.instance_of(int),
            attrs.validators.ge(0),
            attrs.validators.le(150)
        ]
    )
```

### Data Cleaning with Converters
```python
@attrs.define
class Contact:
    # Clean and normalize email
    email: str = attrs.field(
        converter=attrs.converters.pipe(str.strip, str.lower),
        validator=attrs.validators.matches_re(r'^[^@]+@[^@]+\.[^@]+$')
    )
    
    # Parse phone number
    phone: str = attrs.field(
        converter=lambda x: ''.join(filter(str.isdigit, str(x))),
        validator=attrs.validators.min_len(10)
    )
```

### Optional Fields with Defaults
```python
@attrs.define
class APIResponse:
    status: int = attrs.field(validator=attrs.validators.in_([200, 400, 404, 500]))
    
    # Optional message with default
    message: str = attrs.field(
        converter=attrs.converters.default_if_none("OK"),
        validator=attrs.validators.optional(attrs.validators.instance_of(str))
    )
    
    # Optional data that gets empty dict if None
    data: dict = attrs.field(
        converter=attrs.converters.default_if_none(factory=dict)
    )
```