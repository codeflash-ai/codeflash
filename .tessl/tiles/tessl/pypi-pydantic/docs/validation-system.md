# Validation System

Decorators and functions for custom validation logic, including field validators, model validators, and functional validation utilities.

## Capabilities

### Field Validators

Decorators for creating custom field validation logic that runs during model validation.

```python { .api }
def field_validator(*fields, mode='before', check_fields=None):
    """
    Decorator for field validation methods.
    
    Args:
        *fields (str): Field names to validate
        mode (str): Validation mode ('before', 'after', 'wrap', 'plain')
        check_fields (bool, optional): Whether to check if fields exist
        
    Returns:
        Decorator function
    """

@field_validator('field_name')
@classmethod
def validate_field(cls, v):
    """
    Template for field validator method.
    
    Args:
        v: Field value to validate
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If validation fails
    """
```

### Model Validators

Decorators for creating validation logic that operates on the entire model or multiple fields.

```python { .api }
def model_validator(*, mode):
    """
    Decorator for model validation methods.
    
    Args:
        mode (str): Validation mode ('before', 'after', 'wrap')
        
    Returns:
        Decorator function
    """

@model_validator(mode='after')
@classmethod
def validate_model(cls, values):
    """
    Template for model validator method.
    
    Args:
        values: Model values (dict for 'before', model instance for 'after')
        
    Returns:
        Validated values or model instance
        
    Raises:
        ValueError: If validation fails
    """
```

### Function Validation

Decorator to add pydantic validation to regular functions.

```python { .api }
def validate_call(*, config=None, validate_return=False):
    """
    Decorator to validate function arguments and optionally return values.
    
    Args:
        config: Validation configuration
        validate_return (bool): Whether to validate return values
        
    Returns:
        Decorated function with validation
    """
```

### BeforeValidator and AfterValidator

Functional validators that can be used with Annotated types.

```python { .api }
class BeforeValidator:
    """
    Validator that runs before pydantic's internal validation.
    """
    
    def __init__(self, func):
        """
        Initialize validator.
        
        Args:
            func: Validation function
        """

class AfterValidator:
    """
    Validator that runs after pydantic's internal validation.
    """
    
    def __init__(self, func):
        """
        Initialize validator.
        
        Args:
            func: Validation function
        """

class WrapValidator:
    """
    Validator that wraps pydantic's internal validation.
    """
    
    def __init__(self, func):
        """
        Initialize validator.
        
        Args:
            func: Validation function
        """

class PlainValidator:
    """
    Validator that completely replaces pydantic's internal validation.
    """
    
    def __init__(self, func):
        """
        Initialize validator.
        
        Args:
            func: Validation function
        """
```

### Core Schema Classes

Core classes from pydantic-core that are part of the validation API.

```python { .api }
class ValidationInfo:
    """
    Information available during validation.
    """
    
    @property
    def config(self):
        """ConfigDict: Current model configuration"""
    
    @property
    def context(self):
        """dict | None: Validation context"""
    
    @property
    def data(self):
        """dict: Raw input data"""
    
    @property
    def field_name(self):
        """str | None: Current field name"""
    
    @property
    def mode(self):
        """str: Validation mode ('python' or 'json')"""

class ValidatorFunctionWrapHandler:
    """
    Handler for wrap validators.
    """
    
    def __call__(self, value):
        """
        Call the wrapped validator.
        
        Args:
            value: Value to validate
            
        Returns:
            Validated value
        """
```

## Usage Examples

### Field Validators

```python
from pydantic import BaseModel, field_validator
import re

class User(BaseModel):
    name: str
    email: str
    age: int
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.title()
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(pattern, v):
            raise ValueError('Invalid email format')
        return v.lower()
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0:
            raise ValueError('Age cannot be negative')
        if v > 150:
            raise ValueError('Age seems unrealistic')
        return v

# Usage
user = User(name="john doe", email="JOHN@EXAMPLE.COM", age=30)
print(user.name)   # "John Doe"
print(user.email)  # "john@example.com"
```

### Model Validators

```python
from pydantic import BaseModel, model_validator
from typing import Optional

class DateRange(BaseModel):
    start_date: str
    end_date: str
    duration_days: Optional[int] = None
    
    @model_validator(mode='after')
    def validate_date_range(self):
        from datetime import datetime
        
        start = datetime.fromisoformat(self.start_date)
        end = datetime.fromisoformat(self.end_date)
        
        if start >= end:
            raise ValueError('start_date must be before end_date')
        
        # Calculate duration if not provided
        if self.duration_days is None:
            self.duration_days = (end - start).days
        
        return self

# Usage
date_range = DateRange(
    start_date="2023-01-01",
    end_date="2023-01-10"
)
print(date_range.duration_days)  # 9
```

### Function Validation

```python
from pydantic import validate_call
from typing import List

@validate_call
def process_data(
    data: List[int],
    multiplier: float = 1.0,
    max_value: int = 100
) -> List[int]:
    """
    Process list of integers with validation.
    """
    result = []
    for item in data:
        processed = int(item * multiplier)
        if processed > max_value:
            processed = max_value
        result.append(processed)
    return result

# Usage - arguments are validated automatically
result = process_data([1, 2, 3], multiplier=2.5, max_value=50)
print(result)  # [2, 5, 7]

# This would raise ValidationError
# process_data("not a list", multiplier=2.5)
```

### Functional Validators with Annotated

```python
from pydantic import BaseModel, BeforeValidator, AfterValidator
from typing import Annotated

def normalize_string(v):
    """Normalize string by stripping and converting to lowercase."""
    if isinstance(v, str):
        return v.strip().lower()
    return v

def validate_positive(v):
    """Ensure value is positive."""
    if v <= 0:
        raise ValueError('Value must be positive')
    return v

class Product(BaseModel):
    name: Annotated[str, BeforeValidator(normalize_string)]
    price: Annotated[float, AfterValidator(validate_positive)]

# Usage
product = Product(name="  LAPTOP  ", price=999.99)
print(product.name)   # "laptop"
print(product.price)  # 999.99
```

### Validation with Context

```python
from pydantic import BaseModel, field_validator, ValidationInfo

class SecurityModel(BaseModel):
    username: str
    role: str
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v, info: ValidationInfo):
        # Access validation context
        if info.context and 'allowed_roles' in info.context:
            allowed = info.context['allowed_roles']
            if v not in allowed:
                raise ValueError(f'Role must be one of: {allowed}')
        return v

# Usage with context
context = {'allowed_roles': ['admin', 'user', 'guest']}
user = SecurityModel.model_validate(
    {'username': 'john', 'role': 'admin'},
    context=context
)
```