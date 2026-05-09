# Error Handling

Comprehensive exception hierarchy for parsing errors, validation failures, and runtime issues with detailed error reporting. TOML Kit provides specific exceptions for different error conditions to enable precise error handling.

## Capabilities

### Base Exception Classes

Foundation exception classes that other TOML Kit exceptions inherit from.

```python { .api }
class TOMLKitError(Exception):
    """Base exception class for all TOML Kit errors."""

class ParseError(ValueError, TOMLKitError):
    """
    Base class for TOML parsing errors with position information.
    
    Attributes:
    - line: Line number where error occurred
    - col: Column number where error occurred
    """
    
    def __init__(self, line: int, col: int, message: str | None = None) -> None: ...
    
    @property
    def line(self) -> int:
        """Get the line number of the error."""
    
    @property
    def col(self) -> int:
        """Get the column number of the error."""
```

### Parsing Error Classes

Specific exceptions for different TOML syntax and parsing errors.

```python { .api }
class MixedArrayTypesError(ParseError):
    """Array contains elements of different types (invalid in TOML)."""

class InvalidNumberError(ParseError):
    """Numeric value has invalid format."""

class InvalidDateTimeError(ParseError):
    """DateTime value has invalid RFC3339 format."""

class InvalidDateError(ParseError):
    """Date value has invalid format (YYYY-MM-DD)."""

class InvalidTimeError(ParseError):
    """Time value has invalid format."""

class InvalidNumberOrDateError(ParseError):
    """Value cannot be parsed as number or date."""

class InvalidUnicodeValueError(ParseError):
    """Unicode escape sequence is invalid."""

class UnexpectedCharError(ParseError):
    """Unexpected character encountered during parsing."""
    
    def __init__(self, line: int, col: int, char: str) -> None: ...

class EmptyKeyError(ParseError):
    """Key is empty or missing."""

class EmptyTableNameError(ParseError):
    """Table name is empty."""

class InvalidCharInStringError(ParseError):
    """String contains invalid character."""
    
    def __init__(self, line: int, col: int, char: str) -> None: ...

class UnexpectedEofError(ParseError):
    """File ended unexpectedly during parsing."""

class InternalParserError(ParseError):
    """Internal parser error indicating a bug."""

class InvalidControlChar(ParseError):
    """String contains invalid control character."""
    
    def __init__(self, line: int, col: int, char: int, type: str) -> None: ...
```

### Runtime Error Classes

Exceptions that occur during TOML document manipulation and conversion.

```python { .api }
class NonExistentKey(KeyError, TOMLKitError):
    """Attempted to access a key that doesn't exist."""
    
    def __init__(self, key) -> None: ...

class KeyAlreadyPresent(TOMLKitError):
    """Attempted to add a key that already exists."""
    
    def __init__(self, key) -> None: ...

class InvalidStringError(ValueError, TOMLKitError):
    """String value contains invalid character sequences."""
    
    def __init__(self, value: str, invalid_sequences: Collection[str], delimiter: str) -> None: ...

class ConvertError(TypeError, ValueError, TOMLKitError):
    """Failed to convert Python value to TOML item."""
```

## Usage Examples

### Basic Error Handling

```python
import tomlkit
from tomlkit.exceptions import ParseError, TOMLKitError

def safe_parse(content: str):
    """Parse TOML with error handling."""
    try:
        return tomlkit.parse(content)
    except ParseError as e:
        print(f"Parse error at line {e.line}, column {e.col}: {e}")
        return None
    except TOMLKitError as e:
        print(f"TOML Kit error: {e}")
        return None

# Valid TOML
valid_toml = 'title = "My App"'
doc = safe_parse(valid_toml)

# Invalid TOML - syntax error
invalid_toml = 'title = "unclosed string'
doc = safe_parse(invalid_toml)  # Prints error message
```

### Specific Parse Error Handling

```python
import tomlkit
from tomlkit.exceptions import (
    InvalidNumberError, InvalidDateError, MixedArrayTypesError, 
    UnexpectedCharError, EmptyKeyError
)

def parse_with_specific_handling(content: str):
    """Handle specific parsing errors differently."""
    try:
        return tomlkit.parse(content)
    except InvalidNumberError as e:
        print(f"Invalid number format at line {e.line}: {e}")
        # Could attempt to fix or provide suggestions
    except InvalidDateError as e:
        print(f"Invalid date format at line {e.line}: {e}")
        # Could suggest correct date format
    except MixedArrayTypesError as e:
        print(f"Array has mixed types at line {e.line}: {e}")
        # Could explain TOML array type rules
    except UnexpectedCharError as e:
        print(f"Unexpected character at line {e.line}, col {e.col}: {e}")
        # Could highlight the problematic character
    except EmptyKeyError as e:
        print(f"Empty key at line {e.line}: {e}")
        # Could suggest key naming rules

# Test different error types
test_cases = [
    'number = 123.45.67',        # InvalidNumberError
    'date = 2023-13-01',         # InvalidDateError  
    'mixed = [1, "text", true]', # MixedArrayTypesError
    'key = value @',             # UnexpectedCharError
    ' = "value"',                # EmptyKeyError
]

for i, case in enumerate(test_cases):
    print(f"\nTest case {i+1}:")
    parse_with_specific_handling(case)
```

### Runtime Error Handling

```python
import tomlkit
from tomlkit.exceptions import NonExistentKey, KeyAlreadyPresent, ConvertError

def safe_document_operations():
    """Demonstrate runtime error handling."""
    doc = tomlkit.document()
    
    # Key access errors
    try:
        value = doc["nonexistent"]
    except NonExistentKey as e:
        print(f"Key error: {e}")
    
    # Key collision errors
    doc["title"] = "My App"
    try:
        # Attempting to add existing key in certain contexts
        doc.add("title", "Another Title")
    except KeyAlreadyPresent as e:
        print(f"Key already exists: {e}")
    
    # Conversion errors
    class CustomObject:
        pass
    
    try:
        doc["custom"] = CustomObject()  # Cannot convert to TOML
    except ConvertError as e:
        print(f"Conversion error: {e}")

safe_document_operations()
```

### Error Recovery Strategies

```python
import tomlkit
from tomlkit.exceptions import ParseError, InvalidNumberError, InvalidDateError
import re

def parse_with_recovery(content: str):
    """Attempt to parse with error recovery strategies."""
    try:
        return tomlkit.parse(content)
    except InvalidNumberError as e:
        print(f"Attempting to fix number format error at line {e.line}")
        # Simple recovery: remove extra decimal points
        lines = content.split('\n')
        line_content = lines[e.line - 1]
        
        # Fix common number format issues
        fixed_line = re.sub(r'(\d+\.\d+)\.\d+', r'\1', line_content)
        lines[e.line - 1] = fixed_line
        
        try:
            return tomlkit.parse('\n'.join(lines))
        except ParseError:
            print("Recovery failed")
            return None
    
    except InvalidDateError as e:
        print(f"Attempting to fix date format error at line {e.line}")
        # Could implement date format corrections
        return None
    
    except ParseError as e:
        print(f"Unrecoverable parse error: {e}")
        return None

# Test recovery
problematic_toml = '''title = "Test"
version = 1.2.3.4
date = 2023-01-01'''

doc = parse_with_recovery(problematic_toml)
```

### Validation with Error Reporting

```python
import tomlkit
from tomlkit.exceptions import TOMLKitError
from typing import List, Tuple

def validate_config(content: str) -> Tuple[bool, List[str]]:
    """Validate TOML configuration and return errors."""
    errors = []
    
    try:
        doc = tomlkit.parse(content)
    except ParseError as e:
        errors.append(f"Syntax error at line {e.line}, col {e.col}: {str(e)}")
        return False, errors
    
    # Custom validation rules
    try:
        # Check required fields
        if "title" not in doc:
            errors.append("Missing required field: title")
        
        if "version" not in doc:
            errors.append("Missing required field: version")
        
        # Validate types
        if "port" in doc and not isinstance(doc["port"], int):
            errors.append("Port must be an integer")
        
        # Validate ranges
        if "port" in doc and not (1 <= doc["port"] <= 65535):
            errors.append("Port must be between 1 and 65535")
            
    except (KeyError, TypeError, ValueError) as e:
        errors.append(f"Validation error: {e}")
    
    return len(errors) == 0, errors

# Test validation
configs = [
    'title = "Valid App"\nversion = "1.0"\nport = 8080',
    'title = "Invalid App"\nport = "not a number"',
    'version = "1.0"\nport = 70000',  # Missing title, invalid port
]

for i, config in enumerate(configs):
    valid, errors = validate_config(config)
    print(f"\nConfig {i+1}: {'Valid' if valid else 'Invalid'}")
    for error in errors:
        print(f"  - {error}")
```

### Error Logging and Debugging

```python
import tomlkit
from tomlkit.exceptions import ParseError, TOMLKitError
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_with_logging(content: str, source: str = ""):
    """Parse TOML with comprehensive error logging."""
    try:
        doc = tomlkit.parse(content)
        logger.info(f"Successfully parsed TOML from {source}")
        return doc
    
    except ParseError as e:
        # Log detailed parse error information
        lines = content.split('\n')
        error_line = lines[e.line - 1] if e.line <= len(lines) else ""
        
        logger.error(f"Parse error in {source}:")
        logger.error(f"  Line {e.line}, Column {e.col}: {str(e)}")
        logger.error(f"  Content: {error_line}")
        logger.error(f"  Position: {' ' * (e.col - 1)}^")
        
        # Additional context
        start_line = max(0, e.line - 3)
        end_line = min(len(lines), e.line + 2)
        
        logger.error("Context:")
        for i in range(start_line, end_line):
            marker = ">>>" if i == e.line - 1 else "   "
            logger.error(f"  {marker} {i+1:3d}: {lines[i]}")
    
    except TOMLKitError as e:
        logger.error(f"TOML Kit error in {source}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Unexpected error parsing {source}: {str(e)}")
    
    return None

# Test with logging
test_content = '''title = "Test App"
version = 1.0.0
invalid_syntax = @#$%
host = "localhost"'''

doc = parse_with_logging(test_content, "test-config.toml")
```

### Exception Hierarchies for Different Handling

```python
import tomlkit
from tomlkit.exceptions import *

def categorized_error_handling(content: str):
    """Handle errors by category."""
    try:
        return tomlkit.parse(content)
    
    except (InvalidNumberError, InvalidDateError, InvalidTimeError, 
            InvalidDateTimeError, InvalidNumberOrDateError) as e:
        # Data format errors - might be recoverable
        print(f"Data format error: {e}")
        print("Consider checking date/time/number formats")
        return None
    
    except (EmptyKeyError, EmptyTableNameError, InvalidCharInStringError) as e:
        # Structural errors - usually need manual fix
        print(f"Structure error: {e}")
        print("Check TOML syntax and key naming")
        return None
    
    except (UnexpectedCharError, UnexpectedEofError) as e:
        # Syntax errors - need content review
        print(f"Syntax error: {e}")
        print("Review TOML syntax")
        return None
    
    except ParseError as e:
        # General parse errors
        print(f"Parse error: {e}")
        return None
    
    except (NonExistentKey, KeyAlreadyPresent) as e:
        # Runtime errors - programming issues
        print(f"Runtime error: {e}")
        return None
    
    except TOMLKitError as e:
        # Any other TOML Kit error
        print(f"TOML Kit error: {e}")
        return None

# Test different error categories
test_cases = [
    'date = 2023-99-99',      # Data format error
    ' = "empty key"',         # Structure error  
    'key = "value" @',        # Syntax error
]

for case in test_cases:
    categorized_error_handling(case)
```