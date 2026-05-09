# Item Classes and Types

Core item classes representing different TOML value types, containers, and formatting elements with full type safety. These classes provide the foundation for TOML Kit's type system and formatting preservation.

## Capabilities

### Document and Container Classes

Top-level classes for managing TOML documents and organizing items.

```python { .api }
class TOMLDocument(Container):
    """
    Main TOML document class that preserves formatting and structure.
    Inherits all Container functionality with document-specific behavior.
    """

class Container:
    """
    Base container for TOML items with dict-like interface.
    """
    
    def unwrap(self) -> dict[str, Any]:
        """Convert to pure Python dict, losing formatting."""
    
    def value(self) -> dict[str, Any]:
        """Get wrapped dict value with type conversion."""
    
    def add(self, key: Key | Item | str, item: Item | None = None) -> Container:
        """Add item to container."""
    
    def append(self, key: Key | Item | str, item: Item | None = None) -> Container:
        """Append item to container."""
    
    def remove(self, key: Key | Item | str) -> Container:
        """Remove item from container."""
```

### Scalar Value Classes

Classes representing individual TOML scalar values with type safety and formatting preservation.

```python { .api }
class Bool:
    """Boolean TOML item (true/false)."""
    
    def __init__(self, value: bool, trivia: Trivia): ...
    
    @property
    def value(self) -> bool:
        """Get the boolean value."""

class Integer:
    """Integer TOML item with arbitrary precision support."""
    
    def __init__(self, value: int, trivia: Trivia, raw: str): ...
    
    @property  
    def value(self) -> int:
        """Get the integer value."""

class Float:
    """Floating-point TOML item with precision preservation."""
    
    def __init__(self, value: float, trivia: Trivia, raw: str): ...
    
    @property
    def value(self) -> float:
        """Get the float value."""

class String:
    """String TOML item with type and formatting control."""
    
    def __init__(self, value: str, trivia: Trivia, raw: str, multiline: bool): ...
    
    @property
    def value(self) -> str:
        """Get the string value."""
    
    @classmethod
    def from_raw(cls, value: str, type_: StringType = None, escape: bool = True) -> String:
        """Create string with specific formatting type."""
```

### Date and Time Classes

Classes for TOML date, time, and datetime values with RFC3339 compliance.

```python { .api }
class Date:
    """TOML date item (YYYY-MM-DD format)."""
    
    def __init__(self, year: int, month: int, day: int, trivia: Trivia, raw: str): ...
    
    @property
    def value(self) -> datetime.date:
        """Get the date value."""

class Time:
    """TOML time item with optional microseconds and timezone."""
    
    def __init__(self, hour: int, minute: int, second: int, microsecond: int, 
                 tzinfo: tzinfo | None, trivia: Trivia, raw: str): ...
    
    @property
    def value(self) -> datetime.time:
        """Get the time value."""

class DateTime:
    """TOML datetime item with timezone support."""
    
    def __init__(self, year: int, month: int, day: int, hour: int, minute: int,
                 second: int, microsecond: int, tzinfo: tzinfo | None, 
                 trivia: Trivia, raw: str): ...
    
    @property
    def value(self) -> datetime.datetime:
        """Get the datetime value."""
```

### Collection Classes

Classes for TOML arrays, tables, and array of tables with nested structure support.

```python { .api }
class Array:
    """TOML array with type preservation and formatting control."""
    
    def __init__(self, value: list, trivia: Trivia): ...
    
    def append(self, item: Any) -> None:
        """Add item to end of array."""
    
    def extend(self, items: Iterable[Any]) -> None:
        """Add multiple items to array."""
    
    def insert(self, index: int, item: Any) -> None:
        """Insert item at specific index."""
    
    @property
    def value(self) -> list:
        """Get the array as Python list."""

class Table:
    """TOML table with key-value pairs and nested structure."""
    
    def __init__(self, value: Container, trivia: Trivia, is_aot_element: bool,
                 is_super_table: bool | None = None): ...
    
    def append(self, key: str | Key, item: Item) -> Table:
        """Add key-value pair to table."""
    
    @property
    def value(self) -> dict:
        """Get table as Python dict."""

class InlineTable:
    """TOML inline table for single-line table representation."""
    
    def __init__(self, value: Container, trivia: Trivia, new: bool = False): ...
    
    def update(self, other: dict) -> None:
        """Update with key-value pairs from dict."""
    
    @property
    def value(self) -> dict:
        """Get inline table as Python dict."""

class AoT:
    """Array of Tables - TOML's [[table]] syntax."""
    
    def __init__(self, body: list[Table]): ...
    
    def append(self, item: Table | dict) -> None:
        """Add table to array of tables."""
    
    @property
    def value(self) -> list[dict]:
        """Get array of tables as list of dicts."""
```

### Key Classes

Classes for representing TOML keys including simple and dotted keys.

```python { .api }
class Key:
    """Base class for TOML keys."""
    
    @property
    def key(self) -> str:
        """Get the key as string."""

class SingleKey(Key):
    """Simple TOML key without dots."""
    
    def __init__(self, key: str): ...

class DottedKey(Key):
    """Dotted TOML key for nested access."""
    
    def __init__(self, keys: list[Key]): ...
    
    @property  
    def keys(self) -> list[Key]:
        """Get the individual key components."""
```

### Formatting Classes

Classes for whitespace, comments, and formatting preservation.

```python { .api }
class Whitespace:
    """Whitespace item for formatting control."""
    
    def __init__(self, value: str, fixed: bool = False): ...
    
    @property
    def value(self) -> str:
        """Get the whitespace string."""

class Comment:
    """Comment item with formatting preservation."""
    
    def __init__(self, trivia: Trivia): ...
    
    @property
    def value(self) -> str:
        """Get the comment text."""

class Trivia:
    """Formatting metadata for items."""
    
    def __init__(self, indent: str = "", comment_ws: str = "", 
                 comment: str = "", trail: str = ""): ...

class Null:
    """Represents null/empty values in TOML structure."""
    
    @property
    def value(self) -> None:
        """Always returns None."""
```

### Type Enums and Constants

Enums and constants for TOML type classification.

```python { .api }
class StringType(Enum):
    """String type classification for TOML strings."""
    
    BASIC = "basic"          # "string"
    LITERAL = "literal"      # 'string'  
    MLB = "multiline_basic"  # """string"""
    MLL = "multiline_literal" # '''string'''
    
    @classmethod
    def select(cls, literal: bool, multiline: bool) -> StringType:
        """Select appropriate string type based on flags."""
```

### Custom Encoders

Functions for registering and managing custom encoders to extend TOML Kit's type conversion capabilities.

```python { .api }
def register_encoder(encoder: Encoder) -> Encoder:
    """
    Register a custom encoder function for converting Python objects to TOML items.
    
    Parameters:
    - encoder: Function that takes a Python object and returns a TOML Item
    
    Returns:
    The same encoder function (for decorator usage)
    
    Raises:
    ConvertError: If encoder returns invalid TOML item
    """

def unregister_encoder(encoder: Encoder) -> None:
    """
    Remove a previously registered custom encoder.
    
    Parameters:
    - encoder: The encoder function to remove
    """
```

## Usage Examples

### Working with Document Structure

```python
import tomlkit

# Create document and inspect structure
doc = tomlkit.parse('''
title = "My App"
version = "1.0.0"

[server] 
host = "localhost"
port = 8080
''')

# TOMLDocument inherits from Container
print(type(doc))  # <class 'tomlkit.toml_document.TOMLDocument'>
print(isinstance(doc, tomlkit.Container))  # True

# Access container methods
print(doc.value)  # Pure Python dict
print(doc.unwrap())  # Same as .value

# Iterate over items
for key, item in doc.items():
    print(f"{key}: {type(item)} = {item.value}")
```

### Scalar Type Inspection

```python
import tomlkit

doc = tomlkit.parse('''
name = "Example"
count = 42
price = 19.99
enabled = true
created = 2023-01-15
''')

# Inspect item types
name_item = doc._body[0][1]  # First item
print(type(name_item))  # <class 'tomlkit.items.String'>
print(name_item.value)  # "Example"

count_item = doc["count"]
print(type(count_item))  # <class 'tomlkit.items.Integer'>
print(count_item.value)  # 42

# Check if item is specific type
if isinstance(count_item, tomlkit.items.Integer):
    print("Found integer item")
```

### Array Operations

```python
import tomlkit

# Create and manipulate arrays
doc = tomlkit.document()
arr = tomlkit.array()

# Add various types
arr.append("hello")
arr.append(42)
arr.append(True)
arr.extend([1.5, "world"])

doc["items"] = arr

# Array methods
print(len(arr))  # 5
print(arr[0])    # "hello"
print(arr.value) # ["hello", 42, True, 1.5, "world"]

# Insert and modify
arr.insert(1, "inserted")
arr[0] = "modified"
```

### Table Manipulation

```python
import tomlkit

# Create nested table structure
doc = tomlkit.document()

# Standard table
server = tomlkit.table()
server["host"] = "localhost"
server["port"] = 8080
server["ssl"] = True

# Inline table
auth = tomlkit.inline_table()
auth["username"] = "admin"
auth["token"] = "secret123"

# Add to document
doc["server"] = server
doc["auth"] = auth

# Access table properties
print(server.value)  # dict representation
print(type(server))  # <class 'tomlkit.items.Table'>
print(type(auth))    # <class 'tomlkit.items.InlineTable'>
```

### Array of Tables

```python
import tomlkit

# Create array of tables
doc = tomlkit.document()
users = tomlkit.aot()

# Add user tables
user1 = tomlkit.table()
user1["name"] = "Alice"
user1["role"] = "admin"

user2 = tomlkit.table()
user2["name"] = "Bob"
user2["role"] = "user"

users.append(user1)
users.append(user2)

doc["users"] = users

# Access AoT
print(type(users))    # <class 'tomlkit.items.AoT'>
print(users.value)    # [{"name": "Alice", "role": "admin"}, ...]
print(len(users))     # 2
```

### Key Types and Access

```python
import tomlkit

# Different key types
simple = tomlkit.key("title")
dotted = tomlkit.key(["server", "database", "host"])

print(type(simple))  # <class 'tomlkit.items.SingleKey'>
print(type(dotted))  # <class 'tomlkit.items.DottedKey'>

print(simple.key)    # "title"
print(dotted.key)    # "server.database.host"

# Use keys with containers
doc = tomlkit.document()
doc.add(simple, "My Application")
doc.add(dotted, "db.example.com")
```

### Custom Types and Conversion

```python
import tomlkit
from datetime import datetime, date

# Custom encoder for datetime objects
def datetime_encoder(obj):
    if isinstance(obj, datetime):
        return tomlkit.datetime(obj.isoformat())
    elif isinstance(obj, date):
        return tomlkit.date(obj.isoformat())
    raise tomlkit.ConvertError(f"Cannot convert {type(obj)}")

# Register custom encoder
tomlkit.register_encoder(datetime_encoder)

# Now datetime objects work with item()
doc = tomlkit.document()
doc["created"] = datetime.now()
doc["birthday"] = date(1990, 5, 15)

# Unregister when done
tomlkit.unregister_encoder(datetime_encoder)
```

### Formatting Control

```python
import tomlkit

# Create document with precise formatting
doc = tomlkit.document()

# Add comment
doc.add(tomlkit.comment("Configuration file"))
doc.add(tomlkit.nl())

# Add content with whitespace control
doc.add("title", tomlkit.string("My App"))
doc.add(tomlkit.ws("  "))  # Extra spacing
doc.add(tomlkit.comment("Application name"))
doc.add(tomlkit.nl())
doc.add(tomlkit.nl())  # Blank line

# Create table with formatting
server = tomlkit.table()
server.add("host", tomlkit.string("localhost"))
server.add(tomlkit.comment("Server configuration"))

doc.add("server", server)

print(doc.as_string())
# Outputs with preserved formatting and comments
```