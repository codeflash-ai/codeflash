# Item Creation

Functions for creating individual TOML items with proper type representation and formatting control. These functions provide fine-grained control over TOML value creation and formatting.

## Capabilities

### Numeric Items

Create integer and floating-point TOML items from Python numbers or strings.

```python { .api }
def integer(raw: str | int) -> Integer:
    """
    Create an integer TOML item.
    
    Parameters:
    - raw: Integer value as int or string representation
    
    Returns:
    Integer item with proper TOML formatting
    """

def float_(raw: str | float) -> Float:
    """
    Create a float TOML item.
    
    Parameters:
    - raw: Float value as float or string representation
    
    Returns:
    Float item with proper TOML formatting
    """
```

### Boolean Items

Create boolean TOML items from string representations.

```python { .api }
def boolean(raw: str) -> Bool:
    """
    Create a boolean TOML item from string.
    
    Parameters:
    - raw: "true" or "false" string
    
    Returns:
    Bool item with proper TOML representation
    """
```

### String Items

Create string TOML items with extensive formatting control including literal strings, multiline strings, and escape handling.

```python { .api }
def string(
    raw: str,
    *,
    literal: bool = False,
    multiline: bool = False,
    escape: bool = True,
) -> String:
    """
    Create a string TOML item with formatting options.
    
    Parameters:
    - raw: String content
    - literal: If True, create literal string (single quotes)
    - multiline: If True, create multiline string  
    - escape: If True, apply standard TOML escaping rules
    
    Returns:
    String item with specified formatting
    """
```

### Array Items

Create TOML arrays from string representations or empty arrays for programmatic population.

```python { .api }
def array(raw: str = "[]") -> Array:
    """
    Create an array TOML item.
    
    Parameters:
    - raw: String representation of array (default: empty array)
    
    Returns:
    Array item that can be extended with values
    """
```

### Table Items

Create TOML tables and inline tables for structured data organization.

```python { .api }
def table(is_super_table: bool | None = None) -> Table:
    """
    Create a TOML table.
    
    Parameters:
    - is_super_table: If True, create super table for nested sections
    
    Returns:
    Empty Table that can contain key-value pairs
    """

def inline_table() -> InlineTable:
    """
    Create an inline TOML table.
    
    Returns:
    Empty InlineTable for single-line table representation
    """
```

### Generic Item Creation

Convert Python values to appropriate TOML items with automatic type detection.

```python { .api }
def item(value: Any, _parent: Item | None = None, _sort_keys: bool = False) -> Item:
    """
    Convert a Python value to appropriate TOML item.
    
    Parameters:
    - value: Python value to convert
    - _parent: Parent item for context (internal use)
    - _sort_keys: Sort dictionary keys alphabetically
    
    Returns:
    Appropriate TOML item type based on input value
    
    Raises:
    ConvertError: If value cannot be converted to TOML
    """
```

## Usage Examples

### Basic Item Creation

```python
import tomlkit

# Create numeric items
num = tomlkit.integer(42)
pi = tomlkit.float_(3.14159)
big_num = tomlkit.integer("123456789012345678901234567890")

# Create boolean
enabled = tomlkit.boolean("true")
disabled = tomlkit.boolean("false")

# Add to document
doc = tomlkit.document()
doc["count"] = num
doc["pi"] = pi
doc["enabled"] = enabled
```

### String Formatting Options

```python
import tomlkit

# Basic string
basic = tomlkit.string("Hello, World!")

# Literal string (preserves backslashes)
literal = tomlkit.string(r"C:\Users\Name", literal=True)

# Multiline string
multiline = tomlkit.string("""Line 1
Line 2
Line 3""", multiline=True)

# Multiline literal
ml_literal = tomlkit.string("""Raw text\nwith\backslashes""", 
                           literal=True, multiline=True)

# String without escaping
raw_string = tomlkit.string('Contains "quotes"', escape=False)
```

### Array Construction

```python
import tomlkit

# Empty array
arr = tomlkit.array()
arr.append(1)
arr.append(2)
arr.append(3)

# Array from string
parsed_array = tomlkit.array("[1, 2, 3, 4]")

# Array with mixed types
mixed = tomlkit.array()
mixed.extend([1, "hello", True, 3.14])
```

### Table Creation

```python
import tomlkit

# Create standard table
config = tomlkit.table()
config["host"] = "localhost"
config["port"] = 8080
config["ssl"] = True

# Create inline table
credentials = tomlkit.inline_table()
credentials["username"] = "admin"
credentials["password"] = "secret"

# Add to document
doc = tomlkit.document()
doc["server"] = config
doc["auth"] = credentials

print(doc.as_string())
# [server]
# host = "localhost"
# port = 8080
# ssl = true
# auth = {username = "admin", password = "secret"}
```

### Automatic Item Conversion

```python
import tomlkit

# Convert Python values automatically
doc = tomlkit.document()

# These use item() internally
doc["string"] = "Hello"
doc["number"] = 42
doc["float"] = 3.14
doc["bool"] = True
doc["list"] = [1, 2, 3]
doc["dict"] = {"nested": "value"}

# Explicit conversion
python_dict = {"key": "value", "num": 123}
toml_table = tomlkit.item(python_dict)
```

### Complex Structures

```python
import tomlkit

# Build complex document
doc = tomlkit.document()

# Add metadata table
metadata = tomlkit.table()
metadata["title"] = "My Application"
metadata["version"] = "1.0.0"
metadata["authors"] = tomlkit.array()
metadata["authors"].extend(["John Doe", "Jane Smith"])

# Add server configuration
server = tomlkit.table()
server["host"] = "0.0.0.0"
server["port"] = 8080
server["workers"] = 4
server["debug"] = False

# Add inline database config
db = tomlkit.inline_table()
db["url"] = "postgresql://localhost/myapp"
db["pool_size"] = 10

doc["metadata"] = metadata
doc["server"] = server
doc["database"] = db
```