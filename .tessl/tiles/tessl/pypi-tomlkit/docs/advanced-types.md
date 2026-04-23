# Advanced Types

Specialized TOML types including dates, times, keys, and complex data structures with full TOML 1.0.0 compliance. These types handle edge cases and advanced TOML features.

## Capabilities

### Date and Time Types

Create TOML date, time, and datetime items with RFC3339 compliance and timezone support.

```python { .api }
def date(raw: str) -> Date:
    """
    Create a TOML date from RFC3339 string.
    
    Parameters:
    - raw: Date string in format YYYY-MM-DD
    
    Returns:
    Date item representing the date
    
    Raises:
    ValueError: If string is not a valid date format
    """

def time(raw: str) -> Time:
    """
    Create a TOML time from RFC3339 string.
    
    Parameters:
    - raw: Time string in format HH:MM:SS or HH:MM:SS.fff
    
    Returns:
    Time item representing the time
    
    Raises:
    ValueError: If string is not a valid time format
    """

def datetime(raw: str) -> DateTime:
    """
    Create a TOML datetime from RFC3339 string.
    
    Parameters:
    - raw: Datetime string with optional timezone (ISO 8601/RFC3339)
    
    Returns:
    DateTime item representing the datetime
    
    Raises:
    ValueError: If string is not a valid datetime format
    """
```

### Key Types

Create and manipulate TOML keys including simple keys and dotted key paths for nested access.

```python { .api }
def key(k: str | Iterable[str]) -> Key:
    """
    Create a TOML key from string or key path.
    
    Parameters:
    - k: Single key string or iterable of strings for dotted key
    
    Returns:
    Key object (SingleKey or DottedKey)
    """

def key_value(src: str) -> tuple[Key, Item]:
    """
    Parse a key-value pair from string.
    
    Parameters:
    - src: String containing "key = value" pair
    
    Returns:
    Tuple of (Key, Item) representing the parsed pair
    
    Raises:
    ParseError: If string is not valid key-value format
    """
```

### Value Parsing

Parse arbitrary TOML values from strings with automatic type detection.

```python { .api }
def value(raw: str) -> Item:
    """
    Parse a TOML value from string with automatic type detection.
    
    Parameters:
    - raw: String representation of any TOML value
    
    Returns:
    Appropriate Item type based on parsed content
    
    Raises:
    ParseError: If string is not valid TOML value
    """
```

### Array of Tables

Create arrays of tables (AoT) for structured data collections.

```python { .api }
def aot() -> AoT:
    """
    Create an empty array of tables.
    
    Returns:
    AoT object that can contain Table items
    """
```

### Formatting Elements

Create whitespace and comment elements for precise formatting control.

```python { .api }
def ws(src: str) -> Whitespace:
    """
    Create a whitespace element.
    
    Parameters:
    - src: Whitespace string (spaces, tabs)
    
    Returns:
    Whitespace item for formatting control
    """

def nl() -> Whitespace:
    """
    Create a newline element.
    
    Returns:
    Whitespace item representing a newline
    """

def comment(string: str) -> Comment:
    """
    Create a comment element.
    
    Parameters:
    - string: Comment text (without # prefix)
    
    Returns:
    Comment item with proper TOML formatting
    """
```

## Usage Examples

### Date and Time Handling

```python
import tomlkit

# Create date items
birthday = tomlkit.date("1987-07-05")
release_date = tomlkit.date("2023-12-25")

# Create time items  
meeting_time = tomlkit.time("09:30:00")
precise_time = tomlkit.time("14:15:30.123")

# Create datetime items
created_at = tomlkit.datetime("1979-05-27T07:32:00Z")
updated_at = tomlkit.datetime("2023-01-15T10:30:45-08:00")
local_time = tomlkit.datetime("2023-06-01T12:00:00")

# Add to document
doc = tomlkit.document()
doc["user"] = {
    "birthday": birthday,
    "last_login": updated_at,
    "preferred_meeting": meeting_time
}
```

### Key Manipulation

```python
import tomlkit

# Simple keys
simple = tomlkit.key("title")
quoted = tomlkit.key("spaced key")

# Dotted keys for nested access
nested = tomlkit.key(["server", "database", "host"])
deep_nested = tomlkit.key(["app", "features", "auth", "enabled"])

# Parse key-value pairs
key_val = tomlkit.key_value("debug = true")
key_obj, value_obj = key_val
print(key_obj.key)  # "debug"
print(value_obj.value)  # True

# Complex key-value
complex_kv = tomlkit.key_value('database.connection = "postgresql://localhost"')
```

### Value Parsing

```python
import tomlkit

# Parse various value types
integer_val = tomlkit.value("42")
float_val = tomlkit.value("3.14159")
bool_val = tomlkit.value("true")
string_val = tomlkit.value('"Hello, World!"')

# Parse arrays
array_val = tomlkit.value("[1, 2, 3, 4]")
mixed_array = tomlkit.value('["text", 123, true]')

# Parse inline tables
table_val = tomlkit.value('{name = "John", age = 30}')

# Add parsed values to document
doc = tomlkit.document()
doc["count"] = integer_val
doc["ratio"] = float_val
doc["enabled"] = bool_val
doc["items"] = array_val
```

### Array of Tables

```python
import tomlkit

# Create array of tables
products = tomlkit.aot()

# Create individual tables
product1 = tomlkit.table()
product1["name"] = "Widget"
product1["price"] = 19.99
product1["in_stock"] = True

product2 = tomlkit.table()  
product2["name"] = "Gadget"
product2["price"] = 29.99
product2["in_stock"] = False

# Add tables to array
products.append(product1)
products.append(product2)

# Add to document
doc = tomlkit.document()
doc["products"] = products

print(doc.as_string())
# [[products]]
# name = "Widget"
# price = 19.99
# in_stock = true
#
# [[products]]
# name = "Gadget"
# price = 29.99
# in_stock = false
```

### Advanced Formatting

```python
import tomlkit

# Create document with precise formatting
doc = tomlkit.document()

# Add content with comments
doc.add("title", tomlkit.string("My App"))
doc.add(tomlkit.comment("Application metadata"))
doc.add(tomlkit.nl())

# Add table with formatting
server = tomlkit.table()
server.add("host", tomlkit.string("localhost"))
server.add(tomlkit.ws("  "))  # Add extra whitespace
server.add(tomlkit.comment("Server configuration"))

doc.add("server", server)

# Custom whitespace control
doc.add(tomlkit.nl())
doc.add(tomlkit.nl())  # Extra blank line
doc.add(tomlkit.comment("End of configuration"))
```

### Complex Data Structures

```python
import tomlkit

# Build complex nested structure
doc = tomlkit.document()

# Application metadata with dates
app_info = tomlkit.table()
app_info["name"] = "MyApp"
app_info["version"] = "2.1.0"
app_info["created"] = tomlkit.date("2020-01-15")
app_info["last_updated"] = tomlkit.datetime("2023-06-15T14:30:00Z")

# Database configurations array
databases = tomlkit.aot()

# Primary database
primary = tomlkit.table()
primary["name"] = "primary"
primary["host"] = "db1.example.com"
primary["port"] = 5432
primary["ssl"] = True

# Replica database
replica = tomlkit.table()
replica["name"] = "replica"  
replica["host"] = "db2.example.com"
replica["port"] = 5432
replica["ssl"] = True
replica["readonly"] = True

databases.append(primary)
databases.append(replica)

# Schedule with times
schedule = tomlkit.table()
schedule["backup_time"] = tomlkit.time("02:00:00")
schedule["maintenance_start"] = tomlkit.time("01:00:00") 
schedule["maintenance_end"] = tomlkit.time("04:00:00")

doc["app"] = app_info
doc["database"] = databases  
doc["schedule"] = schedule
```