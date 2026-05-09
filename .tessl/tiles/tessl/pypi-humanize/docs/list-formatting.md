# List Formatting

Convert Python lists into natural language strings with proper comma placement and conjunction usage, following standard English grammar rules.

## Capabilities

### Natural List

Converts a list of items into human-readable text with appropriate comma separation and conjunction placement.

```python { .api }
def natural_list(items: list[Any]) -> str:
    """
    Convert a list of items into a human-readable string.
    
    Args:
        items: List of items to convert (any type that can be stringified)
    
    Returns:
        String with commas and 'and' in appropriate places
    
    Examples:
        >>> natural_list(["one", "two", "three"])
        'one, two and three'
        >>> natural_list(["one", "two"])
        'one and two'
        >>> natural_list(["one"])
        'one'
        >>> natural_list([])
        ''
    """
```

## Grammar Rules

The function follows standard English grammar conventions:

- **Single item**: Returns the item as-is
- **Two items**: Joins with " and " (no comma)
- **Three or more items**: Uses commas between all items except the last, which is preceded by " and "

Note: This follows the style without the Oxford comma (no comma before "and").

## Usage Examples

### Basic List Formatting

```python
import humanize

# Single item
print(humanize.natural_list(["apple"]))
# "apple"

# Two items
print(humanize.natural_list(["apple", "banana"]))  
# "apple and banana"

# Three items
print(humanize.natural_list(["apple", "banana", "cherry"]))
# "apple, banana and cherry"

# Many items
items = ["red", "green", "blue", "yellow", "purple"]
print(humanize.natural_list(items))
# "red, green, blue, yellow and purple"
```

### Mixed Data Types

The function works with any items that can be converted to strings:

```python
import humanize

# Numbers
print(humanize.natural_list([1, 2, 3, 4]))
# "1, 2, 3 and 4"

# Mixed types
print(humanize.natural_list(["item", 42, 3.14, True]))
# "item, 42, 3.14 and True"

# Objects with string representations
from datetime import date
dates = [date(2023, 1, 1), date(2023, 6, 15), date(2023, 12, 31)]
print(humanize.natural_list(dates))
# "2023-01-01, 2023-06-15 and 2023-12-31"
```

### Empty and Edge Cases

```python
import humanize

# Empty list
print(humanize.natural_list([]))
# ""

# List with None values
print(humanize.natural_list([None, "something", None]))
# "None, something and None"

# List with empty strings
print(humanize.natural_list(["", "middle", ""]))
# ", middle and "
```

### Practical Applications

#### Error Messages

```python
import humanize

def validate_required_fields(data, required):
    missing = [field for field in required if field not in data]
    if missing:
        fields_text = humanize.natural_list(missing)
        raise ValueError(f"Missing required fields: {fields_text}")

# Example usage
try:
    validate_required_fields({"name": "John"}, ["name", "email", "phone"])
except ValueError as e:
    print(e)  # "Missing required fields: email and phone"
```

#### Status Reports

```python
import humanize

def format_status_report(completed, pending, failed):
    parts = []
    if completed:
        parts.append(f"{len(completed)} completed")
    if pending:
        parts.append(f"{len(pending)} pending")
    if failed:
        parts.append(f"{len(failed)} failed")
    
    return f"Tasks: {humanize.natural_list(parts)}"

# Example usage
print(format_status_report([1, 2, 3], [4, 5], [6]))
# "Tasks: 3 completed, 2 pending and 1 failed"
```

#### User Interface Text

```python
import humanize

def format_permissions(user_permissions):
    readable_permissions = [p.replace('_', ' ').title() for p in user_permissions]
    return f"User has permissions: {humanize.natural_list(readable_permissions)}"

# Example usage
permissions = ["read_posts", "write_comments", "delete_own_content"]
print(format_permissions(permissions))
# "User has permissions: Read Posts, Write Comments and Delete Own Content"
```

## Implementation Notes

- All items in the list are converted to strings using `str()`
- The function handles any iterable that can be converted to a list
- Empty lists return empty strings
- Single-item lists return just that item as a string
- The conjunction "and" is hard-coded in English; for internationalization, use the i18n module