# Document Operations

Core functions for parsing TOML content from strings and files, and serializing TOML documents back to formatted text. These operations maintain perfect formatting preservation while providing standard I/O patterns.

## Capabilities

### Parsing Functions

Convert TOML text into structured TOMLDocument objects that preserve all formatting, comments, and whitespace.

```python { .api }
def parse(string: str | bytes) -> TOMLDocument:
    """
    Parse TOML string or bytes into a TOMLDocument.
    
    Parameters:
    - string: TOML content as string or bytes
    
    Returns:
    TOMLDocument object with preserved formatting
    
    Raises:
    ParseError: If TOML syntax is invalid
    """

def loads(string: str | bytes) -> TOMLDocument:
    """
    Parse TOML string or bytes into a TOMLDocument.
    Alias for parse().
    
    Parameters:
    - string: TOML content as string or bytes
    
    Returns:
    TOMLDocument object with preserved formatting
    """

def load(fp: IO[str] | IO[bytes]) -> TOMLDocument:
    """
    Load TOML content from a file-like object.
    
    Parameters:
    - fp: File-like object to read from
    
    Returns:
    TOMLDocument object with preserved formatting
    
    Raises:
    ParseError: If TOML syntax is invalid
    """
```

### Serialization Functions  

Convert TOML documents and mappings back to properly formatted TOML text strings.

```python { .api }
def dumps(data: Mapping, sort_keys: bool = False) -> str:
    """
    Serialize a TOML document or mapping to a TOML string.
    
    Parameters:
    - data: TOMLDocument, Container, or mapping to serialize
    - sort_keys: If True, sort keys alphabetically
    
    Returns:
    Formatted TOML string
    
    Raises:
    TypeError: If data is not a supported mapping type
    """

def dump(data: Mapping, fp: IO[str], *, sort_keys: bool = False) -> None:
    """
    Serialize a TOML document to a file-like object.
    
    Parameters:
    - data: TOMLDocument, Container, or mapping to serialize  
    - fp: File-like object to write to
    - sort_keys: If True, sort keys alphabetically
    
    Raises:
    TypeError: If data is not a supported mapping type
    """
```

### Document Creation

Create new empty TOML documents for programmatic construction.

```python { .api }
def document() -> TOMLDocument:
    """
    Create a new empty TOMLDocument.
    
    Returns:
    Empty TOMLDocument ready for content
    """
```

## Usage Examples

### Basic Parsing

```python
import tomlkit

# Parse from string
toml_content = '''
title = "My Application"
version = "1.0.0"

[database]
host = "localhost"
port = 5432
'''

doc = tomlkit.parse(toml_content)
print(doc["title"])  # "My Application"
print(doc["database"]["port"])  # 5432

# Parse from bytes
doc_bytes = toml_content.encode('utf-8')
doc = tomlkit.loads(doc_bytes)
```

### File I/O

```python
import tomlkit

# Load from file
with open('config.toml', 'r') as f:
    doc = tomlkit.load(f)

# Modify content
doc["version"] = "1.1.0"
doc["database"]["port"] = 3306

# Save to file
with open('config.toml', 'w') as f:
    tomlkit.dump(doc, f)
```

### Document Construction

```python
import tomlkit

# Create new document
doc = tomlkit.document()

# Add content
doc["title"] = "New Project"
doc["metadata"] = {
    "author": "John Doe",
    "license": "MIT"
}

# Serialize to string
toml_str = tomlkit.dumps(doc)
print(toml_str)

# With sorted keys
sorted_toml = tomlkit.dumps(doc, sort_keys=True)
```

### Format Preservation

```python
import tomlkit

# Original with comments and spacing
original = '''# Configuration file
title = "My App"    # Application name

[server]
host = "localhost"  
port = 8080         # Default port
'''

# Parse and modify
doc = tomlkit.parse(original)
doc["server"]["port"] = 9000

# Comments and formatting preserved
print(doc.as_string())
# Output maintains original comments and spacing
```