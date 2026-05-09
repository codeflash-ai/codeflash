# TOML Kit

A 1.0.0-compliant TOML library for Python that preserves all comments, indentations, whitespace and internal element ordering while making them accessible and editable via an intuitive API. TOML Kit enables programmatic modification of TOML files while preserving their original formatting and structure.

## Package Information

- **Package Name**: tomlkit
- **Language**: Python  
- **Installation**: `pip install tomlkit`

## Core Imports

```python
import tomlkit
```

For direct access to main functions:

```python
from tomlkit import parse, loads, load, dumps, dump, document
```

For creating TOML items:

```python
from tomlkit import integer, float_, boolean, string, array, table, inline_table
```

For file operations:

```python
from tomlkit.toml_file import TOMLFile
```

## Basic Usage

```python
import tomlkit

# Parse existing TOML content
content = '''
# This is a TOML document
title = "Example"

[database]
server = "192.168.1.1"
ports = [ 8001, 8001, 8002 ]
'''

doc = tomlkit.parse(content)
print(doc["title"])  # "Example"

# Modify while preserving formatting
doc["title"] = "Updated Example"
doc["database"]["ports"].append(8003)

# Output preserves original formatting
print(doc.as_string())

# Create new TOML document from scratch  
doc = tomlkit.document()
doc["title"] = "New Document"
doc["server"] = {"host": "localhost", "port": 8080}
print(tomlkit.dumps(doc))
```

## Architecture

TOML Kit uses a layered architecture that separates parsing, item representation, and formatting:

- **Parser**: Converts TOML text into structured document with full formatting preservation
- **Items**: Type-safe representations of TOML values (Integer, String, Table, Array, etc.)
- **Container**: Dict-like interface for organizing and accessing TOML items
- **Trivia**: Metadata that captures whitespace, comments, and formatting details
- **TOMLDocument**: Top-level container that can be serialized back to formatted TOML

This design enables round-trip parsing where input formatting is perfectly preserved while providing a familiar dict-like interface for programmatic access and modification.

## Capabilities

### Document Operations

Core functions for parsing TOML from strings/files and serializing back to TOML format, maintaining perfect formatting preservation.

```python { .api }
def parse(string: str | bytes) -> TOMLDocument: ...
def loads(string: str | bytes) -> TOMLDocument: ...  
def load(fp: IO[str] | IO[bytes]) -> TOMLDocument: ...
def dumps(data: Mapping, sort_keys: bool = False) -> str: ...
def dump(data: Mapping, fp: IO[str], *, sort_keys: bool = False) -> None: ...
def document() -> TOMLDocument: ...
```

[Document Operations](./document-operations.md)

### Item Creation

Functions for creating individual TOML items (integers, strings, tables, etc.) with proper type representation and formatting control.

```python { .api }
def integer(raw: str | int) -> Integer: ...
def float_(raw: str | float) -> Float: ...
def boolean(raw: str) -> Bool: ...
def string(raw: str, *, literal: bool = False, multiline: bool = False, escape: bool = True) -> String: ...
def array(raw: str = "[]") -> Array: ...
def table(is_super_table: bool | None = None) -> Table: ...
def inline_table() -> InlineTable: ...
```

[Item Creation](./item-creation.md)

### Advanced Item Types

Specialized TOML types including dates/times, keys, and complex data structures with full TOML 1.0.0 compliance.

```python { .api }
def date(raw: str) -> Date: ...
def time(raw: str) -> Time: ...
def datetime(raw: str) -> DateTime: ...
def key(k: str | Iterable[str]) -> Key: ...
def value(raw: str) -> Item: ...
def aot() -> AoT: ...
```

[Advanced Types](./advanced-types.md)

### File Operations

High-level interface for reading and writing TOML files with automatic encoding handling and line ending preservation.

```python { .api }
class TOMLFile:
    def __init__(self, path: StrPath) -> None: ...
    def read(self) -> TOMLDocument: ...
    def write(self, data: TOMLDocument) -> None: ...
```

[File Operations](./file-operations.md)

### Item Classes and Types

Core item classes representing different TOML value types, containers, and formatting elements with full type safety.

```python { .api }
class TOMLDocument(Container): ...
class Container: ...
class Bool: ...
class Integer: ...
class Float: ...
class String: ...
class Array: ...
class Table: ...
class InlineTable: ...
```

[Item Classes](./item-classes.md)

### Error Handling

Comprehensive exception hierarchy for parsing errors, validation failures, and runtime issues with detailed error reporting.

```python { .api }
class TOMLKitError(Exception): ...
class ParseError(ValueError, TOMLKitError): ...
class ConvertError(TypeError, ValueError, TOMLKitError): ...
class NonExistentKey(KeyError, TOMLKitError): ...
class KeyAlreadyPresent(TOMLKitError): ...
```

[Error Handling](./error-handling.md)

## Types

```python { .api }
# Type aliases for file paths
StrPath = Union[str, os.PathLike]

# Generic item type
Item = Union[Bool, Integer, Float, String, Date, Time, DateTime, Array, Table, InlineTable, AoT]

# Encoder function type
Encoder = Callable[[Any], Item]

# Document type
TOMLDocument = Container
```