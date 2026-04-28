# URI Utilities

URI handling and path conversion utilities for working with Language Server Protocol URIs and filesystem paths across different platforms.

## Capabilities

### Path and URI Conversion

Functions for converting between filesystem paths and URIs with proper encoding and platform handling.

```python { .api }
def from_fs_path(path: str) -> Optional[str]:
    """
    Convert filesystem path to URI.
    
    Parameters:
    - path: str - Filesystem path to convert
    
    Returns:
    Optional[str] - URI string or None if conversion fails
    """

def to_fs_path(uri: str) -> Optional[str]:
    """
    Convert URI to filesystem path.
    
    Parameters:
    - uri: str - URI string to convert
    
    Returns:
    Optional[str] - Filesystem path or None if conversion fails
    """
```

### URI Manipulation

Functions for parsing, modifying, and constructing URIs with proper encoding handling.

```python { .api }
def uri_scheme(uri: str) -> Optional[str]:
    """
    Extract scheme from URI.
    
    Parameters:
    - uri: str - URI to extract scheme from
    
    Returns:
    Optional[str] - URI scheme or None
    """

def uri_with(
    uri: str,
    scheme: Optional[str] = None,
    netloc: Optional[str] = None, 
    path: Optional[str] = None,
    params: Optional[str] = None,
    query: Optional[str] = None,
    fragment: Optional[str] = None
) -> str:
    """
    Create new URI with modified components.
    
    Parameters:
    - uri: str - Base URI
    - scheme: Optional[str] - New scheme
    - netloc: Optional[str] - New network location
    - path: Optional[str] - New path
    - params: Optional[str] - New parameters  
    - query: Optional[str] - New query string
    - fragment: Optional[str] - New fragment
    
    Returns:
    str - Modified URI
    """

def urlparse(uri: str) -> Tuple[str, str, str, str, str, str]:
    """
    Parse and decode URI into components.
    
    Parameters:
    - uri: str - URI to parse
    
    Returns:
    Tuple[str, str, str, str, str, str] - (scheme, netloc, path, params, query, fragment)
    """

def urlunparse(parts: URLParts) -> str:
    """
    Construct URI from components with proper encoding.
    
    Parameters:
    - parts: URLParts - URI components tuple
    
    Returns:
    str - Constructed URI
    """
```

### Type Definitions

```python { .api }
URLParts = Tuple[str, str, str, str, str, str]
# Type alias for URI components: (scheme, netloc, path, params, query, fragment)
```

## Usage Examples

### Basic Path and URI Conversion

```python
from pygls.uris import from_fs_path, to_fs_path

# Convert filesystem path to URI
path = "/home/user/project/file.py"
uri = from_fs_path(path)
print(uri)  # file:///home/user/project/file.py

# Convert URI back to filesystem path
converted_path = to_fs_path(uri)
print(converted_path)  # /home/user/project/file.py

# Windows path conversion
windows_path = r"C:\Users\user\project\file.py"
windows_uri = from_fs_path(windows_path)
print(windows_uri)  # file:///C:/Users/user/project/file.py
```

### URI Manipulation

```python
from pygls.uris import uri_scheme, uri_with, urlparse

# Extract scheme
uri = "file:///home/user/project/file.py"
scheme = uri_scheme(uri)
print(scheme)  # "file"

# Modify URI components
new_uri = uri_with(uri, path="/home/user/project/another_file.py")
print(new_uri)  # file:///home/user/project/another_file.py

# Parse URI into components
components = urlparse(uri)
print(components)  # ('file', '', '/home/user/project/file.py', '', '', '')
```

### Working with LSP Document URIs

```python
from pygls.uris import from_fs_path, to_fs_path
from pygls.server import LanguageServer

server = LanguageServer("uri-example", "1.0.0")

@server.feature("textDocument/hover")
def hover(params):
    # Get document URI from LSP request
    doc_uri = params.text_document.uri
    
    # Convert to filesystem path for file operations
    file_path = to_fs_path(doc_uri)
    
    if file_path:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Process file content...
        return {"contents": f"File has {len(content)} characters"}
    
    return {"contents": "Could not access file"}

# Convert local paths to URIs for client communication
def send_diagnostics_for_file(file_path):
    uri = from_fs_path(file_path)
    if uri:
        server.publish_diagnostics(uri, [])
```

### URI Validation and Handling

```python
from pygls.uris import uri_scheme, to_fs_path

def handle_document_uri(uri):
    """Safely handle document URIs from LSP requests."""
    
    # Check if it's a file URI
    scheme = uri_scheme(uri)
    if scheme == "file":
        # Convert to filesystem path
        path = to_fs_path(uri)
        if path:
            return path
        else:
            raise ValueError(f"Could not convert URI to path: {uri}")
    elif scheme in ("http", "https"):
        # Handle remote URIs
        return uri
    else:
        raise ValueError(f"Unsupported URI scheme: {scheme}")
```