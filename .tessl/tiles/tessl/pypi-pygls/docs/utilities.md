# Utilities and Helpers

URI handling, position encoding, exception management, and other utility functions for language server development with comprehensive support for cross-platform operations and error handling.

## Capabilities

### Position Encoding Utilities

Comprehensive position encoding system supporting different character encoding methods with backward compatibility for deprecated functions.

```python { .api }
class PositionCodec:
    """
    Position encoding codec for character unit conversion.
    
    Handles conversion between different character encoding methods
    (UTF-8, UTF-16) for LSP position calculations.
    """
    
    @classmethod
    def create_encoding(cls, encoding: str = "utf-16") -> 'PositionCodec':
        """
        Create codec for specific encoding.
        
        Parameters:
        - encoding: str - Encoding type ('utf-8', 'utf-16')
        
        Returns:
        PositionCodec instance
        """
    
    def utf16_unit_offset(self, chars: str) -> int:
        """
        Calculate UTF-16 unit offset.
        
        Parameters:
        - chars: str - Character string
        
        Returns:
        int - UTF-16 unit offset
        """
    
    def client_num_units(self, chars: str) -> int:
        """
        Get number of client character units.
        
        Parameters:
        - chars: str - Character string
        
        Returns:
        int - Number of client units based on encoding
        """

# Deprecated utility functions (maintained for backward compatibility)
def utf16_unit_offset(chars: str) -> int:
    """
    DEPRECATED: Use PositionCodec.utf16_unit_offset instead.
    Calculate UTF-16 unit offset for character string.
    """

def utf16_num_units(chars: str) -> int:
    """
    DEPRECATED: Use PositionCodec.client_num_units instead.
    Get number of UTF-16 units in character string.
    """

def position_from_utf16(lines: List[str], position: Position) -> Position:
    """
    DEPRECATED: Use PositionCodec.position_from_client_units instead.
    Convert position from UTF-16 units to server units.
    """

def position_to_utf16(lines: List[str], position: Position) -> Position:
    """
    DEPRECATED: Use PositionCodec.position_to_client_units instead.
    Convert position from server units to UTF-16 units.
    """

def range_from_utf16(lines: List[str], range: Range) -> Range:
    """
    DEPRECATED: Use PositionCodec.range_from_client_units instead.
    Convert range from UTF-16 units to server units.
    """

def range_to_utf16(lines: List[str], range: Range) -> Range:
    """
    DEPRECATED: Use PositionCodec.range_to_client_units instead.
    Convert range from server units to UTF-16 units.
    """
```

### Constants and Configuration

Configuration constants and attribute definitions for internal pygls operations and feature management.

```python { .api }
# Constants from pygls.constants module

# Dynamically assigned attributes for feature management
ATTR_EXECUTE_IN_THREAD: str = "execute_in_thread"
ATTR_COMMAND_TYPE: str = "command"  
ATTR_FEATURE_TYPE: str = "feature"
ATTR_REGISTERED_NAME: str = "reg_name"
ATTR_REGISTERED_TYPE: str = "reg_type"

# Parameters for server operations
PARAM_LS: str = "ls"
```

### Platform Detection

Platform and environment detection utilities for cross-platform compatibility and runtime environment adaptation.

```python { .api }
# Platform detection from pygls module

IS_WIN: bool = ...
"""True if running on Windows platform."""

IS_PYODIDE: bool = ...
"""True if running in Pyodide environment (browser Python)."""
```

### LSP Type Utilities

Utility functions for working with LSP method types, parameter validation, and capability management.

```python { .api }
# Type utilities from pygls.lsp module

def get_method_params_type(
    method_name: str, 
    lsp_methods_map: dict = METHOD_TO_TYPES
) -> Optional[Type]:
    """
    Get parameter type for LSP method.
    
    Parameters:
    - method_name: str - LSP method name
    - lsp_methods_map: dict - Method to types mapping
    
    Returns:
    Type for method parameters or None
    
    Raises:
    MethodTypeNotRegisteredError if method not found
    """

def get_method_return_type(
    method_name: str,
    lsp_methods_map: dict = METHOD_TO_TYPES  
) -> Optional[Type]:
    """
    Get return type for LSP method.
    
    Parameters:
    - method_name: str - LSP method name
    - lsp_methods_map: dict - Method to types mapping
    
    Returns:
    Type for method return value or None
    
    Raises:
    MethodTypeNotRegisteredError if method not found
    """

def get_method_options_type(
    method_name: str,
    lsp_options_map: dict = METHOD_TO_OPTIONS,
    lsp_methods_map: dict = METHOD_TO_TYPES
) -> Optional[Type]:
    """
    Get options type for LSP method capabilities.
    
    Parameters:
    - method_name: str - LSP method name
    - lsp_options_map: dict - Method to options mapping
    - lsp_methods_map: dict - Method to types mapping
    
    Returns:
    Type for method options or None
    
    Raises:
    MethodTypeNotRegisteredError if method not found
    """

def get_method_registration_options_type(
    method_name: str,
    lsp_methods_map: dict = METHOD_TO_TYPES
) -> Optional[Type]:
    """
    Get registration options type for LSP method.
    
    Parameters:
    - method_name: str - LSP method name
    - lsp_methods_map: dict - Method to types mapping
    
    Returns:
    Type for method registration options or None
    
    Raises:
    MethodTypeNotRegisteredError if method not found
    """

def is_instance(cv: Converter, obj: Any, type_cls: Type) -> bool:
    """
    Check if object is instance of type using cattrs converter.
    
    Parameters:
    - cv: Converter - cattrs converter instance
    - obj: Any - Object to check
    - type_cls: Type - Type to check against
    
    Returns:
    bool - True if object matches type
    """
```

## Usage Examples

### Position Encoding

```python
from pygls.workspace import PositionCodec
from lsprotocol.types import Position, Range

# Create position codec
codec = PositionCodec.create_encoding("utf-16")

# Document with Unicode characters
lines = [
    "def hello():",
    "    print('Hello 世界')",  # Contains non-ASCII characters
    "    return True"
]

# Position at the '世' character
client_position = Position(line=1, character=12)

# Convert from client units to server units
server_position = codec.position_from_client_units(lines, client_position)
print(f"Client pos: {client_position}")
print(f"Server pos: {server_position}")

# Convert back to client units
converted_back = codec.position_to_client_units(lines, server_position)
print(f"Converted back: {converted_back}")

# Handle ranges
client_range = Range(
    start=Position(line=1, character=10),
    end=Position(line=1, character=15)
)

server_range = codec.range_from_client_units(lines, client_range)
print(f"Client range: {client_range}")
print(f"Server range: {server_range}")

# Calculate character units
text = "Hello 世界"
utf16_units = codec.client_num_units(text)
print(f"UTF-16 units for '{text}': {utf16_units}")
```

### Exception Handling

```python
from pygls.exceptions import (
    JsonRpcException,
    JsonRpcMethodNotFound,
    FeatureRequestError,
    PyglsError
)

@server.feature(TEXT_DOCUMENT_HOVER)
def hover_with_error_handling(params):
    try:
        document = server.workspace.get_document(params.text_document.uri)
        
        # Simulate potential errors
        if not document.source.strip():
            raise FeatureRequestError("Document is empty")
        
        # Generate hover content
        position = params.position
        if position.line >= len(document.lines):
            raise JsonRpcException.invalid_params()
        
        word = document.word_at_position(position)
        if not word:
            return None  # No hover content
        
        return Hover(contents=f"Hover for: {word}")
        
    except KeyError:
        # Document not found
        raise JsonRpcMethodNotFound()
        
    except FeatureRequestError:
        # Re-raise feature errors
        raise
        
    except Exception as e:
        # Convert unexpected errors
        raise JsonRpcException.internal_error(str(e))

# Custom error handling
class CustomServerError(PyglsError):
    def __init__(self, message: str, error_code: int = -32000):
        super().__init__(message)
        self.error_code = error_code

@server.command("myServer.riskyOperation")
def risky_operation(params):
    try:
        # Risky operation here
        result = perform_risky_operation(params)
        return result
        
    except CustomServerError as e:
        # Convert to JSON-RPC error
        raise JsonRpcServerError(str(e), e.error_code)
        
    except FileNotFoundError:
        raise JsonRpcException.invalid_params("Required file not found")
        
    except PermissionError:
        raise JsonRpcServerError("Insufficient permissions", -32001)
```

### Type Validation and Method Utilities

```python
from pygls.lsp import (
    get_method_params_type,
    get_method_return_type,
    get_method_options_type
)
from lsprotocol.types import TEXT_DOCUMENT_COMPLETION

# Get type information for LSP methods
params_type = get_method_params_type(TEXT_DOCUMENT_COMPLETION)
print(f"Completion params type: {params_type}")

return_type = get_method_return_type(TEXT_DOCUMENT_COMPLETION)
print(f"Completion return type: {return_type}")

options_type = get_method_options_type(TEXT_DOCUMENT_COMPLETION)
print(f"Completion options type: {options_type}")

# Validate parameters using type information
def validate_completion_params(params):
    expected_type = get_method_params_type(TEXT_DOCUMENT_COMPLETION)
    
    # Use cattrs to validate structure
    try:
        converter = default_converter()
        converter.structure(params, expected_type)
        return True
    except Exception as e:
        print(f"Invalid parameters: {e}")
        return False

# Dynamic feature registration with type checking
def register_feature_with_validation(server, method_name, handler):
    try:
        # Check if method type is registered
        params_type = get_method_params_type(method_name)
        return_type = get_method_return_type(method_name)
        
        print(f"Registering {method_name}")
        print(f"  Params: {params_type}")
        print(f"  Returns: {return_type}")
        
        # Register the feature
        server.feature(method_name)(handler)
        
    except MethodTypeNotRegisteredError:
        print(f"Unknown method: {method_name}")
```

### Platform-Specific Operations

```python
from pygls import IS_WIN, IS_PYODIDE
import os

def get_platform_config():
    """Get platform-specific configuration."""
    config = {
        "line_ending": "\r\n" if IS_WIN else "\n",
        "path_separator": "\\" if IS_WIN else "/",
        "supports_symlinks": not IS_WIN,
        "is_browser": IS_PYODIDE
    }
    
    if IS_PYODIDE:
        # Browser environment limitations
        config.update({
            "supports_file_system": False,
            "supports_threads": False,
            "max_file_size": 1024 * 1024  # 1MB limit
        })
    
    return config

@server.command("myServer.getSystemInfo")
def get_system_info(params):
    """Return system information."""
    config = get_platform_config()
    
    return {
        "platform": "windows" if IS_WIN else "unix",
        "environment": "pyodide" if IS_PYODIDE else "native",
        "python_version": sys.version,
        "config": config
    }

# Platform-specific file operations
def safe_file_operation(file_path: str, operation: str):
    """Perform file operation with platform considerations."""
    
    if IS_PYODIDE:
        return {"error": "File operations not supported in browser"}
    
    try:
        if operation == "read":
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"content": content}
            
        elif operation == "exists":
            exists = os.path.exists(file_path)
            return {"exists": exists}
            
    except PermissionError:
        return {"error": "Permission denied"}
    except FileNotFoundError:
        return {"error": "File not found"}
    except Exception as e:
        return {"error": str(e)}
```