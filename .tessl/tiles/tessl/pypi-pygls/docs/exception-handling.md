# Exception Handling

Exception classes for pygls error handling, JSON-RPC protocol errors, and language server specific exceptions with proper error codes and messages.

## Capabilities

### Base Exception Classes

Core exception classes that form the foundation of pygls error handling.

```python { .api }
class PyglsError(Exception):
    """
    Base exception class for all pygls-specific errors.
    
    Extends the standard Python Exception class to provide
    a common base for all pygls-related exceptions.
    """

class JsonRpcException(Exception):
    """
    Base JSON-RPC exception with error code and message.
    
    Provides structured error information for JSON-RPC
    protocol violations and communication errors.
    """
    
    def __init__(self, code: int, message: str, data: Any = None): ...
```

### JSON-RPC Protocol Exceptions

Standard JSON-RPC error codes and exceptions as defined in the JSON-RPC 2.0 specification.

```python { .api }
class JsonRpcParseError(JsonRpcException):
    """Parse error (-32700) - Invalid JSON received."""
    
class JsonRpcInvalidRequest(JsonRpcException):
    """Invalid request (-32600) - JSON is not a valid request object."""
    
class JsonRpcMethodNotFound(JsonRpcException):
    """Method not found (-32601) - Method does not exist or is not available."""
    
class JsonRpcInvalidParams(JsonRpcException):
    """Invalid parameters (-32602) - Invalid method parameters."""
    
class JsonRpcInternalError(JsonRpcException):
    """Internal error (-32603) - Internal JSON-RPC error."""
```

### LSP-Specific JSON-RPC Exceptions

Language Server Protocol specific error codes and exceptions.

```python { .api }
class JsonRpcServerNotInitialized(JsonRpcException):
    """Server not initialized (-32002) - Server received request before initialize."""
    
class JsonRpcUnknownErrorCode(JsonRpcException):
    """Unknown error code (-32001) - Unknown error."""
    
class JsonRpcRequestCancelled(JsonRpcException):
    """Request cancelled (-32800) - Request was cancelled."""
    
class JsonRpcContentModified(JsonRpcException):
    """Content modified (-32801) - Content was modified while processing request."""
```

### Reserved Error Range Exceptions

Exceptions for JSON-RPC and LSP reserved error code ranges.

```python { .api }
class JsonRpcReservedErrorRangeStart(JsonRpcException):
    """Reserved error range start (-32099)."""
    
class JsonRpcReservedErrorRangeEnd(JsonRpcException):
    """Reserved error range end (-32000)."""
    
class LspReservedErrorRangeStart(JsonRpcException):
    """LSP reserved error range start (-32899)."""
    
class LspReservedErrorRangeEnd(JsonRpcException):
    """LSP reserved error range end (-32800)."""
    
class JsonRpcServerError(JsonRpcException):
    """Server error (custom range -32099 to -32000)."""
```

### pygls Feature Exceptions

Exceptions specific to pygls feature registration and execution.

```python { .api }
class FeatureAlreadyRegisteredError(PyglsError):
    """
    Exception raised when attempting to register a feature that is already registered.
    
    Prevents duplicate feature registration which could cause conflicts
    in method resolution and handler execution.
    """
    
class FeatureRequestError(PyglsError):
    """
    Exception raised during feature request handling.
    
    Used to wrap and propagate errors that occur during the
    execution of registered LSP feature handlers.
    """
    
class FeatureNotificationError(PyglsError):
    """
    Exception raised during feature notification handling.
    
    Used to wrap and propagate errors that occur during the
    execution of registered LSP notification handlers.
    """

class CommandAlreadyRegisteredError(PyglsError):
    """
    Exception raised when attempting to register a command that is already registered.
    
    Prevents duplicate command registration which could cause conflicts
    in command resolution and handler execution.
    """
```

### Protocol and Type Exceptions

Exceptions related to protocol handling and type registration.

```python { .api }
class MethodTypeNotRegisteredError(PyglsError):
    """
    Exception raised when a method type is not registered.
    
    Occurs when attempting to use LSP methods or features
    that have not been properly registered with the server.
    """

class ThreadDecoratorError(PyglsError):
    """
    Exception raised when there are issues with thread decorator usage.
    
    Occurs when the @thread decorator is used incorrectly or
    when thread pool configuration issues prevent proper execution.
    """

class ValidationError(PyglsError):
    """
    Exception raised during data validation.
    
    Used when LSP message parameters or responses fail
    validation against their expected schemas or types.
    """
```

## Usage Examples

### Basic Exception Handling

```python
from pygls.server import LanguageServer
from pygls.exceptions import PyglsError, JsonRpcException
from lsprotocol.types import TEXT_DOCUMENT_COMPLETION

server = LanguageServer("exception-example", "1.0.0")

@server.feature(TEXT_DOCUMENT_COMPLETION)
def completion(params):
    try:
        # Feature implementation that might fail
        document = server.workspace.get_document(params.text_document.uri)
        # Process document...
        return {"items": []}
    except PyglsError as e:
        # Handle pygls-specific errors
        server.show_message(f"pygls error: {e}", MessageType.Error)
        raise
    except JsonRpcException as e:
        # Handle JSON-RPC protocol errors
        server.show_message(f"Protocol error: {e.message}", MessageType.Error)
        raise
    except Exception as e:
        # Handle unexpected errors
        server.show_message(f"Unexpected error: {e}", MessageType.Error)
        raise JsonRpcInternalError(code=-32603, message="Internal server error")
```

### Feature Registration Error Handling

```python
from pygls.server import LanguageServer
from pygls.exceptions import FeatureAlreadyRegisteredError, CommandAlreadyRegisteredError

server = LanguageServer("registration-example", "1.0.0")

def safe_register_feature(feature_name, handler):
    """Safely register a feature with error handling."""
    try:
        @server.feature(feature_name)
        def feature_handler(params):
            return handler(params)
    except FeatureAlreadyRegisteredError:
        server.show_message(f"Feature {feature_name} already registered", MessageType.Warning)

def safe_register_command(command_name, handler):
    """Safely register a command with error handling."""
    try:
        @server.command(command_name)
        def command_handler(params):
            return handler(params)
    except CommandAlreadyRegisteredError:
        server.show_message(f"Command {command_name} already registered", MessageType.Warning)
```

### Protocol Error Handling

```python
from pygls.protocol import LanguageServerProtocol
from pygls.exceptions import (
    JsonRpcParseError, 
    JsonRpcInvalidRequest,
    JsonRpcMethodNotFound
)

class CustomProtocol(LanguageServerProtocol):
    def handle_request(self, request):
        try:
            return super().handle_request(request)
        except JsonRpcParseError:
            # Handle JSON parsing errors
            self.send_notification("window/logMessage", {
                "type": 1,  # Error
                "message": "Failed to parse JSON-RPC request"
            })
            raise
        except JsonRpcMethodNotFound:
            # Handle unknown method calls
            self.send_notification("window/logMessage", {
                "type": 2,  # Warning
                "message": f"Unknown method: {request.get('method', 'unknown')}"
            })
            raise
```

### Validation Error Handling

```python
from pygls.exceptions import ValidationError
from lsprotocol.types import Position, Range

def validate_position(position, lines):
    """Validate position against document lines."""
    if position.line < 0 or position.line >= len(lines):
        raise ValidationError(f"Line {position.line} out of range (0-{len(lines)-1})")
    
    line = lines[position.line]
    if position.character < 0 or position.character > len(line):
        raise ValidationError(f"Character {position.character} out of range (0-{len(line)})")

def validate_range(range_obj, lines):
    """Validate range against document lines."""
    try:
        validate_position(range_obj.start, lines)
        validate_position(range_obj.end, lines)
        
        if range_obj.start.line > range_obj.end.line:
            raise ValidationError("Range start line cannot be after end line")
        
        if (range_obj.start.line == range_obj.end.line and 
            range_obj.start.character > range_obj.end.character):
            raise ValidationError("Range start character cannot be after end character")
            
    except ValidationError:
        # Re-raise validation errors
        raise
    except Exception as e:
        # Convert other errors to validation errors
        raise ValidationError(f"Range validation failed: {e}")
```

### Custom Error Reporting

```python
from pygls.server import LanguageServer
from pygls.exceptions import PyglsError
from lsprotocol.types import MessageType

class CustomLanguageServer(LanguageServer):
    def report_server_error(self, error, source):
        """Override error reporting with custom behavior."""
        if isinstance(error, PyglsError):
            # Handle pygls-specific errors
            self.show_message(f"Server error: {error}", MessageType.Error)
        else:
            # Handle all other errors
            self.show_message(f"Unexpected error: {error}", MessageType.Error)
        
        # Call parent implementation
        super().report_server_error(error, source)
```