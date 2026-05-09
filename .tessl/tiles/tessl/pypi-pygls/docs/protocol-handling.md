# Protocol and Message Handling

Low-level protocol handling for JSON-RPC communication, LSP message processing, custom protocol extensions, and message routing with built-in lifecycle management.

## Capabilities

### Language Server Protocol

Core LSP protocol implementation with built-in handlers for standard LSP methods and extensibility for custom protocols.

```python { .api }
class LanguageServerProtocol(JsonRPCProtocol):
    """
    Language Server Protocol implementation with standard LSP handlers.
    
    Provides built-in handlers for initialize, shutdown, document lifecycle,
    and workspace operations with extensible architecture for custom features.
    """
    
    def get_message_handler(self) -> Callable: ...
    
    def lsp_initialize(self, params: InitializeParams) -> InitializeResult:
        """Handle LSP initialize request."""
    
    def lsp_shutdown(self, params: Any) -> None:
        """Handle LSP shutdown request."""
    
    def lsp_exit(self, params: Any) -> None:
        """Handle LSP exit notification."""
    
    def lsp_text_document_did_open(self, params: DidOpenTextDocumentParams) -> None:
        """Handle document open notification."""
    
    def lsp_text_document_did_change(self, params: DidChangeTextDocumentParams) -> None:
        """Handle document change notification."""
    
    def lsp_text_document_did_close(self, params: DidCloseTextDocumentParams) -> None:
        """Handle document close notification."""
    
    def lsp_workspace_did_change_workspace_folders(self, params: DidChangeWorkspaceFoldersParams) -> None:
        """Handle workspace folder changes."""
    
    def lsp_workspace_execute_command(self, params: ExecuteCommandParams) -> Any:
        """Handle command execution requests."""
```

### JSON-RPC Protocol

Base JSON-RPC protocol implementation for message transport, request/response handling, and connection management.

```python { .api }
class JsonRPCProtocol(asyncio.Protocol):
    """
    Base JSON-RPC protocol for message transport and communication.
    
    Handles connection lifecycle, message parsing, request routing,
    and response management for JSON-RPC communication.
    """
    
    def connection_made(self, transport: asyncio.Transport) -> None:
        """Called when connection is established."""
    
    def connection_lost(self, exc: Exception) -> None:
        """Called when connection is lost."""
    
    def data_received(self, data: bytes) -> None:
        """Process incoming data and parse JSON-RPC messages."""
    
    def send_request(self, method: str, params: Any = None) -> Future:
        """
        Send JSON-RPC request and return future for response.
        
        Parameters:
        - method: str - RPC method name
        - params: Any - Method parameters
        
        Returns:
        Future that resolves to the response
        """
    
    def send_notification(self, method: str, params: Any = None) -> None:
        """
        Send JSON-RPC notification (no response expected).
        
        Parameters:
        - method: str - RPC method name  
        - params: Any - Method parameters
        """
```

### Protocol Message Types

Core message type definitions for JSON-RPC communication with structured request/response handling.

```python { .api }
@attrs.define
class JsonRPCRequestMessage:
    """JSON-RPC request message structure."""
    id: Union[int, str]
    method: str
    params: Any = None
    jsonrpc: str = "2.0"

@attrs.define  
class JsonRPCResponseMessage:
    """JSON-RPC response message structure."""
    id: Union[int, str]
    result: Any = None
    error: Any = None
    jsonrpc: str = "2.0"

@attrs.define
class JsonRPCNotification:
    """JSON-RPC notification message structure."""
    method: str
    params: Any = None
    jsonrpc: str = "2.0"
```

### Protocol Utilities

Utility functions for protocol configuration, message conversion, and type handling.

```python { .api }
def default_converter() -> Converter:
    """
    Create default cattrs converter with LSP-specific hooks.
    
    Returns:
    Configured converter for LSP message serialization/deserialization
    """

def _dict_to_object(d: Any) -> Any:
    """Convert dictionary to nested object structure."""

def _params_field_structure_hook(obj: Dict, cls: Type) -> Any:
    """Structure hook for handling params field in messages."""

def _result_field_structure_hook(obj: Dict, cls: Type) -> Any:
    """Structure hook for handling result field in messages."""
```

### Server Capabilities

System for building and managing server capability declarations for LSP initialization.

```python { .api }
class ServerCapabilitiesBuilder:
    """
    Builder for constructing server capabilities during initialization.
    
    Automatically configures capabilities based on registered features
    and provides manual capability configuration for advanced use cases.
    """
    
    # Capability configuration methods for various LSP features
    # (specific methods depend on LSP specification)
```

## Usage Examples

### Custom Protocol Extension

```python
from pygls.protocol import LanguageServerProtocol
from pygls.server import LanguageServer

class CustomProtocol(LanguageServerProtocol):
    def __init__(self, server, converter):
        super().__init__(server, converter)
        self.custom_state = {}
    
    def lsp_initialize(self, params):
        # Call parent initialization
        result = super().lsp_initialize(params)
        
        # Add custom initialization logic
        self.custom_state['client_name'] = params.client_info.name if params.client_info else "Unknown"
        
        # Extend server capabilities
        result.capabilities.experimental = {
            "customFeature": True,
            "version": "1.0.0"
        }
        
        return result
    
    # Add custom message handler
    @lsp_method("custom/specialRequest")
    def handle_special_request(self, params):
        return {
            "result": "Custom protocol handled",
            "client": self.custom_state.get('client_name')
        }

# Use custom protocol
server = LanguageServer(
    "custom-server", 
    "1.0.0",
    protocol_cls=CustomProtocol
)
```

### Manual Message Sending

```python
from pygls.server import LanguageServer
from lsprotocol.types import MessageType

server = LanguageServer("message-sender", "1.0.0")

@server.feature(TEXT_DOCUMENT_DID_OPEN)
def on_open(params):
    # Send notification to client
    server.lsp.send_notification(
        "window/logMessage",
        {
            "type": MessageType.Info,
            "message": f"Opened document: {params.text_document.uri}"
        }
    )

@server.command("myServer.requestConfiguration")
async def request_config(params):
    # Send request and wait for response
    try:
        config_response = await server.lsp.send_request(
            "workspace/configuration",
            {
                "items": [
                    {"section": "myServer.formatting"},
                    {"section": "myServer.linting"}
                ]
            }
        )
        
        return {"configuration": config_response}
        
    except Exception as e:
        return {"error": str(e)}
```

### Error Handling and Exceptions

```python
from pygls.exceptions import (
    JsonRpcException,
    JsonRpcInternalError,
    FeatureRequestError
)

class RobustProtocol(LanguageServerProtocol):
    def lsp_text_document_did_change(self, params):
        try:
            # Call parent handler
            super().lsp_text_document_did_change(params)
            
            # Custom change processing
            document = self.workspace.get_document(params.text_document.uri)
            self.validate_document(document)
            
        except Exception as e:
            # Log error but don't propagate to avoid breaking protocol
            self.server.logger.error(f"Error processing document change: {e}")
    
    def validate_document(self, document):
        # Custom validation that might raise exceptions
        if len(document.source) > 1000000:
            raise JsonRpcInternalError("Document too large")

@server.feature(TEXT_DOCUMENT_HOVER)
def safe_hover(params):
    try:
        # Hover implementation
        result = generate_hover_content(params)
        return result
        
    except FileNotFoundError:
        # Return None for no hover content
        return None
        
    except Exception as e:
        # Convert to LSP error
        raise FeatureRequestError(f"Hover failed: {str(e)}")
```

### Connection Monitoring

```python
class MonitoredProtocol(LanguageServerProtocol):
    def connection_made(self, transport):
        super().connection_made(transport)
        self.server.logger.info("Client connected")
        
        # Setup connection monitoring
        self.connection_start_time = time.time()
        self.message_count = 0
    
    def connection_lost(self, exc):
        duration = time.time() - self.connection_start_time
        self.server.logger.info(
            f"Client disconnected after {duration:.2f}s, "
            f"processed {self.message_count} messages"
        )
        
        super().connection_lost(exc)
    
    def data_received(self, data):
        self.message_count += 1
        super().data_received(data)
```

### Custom Message Converter

```python
from pygls.protocol import default_converter
import cattrs

def create_custom_converter():
    converter = default_converter()
    
    # Add custom type conversion
    converter.register_structure_hook(
        MyCustomType,
        lambda obj, cls: MyCustomType(**obj)
    )
    
    converter.register_unstructure_hook(
        MyCustomType,
        lambda obj: {"custom_field": obj.value}
    )
    
    return converter

# Use custom converter
server = LanguageServer(
    "custom-converter-server",
    "1.0.0",
    converter_factory=create_custom_converter
)
```