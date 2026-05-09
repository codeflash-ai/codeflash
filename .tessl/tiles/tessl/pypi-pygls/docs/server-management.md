# Server Creation and Management

Core functionality for creating, configuring, and running language servers with support for multiple transport methods, server lifecycle management, and thread pool configuration.

## Capabilities

### LanguageServer Class

Main server class that provides the primary interface for building custom language servers with built-in LSP compliance and feature registration.

```python { .api }
class LanguageServer(Server):
    """
    The default LanguageServer implementation.
    
    Parameters:
    - name: str - Name of the server for client identification
    - version: str - Version of the server for client identification  
    - protocol_cls: Type[LanguageServerProtocol] - Protocol class (defaults to LanguageServerProtocol)
    - max_workers: int - Maximum workers for thread pools (default: 2)
    - text_document_sync_kind: TextDocumentSyncKind - Document sync method (default: Incremental)
    """
    def __init__(
        self,
        name: str,
        version: str, 
        protocol_cls: Type[LanguageServerProtocol] = None,
        max_workers: int = 2,
        text_document_sync_kind: TextDocumentSyncKind = TextDocumentSyncKind.Incremental
    ): ...
```

### Server Transport Methods

Methods for starting the language server with different transport mechanisms to support various editor integrations.

```python { .api }
def start_io(self) -> None:
    """Start server using stdin/stdout transport for direct editor integration."""

def start_tcp(self, host: str, port: int) -> None:
    """
    Start server using TCP transport.
    
    Parameters:
    - host: str - Host address to bind to
    - port: int - Port number to listen on
    """

def start_ws(self, host: str, port: int) -> None:
    """
    Start server using WebSocket transport for browser-based editors.
    
    Parameters:
    - host: str - Host address to bind to  
    - port: int - Port number to listen on
    
    Requires: pip install pygls[ws]
    """

def start_pyodide(self) -> None:
    """
    Start server for Pyodide environment (browser-based Python).
    
    Used when running Python in the browser using Pyodide. Handles
    special considerations for browser-based execution environment.
    """
```

### Base Server Class

Lower-level server implementation that provides the foundation for custom server implementations.

```python { .api }
class Server:
    """
    Base server class for custom implementations.
    
    Parameters:
    - protocol_cls: Type[JsonRPCProtocol] - Protocol implementation class
    - converter_factory: Callable[[], Converter] - Factory for cattrs converter
    - loop: Optional[AbstractEventLoop] - Event loop (creates new if None)
    - max_workers: int - Maximum workers for thread pools (default: 2)
    - sync_kind: TextDocumentSyncKind - Document synchronization method
    """
    def __init__(
        self,
        protocol_cls: Type[JsonRPCProtocol],
        converter_factory: Callable[[], Converter],
        loop: Optional[AbstractEventLoop] = None,
        max_workers: int = 2,
        sync_kind: TextDocumentSyncKind = TextDocumentSyncKind.Incremental
    ): ...
    
    def shutdown(self) -> None:
        """Gracefully shutdown the server and clean up resources."""
```

### Server Properties

Access to server resources and configuration for advanced server management and resource monitoring.

```python { .api }
@property
def workspace(self) -> Workspace:
    """Access to the workspace management instance."""

@property  
def lsp(self) -> LanguageServerProtocol:
    """Access to the underlying protocol instance."""

@property
def loop(self) -> AbstractEventLoop:
    """Access to the asyncio event loop."""

@property
def thread_pool(self) -> ThreadPool:
    """Access to the thread pool for synchronous operations."""

@property
def thread_pool_executor(self) -> ThreadPoolExecutor:
    """Access to the thread pool executor for concurrent operations."""
```

### Client Communication Methods

Methods for communicating with the client, sending notifications, and managing client capabilities.

```python { .api }
def publish_diagnostics(
    self,
    uri: str,
    diagnostics: Optional[List[Diagnostic]] = None,
    version: Optional[int] = None,
    **kwargs
) -> None:
    """
    Send diagnostic notifications to the client.
    
    Parameters:
    - uri: str - Document URI
    - diagnostics: Optional[List[Diagnostic]] - List of diagnostic messages
    - version: Optional[int] - Document version
    """

def show_message(self, message: str, msg_type: MessageType = MessageType.Info) -> None:
    """
    Display message to user.
    
    Parameters:
    - message: str - Message text
    - msg_type: MessageType - Message type (Info, Warning, Error, Log)
    """

def show_message_log(self, message: str, msg_type: MessageType = MessageType.Log) -> None:
    """
    Send message to client's output channel.
    
    Parameters:
    - message: str - Message text
    - msg_type: MessageType - Message type
    """

def send_notification(self, method: str, params: object = None) -> None:
    """
    Send notification to client.
    
    Parameters:
    - method: str - LSP method name
    - params: object - Method parameters
    """

def log_trace(self, message: str, verbose: Optional[str] = None) -> None:
    """
    Send trace notification to client.
    
    Parameters:
    - message: str - Trace message
    - verbose: Optional[str] - Additional verbose information
    """
```

### Client Capability Management

Methods for registering and managing client capabilities dynamically.

```python { .api }
def register_capability(
    self, 
    params: RegistrationParams, 
    callback: Optional[Callable[[], None]] = None
) -> Future:
    """
    Register new capability on client.
    
    Parameters:
    - params: RegistrationParams - Registration parameters
    - callback: Optional[Callable] - Completion callback
    
    Returns:
    Future for registration result
    """

def register_capability_async(self, params: RegistrationParams) -> Future:
    """
    Register new capability on client (async).
    
    Parameters:
    - params: RegistrationParams - Registration parameters
    
    Returns:
    Future for registration result
    """

def unregister_capability(
    self,
    params: UnregistrationParams, 
    callback: Optional[Callable[[], None]] = None
) -> Future:
    """
    Unregister capability from client.
    
    Parameters:
    - params: UnregistrationParams - Unregistration parameters
    - callback: Optional[Callable] - Completion callback
    
    Returns:
    Future for unregistration result
    """

def unregister_capability_async(self, params: UnregistrationParams) -> Future:
    """
    Unregister capability from client (async).
    
    Parameters:
    - params: UnregistrationParams - Unregistration parameters
    
    Returns:
    Future for unregistration result
    """
```

### Configuration and Document Management

Methods for requesting configuration from client and managing workspace edits.

```python { .api }
def get_configuration(
    self,
    params: WorkspaceConfigurationParams,
    callback: ConfigCallbackType
) -> Future:
    """
    Request configuration from client.
    
    Parameters:
    - params: WorkspaceConfigurationParams - Configuration request parameters
    - callback: ConfigCallbackType - Callback for configuration result
    
    Returns:
    Future for configuration result
    """

def get_configuration_async(
    self, 
    params: WorkspaceConfigurationParams
) -> Future:
    """
    Request configuration from client (async).
    
    Parameters:
    - params: WorkspaceConfigurationParams - Configuration request parameters
    
    Returns:
    Future for configuration result
    """

def apply_edit(
    self, 
    edit: WorkspaceEdit, 
    label: Optional[str] = None
) -> WorkspaceApplyEditResponse:
    """
    Apply workspace edit on client.
    
    Parameters:
    - edit: WorkspaceEdit - Workspace edit to apply
    - label: Optional[str] - Edit label for UI
    
    Returns:
    WorkspaceApplyEditResponse - Edit application result
    """

def apply_edit_async(
    self, 
    edit: WorkspaceEdit, 
    label: Optional[str] = None
) -> Future:
    """
    Apply workspace edit on client (async).
    
    Parameters:
    - edit: WorkspaceEdit - Workspace edit to apply
    - label: Optional[str] - Edit label for UI
    
    Returns:
    Future for edit application result  
    """

def show_document(
    self,
    params: ShowDocumentParams,
    callback: ShowDocumentCallbackType
) -> Future:
    """
    Show document in client.
    
    Parameters:
    - params: ShowDocumentParams - Document show parameters
    - callback: ShowDocumentCallbackType - Callback for show result
    
    Returns:
    Future for show document result
    """

def show_document_async(self, params: ShowDocumentParams) -> Future:
    """
    Show document in client (async).
    
    Parameters:
    - params: ShowDocumentParams - Document show parameters
    
    Returns:
    Future for show document result
    """
```

## Usage Examples

### Basic Server Setup

```python
from pygls.server import LanguageServer
from lsprotocol.types import TEXT_DOCUMENT_HOVER, Hover, MarkupContent, MarkupKind

# Create server instance
server = LanguageServer("my-language-server", "1.0.0")

# Register a hover feature
@server.feature(TEXT_DOCUMENT_HOVER)
def hover(params):
    return Hover(
        contents=MarkupContent(
            kind=MarkupKind.Markdown,
            value="**Hello** from my language server!"
        )
    )

# Start with stdio transport
server.start_io()
```

### Custom Protocol Server

```python
from pygls.server import Server
from pygls.protocol import LanguageServerProtocol, default_converter

class CustomProtocol(LanguageServerProtocol):
    def __init__(self, server, converter):
        super().__init__(server, converter)
        # Add custom protocol behavior
        
# Create server with custom protocol
server = Server(
    protocol_cls=CustomProtocol,
    converter_factory=default_converter,
    max_workers=4
)
```

### Multi-Transport Server

```python
import asyncio
from pygls.server import LanguageServer

server = LanguageServer("multi-transport-server", "1.0.0")

# Register features here...

# Start multiple transports
async def run_server():
    # Start TCP server in background
    tcp_task = asyncio.create_task(
        server.start_tcp("localhost", 8080)
    )
    
    # Start WebSocket server in background  
    ws_task = asyncio.create_task(
        server.start_ws("localhost", 8081)
    )
    
    # Start stdio in foreground
    server.start_io()

# Run the server
asyncio.run(run_server())
```