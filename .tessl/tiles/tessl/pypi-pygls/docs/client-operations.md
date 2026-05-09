# Client Operations

Client-side functionality for connecting to language servers, handling server responses, and building language client applications with support for multiple transport methods and message handling.

## Capabilities

### JSON-RPC Client

Base client implementation for connecting to JSON-RPC servers with support for multiple transport methods and message handling.

```python { .api }
class JsonRPCClient:
    """
    JSON-RPC client for connecting to language servers.
    
    Provides client-side JSON-RPC communication with support for stdio,
    TCP, and WebSocket transports, plus message routing and response handling.
    """
    
    def __init__(
        self,
        protocol_cls: Type[JsonRPCProtocol] = None,
        converter_factory: Callable[[], Converter] = None
    ):
        """
        Initialize JSON-RPC client.
        
        Parameters:
        - protocol_cls: Type[JsonRPCProtocol] - Protocol class for communication
        - converter_factory: Callable - Factory for creating message converters
        """
    
    async def start_io(self, cmd: str, *args, **kwargs) -> None:
        """
        Start server process and communicate over stdio.
        
        Parameters:
        - cmd: str - Server command to execute
        - *args: Additional command arguments  
        - **kwargs: Additional keyword arguments for subprocess
        """
    
    def start_tcp(self, host: str, port: int) -> None:
        """
        Start client with TCP transport.
        
        Parameters:
        - host: str - Server host address
        - port: int - Server port number
        """
    
    def start_ws(self, host: str, port: int) -> None:
        """
        Start client with WebSocket transport.
        
        Parameters:
        - host: str - Server host address
        - port: int - Server port number
        """
    
    def feature(self, method_name: str) -> Callable[[F], F]:
        """
        Decorator for registering notification handlers.
        
        Parameters:
        - method_name: str - LSP method name to handle
        
        Returns:
        Decorator function for handler registration
        """
    
    
    async def server_exit(self, server: 'asyncio.subprocess.Process') -> None:
        """
        Called when server process exits (overridable).
        
        Parameters:
        - server: asyncio.subprocess.Process - Exited server process
        """
    
    def report_server_error(
        self,
        error: Exception,
        source: Union[PyglsError, JsonRpcException]
    ) -> None:
        """
        Report server errors (overridable).
        
        Parameters:
        - error: Exception - Error that occurred
        - source: Union[PyglsError, JsonRpcException] - Error source
        """
    
    async def stop(self) -> None:
        """Stop the client and clean up resources."""
    
    @property
    def stopped(self) -> bool:
        """Whether client has been stopped."""
    
    @property
    def protocol(self) -> JsonRPCProtocol:
        """Access to underlying protocol instance."""
```

### Base Language Server Client

Generated LSP client with all standard Language Server Protocol methods for comprehensive server interaction.

```python { .api }
class BaseLanguageClient(JsonRPCClient):
    """
    LSP client with complete Language Server Protocol method support.
    
    Auto-generated client providing all standard LSP requests and
    notifications with proper parameter typing and response handling.
    """
    
    # Note: This class contains numerous auto-generated methods
    # for all LSP features. Key examples include:
    
    async def text_document_completion_async(
        self, 
        params: CompletionParams
    ) -> Union[List[CompletionItem], CompletionList, None]:
        """Send completion request to server."""
    
    async def text_document_hover_async(
        self, 
        params: HoverParams
    ) -> Optional[Hover]:
        """Send hover request to server."""
    
    async def text_document_definition_async(
        self, 
        params: DefinitionParams
    ) -> Union[Location, List[Location], List[LocationLink], None]:
        """Send go-to-definition request to server."""
    
    def text_document_did_open(self, params: DidOpenTextDocumentParams) -> None:
        """Send document open notification."""
    
    def text_document_did_change(self, params: DidChangeTextDocumentParams) -> None:
        """Send document change notification."""
    
    def text_document_did_close(self, params: DidCloseTextDocumentParams) -> None:
        """Send document close notification."""
```

## Usage Examples

### Basic Client Setup

```python
import asyncio
from pygls.client import JsonRPCClient
from lsprotocol.types import (
    InitializeParams,
    ClientCapabilities,
    TextDocumentClientCapabilities,
    CompletionClientCapabilities
)

class LanguageServerClient(JsonRPCClient):
    def __init__(self):
        super().__init__()
        self.server_capabilities = None
    
    async def initialize_server(self):
        """Initialize connection with language server."""
        # Send initialize request
        initialize_params = InitializeParams(
            process_id=os.getpid(),
            root_uri="file:///path/to/project",
            capabilities=ClientCapabilities(
                text_document=TextDocumentClientCapabilities(
                    completion=CompletionClientCapabilities(
                        dynamic_registration=True,
                        completion_item={
                            "snippet_support": True,
                            "documentation_format": ["markdown", "plaintext"]
                        }
                    )
                )
            ),
            initialization_options={}
        )
        
        result = await self.protocol.send_request(
            "initialize", 
            initialize_params
        )
        
        self.server_capabilities = result.capabilities
        
        # Send initialized notification
        self.protocol.send_notification("initialized", {})
        
        return result
    
    async def shutdown_server(self):
        """Gracefully shutdown server connection."""
        await self.protocol.send_request("shutdown", None)
        self.protocol.send_notification("exit", None)

# Usage
async def main():
    client = LanguageServerClient()
    
    try:
        # Start client with stdio to connect to server
        client.start_io(sys.stdin, sys.stdout)
        
        # Initialize server
        init_result = await client.initialize_server()
        print(f"Server initialized: {init_result.server_info.name}")
        
        # Use server features...
        
    finally:
        await client.shutdown_server()

asyncio.run(main())
```

### Handling Server Notifications

```python
from pygls.client import JsonRPCClient

class NotificationHandlingClient(JsonRPCClient):
    def __init__(self):
        super().__init__()
        self.diagnostics = {}
    
    @self.feature("textDocument/publishDiagnostics")
    def handle_diagnostics(self, params):
        """Handle diagnostic notifications from server."""
        uri = params.uri
        diagnostics = params.diagnostics
        
        self.diagnostics[uri] = diagnostics
        
        print(f"Received {len(diagnostics)} diagnostics for {uri}")
        for diagnostic in diagnostics:
            print(f"  {diagnostic.severity}: {diagnostic.message}")
    
    @self.feature("window/logMessage")
    def handle_log_message(self, params):
        """Handle log messages from server."""
        print(f"Server log [{params.type}]: {params.message}")
    
    @self.feature("window/showMessage")
    def handle_show_message(self, params):
        """Handle show message requests from server."""
        print(f"Server message [{params.type}]: {params.message}")
```

### Interactive Client Operations

```python
import asyncio
from lsprotocol.types import (
    DidOpenTextDocumentParams,
    TextDocumentItem,
    CompletionParams,
    Position,
    TextDocumentIdentifier
)

class InteractiveClient(JsonRPCClient):
    async def open_document(self, uri: str, content: str):
        """Open a document on the server."""
        params = DidOpenTextDocumentParams(
            text_document=TextDocumentItem(
                uri=uri,
                language_id="python",
                version=1,
                text=content
            )
        )
        
        self.protocol.send_notification("textDocument/didOpen", params)
        print(f"Opened document: {uri}")
    
    async def get_completions(self, uri: str, line: int, character: int):
        """Request completions at a specific position."""
        params = CompletionParams(
            text_document=TextDocumentIdentifier(uri=uri),
            position=Position(line=line, character=character)
        )
        
        result = await self.protocol.send_request(
            "textDocument/completion",
            params
        )
        
        if result:
            if isinstance(result, list):
                return result
            else:
                return result.items
        return []
    
    async def get_hover(self, uri: str, line: int, character: int):
        """Request hover information at a specific position."""
        params = HoverParams(
            text_document=TextDocumentIdentifier(uri=uri),
            position=Position(line=line, character=character)
        )
        
        result = await self.protocol.send_request(
            "textDocument/hover",
            params
        )
        
        return result

# Interactive usage
async def interactive_session():
    client = InteractiveClient()
    client.start_tcp("localhost", 8080)
    
    await client.initialize_server()
    
    # Open a Python file
    python_code = '''
def hello_world():
    print("Hello, world!")
    return "success"

hello_world().
'''
    
    await client.open_document("file:///test.py", python_code)
    
    # Get completions after the dot
    completions = await client.get_completions("file:///test.py", 4, 15)
    print("Available completions:", [item.label for item in completions])
    
    # Get hover info for function name
    hover = await client.get_hover("file:///test.py", 1, 4)
    if hover:
        print("Hover content:", hover.contents)
    
    await client.shutdown_server()

asyncio.run(interactive_session())
```

### Client with Custom Protocol

```python
from pygls.protocol import JsonRPCProtocol

class CustomClientProtocol(JsonRPCProtocol):
    def __init__(self, client, converter):
        super().__init__(client, converter)
        self.custom_features = {}
    
    async def send_custom_request(self, method: str, params: Any):
        """Send custom request to server."""
        return await self.send_request(f"custom/{method}", params)
    
    def handle_custom_notification(self, method: str, params: Any):
        """Handle custom notifications from server."""
        handler = self.custom_features.get(method)
        if handler:
            handler(params)

class CustomClient(JsonRPCClient):
    def __init__(self):
        super().__init__(protocol_cls=CustomClientProtocol)
    
    def register_custom_feature(self, method: str, handler: Callable):
        """Register handler for custom server notifications."""
        self.protocol.custom_features[method] = handler
    
    async def call_custom_feature(self, feature: str, params: Any):
        """Call custom server feature."""
        return await self.protocol.send_custom_request(feature, params)

# Usage with custom protocol
client = CustomClient()

# Register custom notification handler
client.register_custom_feature(
    "analysis_complete",
    lambda params: print(f"Analysis completed: {params}")
)

# Call custom server feature
result = await client.call_custom_feature("analyze_project", {
    "path": "/path/to/project",
    "deep_analysis": True
})
```

### Error Handling and Reconnection

```python
import time
from pygls.exceptions import JsonRpcException

class RobustClient(JsonRPCClient):
    def __init__(self, max_retries: int = 3):
        super().__init__()
        self.max_retries = max_retries
        self.connected = False
    
    async def connect_with_retry(self, host: str, port: int):
        """Connect with automatic retry logic."""
        for attempt in range(self.max_retries):
            try:
                self.start_tcp(host, port)
                await self.initialize_server()
                self.connected = True
                print(f"Connected to server on attempt {attempt + 1}")
                return
                
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise ConnectionError("Failed to connect after all retry attempts")
    
    async def safe_request(self, method: str, params: Any, timeout: float = 10.0):
        """Send request with error handling and timeout."""
        if not self.connected:
            raise ConnectionError("Not connected to server")
        
        try:
            result = await asyncio.wait_for(
                self.protocol.send_request(method, params),
                timeout=timeout
            )
            return result
            
        except asyncio.TimeoutError:
            print(f"Request {method} timed out after {timeout}s")
            return None
            
        except JsonRpcException as e:
            print(f"LSP error in {method}: {e}")
            return None
            
        except Exception as e:
            print(f"Unexpected error in {method}: {e}")
            return None
    
    def connection_lost(self, exc):
        """Handle connection loss."""
        self.connected = False
        print(f"Connection lost: {exc}")
        
        # Trigger reconnection logic if needed
        asyncio.create_task(self.reconnect())
    
    async def reconnect(self):
        """Attempt to reconnect to server."""
        print("Attempting to reconnect...")
        try:
            await self.connect_with_retry("localhost", 8080)
        except ConnectionError:
            print("Reconnection failed")
```