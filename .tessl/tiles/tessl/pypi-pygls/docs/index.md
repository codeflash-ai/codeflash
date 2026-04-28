# pygls

A pythonic generic language server framework that implements the Language Server Protocol (LSP) for building custom language servers in Python. pygls provides comprehensive server lifecycle management, LSP feature decorators, workspace document management, async/await support, WebSocket capabilities for browser-based editors, and built-in progress reporting.

## Package Information

- **Package Name**: pygls
- **Language**: Python
- **Installation**: `pip install pygls`
- **Optional WebSocket Support**: `pip install pygls[ws]`

## Core Imports

```python
from pygls.server import LanguageServer
```

For protocol customization:

```python
from pygls.protocol import LanguageServerProtocol
from pygls.server import LanguageServer, Server
```

For client development:

```python
from pygls.client import JsonRPCClient
```

For workspace management:

```python
from pygls.workspace import Workspace, TextDocument, PositionCodec
```

For URI utilities:

```python
from pygls.uris import from_fs_path, to_fs_path, uri_scheme
```

For exception handling:

```python
from pygls.exceptions import PyglsError, JsonRpcException, FeatureAlreadyRegisteredError
```

## Basic Usage

```python
from pygls.server import LanguageServer
from lsprotocol.types import (
    TEXT_DOCUMENT_COMPLETION,
    CompletionItem,
    CompletionList,
    CompletionParams,
)

# Create a language server instance
server = LanguageServer("my-language-server", "v1.0.0")

# Register a completion feature
@server.feature(TEXT_DOCUMENT_COMPLETION)
def completions(params: CompletionParams):
    document = server.workspace.get_document(params.text_document.uri)
    current_line = document.lines[params.position.line].strip()
    
    items = []
    if current_line.endswith("hello."):
        items = [
            CompletionItem(label="world"),
            CompletionItem(label="friend"),
        ]
    
    return CompletionList(is_incomplete=False, items=items)

# Register a custom command
@server.command("myCustomCommand")
def my_command(params):
    return {"result": "Command executed successfully"}

# Start the server
if __name__ == "__main__":
    server.start_io()  # Use stdio transport
```

## Architecture

pygls follows a layered architecture designed for maximum flexibility and LSP compliance:

- **Server Layer**: `LanguageServer` and `Server` classes handle server lifecycle, transport management, and feature registration
- **Protocol Layer**: `LanguageServerProtocol` and `JsonRPCProtocol` implement LSP message handling and JSON-RPC communication
- **Workspace Layer**: `Workspace`, `TextDocument`, and `PositionCodec` manage document synchronization and text operations
- **Feature Layer**: Decorators (`@server.feature`, `@server.command`, `@server.thread`) enable declarative feature registration
- **Transport Layer**: Adapters for stdio, TCP, and WebSocket connections support various editor integrations

This design enables custom language servers to integrate seamlessly with VSCode, Vim, Emacs, and other LSP-compatible editors while providing Python-native patterns for rapid development.

## Capabilities

### Server Creation and Management

Core functionality for creating, configuring, and running language servers with support for multiple transport methods and server lifecycle management.

```python { .api }
class LanguageServer(Server):
    def __init__(
        self,
        name: str,
        version: str,
        protocol_cls: Type[LanguageServerProtocol] = None,
        max_workers: int = 2,
        text_document_sync_kind: TextDocumentSyncKind = TextDocumentSyncKind.Incremental
    ): ...
    
    def start_io(self) -> None: ...
    def start_tcp(self, host: str, port: int) -> None: ...
    def start_ws(self, host: str, port: int) -> None: ...
```

[Server Management](./server-management.md)

### Feature Registration and Decorators

Decorator-based system for registering LSP features, custom commands, and thread execution patterns with automatic capability registration.

```python { .api }
def feature(
    self, 
    method_name: str, 
    options: Any = None
) -> Callable[[F], F]: ...

def command(self, command_name: str) -> Callable[[F], F]: ...

def thread(self) -> Callable[[F], F]: ...
```

[Feature Registration](./feature-registration.md)

### Protocol and Message Handling

Low-level protocol handling for JSON-RPC communication, LSP message processing, and custom protocol extensions.

```python { .api }
class LanguageServerProtocol(JsonRPCProtocol):
    def send_request(self, method: str, params: Any = None) -> Future: ...
    def send_notification(self, method: str, params: Any = None) -> None: ...
    def lsp_initialize(self, params: InitializeParams) -> InitializeResult: ...
```

[Protocol Handling](./protocol-handling.md)

### Workspace and Document Management

Comprehensive document synchronization, workspace operations, and text manipulation with position encoding support.

```python { .api }
class Workspace:
    def get_document(self, doc_uri: str) -> TextDocument: ...
    def put_document(self, doc_uri: str, source: str, version: int = None) -> TextDocument: ...
    def remove_document(self, doc_uri: str) -> None: ...
    def apply_edit(self, edit: WorkspaceEdit) -> None: ...

class TextDocument:
    def apply_change(self, change: TextDocumentContentChangeEvent) -> None: ...
    def offset_at_position(self, position: Position) -> int: ...
    def word_at_position(self, position: Position) -> str: ...
```

[Workspace Management](./workspace-management.md)

### Client Operations

Client-side functionality for connecting to language servers and handling server responses.

```python { .api }
class JsonRPCClient:
    def __init__(
        self,
        protocol_cls: Type[JsonRPCProtocol] = None,
        converter_factory: Callable[[], Converter] = None
    ): ...
    
    def start_io(self, stdin: TextIO, stdout: TextIO) -> None: ...
    def start_tcp(self, host: str, port: int) -> None: ...
    def start_ws(self, host: str, port: int) -> None: ...
```

[Client Operations](./client-operations.md)

### Progress Reporting

Built-in progress reporting system for long-running operations with client-side progress bar integration.

```python { .api }
class Progress:
    def create_task(
        self,
        token: ProgressToken,
        title: str,
        cancellable: bool = True,
        message: str = None,
        percentage: int = None
    ) -> Future: ...
    
    def update(
        self,
        token: ProgressToken,
        message: str = None,
        percentage: int = None
    ) -> None: ...
    
    def end(self, token: ProgressToken, message: str = None) -> None: ...
```

[Progress Reporting](./progress-reporting.md)

### Utilities and Helpers

URI handling, position encoding, exception management, and other utility functions for language server development.

```python { .api }
# URI Utilities
def from_fs_path(path: str) -> str: ...
def to_fs_path(uri: str) -> str: ...

# Position Encoding
class PositionCodec:
    def position_from_client_units(self, lines: List[str], position: Position) -> Position: ...
    def position_to_client_units(self, lines: List[str], position: Position) -> Position: ...
```

[Utilities](./utilities.md)

### URI Utilities

Cross-platform URI handling and path conversion utilities for working with Language Server Protocol URIs and filesystem paths across different platforms.

```python { .api }
def from_fs_path(path: str) -> Optional[str]: ...
def to_fs_path(uri: str) -> Optional[str]: ...
def uri_scheme(uri: str) -> Optional[str]: ...
def uri_with(uri: str, **components) -> str: ...
```

[URI Utilities](./uri-utilities.md)

### Exception Handling

Exception classes for pygls error handling, JSON-RPC protocol errors, and language server specific exceptions with proper error codes and messages.

```python { .api }
class PyglsError(Exception): ...
class JsonRpcException(Exception): ...
class JsonRpcParseError(JsonRpcException): ...
class JsonRpcInvalidRequest(JsonRpcException): ...
class JsonRpcMethodNotFound(JsonRpcException): ...
class FeatureAlreadyRegisteredError(PyglsError): ...
class CommandAlreadyRegisteredError(PyglsError): ...
```

[Exception Handling](./exception-handling.md)