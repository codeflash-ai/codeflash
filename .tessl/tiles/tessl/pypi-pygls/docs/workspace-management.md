# Workspace and Document Management

Comprehensive document synchronization, workspace operations, text manipulation with position encoding support, and file system integration for language server implementations.

## Capabilities

### Workspace Class

Central workspace management system that handles document storage, workspace folders, and document lifecycle operations.

```python { .api }
class Workspace:
    """
    Manages documents and workspace operations for language servers.
    
    Provides document storage, retrieval, synchronization, and workspace
    folder management with support for LSP document lifecycle events.
    """
    
    def get_document(self, doc_uri: str) -> TextDocument:
        """
        Retrieve document by URI.
        
        Parameters:
        - doc_uri: str - Document URI
        
        Returns:
        TextDocument instance
        
        Raises:
        KeyError if document not found
        """
    
    def put_document(
        self, 
        doc_uri: str, 
        source: str, 
        version: int = None
    ) -> TextDocument:
        """
        Add or update document in workspace.
        
        Parameters:
        - doc_uri: str - Document URI
        - source: str - Document text content
        - version: int - Document version number
        
        Returns:
        TextDocument instance
        """
    
    def remove_document(self, doc_uri: str) -> None:
        """
        Remove document from workspace.
        
        Parameters:
        - doc_uri: str - Document URI to remove
        """
    
    def apply_edit(self, edit: WorkspaceEdit) -> None:
        """
        Apply workspace edit to documents.
        
        Parameters:
        - edit: WorkspaceEdit - LSP workspace edit structure
        """
    
    def get_text_document(self, doc_uri: str) -> TextDocument:
        """
        Get text document by URI (same as get_document).
        
        Parameters:
        - doc_uri: str - Document URI
        
        Returns:
        TextDocument instance
        """
    
    def put_text_document(
        self, 
        params: DidOpenTextDocumentParams
    ) -> TextDocument:
        """
        Add text document from LSP open parameters.
        
        Parameters:
        - params: DidOpenTextDocumentParams - LSP document open parameters
        
        Returns:
        TextDocument instance
        """
    
    def remove_text_document(self, doc_uri: str) -> None:
        """
        Remove text document by URI.
        
        Parameters:
        - doc_uri: str - Document URI to remove
        """
    
    def update_text_document(
        self, 
        params: DidChangeTextDocumentParams
    ) -> TextDocument:
        """
        Update text document from LSP change parameters.
        
        Parameters:
        - params: DidChangeTextDocumentParams - LSP document change parameters
        
        Returns:
        Updated TextDocument instance
        """
    
    def add_folder(self, folder: WorkspaceFolder) -> None:
        """
        Add workspace folder.
        
        Parameters:
        - folder: WorkspaceFolder - Workspace folder to add
        """
    
    def remove_folder(self, folder_uri: str) -> None:
        """
        Remove workspace folder by URI.
        
        Parameters:
        - folder_uri: str - Workspace folder URI to remove
        """
    
    @property
    def documents(self) -> Dict[str, TextDocument]:
        """Access to document storage mapping."""
    
    @property
    def folders(self) -> List[WorkspaceFolder]:
        """Access to workspace folders."""
    
    @property
    def position_codec(self) -> PositionCodec:
        """Access to position encoding codec."""
    
    @property
    def text_documents(self) -> Dict[str, TextDocument]:
        """Access to text document storage mapping."""
    
    @property
    def root_path(self) -> Optional[str]:
        """Root path of the workspace (deprecated, use root_uri)."""
    
    @property
    def root_uri(self) -> Optional[str]:
        """Root URI of the workspace."""
    
    def is_local(self) -> bool:
        """Check if workspace is local (file:// URI)."""
```

### TextDocument Class

Represents individual text documents with LSP operations, change tracking, and text manipulation utilities.

```python { .api }
class TextDocument:
    """
    Represents a text document with LSP-compatible operations.
    
    Provides text manipulation, position calculations, change application,
    and document metadata management for language server implementations.
    """
    
    def apply_change(self, change: TextDocumentContentChangeEvent) -> None:
        """
        Apply text change to document.
        
        Parameters:
        - change: TextDocumentContentChangeEvent - LSP change event
        """
    
    def offset_at_position(self, position: Position) -> int:
        """
        Get byte offset for position.
        
        Parameters:
        - position: Position - LSP position (line, character)
        
        Returns:
        int - Byte offset in document
        """
    
    def word_at_position(self, position: Position) -> str:
        """
        Get word at specified position.
        
        Parameters:
        - position: Position - LSP position
        
        Returns:
        str - Word at position or empty string
        """
    
    @property
    def uri(self) -> str:
        """Document URI."""
    
    @property  
    def source(self) -> str:
        """Document text content."""
    
    @property
    def lines(self) -> List[str]:
        """Document lines as list."""
    
    @property
    def position_codec(self) -> PositionCodec:
        """Position encoding codec for this document."""
```

### Position Encoding

Position encoding system for handling different character encoding methods between client and server.

```python { .api }
class PositionCodec:
    """
    Handles position encoding between client and server.
    
    Supports different character encoding methods (UTF-8, UTF-16) and
    provides conversion utilities for LSP position calculations.
    """
    
    @classmethod
    def create_encoding(cls, encoding: str) -> 'PositionCodec':
        """
        Create codec for specific encoding.
        
        Parameters:
        - encoding: str - Encoding name ('utf-8', 'utf-16', etc.)
        
        Returns:
        PositionCodec instance
        """
    
    def client_num_units(self, chars: str) -> int:
        """
        Get number of client character units.
        
        Parameters:
        - chars: str - Character string
        
        Returns:
        int - Number of client units
        """
    
    def position_from_client_units(
        self, 
        lines: List[str], 
        position: Position
    ) -> Position:
        """
        Convert position from client units to server units.
        
        Parameters:
        - lines: List[str] - Document lines
        - position: Position - Client position
        
        Returns:
        Position - Server position
        """
    
    def position_to_client_units(
        self, 
        lines: List[str], 
        position: Position
    ) -> Position:
        """
        Convert position from server units to client units.
        
        Parameters:
        - lines: List[str] - Document lines  
        - position: Position - Server position
        
        Returns:
        Position - Client position
        """
    
    def range_from_client_units(
        self, 
        lines: List[str], 
        range: Range
    ) -> Range:
        """
        Convert range from client units to server units.
        
        Parameters:
        - lines: List[str] - Document lines
        - range: Range - Client range
        
        Returns:
        Range - Server range
        """
    
    def range_to_client_units(
        self, 
        lines: List[str], 
        range: Range
    ) -> Range:
        """
        Convert range from server units to client units.
        
        Parameters:
        - lines: List[str] - Document lines
        - range: Range - Server range
        
        Returns:
        Range - Client range
        """
```

## Usage Examples

### Basic Workspace Operations

```python
from pygls.server import LanguageServer
from lsprotocol.types import (
    TEXT_DOCUMENT_DID_OPEN,
    TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_DID_CLOSE,
    Position
)

server = LanguageServer("workspace-example", "1.0.0")

@server.feature(TEXT_DOCUMENT_DID_OPEN)
def on_open(params):
    # Document automatically added to workspace
    document = server.workspace.get_document(params.text_document.uri)
    
    print(f"Opened document: {document.uri}")
    print(f"Line count: {document.line_count}")
    print(f"First line: {document.lines[0] if document.lines else 'Empty'}")

@server.feature(TEXT_DOCUMENT_DID_CHANGE)
def on_change(params):
    # Changes automatically applied to document
    document = server.workspace.get_document(params.text_document.uri)
    
    print(f"Document changed: {document.uri}")
    print(f"New content length: {len(document.source)}")

@server.feature(TEXT_DOCUMENT_DID_CLOSE)
def on_close(params):
    # Remove document from workspace
    server.workspace.remove_document(params.text_document.uri)
    print(f"Closed document: {params.text_document.uri}")
```

### Position and Text Operations

```python
from lsprotocol.types import TEXT_DOCUMENT_HOVER, Position, Range

@server.feature(TEXT_DOCUMENT_HOVER)
def hover(params):
    document = server.workspace.get_document(params.text_document.uri)
    position = params.position
    
    # Get word at cursor position
    word = document.word_at_position(position)
    
    # Get byte offset for position
    offset = document.offset_at_position(position)
    
    # Get line content
    line_text = document.lines[position.line]
    
    return Hover(
        contents=f"Word: '{word}' at offset {offset}\nLine: {line_text}"
    )

@server.command("myServer.getDocumentInfo")
def document_info(params):
    uri = params.arguments[0] if params.arguments else None
    if not uri:
        return {"error": "URI required"}
    
    try:
        document = server.workspace.get_document(uri)
        
        return {
            "uri": document.uri,
            "lineCount": document.line_count,
            "characterCount": len(document.source),
            "firstLine": document.lines[0] if document.lines else "",
            "lastLine": document.lines[-1] if document.lines else ""
        }
    except KeyError:
        return {"error": "Document not found"}
```

### Position Encoding Handling

```python
from pygls.workspace import PositionCodec

@server.feature(TEXT_DOCUMENT_COMPLETION)
def completions(params):
    document = server.workspace.get_document(params.text_document.uri)
    position = params.position
    
    # Use document's position codec for accurate position handling
    codec = document.position_codec
    
    # Convert client position to server position if needed
    server_position = codec.position_from_client_units(
        document.lines, 
        position
    )
    
    # Get completion items
    items = generate_completions(document, server_position)
    
    return CompletionList(items=items)

def generate_completions(document, position):
    # Implementation-specific completion logic
    word = document.word_at_position(position)
    return [CompletionItem(label=f"{word}_completion")]
```

### Workspace Edit Operations

```python
from lsprotocol.types import (
    WorkspaceEdit,
    TextEdit,
    Range,
    Position,
    WORKSPACE_APPLY_EDIT
)

@server.command("myServer.refactorSymbol")
async def refactor_symbol(params):
    old_name = params.arguments[0]
    new_name = params.arguments[1]
    
    # Find all occurrences across workspace
    edits = {}
    
    for uri, document in server.workspace.documents.items():
        text_edits = []
        
        # Find occurrences in document
        for line_idx, line in enumerate(document.lines):
            start = 0
            while True:
                pos = line.find(old_name, start)
                if pos == -1:
                    break
                
                # Create text edit for replacement
                edit_range = Range(
                    start=Position(line=line_idx, character=pos),
                    end=Position(line=line_idx, character=pos + len(old_name))
                )
                text_edits.append(TextEdit(range=edit_range, new_text=new_name))
                start = pos + len(old_name)
        
        if text_edits:
            edits[uri] = text_edits
    
    # Apply workspace edit
    if edits:
        workspace_edit = WorkspaceEdit(changes=edits)
        
        # Send edit request to client
        result = await server.lsp.send_request(
            WORKSPACE_APPLY_EDIT,
            {"edit": workspace_edit}
        )
        
        return {"applied": result.applied if result else False}
    
    return {"applied": False, "message": "No occurrences found"}
```

### Custom Document Handling

```python
from pygls.workspace import TextDocument

class CustomWorkspace:
    def __init__(self, server):
        self.server = server
        self.custom_documents = {}
    
    def process_document(self, uri: str):
        document = self.server.workspace.get_document(uri)
        
        # Custom processing
        processed_lines = []
        for line in document.lines:
            # Apply custom transformations
            processed_line = line.strip().upper()
            processed_lines.append(processed_line)
        
        # Store processed version
        self.custom_documents[uri] = {
            "original": document,
            "processed": processed_lines,
            "timestamp": time.time()
        }
    
    def get_processed_content(self, uri: str) -> List[str]:
        if uri in self.custom_documents:
            return self.custom_documents[uri]["processed"]
        return []

# Use custom workspace handler
custom_workspace = CustomWorkspace(server)

@server.feature(TEXT_DOCUMENT_DID_CHANGE)
def on_document_change(params):
    # Process document after changes
    custom_workspace.process_document(params.text_document.uri)
```