# Feature Registration and Decorators

Decorator-based system for registering LSP features, custom commands, and thread execution patterns with automatic capability registration and declarative programming patterns.

## Capabilities

### Feature Registration

Decorator for registering Language Server Protocol features with automatic capability negotiation and method binding.

```python { .api }
def feature(
    self, 
    method_name: str, 
    options: Any = None
) -> Callable[[F], F]:
    """
    Decorator to register LSP features.
    
    Parameters:
    - method_name: str - LSP method name (e.g., 'textDocument/completion')
    - options: Any - Feature-specific options (e.g., CompletionOptions, HoverOptions)
    
    Returns:
    Decorator function that registers the decorated function as a feature handler
    """

@lsp_method(method_name: str) -> Callable[[F], F]:
    """
    Low-level decorator for registering LSP method handlers.
    
    Parameters:
    - method_name: str - LSP method name
    
    Returns:
    Decorator function for method registration
    """
```

### Command Registration

Decorator for registering custom server commands that can be invoked by clients.

```python { .api }
def command(self, command_name: str) -> Callable[[F], F]:
    """
    Decorator to register custom commands.
    
    Parameters:
    - command_name: str - Unique command identifier
    
    Returns:
    Decorator function that registers the decorated function as a command handler
    """
```

### Thread Execution

Decorator for executing functions in background threads to avoid blocking the main event loop.

```python { .api }
def thread(self) -> Callable[[F], F]:
    """
    Decorator to execute function in thread pool.
    
    Returns:
    Decorator function that wraps the decorated function for thread execution
    """
```

### Feature Manager

Internal feature management system for organizing registered features and commands.

```python { .api }
class FeatureManager:
    def feature(
        self,
        method_name: str,
        options: Any = None
    ) -> Callable: ...
    
    def command(self, command_name: str) -> Callable: ...
    
    def thread(self) -> Callable: ...
    
    @property
    def features(self) -> Dict: ...
    
    @property
    def commands(self) -> Dict: ...
    
    @property
    def feature_options(self) -> Dict: ...
```

## Usage Examples

### Basic Feature Registration

```python
from pygls.server import LanguageServer
from lsprotocol.types import (
    TEXT_DOCUMENT_COMPLETION,
    TEXT_DOCUMENT_HOVER,
    CompletionItem,
    CompletionList,
    CompletionParams,
    Hover,
    HoverParams,
    MarkupContent,
    MarkupKind
)

server = LanguageServer("example-server", "1.0.0")

# Register completion feature
@server.feature(TEXT_DOCUMENT_COMPLETION)
def completions(params: CompletionParams):
    document = server.workspace.get_document(params.text_document.uri)
    items = [
        CompletionItem(label="example"),
        CompletionItem(label="sample")
    ]
    return CompletionList(is_incomplete=False, items=items)

# Register hover feature
@server.feature(TEXT_DOCUMENT_HOVER)
def hover(params: HoverParams):
    return Hover(
        contents=MarkupContent(
            kind=MarkupKind.Markdown,
            value="**Example hover text**"
        )
    )
```

### Feature with Options

```python
from lsprotocol.types import (
    TEXT_DOCUMENT_COMPLETION,
    CompletionOptions,
    CompletionParams
)

# Register completion with trigger characters
@server.feature(
    TEXT_DOCUMENT_COMPLETION,
    CompletionOptions(trigger_characters=['.', '::'])
)
def completions_with_triggers(params: CompletionParams):
    # Handle completion with trigger character context
    trigger_char = params.context.trigger_character if params.context else None
    
    if trigger_char == '.':
        # Provide member completions
        return CompletionList(items=[
            CompletionItem(label="method1"),
            CompletionItem(label="property1")
        ])
    elif trigger_char == '::':
        # Provide namespace completions
        return CompletionList(items=[
            CompletionItem(label="Class1"),
            CompletionItem(label="Function1")
        ])
    
    return CompletionList(items=[])
```

### Custom Commands

```python
from lsprotocol.types import WORKSPACE_EXECUTE_COMMAND

# Register custom command
@server.command("myServer.customCommand")
def custom_command(params):
    # Access command arguments
    args = params.arguments if params.arguments else []
    
    # Perform custom logic
    result = {
        "message": "Command executed successfully",
        "arguments": args,
        "timestamp": time.time()
    }
    
    return result

# Register command that shows configuration
@server.command("myServer.showConfiguration")
async def show_config(params):
    # Get configuration from client
    config = await server.lsp.send_request(
        "workspace/configuration",
        {"items": [{"section": "myServer"}]}
    )
    
    return {"configuration": config}
```

### Thread Execution

```python
import time

# Long-running operation in thread
@server.command("myServer.longOperation")
@server.thread()
def long_operation(params):
    # This runs in a background thread
    time.sleep(5)  # Simulate long-running work
    return {"result": "Operation completed"}

# Blocking I/O operation in thread  
@server.feature(TEXT_DOCUMENT_HOVER)
@server.thread()
def hover_with_file_io(params: HoverParams):
    # File I/O that might block
    with open("some_file.txt", "r") as f:
        content = f.read()
    
    return Hover(
        contents=MarkupContent(
            kind=MarkupKind.PlainText,
            value=f"File content: {content[:100]}..."
        )
    )
```

### Advanced Feature Registration

```python
from lsprotocol.types import (
    TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL,
    SemanticTokensLegend,
    SemanticTokensOptions,
    SemanticTokensParams
)

# Complex feature with comprehensive options
@server.feature(
    TEXT_DOCUMENT_SEMANTIC_TOKENS_FULL, 
    SemanticTokensOptions(
        legend=SemanticTokensLegend(
            token_types=['keyword', 'string', 'number', 'comment'],
            token_modifiers=['deprecated', 'readonly']
        ),
        full=True,
        range=True
    )
)
def semantic_tokens(params: SemanticTokensParams):
    document = server.workspace.get_document(params.text_document.uri)
    
    # Parse document and generate semantic tokens
    tokens = []  # Build semantic token data
    
    return {"data": tokens}
```

### Error Handling in Features

```python
from pygls.exceptions import FeatureRequestError
from lsprotocol.types import TEXT_DOCUMENT_DEFINITION

@server.feature(TEXT_DOCUMENT_DEFINITION)
def goto_definition(params):
    try:
        document = server.workspace.get_document(params.text_document.uri)
        # Definition finding logic here
        
        if not definition_found:
            return None  # No definition found
            
        return definition_location
        
    except Exception as e:
        # Convert to LSP error
        raise FeatureRequestError(f"Failed to find definition: {str(e)}")
```

### Dynamic Feature Registration

```python
from pygls.protocol import LanguageServerProtocol

class CustomProtocol(LanguageServerProtocol):
    def lsp_initialize(self, params):
        result = super().lsp_initialize(params)
        
        # Conditionally register features based on client capabilities
        if params.capabilities.text_document and params.capabilities.text_document.completion:
            self.register_dynamic_completion()
            
        return result
    
    def register_dynamic_completion(self):
        # Register completion feature dynamically
        @self.feature(TEXT_DOCUMENT_COMPLETION)
        def dynamic_completion(params):
            return CompletionList(items=[
                CompletionItem(label="dynamically_registered")
            ])
```