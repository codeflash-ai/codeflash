# Visitor Framework

LibCST's visitor framework provides a powerful pattern for traversing and transforming concrete syntax trees. The framework supports read-only analysis, tree transformation, metadata-dependent processing, and batched operations for performance optimization.

## Capabilities

### Basic Visitor

Read-only traversal of CST trees for analysis and data collection.

```python { .api }
class CSTVisitor:
    """Base class for read-only CST traversal."""
    
    def visit(self, node: CSTNode) -> None:
        """Called when entering a node during traversal."""
    
    def leave(self, node: CSTNode) -> None:
        """Called when leaving a node during traversal."""
    
    # Node-specific visit methods (examples)
    def visit_Module(self, node: Module) -> None: ...
    def visit_FunctionDef(self, node: FunctionDef) -> None: ...
    def visit_Name(self, node: Name) -> None: ...
    def visit_Call(self, node: Call) -> None: ...
    
    # Node-specific leave methods (examples)  
    def leave_Module(self, node: Module) -> None: ...
    def leave_FunctionDef(self, node: FunctionDef) -> None: ...
    def leave_Name(self, node: Name) -> None: ...
    def leave_Call(self, node: Call) -> None: ...
```

### Transformer

Transform CST trees by replacing nodes during traversal.

```python { .api }
class CSTTransformer(CSTVisitor):
    """Base class for CST transformation."""
    
    def leave(self, original_node: CSTNode, updated_node: CSTNode) -> CSTNode:
        """
        Transform node during traversal.
        
        Parameters:
        - original_node: Original node before any child transformations
        - updated_node: Node with transformed children
        
        Returns:
        CSTNode: Replacement node or updated_node to keep unchanged
        """
    
    # Node-specific transformation methods (examples)
    def leave_Module(self, original_node: Module, updated_node: Module) -> Module: ...
    def leave_FunctionDef(self, original_node: FunctionDef, updated_node: FunctionDef) -> Union[FunctionDef, RemovalSentinel]: ...
    def leave_Name(self, original_node: Name, updated_node: Name) -> Name: ...
    def leave_Call(self, original_node: Call, updated_node: Call) -> Union[Call, BaseExpression]: ...
```

### Metadata-Dependent Analysis

Access semantic information during traversal using metadata providers.

```python { .api }
class MetadataDependent:
    """Base class for visitors that depend on metadata."""
    
    METADATA_DEPENDENCIES: Sequence[Type[BaseMetadataProvider]] = ()
    
    def resolve(self, node: CSTNode) -> Any:
        """
        Resolve metadata for a node.
        
        Parameters:
        - node: CST node to resolve metadata for
        
        Returns:
        Any: Metadata value for the node
        """
    
    def resolve_many(self, providers: Sequence[Type[BaseMetadataProvider]]) -> Dict[Type[BaseMetadataProvider], Mapping[CSTNode, Any]]:
        """Resolve multiple metadata providers at once."""
```

### Batched Visitor

Process multiple nodes efficiently with batched operations.

```python { .api }
class BatchableCSTVisitor(CSTVisitor):
    """Base class for visitors that support batched processing."""
    
    def on_visit(self, node: CSTNode) -> bool:
        """
        Determine if this visitor should process the node.
        
        Parameters:
        - node: Node being visited
        
        Returns:
        bool: True if visitor should process this node type
        """
    
    def on_leave(self, original_node: CSTNode) -> None:
        """Process node when leaving during batched traversal."""

def visit_batched(
    nodes: Sequence[CSTNode], 
    visitors: Sequence[BatchableCSTVisitor]
) -> None:
    """
    Visit multiple nodes with multiple visitors efficiently.
    
    Parameters:
    - nodes: Sequence of CST nodes to visit
    - visitors: Sequence of visitors to apply
    """
```

### Node Removal

Remove nodes from the tree during transformation.

```python { .api }
class RemovalSentinel:
    """Sentinel type for indicating node removal."""

# Global instance for node removal
RemoveFromParent: RemovalSentinel

class FlattenSentinel:
    """Sentinel type for flattening sequences."""

class MaybeSentinel:
    """Sentinel type for optional values."""
```

## Usage Examples

### Basic Analysis Visitor

```python
import libcst as cst

class FunctionAnalyzer(cst.CSTVisitor):
    def __init__(self):
        self.function_names = []
        self.call_counts = {}
    
    def visit_FunctionDef(self, node):
        self.function_names.append(node.name.value)
    
    def visit_Call(self, node):
        if isinstance(node.func, cst.Name):
            name = node.func.value
            self.call_counts[name] = self.call_counts.get(name, 0) + 1

# Use the visitor
source = '''
def foo():
    bar()
    baz()
    bar()

def qux():
    foo()
'''

module = cst.parse_module(source)
analyzer = FunctionAnalyzer()
module.visit(analyzer)

print(analyzer.function_names)  # ['foo', 'qux']
print(analyzer.call_counts)     # {'bar': 2, 'baz': 1, 'foo': 1}
```

### Code Transformation

```python
import libcst as cst

class PrintReplacer(cst.CSTTransformer):
    """Replace print() calls with logging.info()."""
    
    def leave_Call(self, original_node, updated_node):
        if isinstance(updated_node.func, cst.Name) and updated_node.func.value == "print":
            # Replace print with logging.info
            new_func = cst.Attribute(
                value=cst.Name("logging"),
                attr=cst.Name("info")
            )
            return updated_node.with_changes(func=new_func)
        return updated_node

# Transform code
source = '''
def greet(name):
    print("Hello, " + name)
    print("Welcome!")
'''

module = cst.parse_module(source)
transformer = PrintReplacer()
new_module = module.visit(transformer)

print(new_module.code)
# Output shows print() replaced with logging.info()
```

### Metadata-Dependent Visitor

```python
import libcst as cst
from libcst.metadata import ScopeProvider, Scope

class VariableAnalyzer(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (ScopeProvider,)
    
    def __init__(self):
        self.variables = {}
    
    def visit_Name(self, node):
        scope = self.resolve(ScopeProvider)
        if isinstance(scope, cst.metadata.FunctionScope):
            scope_name = scope.name
            if scope_name not in self.variables:
                self.variables[scope_name] = []
            self.variables[scope_name].append(node.value)

# Use with metadata wrapper
from libcst.metadata import MetadataWrapper

source = '''
def func1():
    x = 1
    y = 2

def func2():
    a = 3
    b = 4
'''

module = cst.parse_module(source)
wrapper = MetadataWrapper(module)
analyzer = VariableAnalyzer()
wrapper.visit(analyzer)
```

### Node Removal

```python
import libcst as cst

class CommentRemover(cst.CSTTransformer):
    """Remove all comments from code."""
    
    def leave_Comment(self, original_node, updated_node):
        return cst.RemoveFromParent

# Usage
source = '''
# This is a comment
def foo():  # Another comment
    return 42
'''

module = cst.parse_module(source)
transformer = CommentRemover()
new_module = module.visit(transformer)
# Comments will be removed from the output
```

## Types

```python { .api }
# Base visitor types
CSTNodeT = TypeVar("CSTNodeT", bound=CSTNode)
CSTVisitorT = TypeVar("CSTVisitorT", bound=CSTVisitor)

# Visitor base classes
class CSTVisitor:
    """Base class for read-only CST traversal."""

class CSTTransformer(CSTVisitor):
    """Base class for CST transformation."""

class BatchableCSTVisitor(CSTVisitor):
    """Base class for batched visitors."""

class MetadataDependent:
    """Base class for metadata-dependent operations."""
    METADATA_DEPENDENCIES: Sequence[Type[BaseMetadataProvider]]

# Sentinel types for special operations
class RemovalSentinel:
    """Indicates a node should be removed."""

class FlattenSentinel:
    """Indicates a sequence should be flattened."""
    
class MaybeSentinel:
    """Represents an optional value."""

# Global sentinel instances
RemoveFromParent: RemovalSentinel
DoNotCareSentinel: Any

# Metadata provider base type
class BaseMetadataProvider:
    """Base class for all metadata providers."""

# Exception types
class MetadataException(Exception):
    """Raised for metadata-related errors."""
    
class CSTLogicError(Exception):
    """Raised for internal visitor logic errors."""
```