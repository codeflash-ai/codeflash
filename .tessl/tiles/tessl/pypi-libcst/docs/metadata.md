# Metadata Analysis

LibCST's metadata system attaches semantic information to CST nodes, enabling advanced static analysis capabilities including scope analysis, position tracking, qualified name resolution, and type inference. The system uses a provider architecture for extensible metadata computation.

## Capabilities

### Metadata Wrapper

Central coordinator for metadata computation and resolution.

```python { .api }
class MetadataWrapper:
    """Main interface for metadata-enabled CST analysis."""
    
    def __init__(self, module: Module, cache: Optional[Dict] = None) -> None:
        """
        Initialize metadata wrapper for a module.
        
        Parameters:
        - module: CST module to analyze
        - cache: Optional cache for metadata providers
        """
    
    def resolve(self, provider: Type[BaseMetadataProvider]) -> Mapping[CSTNode, Any]:
        """
        Resolve metadata for all nodes using the specified provider.
        
        Parameters:
        - provider: Metadata provider class
        
        Returns:
        Mapping[CSTNode, Any]: Metadata values by node
        """
    
    def resolve_many(self, providers: Sequence[Type[BaseMetadataProvider]]) -> Dict[Type[BaseMetadataProvider], Mapping[CSTNode, Any]]:
        """
        Resolve multiple metadata providers efficiently.
        
        Parameters:
        - providers: Sequence of provider classes
        
        Returns:
        Dict[Type[BaseMetadataProvider], Mapping[CSTNode, Any]]: Results by provider
        """
    
    def visit(self, visitor: MetadataDependent) -> CSTNode:
        """
        Visit module with metadata-dependent visitor.
        
        Parameters:
        - visitor: Visitor that requires metadata
        
        Returns:
        CSTNode: Potentially transformed module
        """
```

### Position Tracking

Track source code positions for CST nodes.

```python { .api }
class PositionProvider(BaseMetadataProvider):
    """Provides line/column positions for CST nodes."""

class WhitespaceInclusivePositionProvider(BaseMetadataProvider):
    """Provides positions including leading/trailing whitespace."""

class ByteSpanPositionProvider(BaseMetadataProvider):
    """Provides byte-based position spans."""

class CodePosition:
    """Represents a line/column position in source code."""
    line: int
    column: int

class CodeRange:
    """Represents a start/end position range."""
    start: CodePosition
    end: CodePosition

class CodeSpan:
    """Represents a byte-based span."""
    start: int
    length: int
```

### Scope Analysis

Analyze variable scopes, assignments, and accesses.

```python { .api }
class ScopeProvider(BaseMetadataProvider):
    """Analyzes variable scopes and assignments."""

class Scope:
    """Base class for all scope types."""
    parent: Optional["Scope"]
    
    def __contains__(self, name: str) -> bool:
        """Check if name is defined in this scope."""
    
    def __getitem__(self, name: str) -> Collection["BaseAssignment"]:
        """Get assignments for a name in this scope."""

class GlobalScope(Scope):
    """Module-level scope."""

class FunctionScope(Scope):
    """Function-level scope."""
    name: str

class ClassScope(Scope):
    """Class-level scope."""
    name: str

class ComprehensionScope(Scope):
    """Comprehension-level scope."""

class BuiltinScope(Scope):
    """Built-in names scope."""

class Assignment:
    """Tracks variable assignments."""
    name: str
    node: CSTNode
    scope: Scope

class BuiltinAssignment(Assignment):
    """Built-in variable assignment."""

class ImportAssignment(Assignment):
    """Import-based assignment."""

class Access:
    """Tracks variable access."""
    name: str
    node: CSTNode
    scope: Scope

# Type aliases
Accesses = Collection[Access]
Assignments = Collection[BaseAssignment]
```

### Qualified Names

Resolve qualified names for imported and defined symbols.

```python { .api }
class QualifiedNameProvider(BaseMetadataProvider):
    """Provides qualified names for nodes."""

class FullyQualifiedNameProvider(BaseMetadataProvider):
    """Provides fully qualified names including module paths."""

class QualifiedName:
    """Represents a qualified name."""
    name: str
    source: QualifiedNameSource

class QualifiedNameSource:
    """Source information for qualified names."""
    IMPORT: ClassVar[int]
    BUILTIN: ClassVar[int]
    LOCAL: ClassVar[int]
```

### Expression Context

Determine Load/Store/Del context for expressions.

```python { .api }
class ExpressionContextProvider(BaseMetadataProvider):
    """Determines expression context (Load/Store/Del)."""

class ExpressionContext:
    """Expression context enumeration."""
    LOAD: ClassVar[int]
    STORE: ClassVar[int] 
    DEL: ClassVar[int]
```

### Parent Node Relationships

Track parent-child relationships in the CST.

```python { .api }
class ParentNodeProvider(BaseMetadataProvider):
    """Provides parent node relationships."""
```

### Advanced Providers

Additional metadata providers for specialized analysis.

```python { .api }
class TypeInferenceProvider(BaseMetadataProvider):
    """Experimental type inference provider."""

class FullRepoManager:
    """Manager for cross-file analysis."""
    
    def __init__(self, repo_root: str, paths: Sequence[str], providers: Sequence[Type[BaseMetadataProvider]]) -> None:
        """
        Initialize repository-wide analysis.
        
        Parameters:
        - repo_root: Root directory of repository
        - paths: Python files to analyze
        - providers: Metadata providers to use
        """

class AccessorProvider(BaseMetadataProvider):
    """Provides accessor metadata."""

class FilePathProvider(BaseMetadataProvider):
    """Provides file path information."""
```

### Provider Base Classes

Foundation for implementing custom metadata providers.

```python { .api }
class BaseMetadataProvider:
    """Base class for all metadata providers."""
    
    def visit_Module(self, node: Module) -> None:
        """Visit module node."""

class BatchableMetadataProvider(BaseMetadataProvider):
    """Base class for batchable providers."""

class VisitorMetadataProvider(BaseMetadataProvider):
    """Base class for visitor-based providers."""

ProviderT = TypeVar("ProviderT", bound=BaseMetadataProvider)
```

## Usage Examples

### Basic Position Tracking

```python
import libcst as cst
from libcst.metadata import MetadataWrapper, PositionProvider

source = '''
def foo():
    x = 42
    return x
'''

module = cst.parse_module(source)
wrapper = MetadataWrapper(module)
positions = wrapper.resolve(PositionProvider)

# Find positions of all Name nodes
for node in cst.metadata.findall(module, cst.Name):
    if node in positions:
        pos = positions[node]
        print(f"Name '{node.value}' at line {pos.start.line}, column {pos.start.column}")
```

### Scope Analysis

```python
import libcst as cst
from libcst.metadata import MetadataWrapper, ScopeProvider

source = '''
x = "global"

def outer():
    y = "outer"
    
    def inner():
        z = "inner"
        print(x, y, z)  # Access variables from different scopes
    
    inner()

outer()
'''

module = cst.parse_module(source)
wrapper = MetadataWrapper(module)
scopes = wrapper.resolve(ScopeProvider)

# Analyze variable assignments and accesses
for node, scope in scopes.items():
    if isinstance(node, cst.Name):
        scope_type = type(scope).__name__
        print(f"Name '{node.value}' in {scope_type}")
        
        # Check if this is an assignment or access
        if node.value in scope:
            assignments = scope[node.value]
            print(f"  Has {len(assignments)} assignments in this scope")
```

### Qualified Name Resolution

```python
import libcst as cst
from libcst.metadata import MetadataWrapper, QualifiedNameProvider

source = '''
import os.path
from collections import Counter
from .local import helper

def process():
    return os.path.join("a", "b")

def analyze(data):
    return Counter(data)

def work():
    return helper.process()
'''

module = cst.parse_module(source)
wrapper = MetadataWrapper(module)
qualified_names = wrapper.resolve(QualifiedNameProvider)

# Find qualified names for all function calls
for node in cst.matchers.findall(module, cst.Call()):
    if node.func in qualified_names:
        qnames = qualified_names[node.func]
        for qname in qnames:
            print(f"Call to: {qname.name}")
```

### Metadata-Dependent Visitor

```python
import libcst as cst
from libcst.metadata import MetadataWrapper, ScopeProvider, PositionProvider

class VariableTracker(cst.MetadataDependent):
    METADATA_DEPENDENCIES = (ScopeProvider, PositionProvider)
    
    def __init__(self):
        self.assignments = []
        self.accesses = []
    
    def visit_Name(self, node):
        scope = self.resolve(ScopeProvider)[node]
        position = self.resolve(PositionProvider)[node]
        
        # Determine if this is assignment or access based on context
        # (simplified - real implementation would check expression context)
        if isinstance(node.parent, cst.Assign) and node in node.parent.targets:
            self.assignments.append({
                'name': node.value,
                'line': position.start.line,
                'scope': type(scope).__name__
            })
        else:
            self.accesses.append({
                'name': node.value, 
                'line': position.start.line,
                'scope': type(scope).__name__
            })

# Usage
source = '''
def example():
    x = 1      # Assignment
    y = x + 2  # Access to x, assignment to y
    return y   # Access to y
'''

module = cst.parse_module(source)
wrapper = MetadataWrapper(module)
tracker = VariableTracker()
wrapper.visit(tracker)

print("Assignments:", tracker.assignments)
print("Accesses:", tracker.accesses)
```

### Cross-File Analysis

```python
from libcst.metadata import FullRepoManager, ScopeProvider, QualifiedNameProvider

# Analyze multiple files in a repository
manager = FullRepoManager(
    repo_root="/path/to/repo",
    paths=["module1.py", "module2.py", "package/__init__.py"],
    providers=[ScopeProvider, QualifiedNameProvider]
)

# Get metadata for all files
repo_metadata = manager.get_cache()
for file_path, file_metadata in repo_metadata.items():
    print(f"Analyzing {file_path}")
    for provider, node_metadata in file_metadata.items():
        print(f"  {provider.__name__}: {len(node_metadata)} nodes")
```

## Types

```python { .api }
# Position types
class CodePosition:
    line: int
    column: int

class CodeRange:
    start: CodePosition  
    end: CodePosition

class CodeSpan:
    start: int
    length: int

# Scope types
class Scope:
    parent: Optional["Scope"]

class GlobalScope(Scope): ...
class FunctionScope(Scope):
    name: str
class ClassScope(Scope):
    name: str
class ComprehensionScope(Scope): ...
class BuiltinScope(Scope): ...

# Assignment and access types
class BaseAssignment:
    name: str
    node: CSTNode
    scope: Scope

class Assignment(BaseAssignment): ...
class BuiltinAssignment(BaseAssignment): ...
class ImportAssignment(BaseAssignment): ...

class Access:
    name: str
    node: CSTNode
    scope: Scope

Accesses = Collection[Access]
Assignments = Collection[BaseAssignment]

# Qualified name types
class QualifiedName:
    name: str
    source: QualifiedNameSource

class QualifiedNameSource:
    IMPORT: ClassVar[int]
    BUILTIN: ClassVar[int]
    LOCAL: ClassVar[int]

# Expression context
class ExpressionContext:
    LOAD: ClassVar[int]
    STORE: ClassVar[int]
    DEL: ClassVar[int]

# Provider types
ProviderT = TypeVar("ProviderT", bound=BaseMetadataProvider)

class BaseMetadataProvider:
    """Base class for metadata providers."""

class BatchableMetadataProvider(BaseMetadataProvider): ...
class VisitorMetadataProvider(BaseMetadataProvider): ...

# Exception types
class MetadataException(Exception):
    """Raised for metadata-related errors."""
```