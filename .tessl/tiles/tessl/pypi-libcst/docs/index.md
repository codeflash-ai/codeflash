# LibCST

A concrete syntax tree parser and serializer library for Python that parses Python 3.0-3.13 source code as a CST tree that preserves all formatting details including comments, whitespaces, and parentheses. LibCST creates a compromise between Abstract Syntax Trees (AST) and traditional Concrete Syntax Trees (CST) by carefully reorganizing node types to create a lossless CST that looks and feels like an AST, making it ideal for building automated refactoring applications and linters.

## Package Information

- **Package Name**: libcst
- **Language**: Python
- **Installation**: `pip install libcst`

## Core Imports

```python
import libcst as cst
```

Common parsing functions:

```python
from libcst import parse_module, parse_statement, parse_expression
```

Visitor framework:

```python
from libcst import CSTVisitor, CSTTransformer, MetadataDependent
```

## Basic Usage

```python
import libcst as cst

# Parse Python code into a CST
source_code = '''
def hello_world():
    print("Hello, World!")
    return 42
'''

# Parse module preserving all formatting
module = cst.parse_module(source_code)

# Transform code using a visitor
class NameCollector(cst.CSTVisitor):
    def __init__(self):
        self.names = []
    
    def visit_Name(self, node):
        self.names.append(node.value)

visitor = NameCollector()
module.visit(visitor)
print(visitor.names)  # ['hello_world', 'print']

# Generate code back from CST (lossless)
generated_code = module.code
print(generated_code)  # Identical to original including whitespace
```

## Architecture

LibCST's design is built around several key components:

- **CST Nodes**: Immutable dataclasses representing every element of Python syntax
- **Visitor Framework**: Pattern for traversing and transforming CST trees
- **Metadata Providers**: System for attaching semantic information to nodes
- **Matcher System**: Declarative pattern matching for finding specific code patterns
- **Codemod Framework**: High-level tools for building code transformation applications

This architecture enables LibCST to serve as the foundation for sophisticated code analysis and transformation tools while maintaining perfect source code fidelity.

## Capabilities

### Core Parsing Functions

Parse Python source code into concrete syntax trees that preserve all formatting details including comments, whitespace, and parentheses.

```python { .api }
def parse_module(source: Union[str, bytes], config: Optional[PartialParserConfig] = None) -> Module: ...
def parse_statement(source: str, config: Optional[PartialParserConfig] = None) -> Union[SimpleStatementLine, BaseCompoundStatement]: ...
def parse_expression(source: str, config: Optional[PartialParserConfig] = None) -> BaseExpression: ...
```

[Parsing](./parsing.md)

### Visitor Framework

Traverse and transform CST trees using the visitor pattern, with support for metadata-dependent analysis and batched processing.

```python { .api }
class CSTVisitor:
    def visit(self, node: CSTNode) -> None: ...
    def leave(self, node: CSTNode) -> None: ...

class CSTTransformer:
    def visit(self, node: CSTNode) -> None: ...
    def leave(self, original_node: CSTNode, updated_node: CSTNode) -> CSTNode: ...

class MetadataDependent:
    METADATA_DEPENDENCIES: Sequence[ProviderT] = ()
    def resolve(self, node: CSTNode) -> Any: ...
```

[Visitor Framework](./visitors.md)

### Pattern Matching

Declarative pattern matching system for finding, extracting, and replacing specific code patterns in CST trees.

```python { .api }
def matches(node: CSTNode, matcher: Union[CSTNode, MatcherPattern]) -> bool: ...
def findall(tree: CSTNode, matcher: Union[CSTNode, MatcherPattern]) -> Sequence[CSTNode]: ...
def extract(node: CSTNode, matcher: Union[CSTNode, MatcherPattern]) -> Dict[str, Union[CSTNode, Sequence[CSTNode]]]: ...
def replace(tree: CSTNode, matcher: Union[CSTNode, MatcherPattern], replacement: Union[CSTNode, Callable]) -> CSTNode: ...
```

[Pattern Matching](./matchers.md)

### Metadata Analysis

Attach semantic information to CST nodes including scope analysis, position tracking, qualified names, and type inference.

```python { .api }
class MetadataWrapper:
    def __init__(self, module: Module, cache: Optional[Dict] = None) -> None: ...
    def resolve(self, provider: Type[BaseMetadataProvider]) -> Mapping[CSTNode, Any]: ...
    def resolve_many(self, providers: Sequence[Type[BaseMetadataProvider]]) -> Dict[Type[BaseMetadataProvider], Mapping[CSTNode, Any]]: ...

class ScopeProvider(BaseMetadataProvider):
    def visit_Module(self, node: Module) -> None: ...

class PositionProvider(BaseMetadataProvider):
    def visit_Module(self, node: Module) -> None: ...
```

[Metadata Analysis](./metadata.md)

### Codemod Framework

High-level framework for building automated code transformation applications with context management, parallel processing, and testing utilities.

```python { .api }
class CodemodCommand:
    def transform_module(self, tree: Module) -> Module: ...

class VisitorBasedCodemodCommand(CodemodCommand):
    def leave_Module(self, original_node: Module, updated_node: Module) -> Module: ...

def transform_module(code: str, command: CodemodCommand) -> TransformResult: ...
def parallel_exec_transform_with_prettyprint(command: CodemodCommand, files: Sequence[str]) -> ParallelTransformResult: ...
```

[Codemod Framework](./codemods.md)

### CST Node Types

Complete set of immutable dataclasses representing every element of Python syntax from expressions and statements to operators and whitespace.

```python { .api }
class Module(CSTNode):
    body: Sequence[Union[SimpleStatementLine, BaseCompoundStatement]]
    header: Sequence[Union[SimpleStatementLine, BaseCompoundStatement]]
    footer: Sequence[Union[SimpleStatementLine, BaseCompoundStatement]]

class Name(BaseExpression):
    value: str
    
class FunctionDef(BaseCompoundStatement):
    name: Name
    params: Parameters
    body: BaseSuite
    decorators: Sequence[Decorator]
```

[CST Nodes](./nodes.md)

### Command-Line Tool

Comprehensive command-line interface for LibCST operations including parsing, visualization, and code transformation execution.

```python { .api }
# Main CLI commands
python -m libcst.tool print <file>     # Display CST tree representation
python -m libcst.tool codemod <command> <files>  # Execute code transformations
python -m libcst.tool list             # Show available codemod commands
python -m libcst.tool initialize       # Create configuration files
```

[CLI Tool Reference](./cli-tool.md)

### Utilities and Helpers

Helper functions, debugging utilities, and testing framework for LibCST development and programmatic usage.

```python { .api }
# Helper functions
from libcst.helpers import calculate_module_and_package, parse_template_module, get_full_name_for_node

# Display utilities  
from libcst.display import dump, dump_graphviz

# Testing framework
from libcst.testing import UnitTest, CodemodTest
```

[Utilities and Helpers](./utilities.md)

## Types

```python { .api }
# Base node types
class CSTNode:
    def visit(self, visitor: CSTVisitor) -> CSTNode: ...
    def with_changes(self, **changes: Any) -> CSTNode: ...

# Core expression and statement base classes  
class BaseExpression(CSTNode): ...
class BaseStatement(CSTNode): ...
class BaseCompoundStatement(BaseStatement): ...
class BaseSmallStatement(BaseStatement): ...

# Visitor types
CSTNodeT = TypeVar("CSTNodeT", bound=CSTNode)
CSTVisitorT = TypeVar("CSTVisitorT", bound="CSTVisitor")

# Exception types
class CSTValidationError(Exception): ...
class ParserSyntaxError(Exception): ...
class CSTLogicError(Exception): ...
class MetadataException(Exception): ...

# Transform results
class TransformResult:
    code: Optional[str]
    encoding: str
    
class TransformSuccess(TransformResult): ...
class TransformFailure(TransformResult): 
    error: Exception
```