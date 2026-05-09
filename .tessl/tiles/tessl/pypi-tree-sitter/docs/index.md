# Tree-sitter

Tree-sitter provides Python bindings to the Tree-sitter parsing library for incremental parsing and syntax tree analysis. It enables developers to parse source code into syntax trees for language analysis, code navigation, syntax highlighting, and building development tools like language servers and code formatters.

## Package Information

- **Package Name**: tree-sitter
- **Language**: Python
- **Installation**: `pip install tree-sitter`
- **Minimum Python Version**: 3.10+

## Core Imports

```python
from tree_sitter import Language, Parser, Tree, Node, TreeCursor
```

For queries and pattern matching:

```python
from tree_sitter import Query, QueryCursor, QueryError, QueryPredicate
```

For utilities and types:

```python
from tree_sitter import Point, Range, LogType, LookaheadIterator
```

Constants:

```python
from tree_sitter import LANGUAGE_VERSION, MIN_COMPATIBLE_LANGUAGE_VERSION
```

## Basic Usage

```python
import tree_sitter_python
from tree_sitter import Language, Parser

# Load a language grammar (requires separate language package)
PY_LANGUAGE = Language(tree_sitter_python.language())

# Create parser and set language
parser = Parser(PY_LANGUAGE)

# Parse source code
source_code = '''
def hello_world():
    print("Hello, world!")
    return True
'''

tree = parser.parse(bytes(source_code, "utf8"))

# Inspect the syntax tree
root_node = tree.root_node
print(f"Root node type: {root_node.type}")
print(f"Tree structure:\n{root_node}")

# Navigate the tree
function_node = root_node.children[0]
function_name = function_node.child_by_field_name("name")
print(f"Function name: {function_name.type}")
```

## Architecture

Tree-sitter uses a layered architecture optimized for incremental parsing:

- **Language**: Represents a grammar for parsing specific programming languages
- **Parser**: Stateful parser that converts source code into syntax trees using a language
- **Tree**: Immutable syntax tree containing the parsed structure
- **Node**: Individual elements in the tree with position, type, and relationship information
- **TreeCursor**: Efficient navigation mechanism for traversing large trees
- **Query System**: Pattern matching for structural queries against syntax trees

The design enables high-performance parsing with incremental updates, making it ideal for real-time applications like editors and development tools.

## Capabilities

### Language and Parser Management

Core functionality for loading language grammars and creating parsers for converting source code into syntax trees.

```python { .api }
class Language:
    def __init__(self, ptr: object) -> None: ...
    @property
    def name(self) -> str | None: ...
    @property
    def abi_version(self) -> int: ...

class Parser:
    def __init__(
        self,
        language: Language | None = None,
        *,
        included_ranges: list[Range] | None = None,
        logger: Callable[[LogType, str], None] | None = None,
    ) -> None: ...
    def parse(
        self,
        source: bytes | Callable[[int, Point], bytes | None],
        old_tree: Tree | None = None,
        encoding: str = "utf8"
    ) -> Tree: ...
```

[Language and Parser Management](./language-parser.md)

### Syntax Tree Navigation

Navigate and inspect parsed syntax trees using nodes and cursors for efficient tree traversal.

```python { .api }
class Tree:
    @property
    def root_node(self) -> Node: ...
    def walk(self) -> TreeCursor: ...
    def changed_ranges(self, new_tree: Tree) -> list[Range]: ...

class Node:
    @property
    def type(self) -> str: ...
    @property
    def children(self) -> list[Node]: ...
    @property
    def start_point(self) -> Point: ...
    @property
    def end_point(self) -> Point: ...
    def child_by_field_name(self, name: str) -> Node | None: ...

class TreeCursor:
    @property
    def node(self) -> Node | None: ...
    def goto_first_child(self) -> bool: ...
    def goto_next_sibling(self) -> bool: ...
    def goto_parent(self) -> bool: ...
```

[Syntax Tree Navigation](./tree-navigation.md)

### Pattern Matching and Queries

Powerful query system for finding patterns in syntax trees using Tree-sitter's query language.

```python { .api }
class Query:
    def __init__(self, language: Language, source: str) -> None: ...
    def pattern_count(self) -> int: ...
    def capture_name(self, index: int) -> str: ...

class QueryCursor:
    def __init__(self, query: Query, *, match_limit: int = 0xFFFFFFFF) -> None: ...
    def captures(
        self,
        node: Node,
        predicate: QueryPredicate | None = None
    ) -> dict[str, list[Node]]: ...
    def matches(
        self,
        node: Node,
        predicate: QueryPredicate | None = None
    ) -> list[tuple[int, dict[str, list[Node]]]]: ...
```

[Pattern Matching and Queries](./queries.md)

### Incremental Parsing

Edit syntax trees and perform incremental parsing for efficient updates when source code changes.

```python { .api }
# Tree editing
def Tree.edit(
    self,
    start_byte: int,
    old_end_byte: int,
    new_end_byte: int,
    start_point: Point | tuple[int, int],
    old_end_point: Point | tuple[int, int],
    new_end_point: Point | tuple[int, int],
) -> None: ...

# Incremental parsing
def Parser.parse(
    self,
    source: bytes,
    old_tree: Tree | None = None,  # Enables incremental parsing
    encoding: str = "utf8"
) -> Tree: ...
```

[Incremental Parsing](./incremental-parsing.md)

## Types

```python { .api }
from typing import NamedTuple
from enum import IntEnum

class Point(NamedTuple):
    row: int
    column: int

class Range:
    def __init__(
        self,
        start_point: Point | tuple[int, int],
        end_point: Point | tuple[int, int],
        start_byte: int,
        end_byte: int,
    ) -> None: ...
    @property
    def start_point(self) -> Point: ...
    @property
    def end_point(self) -> Point: ...
    @property
    def start_byte(self) -> int: ...
    @property
    def end_byte(self) -> int: ...

class LogType(IntEnum):
    PARSE: int
    LEX: int

class QueryError(ValueError): ...

class LookaheadIterator:
    """Iterator for lookahead symbols in parse states."""
    @property
    def language(self) -> Language: ...
    @property
    def current_symbol(self) -> int: ...
    @property
    def current_symbol_name(self) -> str: ...
    def reset(self, state: int, language: Language | None = None) -> bool: ...
    def names(self) -> list[str]: ...
    def symbols(self) -> list[int]: ...
    def __next__(self) -> tuple[int, str]: ...

# Protocol for custom query predicates
class QueryPredicate:
    def __call__(
        self,
        predicate: str,
        args: list[tuple[str, str]],  # str is "capture" | "string"
        pattern_index: int,
        captures: dict[str, list[Node]],
    ) -> bool: ...

# Constants
LANGUAGE_VERSION: int
"""Tree-sitter language ABI version."""

MIN_COMPATIBLE_LANGUAGE_VERSION: int
"""Minimum compatible language ABI version."""
```