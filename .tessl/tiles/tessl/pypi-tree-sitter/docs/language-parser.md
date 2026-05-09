# Language and Parser Management

Core functionality for loading language grammars and creating parsers that convert source code into syntax trees. The Language class represents compiled grammars, while Parser handles the actual parsing process.

## Capabilities

### Language Loading and Introspection

Load language grammars and inspect their properties including node types, fields, and parse states.

```python { .api }
class Language:
    def __init__(self, ptr: object) -> None:
        """
        Create language from language pointer.
        
        Args:
            ptr: Language pointer from external language package
        """

    @property
    def name(self) -> str | None:
        """Language name if available."""

    @property
    def abi_version(self) -> int:
        """ABI version of the language."""

    @property
    def semantic_version(self) -> tuple[int, int, int] | None:
        """Semantic version tuple (major, minor, patch)."""

    @property
    def node_kind_count(self) -> int:
        """Total number of node kinds in the grammar."""

    @property
    def parse_state_count(self) -> int:
        """Total number of parse states in the grammar."""

    @property
    def field_count(self) -> int:
        """Total number of fields in the grammar."""

    def node_kind_for_id(self, id: int) -> str | None:
        """
        Get node kind name for the given ID.
        
        Args:
            id: Node kind ID
            
        Returns:
            Node kind name or None if invalid ID
        """

    def id_for_node_kind(self, kind: str, named: bool) -> int | None:
        """
        Get ID for the given node kind.
        
        Args:
            kind: Node kind name
            named: Whether the node kind is named
            
        Returns:
            Node kind ID or None if not found
        """

    def field_name_for_id(self, field_id: int) -> str | None:
        """
        Get field name for the given field ID.
        
        Args:
            field_id: Field ID
            
        Returns:
            Field name or None if invalid ID
        """

    def field_id_for_name(self, name: str) -> int | None:
        """
        Get field ID for the given field name.
        
        Args:
            name: Field name
            
        Returns:
            Field ID or None if not found
        """

    @property
    def supertypes(self) -> tuple[int, ...]:
        """Tuple of supertype node IDs in the grammar."""

    def subtypes(self, supertype: int) -> tuple[int, ...]:
        """
        Get subtypes for the given supertype.
        
        Args:
            supertype: Supertype node ID
            
        Returns:
            Tuple of subtype node IDs
        """

    def node_kind_is_named(self, id: int) -> bool:
        """
        Check if the node kind is named.
        
        Args:
            id: Node kind ID
            
        Returns:
            True if the node kind is named
        """

    def node_kind_is_visible(self, id: int) -> bool:
        """
        Check if the node kind is visible.
        
        Args:
            id: Node kind ID
            
        Returns:
            True if the node kind is visible
        """

    def node_kind_is_supertype(self, id: int) -> bool:
        """
        Check if the node kind is a supertype.
        
        Args:
            id: Node kind ID
            
        Returns:
            True if the node kind is a supertype
        """

    def next_state(self, state: int, id: int) -> int:
        """
        Get the next parse state given current state and symbol ID.
        
        Args:
            state: Current parse state
            id: Symbol ID
            
        Returns:
            Next parse state
        """

    def lookahead_iterator(self, state: int) -> LookaheadIterator | None:
        """
        Create lookahead iterator for the given parse state.
        
        Args:
            state: Parse state
            
        Returns:
            LookaheadIterator for the state or None if invalid
        """

    def copy(self) -> Language:
        """Create a copy of this language."""

    @property
    def version(self) -> int:
        """Deprecated: Use abi_version instead."""

    def query(self, source: str) -> Query:
        """Deprecated: Use the Query() constructor instead."""
```

### Parser Creation and Configuration

Create parsers and configure them with languages, byte ranges, and logging.

```python { .api }
class Parser:
    def __init__(
        self,
        language: Language | None = None,
        *,
        included_ranges: list[Range] | None = None,
        logger: Callable[[LogType, str], None] | None = None,
    ) -> None:
        """
        Create a new parser.
        
        Args:
            language: Language to use for parsing
            included_ranges: Byte ranges to include in parsing
            logger: Callback for parse/lex log messages
        """

    @property
    def language(self) -> Language | None:
        """Current language (can be get/set/deleted)."""

    @language.setter
    def language(self, language: Language) -> None: ...

    @language.deleter
    def language(self) -> None: ...

    @property
    def included_ranges(self) -> list[Range]:
        """Byte ranges to include in parsing (can be get/set/deleted)."""

    @included_ranges.setter
    def included_ranges(self, ranges: list[Range]) -> None: ...

    @included_ranges.deleter
    def included_ranges(self) -> None: ...

    @property
    def logger(self) -> Callable[[LogType, str], None] | None:
        """Logging callback (can be get/set/deleted)."""

    @logger.setter
    def logger(self, logger: Callable[[LogType, str], None]) -> None: ...

    @logger.deleter
    def logger(self) -> None: ...

    def reset(self) -> None:
        """Reset the parser state."""

    def print_dot_graphs(self, file) -> None:
        """
        Print parse graphs as DOT format for debugging.
        
        Args:
            file: File object with fileno() method or None for stdout
        """
```

### Source Code Parsing

Parse source code from bytes or using a read callback function for large or streaming sources.

```python { .api }
class Parser:
    def parse(
        self,
        source: bytes | bytearray | memoryview,
        old_tree: Tree | None = None,
        encoding: str = "utf8",
    ) -> Tree:
        """
        Parse source code from bytes.
        
        Args:
            source: Source code as bytes
            old_tree: Previous tree for incremental parsing
            encoding: Text encoding ("utf8", "utf16", "utf16le", "utf16be")
            
        Returns:
            Parsed syntax tree
        """

    def parse(
        self,
        read_callback: Callable[[int, Point], bytes | None],
        old_tree: Tree | None = None,
        encoding: str = "utf8",
        progress_callback: Callable[[int, bool], bool] | None = None,
    ) -> Tree:
        """
        Parse source code using a read callback.
        
        Args:
            read_callback: Function that returns bytes for byte offset and Point
            old_tree: Previous tree for incremental parsing
            encoding: Text encoding
            progress_callback: Progress monitoring callback
            
        Returns:
            Parsed syntax tree
        """
```

## Usage Examples

### Loading Language Grammars

```python
import tree_sitter_python
import tree_sitter_javascript
from tree_sitter import Language

# Load Python grammar
py_language = Language(tree_sitter_python.language())
print(f"Python grammar: {py_language.name}")
print(f"Node types: {py_language.node_kind_count}")

# Load JavaScript grammar
js_language = Language(tree_sitter_javascript.language())
print(f"JavaScript ABI version: {js_language.abi_version}")
```

### Basic Parsing

```python
from tree_sitter import Language, Parser
import tree_sitter_python

# Setup
language = Language(tree_sitter_python.language())
parser = Parser(language)

# Parse simple code
code = b'''
def calculate(x, y):
    return x + y
'''

tree = parser.parse(code)
print(f"Root node: {tree.root_node.type}")
print(f"Children: {len(tree.root_node.children)}")
```

### Parsing with Read Callback

```python
# For large files or streaming
source_lines = ["def main():\n", "    print('hello')\n", "    return 0\n"]

def read_by_line(byte_offset, point):
    row, column = point
    if row >= len(source_lines):
        return None
    line = source_lines[row]
    return line[column:].encode("utf8")

tree = parser.parse(read_by_line)
```

### Parser Configuration

```python
from tree_sitter import Parser, Range, LogType

def custom_logger(log_type, message):
    if log_type == LogType.PARSE:
        print(f"Parse: {message}")
    elif log_type == LogType.LEX:
        print(f"Lex: {message}")

# Create parser with configuration
parser = Parser(
    language=language,
    included_ranges=[Range((0, 0), (10, 0), 0, 100)],
    logger=custom_logger
)

# Modify parser settings
parser.language = different_language
parser.included_ranges = [Range((0, 0), (5, 0), 0, 50)]
del parser.logger  # Remove logger
```