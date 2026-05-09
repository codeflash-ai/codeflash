# Syntax Tree Navigation

Navigate and inspect parsed syntax trees using Tree, Node, and TreeCursor objects. Trees are immutable data structures representing the parsed source code, with nodes providing detailed information about each element's position, type, and relationships.

## Capabilities

### Tree Access and Properties

Access the root node and tree-level properties including language information and included ranges.

```python { .api }
class Tree:
    @property
    def root_node(self) -> Node:
        """Root node of the syntax tree."""

    @property
    def included_ranges(self) -> list[Range]:
        """Byte ranges that were included during parsing."""

    @property
    def language(self) -> Language:
        """Language used for parsing this tree."""

    def root_node_with_offset(
        self,
        offset_bytes: int,
        offset_extent: Point | tuple[int, int],
    ) -> Node | None:
        """
        Get root node with byte and extent offset applied.
        
        Args:
            offset_bytes: Byte offset to apply
            offset_extent: Point offset to apply
            
        Returns:
            Root node with offset or None if invalid
        """

    def copy(self) -> Tree:
        """Create a copy of this tree."""

    def walk(self) -> TreeCursor:
        """Create a cursor for efficient tree traversal."""

    def print_dot_graph(self, file) -> None:
        """
        Print tree structure as DOT graph for visualization.
        
        Args:
            file: File object with fileno() method
        """
```

### Node Properties and Position

Access node properties including type, position, content, and structural relationships.

```python { .api }
class Node:
    @property
    def id(self) -> int:
        """Unique identifier for this node."""

    @property
    def kind_id(self) -> int:
        """Node kind ID from the grammar."""

    @property
    def grammar_id(self) -> int:
        """Grammar ID this node belongs to."""

    @property
    def grammar_name(self) -> str:
        """Grammar name this node belongs to."""

    @property
    def type(self) -> str:
        """Node type name (e.g., 'function_definition', 'identifier')."""

    @property
    def is_named(self) -> bool:
        """Whether this node represents a named language construct."""

    @property
    def is_extra(self) -> bool:
        """Whether this node represents extra content (comments, whitespace)."""

    @property
    def has_changes(self) -> bool:
        """Whether this node has been edited."""

    @property
    def has_error(self) -> bool:
        """Whether this node contains parse errors."""

    @property
    def is_error(self) -> bool:
        """Whether this node is an error node."""

    @property
    def is_missing(self) -> bool:
        """Whether this node represents missing required content."""

    @property
    def start_byte(self) -> int:
        """Starting byte position in source code."""

    @property
    def end_byte(self) -> int:
        """Ending byte position in source code."""

    @property
    def byte_range(self) -> tuple[int, int]:
        """Byte range as (start_byte, end_byte) tuple."""

    @property
    def range(self) -> Range:
        """Range object with byte and point information."""

    @property
    def start_point(self) -> Point:
        """Starting position as (row, column) point."""

    @property
    def end_point(self) -> Point:
        """Ending position as (row, column) point."""

    @property
    def text(self) -> bytes | None:
        """Text content of this node as bytes."""

    @property
    def parent(self) -> Node | None:
        """Parent node or None if this is the root."""

    @property
    def descendant_count(self) -> int:
        """Total number of descendant nodes."""

    @property
    def parse_state(self) -> int:
        """Parse state ID for this node."""

    @property
    def next_parse_state(self) -> int:
        """Next parse state ID for this node."""

    def walk(self) -> TreeCursor:
        """Create a cursor for traversing from this node."""
```

### Child Node Access

Access child nodes by index, field name, or position within the source.

```python { .api }
class Node:
    @property
    def children(self) -> list[Node]:
        """All child nodes including unnamed nodes."""

    @property
    def child_count(self) -> int:
        """Total number of child nodes."""

    @property
    def named_children(self) -> list[Node]:
        """Named child nodes only."""

    @property
    def named_child_count(self) -> int:
        """Number of named child nodes."""

    def child(self, index: int) -> Node | None:
        """
        Get child node by index.
        
        Args:
            index: Child index (0-based)
            
        Returns:
            Child node or None if index is out of bounds
        """

    def named_child(self, index: int) -> Node | None:
        """
        Get named child node by index.
        
        Args:
            index: Named child index (0-based)
            
        Returns:
            Named child node or None if index is out of bounds
        """

    def child_by_field_name(self, name: str) -> Node | None:
        """
        Get child node by field name.
        
        Args:
            name: Field name (e.g., 'name', 'body', 'parameters')
            
        Returns:
            Child node with the given field name or None
        """

    def child_by_field_id(self, id: int) -> Node | None:
        """
        Get child node by field ID.
        
        Args:
            id: Field ID from the grammar
            
        Returns:
            Child node with the given field ID or None
        """

    def children_by_field_name(self, name: str) -> list[Node]:
        """
        Get all child nodes with the given field name.
        
        Args:
            name: Field name
            
        Returns:
            List of child nodes with the field name
        """

    def children_by_field_id(self, id: int) -> list[Node]:
        """
        Get all child nodes with the given field ID.
        
        Args:
            id: Field ID
            
        Returns:
            List of child nodes with the field ID
        """

    def field_name_for_child(self, child_index: int) -> str | None:
        """
        Get field name for child at the given index.
        
        Args:
            child_index: Child index (0-based)
            
        Returns:
            Field name for the child or None if no field name
        """

    def field_name_for_named_child(self, child_index: int) -> str | None:
        """
        Get field name for named child at the given index.
        
        Args:
            child_index: Named child index (0-based)
            
        Returns:
            Field name for the named child or None if no field name
        """

    def first_child_for_byte(self, byte: int) -> Node | None:
        """
        Get first child that contains the given byte position.
        
        Args:
            byte: Byte position in source
            
        Returns:
            First child containing the byte or None
        """

    def first_named_child_for_byte(self, byte: int) -> Node | None:
        """
        Get first named child that contains the given byte position.
        
        Args:
            byte: Byte position in source
            
        Returns:
            First named child containing the byte or None
        """
```

### Sibling Node Navigation

Navigate between sibling nodes at the same tree level.

```python { .api }
class Node:
    @property
    def next_sibling(self) -> Node | None:
        """Next sibling node or None if this is the last child."""

    @property
    def prev_sibling(self) -> Node | None:
        """Previous sibling node or None if this is the first child."""

    @property
    def next_named_sibling(self) -> Node | None:
        """Next named sibling node or None."""

    @property
    def prev_named_sibling(self) -> Node | None:
        """Previous named sibling node or None."""
```

### Descendant Search

Find descendant nodes by position ranges or relationships.

```python { .api }
class Node:
    def descendant_for_byte_range(
        self,
        start_byte: int,
        end_byte: int,
    ) -> Node | None:
        """
        Get smallest descendant that spans the given byte range.
        
        Args:
            start_byte: Start of byte range
            end_byte: End of byte range
            
        Returns:
            Descendant node spanning the range or None
        """

    def named_descendant_for_byte_range(
        self,
        start_byte: int,
        end_byte: int,
    ) -> Node | None:
        """
        Get smallest named descendant that spans the given byte range.
        
        Args:
            start_byte: Start of byte range
            end_byte: End of byte range
            
        Returns:
            Named descendant node spanning the range or None
        """

    def descendant_for_point_range(
        self,
        start_point: Point | tuple[int, int],
        end_point: Point | tuple[int, int],
    ) -> Node | None:
        """
        Get smallest descendant that spans the given point range.
        
        Args:
            start_point: Start point (row, column)
            end_point: End point (row, column)
            
        Returns:
            Descendant node spanning the range or None
        """

    def named_descendant_for_point_range(
        self,
        start_point: Point | tuple[int, int],
        end_point: Point | tuple[int, int],
    ) -> Node | None:
        """
        Get smallest named descendant that spans the given point range.
        
        Args:
            start_point: Start point (row, column)
            end_point: End point (row, column)
            
        Returns:
            Named descendant node spanning the range or None
        """

    def child_with_descendant(self, descendant: Node) -> Node | None:
        """
        Get child node that contains the given descendant.
        
        Args:
            descendant: Descendant node to find parent for
            
        Returns:
            Child node containing the descendant or None
        """
```

### Efficient Tree Traversal with TreeCursor

Use TreeCursor for efficient navigation of large syntax trees without creating intermediate Node objects.

```python { .api }
class TreeCursor:
    @property
    def node(self) -> Node | None:
        """Current node at cursor position."""

    @property
    def field_id(self) -> int | None:
        """Field ID of current position or None."""

    @property
    def field_name(self) -> str | None:
        """Field name of current position or None."""

    @property
    def depth(self) -> int:
        """Current depth in the tree."""

    @property
    def descendant_index(self) -> int:
        """Index of current node among all descendants."""

    def copy(self) -> TreeCursor:
        """Create a copy of this cursor."""

    def reset(self, node: Node) -> None:
        """
        Reset cursor to the given node.
        
        Args:
            node: Node to reset cursor to
        """

    def reset_to(self, cursor: TreeCursor) -> None:
        """
        Reset cursor to match another cursor's position.
        
        Args:
            cursor: Cursor to copy position from
        """

    def goto_first_child(self) -> bool:
        """
        Move cursor to first child of current node.
        
        Returns:
            True if moved successfully, False if no children
        """

    def goto_last_child(self) -> bool:
        """
        Move cursor to last child of current node.
        
        Returns:
            True if moved successfully, False if no children
        """

    def goto_parent(self) -> bool:
        """
        Move cursor to parent of current node.
        
        Returns:
            True if moved successfully, False if at root
        """

    def goto_next_sibling(self) -> bool:
        """
        Move cursor to next sibling of current node.
        
        Returns:
            True if moved successfully, False if no next sibling
        """

    def goto_previous_sibling(self) -> bool:
        """
        Move cursor to previous sibling of current node.
        
        Returns:
            True if moved successfully, False if no previous sibling
        """

    def goto_descendant(self, index: int) -> None:
        """
        Move cursor to descendant at the given index.
        
        Args:
            index: Descendant index to move to
        """

    def goto_first_child_for_byte(self, byte: int) -> int | None:
        """
        Move cursor to first child that contains the given byte.
        
        Args:
            byte: Byte position to search for
            
        Returns:
            Child index if found, None otherwise
        """

    def goto_first_child_for_point(self, point: Point | tuple[int, int]) -> int | None:
        """
        Move cursor to first child that contains the given point.
        
        Args:
            point: Point (row, column) to search for
            
        Returns:
            Child index if found, None otherwise
        """
```

## Usage Examples

### Basic Node Navigation

```python
from tree_sitter import Language, Parser
import tree_sitter_python

# Setup and parse
language = Language(tree_sitter_python.language())
parser = Parser(language)

code = b'''
def calculate(x, y):
    result = x + y
    return result
'''

tree = parser.parse(code)
root = tree.root_node

# Navigate to function definition
function_def = root.children[0]
print(f"Function type: {function_def.type}")
print(f"Function position: {function_def.start_point} to {function_def.end_point}")

# Get function name
function_name = function_def.child_by_field_name("name")
print(f"Function name: {function_name.text}")

# Get parameters
parameters = function_def.child_by_field_name("parameters")
print(f"Parameter count: {parameters.named_child_count}")

# Get function body
body = function_def.child_by_field_name("body")
print(f"Body has {body.named_child_count} statements")
```

### TreeCursor for Efficient Traversal

```python
def traverse_tree(tree):
    """Efficiently traverse entire tree using cursor."""
    cursor = tree.walk()
    
    visited_children = False
    while True:
        if not visited_children:
            print(f"Visiting {cursor.node.type} at depth {cursor.depth}")
            if not cursor.goto_first_child():
                visited_children = True
        elif cursor.goto_next_sibling():
            visited_children = False
        elif not cursor.goto_parent():
            break

traverse_tree(tree)
```

### Finding Nodes by Position

```python
# Find node at specific byte position
byte_pos = 25
node_at_pos = root.descendant_for_byte_range(byte_pos, byte_pos + 1)
print(f"Node at byte {byte_pos}: {node_at_pos.type}")

# Find node at specific line/column
point = (2, 4)  # Line 2, column 4
node_at_point = root.descendant_for_point_range(point, point)
print(f"Node at {point}: {node_at_point.type}")
```

### Working with Field Names

```python
# Get field information
function_def = root.children[0]

# Iterate through children with field names
for i, child in enumerate(function_def.children):
    field_name = function_def.field_name_for_child(i)
    if field_name:
        print(f"Child {i} ({child.type}) has field name: {field_name}")
    else:
        print(f"Child {i} ({child.type}) has no field name")

# Get all children with specific field
body_children = function_def.children_by_field_name("body")
print(f"Found {len(body_children)} body elements")
```