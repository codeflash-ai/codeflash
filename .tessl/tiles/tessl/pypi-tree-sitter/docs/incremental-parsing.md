# Incremental Parsing

Tree-sitter's incremental parsing capabilities enable efficient updates to syntax trees when source code changes. Instead of reparsing entire files, incremental parsing reuses unchanged portions of the tree and only reparses modified sections, making it ideal for real-time applications like editors and language servers.

## Capabilities

### Tree Editing

Edit syntax trees to reflect changes in source code before incremental parsing.

```python { .api }
class Tree:
    def edit(
        self,
        start_byte: int,
        old_end_byte: int,
        new_end_byte: int,
        start_point: Point | tuple[int, int],
        old_end_point: Point | tuple[int, int],
        new_end_point: Point | tuple[int, int],
    ) -> None:
        """
        Edit the syntax tree to reflect a change in source code.
        
        Args:
            start_byte: Start byte position of the edit
            old_end_byte: End byte position before the edit
            new_end_byte: End byte position after the edit
            start_point: Start point (row, column) of the edit
            old_end_point: End point before the edit
            new_end_point: End point after the edit
        """
```

```python { .api }
class Node:
    def edit(
        self,
        start_byte: int,
        old_end_byte: int,
        new_end_byte: int,
        start_point: Point | tuple[int, int],
        old_end_point: Point | tuple[int, int],
        new_end_point: Point | tuple[int, int],
    ) -> None:
        """
        Edit this node to reflect a change in source code.
        Same parameters as Tree.edit().
        """
```

### Incremental Parsing

Parse updated source code using a previous tree to enable incremental updates.

```python { .api }
class Parser:
    def parse(
        self,
        source: bytes | Callable[[int, Point], bytes | None],
        old_tree: Tree | None = None,
        encoding: str = "utf8",
        progress_callback: Callable[[int, bool], bool] | None = None,
    ) -> Tree:
        """
        Parse source code, optionally using previous tree for incremental parsing.
        
        Args:
            source: Source code or read callback function
            old_tree: Previous tree to enable incremental parsing
            encoding: Text encoding
            progress_callback: Progress monitoring callback
            
        Returns:
            New syntax tree incorporating changes
        """
```

### Change Detection

Identify ranges in the tree that changed between parse operations.

```python { .api }
class Tree:
    def changed_ranges(self, new_tree: Tree) -> list[Range]:
        """
        Get ranges that changed between this tree and a new tree.
        
        Args:
            new_tree: New tree to compare against
            
        Returns:
            List of ranges where syntactic structure changed
        """
```

## Usage Examples

### Basic Incremental Parsing

```python
from tree_sitter import Language, Parser, Point
import tree_sitter_python

# Setup
language = Language(tree_sitter_python.language())
parser = Parser(language)

# Initial parsing
original_source = b'''
def calculate(x, y):
    result = x + y
    return result
'''

tree = parser.parse(original_source)
print(f"Original tree root: {tree.root_node.type}")

# Simulate editing: change "x + y" to "x * y"
# Edit is at position where "+" appears
edit_start_byte = original_source.find(b'+')
edit_start_point = (1, 14)  # Line 1, column 14

# Edit the tree to reflect the change
tree.edit(
    start_byte=edit_start_byte,
    old_end_byte=edit_start_byte + 1,  # "+" is 1 byte
    new_end_byte=edit_start_byte + 1,  # "*" is also 1 byte
    start_point=edit_start_point,
    old_end_point=(1, 15),  # End of "+"
    new_end_point=(1, 15),  # End of "*"
)

# Create modified source
modified_source = original_source.replace(b'+', b'*')

# Incremental parsing using the edited tree
new_tree = parser.parse(modified_source, old_tree=tree)
print(f"New tree root: {new_tree.root_node.type}")
```

### Detecting Changes

```python
# Find what changed between trees
changed_ranges = tree.changed_ranges(new_tree)

print(f"Found {len(changed_ranges)} changed ranges:")
for i, range_obj in enumerate(changed_ranges):
    print(f"  Range {i}:")
    print(f"    Start: byte {range_obj.start_byte}, point {range_obj.start_point}")
    print(f"    End: byte {range_obj.end_byte}, point {range_obj.end_point}")
    
    # Extract changed text
    changed_text = modified_source[range_obj.start_byte:range_obj.end_byte]
    print(f"    Changed text: {changed_text}")
```

### Complex Text Editing

```python
# Simulate inserting a new line of code
original_source = b'''
def process_data(items):
    results = []
    return results
'''

tree = parser.parse(original_source)

# Insert "filtered = filter_items(items)" before "return results"
insert_position = original_source.find(b'return results')
insert_byte = insert_position
insert_point = (2, 4)  # Line 2, column 4

new_line = b'    filtered = filter_items(items)\n    '
modified_source = (
    original_source[:insert_position] + 
    new_line + 
    original_source[insert_position:]
)

# Edit tree for the insertion
tree.edit(
    start_byte=insert_byte,
    old_end_byte=insert_byte,  # Insertion, so old_end equals start
    new_end_byte=insert_byte + len(new_line),  # New content length
    start_point=insert_point,
    old_end_point=insert_point,  # Insertion
    new_end_point=(3, 4),  # New line added, so row increases
)

# Incremental parsing
new_tree = parser.parse(modified_source, old_tree=tree)
```

### Multiple Edits

```python
# Handle multiple edits in sequence
original_source = b'''
def old_function(a, b):
    temp = a + b
    return temp
'''

tree = parser.parse(original_source)

# Edit 1: Rename function
edit1_start = original_source.find(b'old_function')
tree.edit(
    start_byte=edit1_start,
    old_end_byte=edit1_start + len(b'old_function'),
    new_end_byte=edit1_start + len(b'new_function'),
    start_point=(0, 4),
    old_end_point=(0, 16),
    new_end_point=(0, 16),
)

modified_source = original_source.replace(b'old_function', b'new_function')

# Edit 2: Change operator
edit2_start = modified_source.find(b'+')
tree.edit(
    start_byte=edit2_start,
    old_end_byte=edit2_start + 1,
    new_end_byte=edit2_start + 1,
    start_point=(1, 13),
    old_end_point=(1, 14),
    new_end_point=(1, 14),
)

final_source = modified_source.replace(b'+', b'*')

# Incremental parsing with all edits
final_tree = parser.parse(final_source, old_tree=tree)
```

### Deletion Handling

```python
# Handle text deletion
original_source = b'''
def calculate(x, y, z):
    first = x + y
    second = first * z
    return second
'''

tree = parser.parse(original_source)

# Remove the parameter "z" and its usage
# First, remove ", z" from parameters
param_start = original_source.find(b', z')
param_end = param_start + len(b', z')

# Edit for parameter removal
tree.edit(
    start_byte=param_start,
    old_end_byte=param_end,
    new_end_byte=param_start,  # Deletion, so new_end equals start
    start_point=(0, 16),
    old_end_point=(0, 19),
    new_end_point=(0, 16),  # Deletion
)

# Remove the z usage (need to adjust for previous edit)
modified_source = original_source[:param_start] + original_source[param_end:]

# Remove "second = first * z" line
line_to_remove = b'    second = first * z\n'
line_start = modified_source.find(line_to_remove)
line_end = line_start + len(line_to_remove)

tree.edit(
    start_byte=line_start,
    old_end_byte=line_end,
    new_end_byte=line_start,  # Deletion
    start_point=(2, 0),
    old_end_point=(3, 0),
    new_end_point=(2, 0),
)

final_source = modified_source[:line_start] + modified_source[line_end:]

# Update return statement
final_source = final_source.replace(b'second', b'first')

# Final incremental parse
final_tree = parser.parse(final_source, old_tree=tree)
```

### Performance Monitoring

```python
import time

def benchmark_parsing(source, old_tree=None):
    """Benchmark parsing performance."""
    start_time = time.time()
    tree = parser.parse(source, old_tree=old_tree)
    end_time = time.time()
    
    parse_type = "incremental" if old_tree else "full"
    print(f"{parse_type.title()} parsing took {end_time - start_time:.4f} seconds")
    return tree

# Compare full vs incremental parsing
original_tree = benchmark_parsing(original_source)

# Make edit
tree.edit(start_byte=50, old_end_byte=51, new_end_byte=51,
          start_point=(1, 10), old_end_point=(1, 11), new_end_point=(1, 11))

modified_source = original_source[:50] + b'*' + original_source[51:]

# Incremental parsing should be faster
new_tree = benchmark_parsing(modified_source, old_tree=tree)
```

### Editor Integration Pattern

```python
class IncrementalEditor:
    """Example pattern for editor integration."""
    
    def __init__(self, language):
        self.parser = Parser(language)
        self.current_tree = None
        self.current_source = b""
    
    def set_content(self, source):
        """Set initial content."""
        self.current_source = source
        self.current_tree = self.parser.parse(source)
    
    def apply_edit(self, start_byte, old_end_byte, new_end_byte,
                   start_point, old_end_point, new_end_point, new_text):
        """Apply an edit and reparse incrementally."""
        if self.current_tree:
            # Edit the tree
            self.current_tree.edit(
                start_byte, old_end_byte, new_end_byte,
                start_point, old_end_point, new_end_point
            )
        
        # Update source
        self.current_source = (
            self.current_source[:start_byte] + 
            new_text + 
            self.current_source[old_end_byte:]
        )
        
        # Incremental parsing
        self.current_tree = self.parser.parse(
            self.current_source, 
            old_tree=self.current_tree
        )
    
    def get_changed_ranges(self, old_tree):
        """Get ranges that changed since old tree."""
        if old_tree and self.current_tree:
            return old_tree.changed_ranges(self.current_tree)
        return []

# Usage
editor = IncrementalEditor(language)
editor.set_content(b"def hello(): pass")

# Simulate typing
old_tree = editor.current_tree.copy()
editor.apply_edit(
    start_byte=13, old_end_byte=13, new_end_byte=20,
    start_point=(0, 13), old_end_point=(0, 13), new_end_point=(0, 20),
    new_text=b"world()"
)

# Check what changed
changes = editor.get_changed_ranges(old_tree)
print(f"Edit affected {len(changes)} ranges")
```

## Performance Tips

1. **Always edit the tree before incremental parsing** - This allows Tree-sitter to reuse unchanged subtrees
2. **Batch edits when possible** - Apply multiple edits to the same tree before reparsing
3. **Use precise edit ranges** - Accurate byte and point positions improve incremental parsing efficiency  
4. **Monitor change ranges** - Use `changed_ranges()` to identify which parts of your application need updates
5. **Consider parse performance** - Incremental parsing is most beneficial for large files with small changes