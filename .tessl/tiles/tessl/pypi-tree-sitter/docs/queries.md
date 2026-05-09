# Pattern Matching and Queries

Tree-sitter's query system enables powerful pattern matching against syntax trees using a specialized query language. Queries can find specific code patterns, extract information, and perform structural analysis across parsed source code.

## Capabilities

### Query Creation and Introspection

Create queries from query strings and inspect their structure including patterns, captures, and strings.

```python { .api }
class Query:
    def __init__(self, language: Language, source: str) -> None:
        """
        Create a query from query source string.
        
        Args:
            language: Language to create query for
            source: Query string in Tree-sitter query syntax
            
        Raises:
            QueryError: If query syntax is invalid
        """

    def pattern_count(self) -> int:
        """Number of patterns in this query."""

    def capture_count(self) -> int:
        """Number of captures defined in this query."""

    def string_count(self) -> int:
        """Number of string literals in this query."""

    def start_byte_for_pattern(self, index: int) -> int:
        """
        Get start byte position of pattern in query source.
        
        Args:
            index: Pattern index
            
        Returns:
            Start byte position of pattern
        """

    def end_byte_for_pattern(self, index: int) -> int:
        """
        Get end byte position of pattern in query source.
        
        Args:
            index: Pattern index
            
        Returns:
            End byte position of pattern
        """

    def capture_name(self, index: int) -> str:
        """
        Get capture name by index.
        
        Args:
            index: Capture index
            
        Returns:
            Capture name
        """

    def capture_quantifier(
        self,
        pattern_index: int,
        capture_index: int,
    ) -> str:
        """
        Get capture quantifier for pattern and capture.
        
        Args:
            pattern_index: Pattern index
            capture_index: Capture index within pattern
            
        Returns:
            Quantifier string: "", "?", "*", or "+"
        """

    def string_value(self, index: int) -> str:
        """
        Get string literal value by index.
        
        Args:
            index: String index
            
        Returns:
            String literal value
        """
```

### Query Pattern Analysis

Analyze pattern properties and behavior including root status and locality.

```python { .api }
class Query:
    def is_pattern_rooted(self, index: int) -> bool:
        """
        Check if pattern is rooted (starts at tree root).
        
        Args:
            index: Pattern index
            
        Returns:
            True if pattern must start at root
        """

    def is_pattern_non_local(self, index: int) -> bool:
        """
        Check if pattern is non-local (can match across subtrees).
        
        Args:
            index: Pattern index
            
        Returns:
            True if pattern is non-local
        """

    def is_pattern_guaranteed_at_step(self, index: int) -> bool:
        """
        Check if pattern has guaranteed matches at step.
        
        Args:
            index: Pattern index
            
        Returns:
            True if pattern has guaranteed step matches
        """

    def pattern_settings(self, index: int) -> dict[str, str | None]:
        """
        Get pattern settings as key-value pairs.
        
        Args:
            index: Pattern index
            
        Returns:
            Dictionary of pattern settings
        """

    def pattern_assertions(self, index: int) -> dict[str, tuple[str | None, bool]]:
        """
        Get pattern assertions.
        
        Args:
            index: Pattern index
            
        Returns:
            Dictionary mapping assertion names to (value, negated) tuples
        """
```

### Query Modification

Disable specific captures or patterns to customize query behavior.

```python { .api }
class Query:
    def disable_capture(self, name: str) -> None:
        """
        Disable capture by name.
        
        Args:
            name: Capture name to disable
        """

    def disable_pattern(self, index: int) -> None:
        """
        Disable pattern by index.
        
        Args:
            index: Pattern index to disable
        """
```

### Query Execution with QueryCursor

Execute queries against syntax trees and retrieve matches or captures.

```python { .api }
class QueryCursor:
    def __init__(
        self,
        query: Query,
        *,
        match_limit: int = 0xFFFFFFFF,
    ) -> None:
        """
        Create query cursor for executing queries.
        
        Args:
            query: Query to execute
            match_limit: Maximum number of matches to return
        """

    @property
    def match_limit(self) -> int:
        """Maximum number of matches (can be get/set/deleted)."""

    @match_limit.setter
    def match_limit(self, limit: int) -> None: ...

    @match_limit.deleter
    def match_limit(self) -> None: ...

    @property
    def did_exceed_match_limit(self) -> bool:
        """Whether the last query execution exceeded match limit."""

    def set_max_start_depth(self, depth: int) -> None:
        """
        Set maximum depth to start matching patterns.
        
        Args:
            depth: Maximum starting depth
        """

    def set_byte_range(self, start: int, end: int) -> None:
        """
        Limit query execution to specific byte range.
        
        Args:
            start: Start byte position
            end: End byte position
        """

    def set_point_range(
        self,
        start: Point | tuple[int, int],
        end: Point | tuple[int, int],
    ) -> None:
        """
        Limit query execution to specific point range.
        
        Args:
            start: Start point (row, column)
            end: End point (row, column)
        """
```

### Capture Extraction

Extract all captures from query execution, grouped by capture name.

```python { .api }
class QueryCursor:
    def captures(
        self,
        node: Node,
        predicate: QueryPredicate | None = None,
        progress_callback: Callable[[int], bool] | None = None,
    ) -> dict[str, list[Node]]:
        """
        Execute query and return all captures grouped by name.
        
        Args:
            node: Root node to search from
            predicate: Custom predicate function for filtering
            progress_callback: Progress monitoring callback
            
        Returns:
            Dictionary mapping capture names to lists of matching nodes
        """
```

### Match Extraction

Extract complete matches with pattern information and grouped captures.

```python { .api }
class QueryCursor:
    def matches(
        self,
        node: Node,
        predicate: QueryPredicate | None = None,
        progress_callback: Callable[[int], bool] | None = None,
    ) -> list[tuple[int, dict[str, list[Node]]]]:
        """
        Execute query and return complete matches.
        
        Args:
            node: Root node to search from
            predicate: Custom predicate function for filtering
            progress_callback: Progress monitoring callback
            
        Returns:
            List of (pattern_index, captures) tuples where captures
            is a dictionary mapping capture names to lists of nodes
        """
```

### Custom Query Predicates

Implement custom logic for query filtering using the QueryPredicate protocol.

```python { .api }
class QueryPredicate:
    def __call__(
        self,
        predicate: str,
        args: list[tuple[str, str]],
        pattern_index: int,
        captures: dict[str, list[Node]],
    ) -> bool:
        """
        Custom predicate function for query filtering.
        
        Args:
            predicate: Predicate name used in query
            args: List of (value, type) argument tuples
            pattern_index: Index of current pattern being matched
            captures: Current captures for this pattern
            
        Returns:
            True if predicate matches, False otherwise
        """
```

### Query Errors

Handle query syntax and execution errors.

```python { .api }
class QueryError(ValueError):
    """Raised when query syntax is invalid or execution fails."""
```

## Usage Examples

### Basic Query Usage

```python
from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_python

# Setup
language = Language(tree_sitter_python.language())
parser = Parser(language)

code = b'''
def calculate(x, y):
    result = x + y
    return result

def process(data):
    value = data * 2
    return value
'''

tree = parser.parse(code)

# Create query to find function definitions
query = Query(language, '''
(function_definition
  name: (identifier) @function.name
  parameters: (parameters) @function.params
  body: (block) @function.body)
''')

# Execute query
cursor = QueryCursor(query)
captures = cursor.captures(tree.root_node)

print(f"Found {len(captures['function.name'])} functions:")
for func_name in captures['function.name']:
    print(f"  - {func_name.text.decode()}")
```

### Complex Query Patterns

```python
# Query for variable assignments within functions
assignment_query = Query(language, '''
(function_definition
  name: (identifier) @func.name
  body: (block
    (expression_statement
      (assignment
        left: (identifier) @var.name
        right: (_) @var.value))))
''')

cursor = QueryCursor(assignment_query)
matches = cursor.matches(tree.root_node)

for pattern_idx, match_captures in matches:
    func_name = match_captures['func.name'][0].text.decode()
    var_name = match_captures['var.name'][0].text.decode()
    print(f"In function {func_name}, variable {var_name} is assigned")
```

### Query with Predicates

```python
# Query with string matching predicate
predicate_query = Query(language, '''
(call
  function: (identifier) @func.name
  arguments: (argument_list
    (string) @arg.string))

(#eq? @func.name "print")
''')

# Custom predicate function
def custom_predicate(predicate, args, pattern_index, captures):
    if predicate == "eq?":
        capture_name, expected_value = args[0][0], args[1][0]
        if capture_name in captures:
            actual_value = captures[capture_name][0].text.decode()
            return actual_value == expected_value
    return False

cursor = QueryCursor(predicate_query)
matches = cursor.matches(tree.root_node, predicate=custom_predicate)
```

### Query Optimization

```python
# Configure cursor for performance
cursor = QueryCursor(query, match_limit=100)
cursor.set_max_start_depth(3)  # Don't start matching too deep
cursor.set_byte_range(0, 500)  # Limit to first 500 bytes

# Check if limit was exceeded
captures = cursor.captures(tree.root_node)
if cursor.did_exceed_match_limit:
    print("Query hit match limit - results may be incomplete")
```

### Query Introspection

```python
# Analyze query structure
print(f"Query has {query.pattern_count()} patterns")
print(f"Query has {query.capture_count()} captures")

for i in range(query.capture_count()):
    capture_name = query.capture_name(i)
    print(f"Capture {i}: @{capture_name}")

# Check pattern properties
for i in range(query.pattern_count()):
    print(f"Pattern {i}:")
    print(f"  Rooted: {query.is_pattern_rooted(i)}")
    print(f"  Non-local: {query.is_pattern_non_local(i)}")
    print(f"  Settings: {query.pattern_settings(i)}")
```

### Progress Monitoring

```python
def progress_callback(current_step):
    """Progress callback that can cancel long-running queries."""
    print(f"Query progress: step {current_step}")
    # Return False to cancel query execution
    return current_step < 1000

cursor = QueryCursor(query)
captures = cursor.captures(
    tree.root_node,
    progress_callback=progress_callback
)
```

## Query Language Reference

Tree-sitter queries use S-expression syntax to match tree structures:

- `(node_type)` - Match nodes of specific type
- `@capture.name` - Capture matched nodes with a name
- `field: (pattern)` - Match nodes in specific fields
- `"literal"` - Match literal text content
- `(_)` - Match any node type
- `(#predicate? @capture "value")` - Apply predicates to filter matches

Quantifiers:
- `pattern?` - Optional (0 or 1 matches)
- `pattern*` - Zero or more matches  
- `pattern+` - One or more matches

Advanced features:
- `(#eq? @capture "value")` - String equality predicate
- `(#match? @capture "regex")` - Regex matching predicate
- `(#set! key "value")` - Set pattern metadata