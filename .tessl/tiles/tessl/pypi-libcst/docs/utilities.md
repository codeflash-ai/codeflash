# LibCST Utilities

LibCST provides a comprehensive suite of utility functions to help developers work with Concrete Syntax Trees efficiently. These utilities cover everything from module resolution and template parsing to node introspection and testing helpers.

## Helper Functions

LibCST's helper functions provide essential utilities for working with CST nodes, module imports, templates, and more.

### Module and Package Utilities

These functions help with module resolution, import handling, and calculating module paths:

```python { .api }
from libcst.helpers import (
    calculate_module_and_package,
    get_absolute_module,
    get_absolute_module_for_import,
    get_absolute_module_for_import_or_raise,
    get_absolute_module_from_package,
    get_absolute_module_from_package_for_import,
    get_absolute_module_from_package_for_import_or_raise,
    insert_header_comments,
    ModuleNameAndPackage,
)
```

**calculate_module_and_package**
```python
def calculate_module_and_package(
    repo_root: StrPath, 
    filename: StrPath, 
    use_pyproject_toml: bool = False
) -> ModuleNameAndPackage
```

Calculates the Python module name and package for a given file path relative to a repository root.

```python
from libcst.helpers import calculate_module_and_package
from pathlib import Path

# Calculate module info for a file
result = calculate_module_and_package(
    repo_root="/project/root",
    filename="/project/root/src/mypackage/utils.py"
)
# result.name = "src.mypackage.utils"
# result.package = "src.mypackage"
```

**get_absolute_module**
```python
def get_absolute_module(
    current_module: Optional[str], 
    module_name: Optional[str], 
    num_dots: int
) -> Optional[str]
```

Resolves relative imports to absolute module names based on the current module and number of dots.

```python
from libcst.helpers import get_absolute_module

# Resolve relative import
absolute = get_absolute_module(
    current_module="mypackage.submodule.file",
    module_name="utils", 
    num_dots=2  # from ..utils import something
)
# Returns: "mypackage.utils"
```

**get_absolute_module_for_import**
```python
def get_absolute_module_for_import(
    current_module: Optional[str], 
    import_node: ImportFrom
) -> Optional[str]
```

Extracts and resolves the absolute module name from an ImportFrom CST node.

**insert_header_comments**
```python
def insert_header_comments(node: Module, comments: List[str]) -> Module
```

Inserts comments after the last non-empty line in a module header, useful for adding copyright notices or other header comments.

```python
import libcst as cst
from libcst.helpers import insert_header_comments

# Parse a module
module = cst.parse_module("print('hello')")

# Add header comments
new_module = insert_header_comments(module, [
    "# Copyright 2024 MyCompany",
    "# Licensed under MIT"
])
```

### Template Parsing Functions

Template functions allow you to parse Python code templates with variable substitution:

```python { .api }
from libcst.helpers import (
    parse_template_expression,
    parse_template_module,
    parse_template_statement,
)
```

**parse_template_module**
```python
def parse_template_module(
    template: str,
    config: PartialParserConfig = _DEFAULT_PARTIAL_PARSER_CONFIG,
    **template_replacements: ValidReplacementType,
) -> Module
```

Parses an entire Python module template with variable substitution.

```python
import libcst as cst
from libcst.helpers import parse_template_module

# Parse a module template with substitutions
module = parse_template_module(
    "from {mod} import {name}\n\ndef {func}():\n    return {value}",
    mod=cst.Name("math"),
    name=cst.Name("pi"),
    func=cst.Name("get_pi"),
    value=cst.Name("pi")
)
```

**parse_template_statement**
```python
def parse_template_statement(
    template: str,
    config: PartialParserConfig = _DEFAULT_PARTIAL_PARSER_CONFIG,
    **template_replacements: ValidReplacementType,
) -> Union[SimpleStatementLine, BaseCompoundStatement]
```

Parses a statement template with variable substitution.

```python
from libcst.helpers import parse_template_statement

# Parse a statement template
stmt = parse_template_statement(
    "assert {condition}, {message}",
    condition=cst.parse_expression("x > 0"),
    message=cst.SimpleString('"Value must be positive"')
)
```

**parse_template_expression**
```python
def parse_template_expression(
    template: str,
    config: PartialParserConfig = _DEFAULT_PARTIAL_PARSER_CONFIG,
    **template_replacements: ValidReplacementType,
) -> BaseExpression
```

Parses an expression template with variable substitution.

```python
from libcst.helpers import parse_template_expression

# Parse an expression template
expr = parse_template_expression(
    "{left} + {right}",
    left=cst.Name("x"),
    right=cst.Integer("42")
)
```

### Node Introspection Utilities

These functions help examine and analyze CST node structure:

```python { .api }
from libcst.helpers import (
    get_node_fields,
    get_field_default_value,
    is_whitespace_node_field,
    is_syntax_node_field,
    is_default_node_field,
    filter_node_fields,
)
```

**get_node_fields**
```python
def get_node_fields(node: CSTNode) -> Sequence[dataclasses.Field[CSTNode]]
```

Returns all dataclass fields for a CST node.

```python
import libcst as cst
from libcst.helpers import get_node_fields

# Get all fields of a Name node
name_node = cst.Name("variable")
fields = get_node_fields(name_node)
for field in fields:
    print(f"Field: {field.name}, Type: {field.type}")
```

**filter_node_fields**
```python
def filter_node_fields(
    node: CSTNode,
    *,
    show_defaults: bool,
    show_syntax: bool,
    show_whitespace: bool,
) -> Sequence[dataclasses.Field[CSTNode]]
```

Returns a filtered list of node fields based on various criteria.

```python
from libcst.helpers import filter_node_fields

# Get only non-default, non-whitespace fields
filtered_fields = filter_node_fields(
    node,
    show_defaults=False,
    show_syntax=True,
    show_whitespace=False
)
```

### Name Resolution Helpers

Functions for extracting qualified names from CST nodes:

```python { .api }
from libcst.helpers import (
    get_full_name_for_node,
    get_full_name_for_node_or_raise,
)
```

**get_full_name_for_node**
```python
def get_full_name_for_node(node: Union[str, CSTNode]) -> Optional[str]
```

Extracts the full qualified name from various CST node types (Name, Attribute, Call, etc.).

```python
import libcst as cst
from libcst.helpers import get_full_name_for_node

# Extract name from various node types
attr_node = cst.parse_expression("os.path.join")
name = get_full_name_for_node(attr_node)  # Returns: "os.path.join"

call_node = cst.parse_expression("math.sqrt(4)")
name = get_full_name_for_node(call_node)  # Returns: "math.sqrt"
```

### Type Checking Utilities

```python { .api }
from libcst.helpers import ensure_type
```

**ensure_type**
```python
def ensure_type(node: object, nodetype: Type[T]) -> T
```

Type-safe casting with runtime validation for CST nodes.

```python
from libcst.helpers import ensure_type
import libcst as cst

# Safely cast to a specific node type
expr = cst.parse_expression("variable_name")
name_node = ensure_type(expr, cst.Name)  # Validates it's actually a Name node
```

### Path Utilities

```python { .api }
from libcst.helpers.paths import chdir
```

**chdir**
```python
@contextmanager
def chdir(path: StrPath) -> Generator[Path, None, None]
```

Context manager for temporarily changing the working directory.

```python
from libcst.helpers.paths import chdir
from pathlib import Path

# Temporarily change to another directory
with chdir("/some/other/path") as new_path:
    # Work in the new directory
    current = Path.cwd()  # Points to /some/other/path
# Automatically returns to original directory
```

### Matcher Conversion Utilities

```python { .api }
from libcst.helpers.matchers import node_to_matcher
```

**node_to_matcher**
```python
def node_to_matcher(
    node: CSTNode, 
    *, 
    match_syntactic_trivia: bool = False
) -> matchers.BaseMatcherNode
```

Converts a concrete CST node into a matcher for pattern matching.

```python
import libcst as cst
from libcst.helpers.matchers import node_to_matcher

# Convert node to matcher
node = cst.parse_expression("variable_name")
matcher = node_to_matcher(node)

# Use matcher to find similar patterns
# (useful in codemods and analysis tools)
```

## Display and Debugging Utilities

LibCST provides utilities for visualizing and debugging CST structures.

### Text Display

```python { .api }
from libcst.display import dump
```

**dump**
```python
def dump(
    node: CSTNode,
    *,
    indent: str = "  ",
    show_defaults: bool = False,
    show_syntax: bool = False,
    show_whitespace: bool = False,
) -> str
```

Returns a string representation of a CST node with configurable detail levels.

```python
import libcst as cst
from libcst.display import dump

# Parse some code
node = cst.parse_expression("x + y * 2")

# Basic representation (minimal)
print(dump(node))

# Detailed representation showing all fields
print(dump(node, show_defaults=True, show_syntax=True, show_whitespace=True))

# Output example:
# BinaryOperation(
#   left=Name("x"),
#   operator=Add(),
#   right=BinaryOperation(
#     left=Name("y"),
#     operator=Multiply(),
#     right=Integer("2")
#   )
# )
```

### GraphViz Visualization

```python { .api }
from libcst.display import dump_graphviz
```

**dump_graphviz**
```python
def dump_graphviz(
    node: object,
    *,
    show_defaults: bool = False,
    show_syntax: bool = False,
    show_whitespace: bool = False,
) -> str
```

Generates a GraphViz .dot representation for visualizing CST structure as a graph.

```python
import libcst as cst
from libcst.display import dump_graphviz

# Parse some code
node = cst.parse_expression("len(items)")

# Generate GraphViz representation
dot_content = dump_graphviz(node)

# Save to file and render with GraphViz tools
with open("ast_graph.dot", "w") as f:
    f.write(dot_content)

# Then use: dot -Tpng ast_graph.dot -o ast_graph.png
```

The generated GraphViz output can be rendered to various formats (PNG, SVG, PDF) using GraphViz tools, providing visual tree representations of CST structures that are invaluable for understanding complex code structures.

## Testing Utilities

LibCST provides comprehensive testing utilities for unit tests and codemod development.

### Base Testing Framework

```python { .api }
from libcst.testing import UnitTest
from libcst.testing.utils import data_provider, none_throws
```

**UnitTest**
```python
class UnitTest(TestCase, metaclass=BaseTestMeta)
```

Enhanced unittest.TestCase with data provider support and other testing conveniences.

```python
from libcst.testing import UnitTest, data_provider

class MyTests(UnitTest):
    @data_provider([
        ("input1", "expected1"),
        ("input2", "expected2"),
    ])
    def test_with_data(self, input_val: str, expected: str) -> None:
        # Test implementation
        pass
```

**data_provider**
```python
def data_provider(
    static_data: StaticDataType, 
    *, 
    test_limit: int = DEFAULT_TEST_LIMIT
) -> Callable[[Callable], Callable]
```

Decorator that generates multiple test methods from a data set.

```python
from libcst.testing.utils import data_provider

@data_provider([
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
])
def test_person_data(self, name: str, age: int) -> None:
    # This creates test_person_data_0 and test_person_data_1
    pass
```

**none_throws**
```python
def none_throws(value: Optional[T], message: str = "Unexpected None value") -> T
```

Assertion helper for non-None values with better error messages.

```python
from libcst.testing.utils import none_throws

def test_something(self) -> None:
    result = get_optional_value()
    # Safely unwrap Optional value with clear error message
    actual_value = none_throws(result, "Expected non-None result from get_optional_value")
```

### Codemod Testing

```python { .api }
from libcst.codemod import CodemodTest
```

**CodemodTest**
```python
class CodemodTest(_CodemodTest, UnitTest):
    TRANSFORM: Type[Codemod] = ...  # Set to your codemod class
```

Base class for testing codemods with convenient assertion methods.

```python
from libcst.codemod import CodemodTest
from my_codemod import MyCodemod

class TestMyCodemod(CodemodTest):
    TRANSFORM = MyCodemod
    
    def test_simple_transformation(self) -> None:
        # Test a codemod transformation
        self.assertCodemod(
            before="""
                def old_function():
                    pass
            """,
            after="""
                def new_function():
                    pass
            """,
            # Any arguments to pass to the codemod constructor
        )
```

**assertCodemod**
```python
def assertCodemod(
    self,
    before: str,
    after: str,
    *args: object,
    context_override: Optional[CodemodContext] = None,
    python_version: Optional[str] = None,
    expected_warnings: Optional[Sequence[str]] = None,
    expected_skip: bool = False,
    **kwargs: object,
) -> None
```

Comprehensive codemod testing with support for warnings, skip conditions, and custom contexts.

```python
def test_codemod_with_warnings(self) -> None:
    self.assertCodemod(
        before="old_code",
        after="new_code",
        expected_warnings=["Deprecated function usage detected"],
        python_version="3.8"
    )

def test_codemod_skip(self) -> None:
    self.assertCodemod(
        before="code_that_should_be_skipped",
        after="code_that_should_be_skipped",  # Same as before
        expected_skip=True
    )
```

**assertCodeEqual**
```python
def assertCodeEqual(self, expected: str, actual: str) -> None
```

Code-aware string comparison that normalizes whitespace and handles multi-line strings.

**make_fixture_data**
```python
@staticmethod
def make_fixture_data(data: str) -> str
```

Normalizes test fixture code by removing leading/trailing whitespace and ensuring proper newlines.

```python
def test_fixture_normalization(self) -> None:
    # These strings are automatically normalized
    code = self.make_fixture_data("""
        def example():
            return True
    """)
    # Becomes: "def example():\n    return True\n"
```

## Usage Examples

### Complete Workflow Example

Here's a comprehensive example showing how to use various LibCST utilities together:

```python
import libcst as cst
from libcst.helpers import (
    parse_template_module,
    get_full_name_for_node,
    calculate_module_and_package,
    insert_header_comments
)
from libcst.display import dump
from libcst.testing import UnitTest, data_provider

class CodeAnalysisExample(UnitTest):
    
    @data_provider([
        ("simple_function", "def foo(): pass"),
        ("class_definition", "class Bar: pass"),
    ])
    def test_code_analysis(self, name: str, code: str) -> None:
        # Parse the code
        module = cst.parse_module(code)
        
        # Analyze and display structure
        print(f"Analysis of {name}:")
        print(dump(module, show_syntax=True))
        
        # Extract function/class names
        class NameCollector(cst.CSTVisitor):
            def __init__(self) -> None:
                self.names: List[str] = []
                
            def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
                name = get_full_name_for_node(node.name)
                if name:
                    self.names.append(f"function: {name}")
                    
            def visit_ClassDef(self, node: cst.ClassDef) -> None:
                name = get_full_name_for_node(node.name)
                if name:
                    self.names.append(f"class: {name}")
        
        collector = NameCollector()
        module.visit(collector)
        
        # Verify we found expected definitions
        self.assertGreater(len(collector.names), 0)

def generate_boilerplate_code():
    """Example of generating code with templates"""
    
    # Create a module template
    template = """
    {header_comment}
    
    from typing import {type_imports}
    
    class {class_name}:
        def __init__(self, {init_params}) -> None:
            {init_body}
            
        def {method_name}(self) -> {return_type}:
            {method_body}
    """
    
    # Generate code using template
    module = parse_template_module(
        template,
        header_comment=cst.SimpleStatementLine([
            cst.Expr(cst.SimpleString('"""Generated class module."""'))
        ]),
        type_imports=cst.Name("Optional, List"),
        class_name=cst.Name("DataProcessor"),
        init_params=cst.Parameters([
            cst.Param(cst.Name("data"), cst.Annotation(cst.Name("List[str]")))
        ]),
        init_body=cst.SimpleStatementLine([
            cst.Assign([cst.AssignTarget(cst.Attribute(cst.Name("self"), cst.Name("data")))], cst.Name("data"))
        ]),
        method_name=cst.Name("process"),
        return_type=cst.Name("Optional[str]"),
        method_body=cst.SimpleStatementLine([
            cst.Return(cst.Name("None"))
        ])
    )
    
    # Add copyright header
    final_module = insert_header_comments(module, [
        "# Copyright 2024 Example Corp",
        "# Auto-generated code - do not edit manually"
    ])
    
    return final_module.code

if __name__ == "__main__":
    # Generate and print example code
    generated_code = generate_boilerplate_code()
    print("Generated Code:")
    print(generated_code)
    
    # Calculate module information
    module_info = calculate_module_and_package(
        repo_root="/project/root",
        filename="/project/root/src/generated/processor.py"
    )
    print(f"\nModule: {module_info.name}")
    print(f"Package: {module_info.package}")
```

### Advanced Template Usage

```python
from libcst.helpers import parse_template_statement, parse_template_expression

def create_error_handling_wrapper():
    """Create error handling code using templates"""
    
    # Template for try-catch wrapper
    try_template = """
    try:
        {original_call}
    except {exception_type} as e:
        {error_handler}
    """
    
    # Create the wrapped statement
    wrapped = parse_template_statement(
        try_template,
        original_call=parse_template_expression("process_data(input_value)"),
        exception_type=cst.Name("ValueError"),
        error_handler=parse_template_statement(
            'logger.error("Processing failed: %s", str(e))'
        )
    )
    
    return wrapped
```

This comprehensive utilities documentation provides developers with all the tools they need to work effectively with LibCST, from basic node manipulation to advanced code generation and testing workflows.