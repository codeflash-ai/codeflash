# Codemod Framework

LibCST's codemod framework provides a high-level system for building automated code transformation applications. It handles context management, parallel processing, file discovery, error handling, and testing utilities for large-scale refactoring operations.

## Capabilities

### Base Codemod Classes

Foundation classes for building code transformation commands.

```python { .api }
class CodemodCommand:
    """Base class for codemod commands."""
    
    def transform_module(self, tree: Module) -> Module:
        """
        Transform a module CST.
        
        Parameters:
        - tree: Module CST to transform
        
        Returns:
        Module: Transformed module CST
        """
    
    def get_description(self) -> str:
        """Get human-readable description of the transformation."""

class VisitorBasedCodemodCommand(CodemodCommand):
    """Base class for visitor-based codemod commands."""
    
    def leave_Module(self, original_node: Module, updated_node: Module) -> Module:
        """Transform module using visitor pattern."""

class MagicArgsCodemodCommand(CodemodCommand):
    """Base class for commands with automatic argument handling."""
```

### Context Management

Manage execution context and state during transformations.

```python { .api }
class CodemodContext:
    """Execution context for codemod operations."""
    
    def __init__(self, metadata_wrapper: MetadataWrapper) -> None:
        """
        Initialize codemod context.
        
        Parameters:
        - metadata_wrapper: Metadata wrapper for the module
        """
    
    @property
    def metadata_wrapper(self) -> MetadataWrapper:
        """Access to metadata for the current module."""

class ContextAwareTransformer(CSTTransformer):
    """Transformer with access to codemod context."""
    
    def __init__(self, context: CodemodContext) -> None:
        """
        Initialize context-aware transformer.
        
        Parameters:
        - context: Codemod execution context
        """
    
    @property
    def context(self) -> CodemodContext:
        """Access to codemod context."""

class ContextAwareVisitor(CSTVisitor):
    """Visitor with access to codemod context."""
    
    def __init__(self, context: CodemodContext) -> None:
        """
        Initialize context-aware visitor.
        
        Parameters:
        - context: Codemod execution context
        """
    
    @property  
    def context(self) -> CodemodContext:
        """Access to codemod context."""
```

### Transform Results

Result types for tracking transformation outcomes.

```python { .api }
class TransformResult:
    """Base class for transformation results."""
    code: Optional[str]
    encoding: str

class TransformSuccess(TransformResult):
    """Successful transformation result."""

class TransformFailure(TransformResult):
    """Failed transformation result."""
    error: Exception

class TransformSkip(TransformResult):
    """Skipped transformation result."""
    skip_reason: SkipReason

class TransformExit(TransformResult):
    """Early exit transformation result."""

class SkipFile(TransformResult):
    """File skip result."""

class SkipReason:
    """Enumeration of skip reasons."""
    BLACKLISTED: ClassVar[str]
    OTHER: ClassVar[str]

class ParallelTransformResult:
    """Results from parallel transformation execution."""
    successes: int
    failures: int
    skips: int
    warnings: int
```

### Execution Functions

High-level functions for running codemod transformations.

```python { .api }
def transform_module(
    code: str, 
    command: CodemodCommand,
    *, 
    python_version: str = "3.8"
) -> TransformResult:
    """
    Transform a single module with a codemod command.
    
    Parameters:
    - code: Source code to transform
    - command: Codemod command to apply
    - python_version: Target Python version
    
    Returns:
    TransformResult: Result of the transformation
    """

def gather_files(
    paths: Sequence[str], 
    *, 
    include_generated: bool = False,
    exclude_patterns: Sequence[str] = ()
) -> Sequence[str]:
    """
    Gather Python files for transformation.
    
    Parameters:
    - paths: Directories or files to search
    - include_generated: Include generated files
    - exclude_patterns: Patterns to exclude
    
    Returns:
    Sequence[str]: List of Python file paths
    """

def exec_transform_with_prettyprint(
    command: CodemodCommand,
    files: Sequence[str],
    *,
    jobs: int = 1,
    show_diff: bool = True,
    unified_diff_lines: int = 5,
    hide_generated: bool = True,
    hide_blacklisted: bool = True,
    hide_progress: bool = False
) -> int:
    """
    Execute transformation with formatted output.
    
    Parameters:
    - command: Codemod command to execute
    - files: Files to transform
    - jobs: Number of parallel jobs
    - show_diff: Show code differences
    - unified_diff_lines: Context lines in diff
    - hide_generated: Hide generated files from output
    - hide_blacklisted: Hide blacklisted files
    - hide_progress: Hide progress indicators
    
    Returns:
    int: Exit code (0 for success)
    """

def parallel_exec_transform_with_prettyprint(
    command: CodemodCommand,
    files: Sequence[str],
    *,
    jobs: int = 1,
    **kwargs
) -> ParallelTransformResult:
    """
    Execute transformation in parallel with formatted output.
    
    Parameters:
    - command: Codemod command to execute
    - files: Files to transform
    - jobs: Number of parallel jobs
    - kwargs: Additional arguments for exec_transform_with_prettyprint
    
    Returns:
    ParallelTransformResult: Aggregated results
    """

def diff_code(
    old_code: str, 
    new_code: str, 
    *, 
    filename: str = "<unknown>",
    lines: int = 5
) -> str:
    """
    Generate unified diff between code versions.
    
    Parameters:
    - old_code: Original code
    - new_code: Transformed code
    - filename: Filename for diff header
    - lines: Context lines
    
    Returns:
    str: Unified diff output
    """
```

### Testing Framework

Testing utilities for codemod development and validation.

```python { .api }
class CodemodTest:
    """Base class for codemod testing."""
    
    def assert_transform(
        self,
        before: str,
        after: str,
        *,
        command: Optional[CodemodCommand] = None,
        python_version: str = "3.8"
    ) -> None:
        """
        Assert that transformation produces expected result.
        
        Parameters:
        - before: Source code before transformation
        - after: Expected code after transformation
        - command: Codemod command (defaults to self.command)
        - python_version: Target Python version
        """
    
    def assert_unchanged(
        self,
        code: str,
        *,
        command: Optional[CodemodCommand] = None,
        python_version: str = "3.8"
    ) -> None:
        """
        Assert that transformation leaves code unchanged.
        
        Parameters:
        - code: Source code to test
        - command: Codemod command (defaults to self.command)
        - python_version: Target Python version
        """
```

## Usage Examples

### Simple Codemod Command

```python
import libcst as cst
from libcst.codemod import VisitorBasedCodemodCommand

class RemovePrintStatements(VisitorBasedCodemodCommand):
    """Remove all print() calls from code."""
    
    def leave_Call(self, original_node, updated_node):
        # Remove print function calls
        if isinstance(updated_node.func, cst.Name) and updated_node.func.value == "print":
            return cst.RemoveFromParent
        return updated_node
    
    def get_description(self):
        return "Remove all print() statements"

# Usage
from libcst.codemod import transform_module

source = '''
def process_data(data):
    print("Processing data...")
    result = data * 2
    print(f"Result: {result}")
    return result
'''

command = RemovePrintStatements()
result = transform_module(source, command)

if isinstance(result, cst.codemod.TransformSuccess):
    print("Transformation successful:")
    print(result.code)
else:
    print(f"Transformation failed: {result.error}")
```

### Context-Aware Codemod

```python
import libcst as cst
from libcst.codemod import VisitorBasedCodemodCommand, CodemodContext
from libcst.metadata import ScopeProvider

class SafeVariableRenamer(VisitorBasedCodemodCommand):
    """Rename variables while avoiding conflicts."""
    
    METADATA_DEPENDENCIES = (ScopeProvider,)
    
    def __init__(self, old_name: str, new_name: str):
        super().__init__()
        self.old_name = old_name
        self.new_name = new_name
    
    def leave_Name(self, original_node, updated_node):
        if updated_node.value == self.old_name:
            # Check if new name conflicts in current scope
            scope = self.resolve(ScopeProvider)[updated_node]
            if self.new_name not in scope:
                return updated_node.with_changes(value=self.new_name)
        return updated_node
    
    def get_description(self):
        return f"Rename '{self.old_name}' to '{self.new_name}' safely"

# Usage
source = '''
def example():
    x = 1
    y = x + 2
    return y
'''

command = SafeVariableRenamer("x", "input_value")
result = transform_module(source, command)
```

### Batch File Processing

```python
from libcst.codemod import (
    gather_files, 
    exec_transform_with_prettyprint,
    parallel_exec_transform_with_prettyprint
)

class UpdateImports(VisitorBasedCodemodCommand):
    """Update deprecated import statements."""
    
    def leave_ImportFrom(self, original_node, updated_node):
        if isinstance(updated_node.module, cst.Attribute):
            if updated_node.module.value.value == "oldmodule":
                new_module = updated_node.module.with_changes(
                    value=cst.Name("newmodule")
                )
                return updated_node.with_changes(module=new_module)
        return updated_node

# Process multiple files
command = UpdateImports()
files = gather_files(["src/", "tests/"], exclude_patterns=["**/generated/**"])

# Sequential processing
exit_code = exec_transform_with_prettyprint(
    command,
    files,
    show_diff=True,
    unified_diff_lines=3
)

# Parallel processing for large codebases
result = parallel_exec_transform_with_prettyprint(
    command,
    files,
    jobs=4,
    show_diff=True
)

print(f"Processed {result.successes} files successfully")
print(f"Failed on {result.failures} files")
print(f"Skipped {result.skips} files")
```

### Advanced Pattern-Based Codemod

```python
import libcst as cst
import libcst.matchers as m
from libcst.codemod import VisitorBasedCodemodCommand

class ModernizeExceptionHandling(VisitorBasedCodemodCommand):
    """Modernize exception handling patterns."""
    
    def leave_ExceptHandler(self, original_node, updated_node):
        # Transform "except Exception, e:" to "except Exception as e:"
        if (isinstance(updated_node.type, cst.Name) and 
            isinstance(updated_node.name, cst.AsName) and
            isinstance(updated_node.name.name, cst.Name)):
            
            # Check if this is old-style exception syntax
            if updated_node.name.asname is None:
                return updated_node.with_changes(
                    name=cst.AsName(
                        name=updated_node.name.name,
                        asname=cst.AsName(name=updated_node.name.name)
                    )
                )
        return updated_node
    
    def get_description(self):
        return "Modernize exception handling syntax"

# Usage with matcher-based replacement
class ReplaceAssertWithRaise(VisitorBasedCodemodCommand):
    """Replace assert False with raise statements."""
    
    def leave_Assert(self, original_node, updated_node):
        # Replace "assert False, msg" with "raise AssertionError(msg)"
        if m.matches(updated_node.test, m.Name(value="False")):
            if updated_node.msg:
                return cst.Raise(
                    exc=cst.Call(
                        func=cst.Name("AssertionError"),
                        args=[cst.Arg(value=updated_node.msg)]
                    )
                )
            else:
                return cst.Raise(exc=cst.Call(func=cst.Name("AssertionError")))
        return updated_node
```

### Codemod Testing

```python
import unittest
from libcst.codemod import CodemodTest

class TestRemovePrintStatements(CodemodTest):
    command = RemovePrintStatements()
    
    def test_removes_print_calls(self):
        before = '''
def example():
    print("hello")
    x = 42
    print("world")
    return x
'''
        after = '''
def example():
    x = 42
    return x
'''
        self.assert_transform(before, after)
    
    def test_preserves_other_calls(self):
        code = '''
def example():
    log("message")
    return calculate()
'''
        self.assert_unchanged(code)
    
    def test_removes_nested_prints(self):
        before = '''
if condition:
    print("debug")
    process()
'''
        after = '''
if condition:
    process()
'''
        self.assert_transform(before, after)

if __name__ == "__main__":
    unittest.main()
```

## Types

```python { .api }
# Base command types
class CodemodCommand:
    """Base class for codemod commands."""

class VisitorBasedCodemodCommand(CodemodCommand):
    """Visitor-based codemod command."""

class MagicArgsCodemodCommand(CodemodCommand):
    """Command with automatic argument handling."""

# Context types
class CodemodContext:
    """Execution context for codemods."""
    metadata_wrapper: MetadataWrapper

class ContextAwareTransformer(CSTTransformer):
    """Transformer with context access."""
    context: CodemodContext

class ContextAwareVisitor(CSTVisitor):
    """Visitor with context access."""
    context: CodemodContext

# Result types
class TransformResult:
    """Base transformation result."""
    code: Optional[str]
    encoding: str

class TransformSuccess(TransformResult): ...
class TransformFailure(TransformResult):
    error: Exception
class TransformSkip(TransformResult):
    skip_reason: SkipReason
class TransformExit(TransformResult): ...
class SkipFile(TransformResult): ...

class SkipReason:
    BLACKLISTED: ClassVar[str]
    OTHER: ClassVar[str]

class ParallelTransformResult:
    """Results from parallel execution."""
    successes: int
    failures: int
    skips: int
    warnings: int

# Testing types
class CodemodTest:
    """Base class for codemod testing."""
    command: CodemodCommand
```