# Plugin System

Extensible plugin architecture for customizing type checking behavior and adding support for specific libraries or frameworks. Mypy's plugin system allows deep integration with third-party libraries and custom type checking logic.

## Capabilities

### Core Plugin Classes

Base classes for creating mypy plugins that customize type checking behavior.

```python { .api }
class Plugin:
    """
    Base class for mypy plugins.
    
    Plugins can customize various aspects of type checking by providing
    hooks that are called during different phases of analysis.
    
    Methods to override:
    - get_type_analyze_hook(self, fullname: str) -> Callable | None
    - get_function_hook(self, fullname: str) -> Callable | None  
    - get_method_hook(self, fullname: str) -> Callable | None
    - get_attribute_hook(self, fullname: str) -> Callable | None
    - get_class_decorator_hook(self, fullname: str) -> Callable | None
    - get_metaclass_hook(self, fullname: str) -> Callable | None
    - get_base_class_hook(self, fullname: str) -> Callable | None
    """
    
    def __init__(self, options: Options):
        """Initialize plugin with mypy options."""

class CommonPluginApi:
    """
    Common API available to plugin callbacks.
    
    Provides access to type analysis utilities, name lookup,
    and type construction functions used by plugins.
    
    Attributes:
    - modules: dict[str, MypyFile] - Loaded modules
    - msg: MessageBuilder - Error reporting
    - options: Options - Mypy configuration
    
    Methods:
    - named_generic_type(name: str, args: list[Type]) -> Instance
    - named_type(name: str) -> Instance
    - lookup_fully_qualified(name: str) -> SymbolTableNode | None
    - fail(msg: str, ctx: Context) -> None
    """

class SemanticAnalyzerPluginInterface(CommonPluginApi):
    """
    API available during semantic analysis phase.
    
    Used for plugins that need to analyze code structure,
    modify AST nodes, or add new symbol table entries.
    
    Additional methods:
    - add_symbol_table_node(name: str, stnode: SymbolTableNode) -> None
    - lookup_current_scope(name: str) -> SymbolTableNode | None
    - defer_node(node: Node, enclosing_class: TypeInfo | None) -> None
    """

class CheckerPluginInterface(CommonPluginApi):
    """
    API available during type checking phase.
    
    Used for plugins that need to perform custom type checking,
    validate specific patterns, or integrate with type inference.
    
    Additional methods:
    - check_subtype(left: Type, right: Type, ctx: Context) -> bool
    - type_check_expr(expr: Expression, type_context: Type | None) -> Type
    """
```

### Plugin Context Classes

Context objects passed to plugin hooks containing information about the analysis context.

```python { .api }
class FunctionContext:
    """
    Context for function call analysis hooks.
    
    Attributes:
    - default_return_type: Type - Default return type
    - arg_types: list[list[Type]] - Argument types for each argument
    - arg_names: list[list[str | None]] - Argument names
    - callee_type: Type - Type of the function being called
    - context: Context - AST context for error reporting
    - api: CheckerPluginInterface - Type checker API
    """

class AttributeContext:
    """
    Context for attribute access analysis hooks.
    
    Attributes:
    - default_attr_type: Type - Default attribute type
    - type: Type - Type of the object being accessed
    - context: Context - AST context
    - api: CheckerPluginInterface - Type checker API
    """

class ClassDefContext:
    """
    Context for class definition analysis hooks.
    
    Attributes:
    - cls: ClassDef - Class definition AST node
    - reason: Type - Reason for hook invocation
    - api: SemanticAnalyzerPluginInterface - Semantic analyzer API
    """

class BaseClassContext:
    """
    Context for base class analysis hooks.
    
    Attributes:
    - cls: ClassDef - Class definition
    - arg: Expression - Base class expression
    - default_base: Type - Default base class type
    - api: SemanticAnalyzerPluginInterface - API access
    """
```

## Built-in Plugins

### Default Plugin

Core plugin providing built-in type checking functionality.

```python { .api }
class DefaultPlugin(Plugin):
    """
    Default plugin with built-in type handling.
    
    Provides standard type checking for:
    - Built-in functions and types
    - Standard library modules
    - Common Python patterns
    - Generic type instantiation
    """
```

### Library-Specific Plugins

Pre-built plugins for popular Python libraries.

```python { .api }
# Available built-in plugins in mypy.plugins:

class AttrsPlugin(Plugin):
    """Support for attrs library decorators and classes."""

class DataclassesPlugin(Plugin):
    """Support for dataclasses with proper type inference."""

class EnumsPlugin(Plugin):
    """Enhanced support for enum.Enum classes."""

class FunctoolsPlugin(Plugin):
    """Support for functools decorators like @lru_cache."""

class CtypesPlugin(Plugin):
    """Support for ctypes library type checking."""

class SqlAlchemyPlugin(Plugin):
    """Support for SQLAlchemy ORM type checking."""
```

## Creating Custom Plugins

### Basic Plugin Structure

```python
from mypy.plugin import Plugin, FunctionContext
from mypy.types import Type, Instance
from mypy.nodes import ARG_POS, Argument, Var, PassStmt

class CustomPlugin(Plugin):
    """Example custom plugin for specialized type checking."""
    
    def get_function_hook(self, fullname: str):
        """Return hook for specific function calls."""
        if fullname == "mylib.special_function":
            return self.handle_special_function
        elif fullname == "mylib.create_instance":
            return self.handle_create_instance
        return None
    
    def handle_special_function(self, ctx: FunctionContext) -> Type:
        """Custom handling for mylib.special_function."""
        # Validate arguments
        if len(ctx.arg_types) != 2:
            ctx.api.fail("special_function requires exactly 2 arguments", ctx.context)
            return ctx.default_return_type
        
        # Check first argument is string
        first_arg = ctx.arg_types[0][0] if ctx.arg_types[0] else None
        if not isinstance(first_arg, Instance) or first_arg.type.fullname != 'builtins.str':
            ctx.api.fail("First argument must be a string", ctx.context)
        
        # Return custom type based on analysis
        return ctx.api.named_type('mylib.SpecialResult')
    
    def handle_create_instance(self, ctx: FunctionContext) -> Type:
        """Custom factory function handling."""
        if ctx.arg_types and ctx.arg_types[0]:
            # Return instance of type specified in first argument
            type_arg = ctx.arg_types[0][0]
            if isinstance(type_arg, Instance):
                return type_arg
        
        return ctx.default_return_type

# Plugin entry point
def plugin(version: str):
    """Entry point for mypy plugin discovery."""
    return CustomPlugin
```

### Advanced Plugin Features

```python
from mypy.plugin import Plugin, ClassDefContext, BaseClassContext
from mypy.types import Type, Instance, CallableType
from mypy.nodes import ClassDef, FuncDef, Decorator

class AdvancedPlugin(Plugin):
    """Advanced plugin with class and decorator handling."""
    
    def get_class_decorator_hook(self, fullname: str):
        """Handle class decorators."""
        if fullname == "mylib.special_class":
            return self.handle_special_class_decorator
        return None
    
    def get_base_class_hook(self, fullname: str):
        """Handle specific base classes."""
        if fullname == "mylib.BaseModel":
            return self.handle_base_model
        return None
    
    def handle_special_class_decorator(self, ctx: ClassDefContext) -> None:
        """Process @special_class decorator."""
        # Add special methods to the class
        self.add_magic_method(ctx.cls, "__special__", 
                            ctx.api.named_type('builtins.str'))
    
    def handle_base_model(self, ctx: BaseClassContext) -> Type:
        """Handle BaseModel inheritance."""
        # Analyze class for special fields
        if isinstance(ctx.cls, ClassDef):
            self.process_model_fields(ctx.cls, ctx.api)
        
        return ctx.default_base
    
    def add_magic_method(self, cls: ClassDef, method_name: str, 
                        return_type: Type) -> None:
        """Add a magic method to class definition."""
        # Create method signature
        method_type = CallableType(
            arg_types=[Instance(cls.info, [])],  # self parameter
            arg_kinds=[ARG_POS],
            arg_names=['self'],
            return_type=return_type,
            fallback=self.lookup_typeinfo('builtins.function')
        )
        
        # Add to symbol table
        method_node = FuncDef(method_name, [], None, None)
        method_node.type = method_type
        cls.info.names[method_name] = method_node
    
    def process_model_fields(self, cls: ClassDef, api) -> None:
        """Process model field definitions."""
        for stmt in cls.defs.body:
            if isinstance(stmt, AssignmentStmt):
                # Analyze field assignments
                self.analyze_field_assignment(stmt, api)

def plugin(version: str):
    return AdvancedPlugin
```

### Plugin with Type Analysis

```python
from mypy.plugin import Plugin, AttributeContext
from mypy.types import Type, Instance, AnyType, TypeOfAny

class TypeAnalysisPlugin(Plugin):
    """Plugin demonstrating type analysis capabilities."""
    
    def get_attribute_hook(self, fullname: str):
        """Handle attribute access."""
        if fullname == "mylib.DynamicObject.__getattr__":
            return self.handle_dynamic_getattr
        return None
    
    def handle_dynamic_getattr(self, ctx: AttributeContext) -> Type:
        """Handle dynamic attribute access."""
        # Analyze the attribute name
        attr_name = self.get_attribute_name(ctx)
        
        if attr_name and attr_name.startswith('computed_'):
            # Return specific type for computed attributes
            return ctx.api.named_type('builtins.float')
        elif attr_name and attr_name.startswith('cached_'):
            # Return cached value type
            return self.get_cached_type(attr_name, ctx)
        
        # Default to Any for unknown dynamic attributes
        return AnyType(TypeOfAny.from_error)
    
    def get_attribute_name(self, ctx: AttributeContext) -> str | None:
        """Extract attribute name from context."""
        # This would need to analyze the AST context
        # Implementation depends on specific use case
        return None
    
    def get_cached_type(self, attr_name: str, ctx: AttributeContext) -> Type:
        """Determine type for cached attributes."""
        # Custom logic for determining cached value types
        cache_map = {
            'cached_count': ctx.api.named_type('builtins.int'),
            'cached_name': ctx.api.named_type('builtins.str'),
            'cached_data': ctx.api.named_type('builtins.list')
        }
        
        return cache_map.get(attr_name, AnyType(TypeOfAny.from_error))

def plugin(version: str):
    return TypeAnalysisPlugin
```

## Plugin Configuration and Loading

### Plugin Entry Points

```python
# setup.py or pyproject.toml configuration for plugin distribution
from setuptools import setup

setup(
    name="mypy-custom-plugin",
    entry_points={
        "mypy.plugins": [
            "custom_plugin = mypy_custom_plugin.plugin:plugin"
        ]
    }
)
```

### Plugin Loading in mypy.ini

```ini
[mypy]
plugins = mypy_custom_plugin.plugin, another_plugin

[mypy-mylib.*]
# Plugin-specific configuration
ignore_errors = false
```

### Plugin Loading Programmatically

```python
from mypy.build import build, BuildSource
from mypy.options import Options

# Load plugin programmatically
options = Options()
options.plugins = ['mypy_custom_plugin.plugin']

# Custom plugin instance
plugin_instance = CustomPlugin(options)

sources = [BuildSource("myfile.py", None, None)]
result = build(sources, options, extra_plugins=[plugin_instance])
```

## Testing Plugins

### Plugin Test Framework

```python
import tempfile
import os
from mypy import api
from mypy.test.helpers import Suite

class PluginTestCase:
    """Test framework for mypy plugins."""
    
    def run_with_plugin(self, source_code: str, plugin_path: str) -> tuple[str, str, int]:
        """Run mypy with plugin on source code."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(source_code)
            temp_file = f.name
        
        try:
            # Run mypy with plugin
            result = api.run([
                '--plugins', plugin_path,
                '--show-error-codes',
                temp_file
            ])
            return result
        finally:
            os.unlink(temp_file)
    
    def assert_no_errors(self, result: tuple[str, str, int]):
        """Assert that mypy found no errors."""
        stdout, stderr, exit_code = result
        assert exit_code == 0, f"Expected no errors, got: {stderr}"
    
    def assert_error_contains(self, result: tuple[str, str, int], 
                            expected_message: str):
        """Assert that error output contains expected message."""
        stdout, stderr, exit_code = result
        assert expected_message in stderr, f"Expected '{expected_message}' in: {stderr}"

# Usage
def test_custom_plugin():
    """Test custom plugin functionality."""
    source = '''
from mylib import special_function

result = special_function("hello", 42)  # Should pass
result2 = special_function(123, "world")  # Should fail
'''
    
    test_case = PluginTestCase()
    result = test_case.run_with_plugin(source, "mypy_custom_plugin.plugin")
    
    # Should have one error for the second call
    test_case.assert_error_contains(result, "First argument must be a string")
```

### Integration Testing

```python
import pytest
from mypy import api

class TestPluginIntegration:
    """Integration tests for plugin with real codebases."""
    
    @pytest.fixture
    def sample_project(self, tmp_path):
        """Create sample project for testing."""
        # Create project structure
        (tmp_path / "mylib").mkdir()
        (tmp_path / "mylib" / "__init__.py").write_text("")
        
        (tmp_path / "mylib" / "core.py").write_text('''
class SpecialResult:
    def __init__(self, value: str):
        self.value = value

def special_function(name: str, count: int) -> SpecialResult:
    return SpecialResult(f"{name}_{count}")
''')
        
        (tmp_path / "main.py").write_text('''
from mylib.core import special_function

result = special_function("test", 5)
print(result.value)
''')
        
        return tmp_path
    
    def test_plugin_with_project(self, sample_project):
        """Test plugin with complete project."""
        os.chdir(sample_project)
        
        result = api.run([
            '--plugins', 'mypy_custom_plugin.plugin',
            'main.py'
        ])
        
        stdout, stderr, exit_code = result
        assert exit_code == 0, f"Plugin failed on project: {stderr}"
```

## Plugin Best Practices

### Performance Considerations

```python
class EfficientPlugin(Plugin):
    """Example of performance-conscious plugin design."""
    
    def __init__(self, options):
        super().__init__(options)
        # Cache expensive computations
        self._type_cache = {}
        self._analyzed_classes = set()
    
    def get_function_hook(self, fullname: str):
        # Use early returns to avoid unnecessary work
        if not fullname.startswith('mylib.'):
            return None
        
        # Cache hook lookups
        if fullname not in self._hook_cache:
            self._hook_cache[fullname] = self._compute_hook(fullname)
        
        return self._hook_cache[fullname]
    
    def handle_expensive_operation(self, ctx: FunctionContext) -> Type:
        """Cache expensive type computations."""
        cache_key = (ctx.callee_type, tuple(str(t) for t in ctx.arg_types[0]))
        
        if cache_key in self._type_cache:
            return self._type_cache[cache_key]
        
        # Perform expensive computation
        result = self._compute_type(ctx)
        self._type_cache[cache_key] = result
        return result
```

### Error Handling in Plugins

```python
class RobustPlugin(Plugin):
    """Plugin with proper error handling."""
    
    def handle_function_call(self, ctx: FunctionContext) -> Type:
        """Safely handle function calls with error recovery."""
        try:
            # Validate context
            if not ctx.arg_types:
                ctx.api.fail("Missing arguments", ctx.context)
                return ctx.default_return_type
            
            # Perform analysis
            return self._analyze_call(ctx)
            
        except Exception as e:
            # Log error and fall back to default behavior
            ctx.api.fail(f"Plugin error: {e}", ctx.context)
            return ctx.default_return_type
    
    def _analyze_call(self, ctx: FunctionContext) -> Type:
        """Internal analysis with proper error handling."""
        # Implementation with validation at each step
        pass
```