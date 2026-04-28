# Build System Integration

Advanced APIs for custom build systems and development tools that need fine-grained control over the type checking process. These APIs provide programmatic access to mypy's internal build system and analysis engine.

## Capabilities

### Main Build Function

The primary entry point for programmatic type analysis with full control over the build process.

```python { .api }
def build(sources, options, alt_lib_path=None, flush_errors=None, 
          fscache=None, stdout=None, stderr=None, extra_plugins=None) -> BuildResult:
    """
    Main function to analyze a program programmatically.
    
    Parameters:
    - sources: list[BuildSource] - List of BuildSource objects to analyze
    - options: Options - Options object with configuration settings
    - alt_lib_path: str | None - Alternative library path for imports
    - flush_errors: callable | None - Callback for flushing error messages
    - fscache: FileSystemCache | None - File system cache for performance
    - stdout: TextIO | None - Output stream for messages
    - stderr: TextIO | None - Error stream for diagnostics
    - extra_plugins: list[Plugin] | None - Additional plugins to load
    
    Returns:
    - BuildResult: Analysis results with errors, types, and file information
    
    Usage:
    result = build(sources, options)
    """
```

### Build Source Definition

Represents source files or modules to be analyzed.

```python { .api }
class BuildSource:
    """
    Represents a source file or module to be analyzed.
    
    Attributes:
    - path: str | None - File path (None for in-memory modules)
    - module: str | None - Module name (None for file-based sources)
    - text: str | None - Source code text (None to read from file)
    - followed: bool - Whether this source was found by following imports
    
    Usage:
    # File-based source
    source = BuildSource("myfile.py", module=None, text=None)
    
    # Module-based source
    source = BuildSource(path=None, module="mypackage.mymodule", text=None)
    
    # In-memory source
    source = BuildSource(path="<string>", module=None, text="def func(): pass")
    """
    
    def __init__(self, path: str | None, module: str | None, text: str | None, followed: bool = False):
        """Initialize a build source."""
```

### Build Results

Contains the results of type checking analysis.

```python { .api }
class BuildResult:
    """
    Contains the results of type checking analysis.
    
    Attributes:
    - errors: list[str] - List of error message strings
    - files: dict[str, MypyFile] - Mapping of module names to AST nodes
    - types: dict[Expression, Type] - Type information for expressions
    - used_cache: bool - Whether cached results were used
    - manager: BuildManager | None - Build manager instance
    
    Usage:
    if result.errors:
        for error in result.errors:
            print(error)
    """
```

### Build Manager

Advanced build management for fine-grained control over the type checking process.

```python { .api }
class BuildManager:
    """
    Build manager that coordinates parsing, import processing, semantic analysis and type checking.
    
    This class holds shared state for building a mypy program and is used for advanced
    build scenarios that need fine-grained control over the analysis process.
    
    Attributes:
    - data_dir: str - Mypy data directory (contains stubs)
    - search_paths: SearchPaths - Module search paths configuration
    - modules: dict[str, MypyFile] - Mapping of module ID to MypyFile
    - options: Options - Build options
    - missing_modules: set[str] - Set of modules that could not be imported
    - stale_modules: set[str] - Set of modules that needed to be rechecked
    - version_id: str - The current mypy version identifier
    - plugin: Plugin | None - Active mypy plugin(s)
    
    Usage:
    manager = BuildManager(data_dir, search_paths, options, version_id)
    result = manager.build(sources)
    """
```

### File System Cache

Caching system for improved performance during type checking operations.

```python { .api }
class FileSystemCache:
    """
    File system cache for improved performance during mypy operations.
    
    Provides caching of file system operations and module information
    to speed up repeated type checking runs.
    
    Methods:
    - flush(): Clear all cached data
    - set_package_root(package_root: list[str]): Set package root directories
    
    Usage:
    cache = FileSystemCache()
    cache.set_package_root(['/path/to/package'])
    result = build(sources, options, fscache=cache)
    """
```

### Search Paths Configuration

Module search path configuration for finding Python modules and packages.

```python { .api }
class SearchPaths:
    """
    Configuration for module search paths used during import resolution.
    
    Attributes:
    - python_path: tuple[str, ...] - Where user code is found
    - mypy_path: tuple[str, ...] - From $MYPYPATH or config variable
    - package_path: tuple[str, ...] - From get_site_packages_dirs()
    - typeshed_path: tuple[str, ...] - Paths in typeshed
    
    Usage:
    paths = SearchPaths(python_path, mypy_path, package_path, typeshed_path)
    """
```

### Configuration Options

Comprehensive configuration object for controlling type checking behavior.

```python { .api }
class Options:
    """
    Configuration container for all mypy settings.
    
    Build Configuration:
    - python_version: tuple[int, int] - Target Python version (default: current)
    - platform: str - Target platform ('linux', 'win32', 'darwin')
    - custom_typing_module: str | None - Custom typing module name
    - mypy_path: list[str] - Additional module search paths
    - namespace_packages: bool - Support namespace packages
    - explicit_package_bases: bool - Require __init__.py for packages
    
    Type Checking Options:
    - strict_mode: bool - Enable all strict mode flags
    - disallow_untyped_defs: bool - Disallow untyped function definitions
    - disallow_incomplete_defs: bool - Disallow partially typed definitions
    - disallow_untyped_decorators: bool - Disallow untyped decorators
    - disallow_any_generics: bool - Disallow generic types without parameters
    - disallow_any_unimported: bool - Disallow Any from missing imports
    - disallow_subclassing_any: bool - Disallow subclassing Any
    - warn_redundant_casts: bool - Warn about redundant casts
    - warn_unused_ignores: bool - Warn about unused # type: ignore
    - warn_return_any: bool - Warn about returning Any from functions
    
    Error Reporting:
    - show_error_codes: bool - Show error codes in output
    - show_column_numbers: bool - Show column numbers in errors
    - show_absolute_path: bool - Show absolute paths in errors
    - ignore_missing_imports: bool - Ignore missing import errors
    - follow_imports: str - Import following mode ('normal', 'silent', 'skip')
    
    Performance:
    - incremental: bool - Enable incremental mode
    - cache_dir: str - Directory for cache files
    - sqlite_cache: bool - Use SQLite for caching
    - skip_version_check: bool - Skip cache version checks
    
    Usage:
    options = Options()
    options.strict_mode = True
    options.python_version = (3, 11)
    """
```

## Usage Examples

### Basic Build Integration

```python
from mypy.build import build, BuildSource
from mypy.options import Options

# Configure options
options = Options()
options.strict_mode = True
options.python_version = (3, 11)
options.show_error_codes = True

# Define sources to analyze
sources = [
    BuildSource("src/main.py", module=None, text=None),
    BuildSource("src/utils.py", module=None, text=None),
]

# Perform analysis
result = build(sources, options)

# Process results
if result.errors:
    print("Type errors found:")
    for error in result.errors:
        print(f"  {error}")
else:
    print("No type errors!")

# Access type information
for expr, typ in result.types.items():
    print(f"Expression {expr} has type {typ}")
```

### Advanced Build Configuration

```python
from mypy.build import build, BuildSource
from mypy.options import Options
from mypy.fscache import FileSystemCache
import sys

class AdvancedTypeChecker:
    """Advanced type checker with custom configuration."""
    
    def __init__(self, cache_dir=".mypy_cache"):
        self.options = Options()
        self.cache = FileSystemCache(cache_dir)
        self.setup_options()
    
    def setup_options(self):
        """Configure type checking options."""
        # Strict type checking
        self.options.strict_mode = True
        
        # Target configuration
        self.options.python_version = (3, 11)
        self.options.platform = sys.platform
        
        # Error reporting
        self.options.show_error_codes = True
        self.options.show_column_numbers = True
        self.options.color_output = True
        
        # Performance
        self.options.incremental = True
        self.options.cache_dir = ".mypy_cache"
        
        # Import handling
        self.options.follow_imports = "normal"
        self.options.namespace_packages = True
    
    def check_project(self, source_files):
        """Type check a project."""
        # Convert file paths to BuildSource objects
        sources = []
        for filepath in source_files:
            source = BuildSource(filepath, module=None, text=None)
            sources.append(source)
        
        # Perform type checking
        result = build(
            sources=sources,
            options=self.options,
            fscache=self.cache,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        
        return result
    
    def check_module(self, module_name, source_text):
        """Type check a module from source text."""
        source = BuildSource(
            path=f"<{module_name}>",
            module=module_name,
            text=source_text
        )
        
        result = build([source], self.options)
        return result

# Usage
checker = AdvancedTypeChecker()
result = checker.check_project(['src/main.py', 'src/utils.py'])

if result.errors:
    print(f"Found {len(result.errors)} type errors")
```

### Custom Error Handling

```python
from mypy.build import build, BuildSource
from mypy.options import Options
from mypy.errors import CompileError
import sys
from io import StringIO

def type_check_with_custom_errors(sources, options):
    """Type check with custom error handling and formatting."""
    
    # Capture error output
    error_stream = StringIO()
    
    try:
        result = build(
            sources=sources,
            options=options,
            stderr=error_stream
        )
        
        # Parse and format errors
        errors = []
        if result.errors:
            for error_line in result.errors:
                # Parse error components
                parts = error_line.split(":", 3)
                if len(parts) >= 4:
                    file_path = parts[0]
                    line_num = parts[1]
                    level = parts[2].strip()
                    message = parts[3].strip()
                    
                    errors.append({
                        'file': file_path,
                        'line': int(line_num) if line_num.isdigit() else 0,
                        'level': level,
                        'message': message
                    })
        
        return {
            'success': len(errors) == 0,
            'errors': errors,
            'result': result
        }
        
    except CompileError as e:
        return {
            'success': False,
            'errors': [{'file': '<build>', 'line': 0, 'level': 'error', 'message': str(e)}],
            'result': None
        }

# Usage
options = Options()
options.strict_mode = True

sources = [BuildSource("myfile.py", None, None)]
analysis = type_check_with_custom_errors(sources, options)

if not analysis['success']:
    print("Type checking failed:")
    for error in analysis['errors']:
        print(f"  {error['file']}:{error['line']}: {error['message']}")
```

### Integration with Custom Plugins

```python
from mypy.build import build, BuildSource
from mypy.options import Options
from mypy.plugin import Plugin

class CustomPlugin(Plugin):
    """Custom plugin for specialized type checking."""
    
    def get_function_hook(self, fullname):
        """Hook for function calls."""
        if fullname == "mypackage.special_function":
            return self.handle_special_function
        return None
    
    def handle_special_function(self, ctx):
        """Custom handling for special function."""
        # Custom type checking logic
        return ctx.default_return_type

def build_with_plugin():
    """Build with custom plugin."""
    options = Options()
    options.strict_mode = True
    
    # Create custom plugin instance
    plugin = CustomPlugin(options, "custom_plugin")
    
    sources = [BuildSource("mycode.py", None, None)]
    
    result = build(
        sources=sources,
        options=options,
        extra_plugins=[plugin]
    )
    
    return result

# Usage
result = build_with_plugin()
```

## Build Manager Integration

### Advanced Build Control

```python
from mypy.build import BuildManager, BuildSource
from mypy.options import Options
from mypy.fscache import FileSystemCache
from mypy.modulefinder import SearchPaths

class CustomBuildManager:
    """Custom build manager for advanced scenarios."""
    
    def __init__(self, data_dir=".mypy_cache"):
        self.options = Options()
        self.fscache = FileSystemCache()
        self.data_dir = data_dir
        
        # Configure search paths
        self.search_paths = SearchPaths(
            python_path=(),
            mypy_path=(),
            package_path=(),
            typeshed_path=()
        )
    
    def incremental_build(self, sources, changed_files=None):
        """Perform incremental build with BuildManager."""
        manager = BuildManager(
            data_dir=self.data_dir,
            search_paths=self.search_paths,
            ignore_prefix="",
            source_set=set(),
            reports=None,
            options=self.options,
            version_id="custom_build",
            plugin=None,
            fscache=self.fscache,
            t0=0
        )
        
        # Build with incremental support
        if changed_files:
            # Invalidate cache for changed files
            for filepath in changed_files:
                manager.flush_cache(filepath)
        
        return manager.build(sources)

# Usage for development tools that need incremental checking
manager = CustomBuildManager()
sources = [BuildSource("myfile.py", None, None)]
result = manager.incremental_build(sources, changed_files=["myfile.py"])
```

## Performance Optimization

### Caching Strategies

```python
from mypy.build import build, BuildSource
from mypy.options import Options
from mypy.fscache import FileSystemCache

# Persistent cache for better performance
cache = FileSystemCache(".mypy_cache")

# Reuse cache across builds
options = Options()
options.incremental = True
options.cache_dir = ".mypy_cache"

sources = [BuildSource("myfile.py", None, None)]

# First build (slower)
result1 = build(sources, options, fscache=cache)

# Subsequent builds (faster)
result2 = build(sources, options, fscache=cache)
```

### Batch Processing

```python
from mypy.build import build, BuildSource
from mypy.options import Options

def batch_type_check(file_groups):
    """Process multiple file groups efficiently."""
    options = Options()
    options.incremental = True
    
    results = []
    
    for group in file_groups:
        sources = [BuildSource(f, None, None) for f in group]
        result = build(sources, options)
        results.append(result)
    
    return results

# Usage
file_groups = [
    ["core/main.py", "core/utils.py"],
    ["tests/test_main.py", "tests/test_utils.py"],
    ["plugins/plugin1.py", "plugins/plugin2.py"]
]

results = batch_type_check(file_groups)
```