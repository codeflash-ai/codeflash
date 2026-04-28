# Mypy

A static type checker for Python that uses type hints to analyze code without running it. Mypy enables gradual typing, allowing developers to add type annotations incrementally to existing codebases while providing powerful static analysis capabilities including type inference, generics, union types, and structural subtyping.

## Package Information

- **Package Name**: mypy
- **Language**: Python
- **Installation**: `pip install mypy`
- **Python Version**: >=3.9

## Core Imports

```python
import mypy
```

For programmatic usage:

```python
from mypy import api
```

For advanced build integration:

```python
from mypy.build import build, BuildSource, BuildResult
from mypy.options import Options
```

## Basic Usage

### Command Line Usage

```python
# Basic type checking
import subprocess
result = subprocess.run(['mypy', 'myfile.py'], capture_output=True, text=True)
print(result.stdout)
```

### Programmatic API Usage

```python
from mypy import api

# Type check files programmatically
result = api.run(['myfile.py'])
stdout, stderr, exit_code = result

if exit_code == 0:
    print("Type checking passed!")
else:
    print("Type errors found:")
    print(stderr)
```

### Advanced Build Integration

```python
from mypy.build import build, BuildSource
from mypy.options import Options

# Configure options
options = Options()
options.strict_mode = True
options.python_version = (3, 11)

# Define sources to analyze
sources = [BuildSource("myfile.py", module=None, text=None)]

# Perform analysis
result = build(sources, options)

# Check for errors
if result.errors:
    for error in result.errors:
        print(f"{error.file}:{error.line}: {error.message}")
else:
    print("No type errors found!")
```

## Architecture

Mypy's architecture consists of several key components:

- **Parser**: Converts Python source code into mypy's AST representation
- **Semantic Analyzer**: Builds symbol tables and resolves names and imports
- **Type Checker**: Performs static type analysis using the type system
- **Type System**: Comprehensive type representations including generics, unions, and protocols
- **Plugin System**: Extensible architecture for custom type checking logic
- **Build System**: Manages incremental analysis and dependency tracking

The type checker supports Python's full type system including type variables, generic types, callable types, union types, literal types, TypedDict, Protocol classes, and advanced features like ParamSpec and TypeVarTuple.

## Capabilities

### Command Line Tools

Collection of command-line utilities for type checking, stub generation, stub testing, daemon mode, and Python-to-C compilation.

```python { .api }
# Entry points available as console commands:
# mypy - Main type checker
# stubgen - Generate stub files (.pyi)
# stubtest - Test stub files for accuracy
# dmypy - Daemon mode for faster checking
# mypyc - Compile Python to C extensions
```

[Command Line Tools](./command-line-tools.md)

### Programmatic API

Simple API functions for integrating mypy into Python applications, providing programmatic access to type checking functionality.

```python { .api }
def run(args: list[str]) -> tuple[str, str, int]:
    """
    Run mypy programmatically with command line arguments.
    
    Parameters:
    - args: Command line arguments as list of strings
    
    Returns:
    - tuple: (stdout, stderr, exit_code)
    """

def run_dmypy(args: list[str]) -> tuple[str, str, int]:
    """
    Run dmypy daemon client programmatically.
    
    Parameters:
    - args: Command line arguments as list of strings
    
    Returns:
    - tuple: (stdout, stderr, exit_code)
    
    Note: Not thread-safe, modifies sys.stdout/stderr
    """
```

[Programmatic API](./programmatic-api.md)

### Build System Integration

Advanced APIs for custom build systems and development tools that need fine-grained control over the type checking process.

```python { .api }
def build(sources, options, alt_lib_path=None, flush_errors=None, 
          fscache=None, stdout=None, stderr=None, extra_plugins=None) -> BuildResult:
    """
    Main function to analyze a program programmatically.
    
    Parameters:
    - sources: List of BuildSource objects to analyze
    - options: Options object with configuration settings
    - alt_lib_path: Alternative library path (optional)
    - flush_errors: Error flushing callback (optional)
    - fscache: File system cache (optional)
    - stdout: Output stream (optional)
    - stderr: Error stream (optional)
    - extra_plugins: Additional plugins (optional)
    
    Returns:
    - BuildResult: Analysis results with errors and file information
    """

class BuildSource:
    """Represents a source file or module to be analyzed."""

class BuildResult:
    """Contains the results of type checking analysis."""

class Options:
    """Configuration container for all mypy settings."""
```

[Build System Integration](./build-system.md)

### Type System

Comprehensive type representations used by mypy's static analysis engine, including all Python type system features.

```python { .api }
class Type:
    """Abstract base class for all type representations."""

class Instance:
    """Represents instances of classes and built-in types."""

class CallableType:
    """Represents function and method types with call signatures."""

class UnionType:
    """Represents union types (X | Y or Union[X, Y])."""

class TypeVarType:
    """Represents type variables for generic programming."""

class LiteralType:
    """Represents literal value types (Literal['value'])."""
```

[Type System](./type-system.md)

### Error System

Error codes, error handling classes, and error reporting functionality for comprehensive type checking diagnostics.

```python { .api }
class ErrorCode:
    """Represents specific error types with descriptions and categories."""

class Errors:
    """Collects and formats error messages during type checking."""

class CompileError(Exception):
    """Exception raised when type checking fails."""

# Key error codes
ATTR_DEFINED: ErrorCode
NAME_DEFINED: ErrorCode
CALL_ARG: ErrorCode
TYPE_ARG: ErrorCode
RETURN_VALUE: ErrorCode
```

[Error System](./error-system.md)

### AST and Node System

Abstract syntax tree representation and node types used internally by mypy for code analysis.

```python { .api }
class Node:
    """Abstract base class for all AST nodes."""

class MypyFile:
    """Represents an entire source file."""

class ClassDef:
    """Represents class definitions."""

class FuncDef:
    """Represents function and method definitions."""

class Expression:
    """Base class for expression nodes."""

class Statement:
    """Base class for statement nodes."""
```

[AST and Node System](./ast-nodes.md)

### Plugin System

Extensible plugin architecture for customizing type checking behavior and adding support for specific libraries or frameworks.

```python { .api }
class Plugin:
    """Base class for mypy plugins."""

class CommonPluginApi:
    """Common API available to plugin callbacks."""

class SemanticAnalyzerPluginInterface:
    """API available during semantic analysis phase."""

class CheckerPluginInterface:
    """API available during type checking phase."""
```

[Plugin System](./plugin-system.md)

### Stub Generation and Testing

Tools for generating and validating Python stub files (.pyi) for type checking without runtime dependencies.

```python { .api }
# Available through stubgen command
def main():  # stubgen.main
    """Main entry point for stub generation."""

# Available through stubtest command  
def main():  # stubtest.main
    """Main entry point for stub validation."""
```

[Stub Tools](./stub-tools.md)

### Daemon Mode

High-performance daemon mode for faster incremental type checking in development environments.

```python { .api }
# Available through dmypy command and run_dmypy() API
# Provides faster incremental checking for large codebases
```

[Daemon Mode](./daemon-mode.md)

### MypyC Compiler

Python-to-C compiler that generates efficient C extensions from Python code with type annotations.

```python { .api }
# Available through mypyc command
def main():  # mypyc.__main__.main
    """Main compiler entry point."""

def mypycify(paths, **kwargs):  # mypyc.build.mypycify
    """Compile Python modules to C extensions."""
```

[MypyC Compiler](./mypyc-compiler.md)

## Types

### Core Configuration Types

```python { .api }
class Options:
    """
    Configuration container for all mypy settings.
    
    Attributes:
    - python_version: tuple[int, int] - Target Python version
    - platform: str - Target platform
    - strict_mode: bool - Enable all strict mode flags
    - show_error_codes: bool - Show error codes in output
    - ignore_missing_imports: bool - Ignore missing import errors
    - disallow_untyped_defs: bool - Disallow untyped function definitions
    - disallow_any_generics: bool - Disallow generic types without type parameters
    - warn_redundant_casts: bool - Warn about redundant casts
    - warn_unused_ignores: bool - Warn about unused # type: ignore comments
    """

BuildType = int  # Constants: STANDARD, MODULE, PROGRAM_TEXT
```

### Build System Types

```python { .api }
class BuildSource:
    """
    Represents a source file or module to be analyzed.
    
    Attributes:
    - path: str - File path (or None for modules)
    - module: str - Module name (or None for files)
    - text: str - Source text (or None to read from file)
    """

class BuildResult:
    """
    Contains the results of type checking analysis.
    
    Attributes:
    - errors: list[str] - List of error messages
    - files: dict - Information about analyzed files
    - types: dict - Type information mapping
    - manager: BuildManager | None - Build manager instance
    """

class BuildManager:
    """
    Build manager that coordinates parsing, import processing, 
    semantic analysis and type checking.
    
    Attributes:
    - data_dir: str - Mypy data directory
    - search_paths: SearchPaths - Module search paths
    - modules: dict[str, MypyFile] - Loaded modules
    - options: Options - Build options
    """

class FileSystemCache:
    """
    File system cache for improved performance during mypy operations.
    
    Methods:
    - flush(): Clear all cached data
    - set_package_root(package_root: list[str]): Set package root directories
    """

class SearchPaths:
    """
    Configuration for module search paths used during import resolution.
    
    Attributes:
    - python_path: tuple[str, ...] - User code locations
    - mypy_path: tuple[str, ...] - MYPYPATH directories
    - package_path: tuple[str, ...] - Site-packages directories
    - typeshed_path: tuple[str, ...] - Typeshed locations
    """
```