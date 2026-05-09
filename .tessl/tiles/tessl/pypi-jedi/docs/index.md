# Jedi

Jedi is a comprehensive static analysis library for Python that provides intelligent autocompletion, code navigation (goto definitions), documentation lookup, and refactoring capabilities primarily designed for integration into IDEs, editors, and development tools. The library offers a simple and powerful API that can analyze Python code without executing it, supporting features like method completion, parameter suggestions, variable inference, reference finding, and code search across projects and modules.

## Package Information

- **Package Name**: jedi
- **Version**: 0.19.2
- **Language**: Python
- **Installation**: `pip install jedi`

## Core Imports

```python
import jedi
```

For basic script analysis:

```python
from jedi import Script
```

For REPL/interpreter usage:

```python
from jedi import Interpreter
```

For environment and project management:

```python
from jedi import Project, get_default_project
from jedi.api.environment import find_virtualenvs, get_default_environment
```

## Basic Usage

```python
import jedi

# Analyze code for completions
source_code = '''
import json
json.lo'''

script = jedi.Script(code=source_code, path='example.py')

# Get completions at position
completions = script.complete(line=3, column=len('json.lo'))
for completion in completions:
    print(f"{completion.name}: {completion.complete}")

# Get function signatures
signatures = script.get_signatures(line=3, column=len('json.loads('))
for sig in signatures:
    print(f"Function: {sig.name}")
    for param in sig.params:
        print(f"  {param.name}: {param.description}")

# Type inference
definitions = script.infer(line=2, column=len('json'))
for definition in definitions:
    print(f"Type: {definition.type}, Module: {definition.module_name}")
```

## Architecture

Jedi's architecture centers around static analysis and inference:

- **Script**: Main entry point for analyzing source code files with full project context
- **Interpreter**: Specialized for REPL environments with runtime namespace integration
- **InferenceState**: Core inference engine that tracks types, scopes, and relationships
- **Environment**: Python environment abstraction supporting virtualenvs and multiple Python versions
- **Project**: Project-level configuration including sys.path, extensions, and search scope

The inference system works by parsing code with Parso, building syntax trees, and applying type inference rules to determine completions, definitions, and references without code execution. This enables safe analysis of any Python code while maintaining accuracy through sophisticated static analysis techniques.

## Capabilities

### Script Analysis

Core functionality for analyzing Python source code, providing completions, type inference, definition lookup, and navigation. Supports both file-based and string-based code analysis with full project context.

```python { .api }
class Script:
    def __init__(self, code=None, *, path=None, environment=None, project=None): ...
    def complete(self, line=None, column=None, *, fuzzy=False): ...
    def infer(self, line=None, column=None, *, only_stubs=False, prefer_stubs=False): ...
    def goto(self, line=None, column=None, *, follow_imports=False, follow_builtin_imports=False, only_stubs=False, prefer_stubs=False): ...
```

[Script Analysis](./script-analysis.md)

### Interpreter Integration

REPL and interactive environment support with runtime namespace integration. Enables completions and analysis in interactive Python sessions using actual runtime objects and namespaces.

```python { .api }
class Interpreter(Script):
    def __init__(self, code, namespaces, *, project=None, **kwds): ...
```

[Interpreter Integration](./interpreter-integration.md)

### Code Navigation and Search

Advanced code navigation including reference finding, symbol search, context analysis, and help lookup. Provides IDE-level navigation capabilities across entire projects.

```python { .api }
def get_references(self, line=None, column=None, **kwargs): ...
def search(self, string, *, all_scopes=False): ...
def help(self, line=None, column=None): ...
def get_context(self, line=None, column=None): ...
```

[Code Navigation](./code-navigation.md)

### Environment Management

Python environment detection and management supporting virtualenvs, system environments, and custom Python installations. Provides environment-aware analysis and completion.

```python { .api }
def find_virtualenvs(paths=None, *, safe=True, use_environment_vars=True): ...
def find_system_environments(*, env_vars=None): ...
def get_default_environment(): ...
def create_environment(path, *, safe=True, env_vars=None): ...
```

[Environment Management](./environment-management.md)

### Project Configuration

Project-level configuration and management including sys.path customization, extension loading, and multi-file analysis scope. Enables project-aware analysis and search.

```python { .api }
class Project:
    def __init__(self, path, *, environment_path=None, load_unsafe_extensions=False, sys_path=None, added_sys_path=(), smart_sys_path=True): ...
    def search(self, string, *, all_scopes=False): ...
```

[Project Configuration](./project-configuration.md)

### Code Refactoring

Code refactoring operations including rename, extract variable, extract function, and inline operations. Provides safe refactoring with proper scope analysis and conflict detection.

```python { .api }
def rename(self, line=None, column=None, *, new_name): ...
def extract_variable(self, line, column, *, new_name, until_line=None, until_column=None): ...
def extract_function(self, line, column, *, new_name, until_line=None, until_column=None): ...
def inline(self, line=None, column=None): ...
```

[Code Refactoring](./code-refactoring.md)

### Configuration and Debugging

Global configuration settings and debugging utilities for controlling jedi behavior, performance tuning, and troubleshooting analysis issues.

```python { .api }
def set_debug_function(func_cb=debug.print_to_stdout, warnings=True, notices=True, speed=True): ...
def preload_module(*modules): ...
```

[Configuration](./configuration.md)

## Constants

```python { .api }
__version__: str = "0.19.2"  # Library version
```

## Result Types

Core result types returned by jedi analysis operations:

```python { .api }
class BaseName:
    name: str
    type: str  # 'module', 'class', 'instance', 'function', 'param', 'path', 'keyword', 'property', 'statement'
    module_name: str
    module_path: str
    line: int
    column: int
    description: str
    full_name: str
    
    def docstring(self, raw=False, fast=True): ...
    def goto(self, **kwargs): ...
    def infer(self, **kwargs): ...
    def get_line_code(self, before=0, after=0): ...
    def get_signatures(self): ...
    def execute(self): ...
    def get_type_hint(self): ...
    def is_stub(self): ...
    def is_side_effect(self): ...
    def in_builtin_module(self): ...
    def parent(self): ...

class Name(BaseName):
    def defined_names(self): ...
    def is_definition(self): ...

class Completion(BaseName):
    complete: str
    name_with_symbols: str

class Signature(BaseName):
    params: list
    index: int
    bracket_start: tuple

class ParamName(Name):
    kind: str
    def infer_default(self): ...
    def infer_annotation(self, **kwargs): ...
```

## Exception Types

```python { .api }
class InternalError(Exception): ...
class RefactoringError(Exception): ...
class InvalidPythonEnvironment(Exception): ...
class WrongVersion(Exception): ...  # Reserved for future use
```