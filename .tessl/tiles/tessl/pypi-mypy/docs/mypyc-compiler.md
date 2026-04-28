# MypyC Compiler

Python-to-C compiler that generates efficient C extensions from Python code with type annotations. MypyC can significantly improve runtime performance for type-annotated Python code.

## Capabilities

### Compiler Entry Points

```python { .api }
def main() -> None:  
    """Main compiler entry point (mypyc.__main__.main)."""

def mypycify(
    paths: list[str],
    *,
    only_compile_paths: Iterable[str] | None = None,
    verbose: bool = False,
    opt_level: str = "3",
    debug_level: str = "1",
    strip_asserts: bool = False,
    multi_file: bool = False,
    separate: bool | list[tuple[list[str], str | None]] = False,
    skip_cgen_input: Any | None = None,
    target_dir: str | None = None,
    include_runtime_files: bool | None = None,
    strict_dunder_typing: bool = False,
    group_name: str | None = None,
) -> list[Extension]:
    """
    Main entry point to building using mypyc. Compile Python modules to C extensions.
    
    Parameters:
    - paths: list[str] - File paths to build (may also contain mypy options)
    - only_compile_paths: Iterable[str] | None - Only compile these specific paths
    - verbose: bool - Enable verbose output (default: False)
    - opt_level: str - Optimization level "0"-"3" (default: "3")
    - debug_level: str - Debug information level "0"-"3" (default: "1")
    - strip_asserts: bool - Strip assert statements (default: False)
    - multi_file: bool - Generate multiple C files (default: False)
    - separate: bool | list - Compile modules separately (default: False)
    - skip_cgen_input: Any | None - Skip code generation input (advanced)
    - target_dir: str | None - Target directory for output files
    - include_runtime_files: bool | None - Include runtime support files
    - strict_dunder_typing: bool - Strict typing for dunder methods (default: False)
    - group_name: str | None - Group name for extensions
    
    Returns:
    - list[Extension]: Setuptools Extension objects for compiled modules
    """
```

#### Usage Examples

```bash
# Compile a module  
mypyc mymodule.py

# Compile with optimizations
mypyc --opt-level 3 mymodule.py

# Multi-file compilation
mypyc --multi-file package/
```

### Integration with Build Systems

```python
# Basic setup.py integration
from mypyc.build import mypycify
from setuptools import setup

ext_modules = mypycify([
    "mypackage/core.py",
    "mypackage/utils.py"
], opt_level="3")

setup(
    name="mypackage",
    ext_modules=ext_modules
)

# Advanced setup.py with full options
ext_modules = mypycify([
    "mypackage/core.py",
    "mypackage/utils.py",
    "mypackage/algorithms.py"
], 
    opt_level="3",
    debug_level="1",
    multi_file=True,
    strip_asserts=True,
    verbose=True,
    target_dir="build",
    strict_dunder_typing=True
)

# Selective compilation - only compile performance-critical modules
ext_modules = mypycify([
    "mypackage/core.py",
    "mypackage/utils.py",
    "mypackage/algorithms.py"  # This won't be compiled
], 
    only_compile_paths=["mypackage/core.py", "mypackage/utils.py"],
    opt_level="3"
)
```

For complete mypyc usage patterns, see [Command Line Tools](./command-line-tools.md) documentation.