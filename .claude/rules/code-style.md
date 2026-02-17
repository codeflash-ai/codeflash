# Code Style

- **Line length**: 120 characters
- **Python**: 3.9+ syntax
- **Package management**: Always use `uv`, never `pip`
- **Tooling**: Ruff for linting/formatting, mypy strict mode, prek for pre-commit checks
- **Comments**: Minimal - only explain "why", not "what"
- **Docstrings**: Do not add unless explicitly requested
- **Naming**: NEVER use leading underscores (`_function_name`) - Python has no true private functions, use public names
- **Paths**: Always use absolute paths, handle encoding explicitly (UTF-8)
