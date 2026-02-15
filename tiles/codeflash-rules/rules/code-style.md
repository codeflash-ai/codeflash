# Code Style

- **Line length**: 120 characters
- **Python**: 3.9+ syntax (use `from __future__ import annotations` for type hints)
- **Package management**: Always use `uv`, never `pip` — run commands via `uv run`
- **Tooling**: Ruff for linting/formatting, mypy strict mode, prek for pre-commit checks (`uv run prek run`)
- **Comments**: Minimal — only explain "why", not "what"
- **Docstrings**: Do not add unless explicitly requested
- **Naming**: NEVER use leading underscores (`_function_name`) — Python has no true private functions, use public names
- **Paths**: Always use absolute `Path` objects, handle encoding explicitly (UTF-8)
- **Source transforms**: Use `libcst` for code modification/transformation to preserve formatting; `ast` is acceptable for read-only analysis and parsing
