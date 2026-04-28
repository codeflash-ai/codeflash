# Code Style

- **Line length**: 120 characters
- **Python**: 3.9+ syntax
- **Package management**: Always use `uv`, never `pip`
- **Tooling**: Ruff for linting/formatting, mypy strict mode, prek for pre-commit checks
- **Comments**: Minimal — only explain "why", not "what"
- **Docstrings**: Do not add docstrings unless the user explicitly asks
- **Types**: Match the type annotation style of surrounding code
- **Naming**: No leading underscores (`_function_name`) — Python has no true private functions
- **Paths**: Always use absolute paths
- **Encoding**: Always pass `encoding="utf-8"` to `open()`, `read_text()`, `write_text()` in new or changed code
- **Verification**: Use `uv run prek` — it handles ruff, ty, mypy in one pass. Don't run them separately
- **Code transforms**: Use `libcst` for code modification/transformation. `ast` is acceptable for read-only analysis
