# Code Style

- **Line length**: 120 characters
- **Python**: 3.9+ syntax
- **Package management**: Always use `uv`, never `pip`
- **Tooling**: Ruff for linting/formatting, mypy strict mode, prek for pre-commit checks
- **Comments**: Minimal - only explain "why", not "what"
- **Docstrings**: Do not add docstrings to new or changed code unless the user explicitly asks for them — not even one-liners. The codebase intentionally keeps functions self-documenting through clear naming and type annotations
- **Types**: Match the type annotation style of surrounding code — the codebase uses annotations, so add them in new code
- **Naming**: NEVER use leading underscores (`_function_name`) - Python has no true private functions, use public names
- **Paths**: Always use absolute paths
- **Encoding**: Always pass `encoding="utf-8"` to `open()`, `read_text()`, `write_text()`, etc. in new or changed code — Windows defaults to `cp1252` which breaks on non-ASCII content. Don't flag pre-existing code that lacks it unless you're already modifying that line.
- **Verification**: Use `uv run prek` to verify code — it handles ruff, ty, mypy in one pass. Don't run `ruff`, `mypy`, or `python -c "import ..."` separately; `prek` is the single verification command
