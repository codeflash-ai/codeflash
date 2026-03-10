# Code Style

- **Line length**: 120 characters
- **Python**: 3.9+ syntax
- **Package management**: Always use `uv`, never `pip`
- **Tooling**: Ruff for linting/formatting, mypy strict mode, prek for pre-commit checks
- **Comments**: Minimal - only explain "why", not "what"
- **Docstrings**: Do not add unless explicitly requested
- **Naming**: NEVER use leading underscores (`_function_name`) - Python has no true private functions, use public names
- **Paths**: Always use absolute paths
- **Encoding**: Always pass `encoding="utf-8"` to `open()`, `read_text()`, `write_text()`, etc. in new or changed code — Windows defaults to `cp1252` which breaks on non-ASCII content. Don't flag pre-existing code that lacks it unless you're already modifying that line.
- **Pre-commit**: Run `uv run prek` before committing — fix any issues before creating the commit
- **Pre-push**: Before pushing, run `uv run prek run --from-ref origin/<base>` to check all changed files against the PR base — this matches CI behavior and catches issues that per-commit prek misses. To detect the base branch: `gh pr view --json baseRefName -q .baseRefName 2>/dev/null || echo main`
