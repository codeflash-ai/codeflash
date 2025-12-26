# CodeFlash AI Agent Instructions

## Project Overview

CodeFlash is an AI-powered Python code optimizer that automatically improves code performance while maintaining correctness.

## Architecture

```
codeflash/
├── main.py                 # CLI entry point
├── cli_cmds/               # Command handling, console output (Rich)
├── discovery/              # Find optimizable functions
├── context/                # Extract code dependencies and imports
├── optimization/           # Generate optimized code via AI
├── verification/           # Run deterministic tests (pytest plugin)
├── benchmarking/           # Performance measurement
├── github/                 # PR creation
├── api/                    # AI service communication
├── code_utils/             # Code parsing, git utilities
├── models/                 # Pydantic models and types
├── tracing/                # Function call tracing
├── lsp/                    # IDE integration
├── telemetry/              # Sentry, PostHog
└── either.py               # Functional error handling
```

## Critical Development Patterns

### Use uv, NEVER pip
**NEVER use `pip install` or `pip` commands.** This project uses `uv` exclusively for package management.
```bash
uv sync                    # Install dependencies (NOT pip install -r requirements.txt)
uv sync --group dev        # Dev dependencies (NOT pip install -e .)
uv run pytest              # Run commands (NOT python -m pytest)
uv add package             # Add packages (NOT pip install package)
```

### Use libcst, not ast
Always use `libcst` for code parsing/modification to preserve formatting.

### Use Either pattern for errors
```python
from codeflash.either import is_successful
result = aiservice_client.call_llm(...)
if is_successful(result):
    optimized_code = result.value
else:
    error = result.error
```

### Git worktree isolation
Optimizations run in isolated worktrees:
```python
from codeflash.code_utils.git_worktree_utils import create_detached_worktree, remove_worktree
```

## Code Style & Conventions

- **Tooling**: Ruff for linting/formatting, mypy strict mode, pre-commit hooks
- **Line length**: 120 characters
- **Python**: 3.9+ syntax
- **Comments**: Minimal - only explain "why", not "what"
- **Docstrings**: Do not add unless explicitly requested
- **Naming**: Prefer public functions (no leading underscore) - Python doesn't have true private functions
- **Paths**: Always use absolute paths, handle encoding explicitly (UTF-8)

## PR Review Guidelines

- **Limit review scope** - only review code actually modified in the PR, not other parts of the codebase
- **Single comment** - consolidate all feedback into one comment per review
- **Edit existing comments** - if you already commented on the PR, edit that comment instead of creating a new one

# Agent Rules <!-- tessl-managed -->

@.tessl/RULES.md follow the [instructions](.tessl/RULES.md)
