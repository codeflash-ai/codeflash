# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CodeFlash is an AI-powered Python code optimizer that automatically improves code performance while maintaining correctness. It uses LLMs to generate optimization candidates, verifies correctness through test execution, and benchmarks performance improvements.

## Common Commands

```bash
# Package management (NEVER use pip)
uv sync                          # Install dependencies
uv sync --group dev              # Install dev dependencies
uv add <package>                 # Add a package

# Running tests
uv run pytest tests/             # Run all tests
uv run pytest tests/test_foo.py  # Run specific test file
uv run pytest tests/test_foo.py::test_bar -v  # Run single test

# Type checking and linting
uv run mypy codeflash/           # Type check
uv run ruff check codeflash/     # Lint
uv run ruff format codeflash/    # Format

# Pre-commit (run before committing)
uv run pre-commit run --all-files

# Running the CLI
uv run codeflash --help
uv run codeflash init            # Initialize in a project
uv run codeflash --all           # Optimize entire codebase
```

## Architecture

```
codeflash/
├── main.py                 # CLI entry point
├── cli_cmds/               # Command handling, console output (Rich)
├── discovery/              # Find optimizable functions
├── context/                # Extract code dependencies and imports
├── optimization/           # Generate optimized code via AI
│   ├── optimizer.py        # Main optimization orchestration
│   └── function_optimizer.py  # Per-function optimization logic
├── verification/           # Run deterministic tests (pytest plugin)
├── benchmarking/           # Performance measurement
├── github/                 # PR creation
├── api/                    # AI service communication
├── code_utils/             # Code parsing, git utilities
├── models/                 # Pydantic models and types
├── tracing/                # Function call tracing
├── lsp/                    # IDE integration (Language Server Protocol)
├── telemetry/              # Sentry, PostHog
├── either.py               # Functional Result type for error handling
└── result/                 # Result types and handling
```

### Key Patterns

**Either/Result pattern for errors:**
```python
from codeflash.either import is_successful, Success, Failure
result = some_operation()
if is_successful(result):
    value = result.unwrap()
else:
    error = result.failure()
```

**Git worktree isolation** - Optimizations run in isolated worktrees to avoid affecting the main repo:
```python
from codeflash.code_utils.git_worktree_utils import create_detached_worktree, remove_worktree
```

**Use libcst, not ast** - Always use `libcst` for code parsing/modification to preserve formatting.

## Code Style

- **Line length**: 120 characters
- **Python**: 3.9+ syntax
- **Tooling**: Ruff for linting/formatting, mypy strict mode, pre-commit hooks
- **Comments**: Minimal - only explain "why", not "what"
- **Docstrings**: Do not add unless explicitly requested
- **Naming**: Prefer public functions (no leading underscore) - Python doesn't have true private functions
- **Paths**: Always use absolute paths, handle encoding explicitly (UTF-8)

## Git Commits & Pull Requests

- Use conventional commit format: `fix:`, `feat:`, `refactor:`, `docs:`, `test:`, `chore:`
- Keep commits atomic - one logical change per commit
- Commit message body should be concise (1-2 sentences max)
- PR titles should also use conventional format

# Agent Rules <!-- tessl-managed -->

@.tessl/RULES.md follow the [instructions](.tessl/RULES.md)
