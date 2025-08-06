# Bash commands
- `ruff check --fix --exit-non-zero-on-fix --config=pyproject.toml .`: Run the linter
- `ruff format .`: Run the formatter
- `mypy --non-interactive --config-file pyproject.toml @mypy_allowlist.txt`: mypy typechecker

# Running tests
- Run tests with `pytest tests/`

# Workflow
- Be sure to typecheck when you’re done making a series of code changes
- Prefer running single tests, and not the whole test suite, for performance
