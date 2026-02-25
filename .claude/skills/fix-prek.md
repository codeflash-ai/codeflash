# Fix prek failures

When prek (pre-commit) checks fail:

1. Run `uv run prek run` to see failures (local, checks staged files)
2. In CI, the equivalent is `uv run prek run --from-ref origin/main`
3. prek runs ruff format, ruff check, and mypy on changed files
4. Fix issues in order: formatting → lint → type errors
5. Re-run `uv run prek run` to verify all checks pass
