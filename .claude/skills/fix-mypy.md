# Fix mypy errors

When modifying code, fix any mypy type errors in the files you changed:

```bash
uv run mypy --non-interactive --config-file pyproject.toml <changed_files>
```

- Fix type annotation issues: missing return types, incorrect types, Optional/None unions, import errors for type hints
- Do NOT add `# type: ignore` comments — always fix the root cause
- Do NOT fix type errors that require logic changes, complex generic type rework, or anything that could change runtime behavior
- Files in `mypy_allowlist.txt` are checked in CI — ensure they remain error-free
