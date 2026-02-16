# Troubleshooting

| Problem | Likely cause | Fix |
|---------|-------------|-----|
| Circular import at startup | Importing from `models/` in a module loaded early | Move import into `TYPE_CHECKING` block or use lazy import |
| `UnsupportedLanguageError` | Language modules not registered yet | Call `_ensure_languages_registered()` or use `get_language_support()` which does it automatically |
| Tests pass locally but fail in CI | Path differences (absolute vs relative) | Always use `.resolve()` on Path objects |
| `ValidationError` from Pydantic | Invalid code passed to `CodeString` | Check that generated code passes syntax validation for the target language |
| `encoded_tokens_len` exceeds limit | Context too large | Reduce helper functions or split into read-only vs read-writable |
