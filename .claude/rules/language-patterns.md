---
paths:
  - "codeflash/languages/**/*.py"
---

# Language Support Patterns

- Current language is a module-level singleton in `languages/current.py` — use `set_current_language()` / `current_language()`, never pass language as a parameter through call chains
- Use `get_language_support(identifier)` from `languages/registry.py` — never import language classes directly
- New language support classes must use the `@register_language` decorator
- `languages/__init__.py` uses `__getattr__` for lazy imports to avoid circular dependencies
- Prefer `LanguageSupport` protocol dispatch over `is_python()`/`is_javascript()` guards
- `is_javascript()` returns `True` for both JavaScript and TypeScript (still used in ~15 call sites pending migration)
