# Language Support Rules

- Current language is a module-level singleton in `languages/current.py` — use `set_current_language()` / `current_language()`, never pass language as a parameter through call chains
- Use `get_language_support(identifier)` from `languages/registry.py` to get a `LanguageSupport` instance — accepts `Path`, `Language` enum, or string; never import language classes directly
- New language support classes must use the `@register_language` decorator to register with the extension and language registries
- `languages/__init__.py` uses `__getattr__` for lazy imports to avoid circular dependencies — follow this pattern when adding new exports
- `is_javascript()` returns `True` for both JavaScript and TypeScript
- Language modules are lazily imported on first `get_language_support()` call via `_ensure_languages_registered()` — the `@register_language` decorator fires on import and populates `_EXTENSION_REGISTRY` and `_LANGUAGE_REGISTRY`
- `LanguageSupport` instances are cached in `_SUPPORT_CACHE` — use `clear_cache()` only in tests
