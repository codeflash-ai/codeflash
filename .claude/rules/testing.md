---
paths:
  - "tests/**"
  - "codeflash/**/*test*.py"
---

# Testing Conventions

- Code context extraction and replacement tests must always assert for full string equality, no substring matching.
- Use pytest's `tmp_path` fixture for temp directories — do not use `tempfile.mkdtemp()`, `tempfile.TemporaryDirectory()`, or `NamedTemporaryFile`. Some existing tests still use `tempfile` but new tests must use `tmp_path`.
- Always call `.resolve()` on Path objects before passing them to functions under test — this ensures absolute paths and resolves symlinks. Example: `source_file = (tmp_path / "example.py").resolve()`
- Use `.as_posix()` when converting resolved paths to strings (normalizes to forward slashes).
- Any new feature or bug fix that can be tested automatically must have test cases.
- If changes affect existing test expectations, update the tests accordingly. Tests must always pass after changes.
- The pytest plugin patches `time`, `random`, `uuid`, and `datetime` for deterministic test execution — never assume real randomness or real time in verification tests.
- `conftest.py` uses an autouse fixture that calls `reset_current_language()` — tests always start with Python as the default language.
