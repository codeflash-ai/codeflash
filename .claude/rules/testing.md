---
paths:
  - "tests/**"
  - "codeflash/**/*test*.py"
---

# Testing

- Full string equality for context extraction/replacement tests — no substring matching
- Use pytest's `tmp_path` fixture — not `tempfile.mkdtemp()` or `NamedTemporaryFile`
- Always call `.resolve()` on Path objects before passing to functions under test
- Use `.as_posix()` when converting resolved paths to strings
- New features and bug fixes must have test cases
- The pytest plugin patches `time`, `random`, `uuid`, `datetime` for deterministic execution
- `conftest.py` autouse fixture calls `reset_current_language()` — tests start with Python as default
- Prefer running individual tests over full suites: `uv run pytest tests/test_foo.py::TestBar::test_baz -v`
- Only run the full suite when explicitly asked or before pushing
