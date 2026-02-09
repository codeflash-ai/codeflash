---
paths:
  - "tests/**"
  - "codeflash/**/*test*.py"
---

# Testing Conventions

- Code context extraction and replacement tests must always assert for full string equality, no substring matching.
- Use pytest's `tmp_path` fixture for temp directories (it's a `Path` object).
- Write temp files inside `tmp_path`, never use `NamedTemporaryFile` (causes Windows file contention).
- Always call `.resolve()` on Path objects to ensure absolute paths and resolve symlinks.
- Use `.as_posix()` when converting resolved paths to strings (normalizes to forward slashes).
- Any new feature or bug fix that can be tested automatically must have test cases.
- If changes affect existing test expectations, update the tests accordingly. Tests must always pass after changes.
