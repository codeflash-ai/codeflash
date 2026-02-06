---
paths:
  - "codeflash/**/*.py"
---

# Source Code Rules

- Use libcst, not ast - always use `libcst` for code parsing/modification to preserve formatting.
- NEVER use leading underscores for function names (e.g., `_helper`). Python has no true private functions. Always use public names.
- Any new feature or bug fix that can be tested automatically must have test cases.
- If changes affect existing test expectations, update the tests accordingly. Tests must always pass after changes.
