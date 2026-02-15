---
name: add-codeflash-feature
description: >
  Guides implementation of new functionality in the codeflash optimization engine.
  Use when adding a feature, building new functionality, implementing a new
  optimization strategy, adding a language backend, creating an API endpoint,
  extending the verification pipeline, or developing any new codeflash capability.
  Covers module identification, Result type patterns, config, types, tests, and
  quality checks.
---

# Add Codeflash Feature

Use this workflow when implementing new functionality in the codeflash codebase — new optimization strategies, language backends, API endpoints, CLI commands, config options, or pipeline extensions.

## Step 1: Identify Target Modules

Determine which module(s) need modification. See [MODULE_REFERENCE.md](MODULE_REFERENCE.md) for the full mapping of feature areas to modules and key files.

**Checkpoint**: Read the target files and understand existing patterns before writing any code. Look for similar features already implemented as reference.

## Step 2: Follow Result Type Pattern

Use the `Result[L, R]` type from `either.py` for error handling in pipeline operations:

```python
from codeflash.either import Success, Failure, is_successful

def my_operation() -> Result[str, MyResultType]:
    if error_condition:
        return Failure("descriptive error message")
    return Success(result_value)

# Usage:
result = my_operation()
if not is_successful(result):
    logger.error(result.failure())
    return
value = result.unwrap()
```

**Checkpoint**: Verify your function signatures match the `Result` pattern used in surrounding code. Not all functions use `Result` — match the convention of the module you're modifying.

## Step 3: Add Configuration Constants

If the feature needs configurable thresholds or limits:

1. Add constants to `code_utils/config_consts.py`
2. If effort-dependent, add to `EFFORT_VALUES` dict with values for all three levels:
   ```python
   # In config_consts.py:
   class EffortKeys(str, Enum):
       MY_NEW_KEY = "MY_NEW_KEY"

   EFFORT_VALUES: dict[str, dict[EffortLevel, Any]] = {
       # ... existing entries ...
       EffortKeys.MY_NEW_KEY.value: {
           EffortLevel.LOW: 1,
           EffortLevel.MEDIUM: 3,
           EffortLevel.HIGH: 5,
       },
   }
   ```
3. Access via `get_effort_value(EffortKeys.MY_NEW_KEY, effort_level)`

**Checkpoint**: Skip this step if the feature doesn't need configuration. Not every feature requires new constants.

## Step 4: Add Domain Types

If new data structures are needed:

1. Add Pydantic models or frozen dataclasses to `models/models.py` or `models/function_types.py`
2. Use `@dataclass(frozen=True)` for immutable data, `BaseModel` for models that need serialization
3. Keep `function_types.py` dependency-free — no imports from other codeflash modules

Example following existing patterns:
```python
# In models/models.py:
@dataclass(frozen=True)
class MyNewType:
    name: str
    value: int
    source: OptimizedCandidateSource

# For serializable models:
class MyNewModel(BaseModel):
    items: list[MyNewType] = []
```

**Checkpoint**: Skip this step if you can reuse existing types. Check `models/models.py` for types that already fit your needs.

## Step 5: Write Tests

Follow existing test patterns:

1. Create test files in `tests/` mirroring the source structure (e.g., `tests/test_optimization/test_my_feature.py`)
2. Use pytest's `tmp_path` fixture for temp directories — never `NamedTemporaryFile`
3. Always call `.resolve()` on Path objects and `.as_posix()` for string conversion
4. Assert full string equality for code context tests — no substring matching
5. The pytest plugin patches `time`, `random`, `uuid`, `datetime` — never rely on real values in verification tests

```python
def test_my_feature(tmp_path: Path) -> None:
    test_file = tmp_path / "test_module.py"
    test_file.write_text("def foo(): return 1", encoding="utf-8")
    result = my_operation(test_file.resolve())
    assert is_successful(result)
    assert result.unwrap() == expected_value
```

**Checkpoint**: Run the new tests in isolation before proceeding: `uv run pytest tests/path/to/test_file.py -x`

## Step 6: Run Quality Checks

Run all validation before committing:

```bash
# Pre-commit checks (ruff format + lint)
uv run prek run

# Type checking
uv run mypy codeflash/

# Run relevant tests
uv run pytest tests/path/to/relevant/tests -x
```

**If checks fail**:
- `prek run` failures: Fix formatting/lint issues reported by ruff, then re-run
- `mypy` failures: Fix type errors — common issues are missing return types, wrong `Optional` usage, or missing imports in `TYPE_CHECKING` block
- Test failures: Fix the failing test or the implementation, then re-run

## Step 7: Language Support Considerations

If the feature needs to work across languages:

1. Use `get_language_support(identifier)` from `languages/registry.py` — never import language classes directly
2. Current language is a singleton: `set_current_language()` / `current_language()` from `languages/current.py`
3. Use `is_python()` / `is_javascript()` guards for language-specific branches
4. New language support classes must use `@register_language` decorator and be instantiable without arguments

**Checkpoint**: Skip this step if the feature is Python-only. Most features don't need multi-language support.

## Troubleshooting

If you run into issues, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common problems and fixes (circular imports, `UnsupportedLanguageError`, CI path failures, Pydantic validation errors, token limit exceeded).
