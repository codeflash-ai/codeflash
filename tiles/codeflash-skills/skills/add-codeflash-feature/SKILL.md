---
name: add-codeflash-feature
description: Step-by-step workflow for adding a new feature to the codeflash codebase
---

# Add Codeflash Feature

Use this workflow when implementing a new feature in the codeflash codebase.

## Step 1: Identify Target Modules

Determine which module(s) need modification based on the feature:

| Feature area | Primary module | Key files |
|-------------|----------------|-----------|
| New optimization strategy | `optimization/` | `function_optimizer.py`, `optimizer.py` |
| New test type | `verification/`, `models/` | `test_runner.py`, `pytest_plugin.py`, `test_type.py` |
| New AI service endpoint | `api/` | `aiservice.py` |
| New language support | `languages/` | Create new `languages/<lang>/support.py` |
| Context extraction change | `context/` | `code_context_extractor.py` |
| New CLI command | `cli_cmds/` | `cli.py` |
| New config option | `setup/`, `code_utils/` | `config_consts.py`, `setup/detector.py` |
| Discovery filter | `discovery/` | `functions_to_optimize.py` |
| PR/result changes | `github/`, `result/` | Relevant handlers |

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

## Step 3: Add Configuration Constants

If the feature needs configurable thresholds or limits:

1. Add constants to `code_utils/config_consts.py`
2. If effort-dependent, add to `EFFORT_VALUES` dict with values for `LOW`, `MEDIUM`, `HIGH`
3. Add a corresponding `EffortKeys` enum entry
4. Access via `get_effort_value(EffortKeys.MY_KEY, effort_level)`

## Step 4: Add Domain Types

If new data structures are needed:

1. Add Pydantic models or frozen dataclasses to `models/models.py` or `models/function_types.py`
2. Use `@dataclass(frozen=True)` for immutable data
3. Use `BaseModel` for models that need serialization
4. Keep `function_types.py` dependency-free (no imports from other codeflash modules)

## Step 5: Write Tests

Follow existing test patterns:

1. Create test files in the `tests/` directory mirroring the source structure
2. Use pytest's `tmp_path` fixture for temp directories
3. Always call `.resolve()` on Path objects
4. Assert full string equality for code context tests — no substring matching
5. Remember the pytest plugin patches `time`, `random`, `uuid`, `datetime` — don't rely on real values

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

## Step 7: Language Support Considerations

If the feature needs to work across languages:

1. Check if the feature uses language-specific APIs — use `get_language_support(identifier)` from `languages/registry.py`
2. Current language is a singleton: `set_current_language()` / `current_language()` from `languages/current.py`
3. Use `is_python()` / `is_javascript()` guards for language-specific branches
4. New language support classes must use `@register_language` decorator
