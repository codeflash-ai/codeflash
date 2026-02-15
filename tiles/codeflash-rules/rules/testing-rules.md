# Testing Rules

- Code context extraction and replacement tests must assert full string equality — no substring matching
- Use pytest's `tmp_path` fixture for temp directories (it's a `Path` object)
- Write temp files inside `tmp_path`, never use `NamedTemporaryFile` (causes Windows file contention)
- Always call `.resolve()` on Path objects to ensure absolute paths and resolve symlinks
- Use `.as_posix()` when converting resolved paths to strings (normalizes to forward slashes)
- Any new feature or bug fix that can be tested automatically must have test cases
- If changes affect existing test expectations, update the tests accordingly — tests must always pass after changes
- The pytest plugin patches `time`, `random`, `uuid`, `datetime`, `os.urandom`, and `numpy.random` for deterministic test execution — never assume real randomness or real time in verification tests
- `conftest.py` uses an autouse fixture that calls `reset_current_language()` — tests always start with Python as the default language
- Test types are defined by the `TestType` enum: `EXISTING_UNIT_TEST`, `INSPIRED_REGRESSION`, `GENERATED_REGRESSION`, `REPLAY_TEST`, `CONCOLIC_COVERAGE_TEST`, `INIT_STATE_TEST`
- Verification runs tests in a subprocess using a custom pytest plugin (`verification/pytest_plugin.py`) — behavioral tests use blocklisted plugins (`benchmark`, `codspeed`, `xdist`, `sugar`), benchmarking tests additionally block `cov` and `profiling`
