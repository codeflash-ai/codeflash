# Debugging

## Root cause first

When encountering a bug, investigate the root cause. Don't patch symptoms. If you're about to add a try/except, a fallback default, or a defensive check — ask whether the real fix is upstream.

## Isolated testing

Prefer running individual test functions over full suites. Only run the full suite when explicitly asked or before pushing.

- Single function: `uv run pytest tests/test_foo.py::TestBar::test_baz -v`
- Single module: `uv run pytest tests/test_foo.py -v`
- Full suite: only when asked, or before `git push`

When debugging a specific endpoint or integration, test it directly instead of running the entire pipeline end-to-end.

## Subprocess failures

When a subprocess fails, always log stdout and stderr. "Exit code 1" with no output is useless.
