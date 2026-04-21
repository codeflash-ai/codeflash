# Contributing to Codeflash

Thanks for your interest in contributing. This guide covers both contributing changes back to Codeflash itself and running Codeflash from this repository in editable mode to optimize your own projects.

## Table of contents

- [Quick links](#quick-links)
- [Ways to contribute](#ways-to-contribute)
- [Development environment](#development-environment)
- [Running tests and checks](#running-tests-and-checks)
- [Code style](#code-style)
- [Branches, commits, and pull requests](#branches-commits-and-pull-requests)
- [Using Codeflash in editable mode](#using-codeflash-in-editable-mode)
- [Reporting bugs and requesting features](#reporting-bugs-and-requesting-features)
- [Security issues](#security-issues)

## Quick links

- Issues: https://github.com/codeflash-ai/codeflash/issues
- Discussions and support: [Discord](https://www.codeflash.ai/discord)
- Documentation: https://docs.codeflash.ai
- Security policy: [`SECURITY.md`](SECURITY.md)
- Project conventions for AI agents and humans alike: [`CLAUDE.md`](CLAUDE.md)

## Ways to contribute

- **Bug reports**: open an issue with a minimal reproducer that fails on `main`.
- **Bug fixes**: follow the bug-fix workflow in [`CLAUDE.md`](CLAUDE.md) - read the code, write a failing test, apply the fix, confirm the test now passes.
- **Features**: open an issue first for anything non-trivial so the approach can be agreed before implementation.
- **Documentation**: the full documentation lives at https://docs.codeflash.ai. Fixes to README, docstrings, and this guide can be submitted as PRs here.
- **Language support**: Codeflash supports Python, JavaScript / TypeScript, and Java today. New language support is a significant effort - please start with an issue.

## Development environment

### Prerequisites

- Python 3.9 or newer
- [`uv`](https://github.com/astral-sh/uv) for dependency management (required - do not use `pip` directly)
- `git`
- For JavaScript end-to-end tests: Node.js and `npm`
- For Java end-to-end tests: a JDK (see `.github/workflows/java-e2e.yaml` for the tested version)

### Setup

Fork the repository, clone your fork, and install the dev dependencies with `uv`:

```bash
git clone https://github.com/<your-username>/codeflash.git
cd codeflash
uv sync
```

`uv sync` installs Codeflash plus the `dev` dependency group (ruff, mypy, ipython, type stubs). The `codeflash` CLI is installed into the virtualenv and can be invoked via `uv run codeflash ...`.

### Optional: point at your fork's upstream

```bash
git remote add upstream https://github.com/codeflash-ai/codeflash.git
git fetch upstream
```

## Running tests and checks

Use `uv run prek` as the single verification command. It runs ruff (lint + format), mypy (strict), and related checks in one pass, matching what CI runs.

```bash
# Check every changed file against the pre-commit hooks locally
uv run prek

# Match CI behavior: check everything changed against the PR base branch
BASE=$(gh pr view --json baseRefName -q .baseRefName 2>/dev/null || echo main)
uv run prek run --from-ref origin/$BASE
```

Run the test suite with pytest via `uv`:

```bash
uv run pytest tests/
```

To run a subset:

```bash
uv run pytest tests/code_utils/ -k "test_something"
```

End-to-end tests live under `code_to_optimize/` and are exercised by CI (`.github/workflows/ci.yaml`, `java-e2e.yaml`). They can be run locally by invoking the scripts referenced from those workflows if you have the relevant runtime installed.

## Code style

The full ruleset is in [`.claude/rules/code-style.md`](.claude/rules/code-style.md). Highlights:

- **Line length**: 120 characters.
- **Python**: 3.9+ syntax. Use type annotations consistent with surrounding code.
- **Package management**: `uv` only. Do not add dependencies with `pip install`.
- **Docstrings**: do not add docstrings to new or changed code unless explicitly requested. The codebase intentionally keeps functions self-documenting through clear naming and type annotations.
- **Naming**: no leading underscores on Python names (`_private` style). Python has no true private functions; use public names.
- **File I/O**: always pass `encoding="utf-8"` to `open()`, `Path.read_text()`, `Path.write_text()`, and similar calls in new or changed code. Windows defaults to `cp1252`, which breaks on non-ASCII content.
- **Paths**: prefer absolute paths internally.
- **Verification**: `uv run prek` is the canonical check. Don't run `ruff`, `mypy`, or `python -c "import ..."` separately; `prek` handles them together.

## Branches, commits, and pull requests

- Create a feature branch off an up-to-date `main`. Never commit directly to `main`.
- Use conventional-commit prefixes: `fix:`, `feat:`, `refactor:`, `docs:`, `test:`, `chore:`. Keep commit messages concise (1-2 sentence body max).
- Keep commits atomic - one logical change per commit.
- PR titles also use the conventional format. The PR body should be short and link any related issues.
- If the change corresponds to a Linear ticket, include `CF-#<number>` in the PR body.
- Run `uv run prek` (or `uv run prek run --from-ref origin/main`) before pushing. CI will block merge if hooks fail.

## Using Codeflash in editable mode

If you want to use Codeflash itself to optimize your own Python projects while developing or testing changes to Codeflash, you can install it in editable mode from this repository.

### Install as an editable dependency

From your target project's directory:

```bash
# Using uv (recommended)
uv add --editable /absolute/path/to/your/codeflash/checkout

# Or, if you use pip inside a virtualenv
pip install -e /absolute/path/to/your/codeflash/checkout
```

From the Codeflash repository root you can also run the CLI directly without installing into the target project:

```bash
cd /absolute/path/to/your/codeflash/checkout
uv run codeflash init              # in the target project, cwd matters
uv run codeflash --all             # optimize the entire target codebase
uv run codeflash optimize script.py
```

You will still need a Codeflash API key - `uv run codeflash init` walks through key generation and GitHub app setup. See the [Quick Start in the README](README.md#quick-start) for the full flow.

### When to use editable mode

- You are iterating on a Codeflash change and want to dogfood it against a real codebase.
- You need to reproduce a bug your target project hits, with your local patches applied.
- You are developing a new optimization rule, heuristic, or language integration and want end-to-end coverage beyond `tests/`.

For day-to-day optimization of a project you are not hacking on Codeflash itself, install the released package from PyPI (`pip install codeflash` or `uv add codeflash`) instead.

## Reporting bugs and requesting features

Before filing a new issue, please:

1. Search existing [open and closed issues](https://github.com/codeflash-ai/codeflash/issues?q=is%3Aissue) to avoid duplicates.
2. Include the Codeflash version (`codeflash --version`) and Python / uv versions.
3. Include the smallest reproducer you can. For bugs, a failing test that exercises the behavior is ideal.

## Security issues

Do not report suspected security issues in public GitHub issues. See [`SECURITY.md`](SECURITY.md) for the reporting process.
