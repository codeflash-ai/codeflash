# Copilot Instructions for AI Coding Agents

Welcome to the Codeflash codebase! This guide is tailored for AI coding agents to be immediately productive and effective in this repository.

## Big Picture Architecture
- **Core Purpose:** Codeflash is a Python code optimizer that leverages LLMs to suggest, test, and benchmark code improvements, then creates merge-ready PRs.
- **Main Components:**
  - `codeflash/`: Core logic, including optimization, tracing, benchmarking, and API integrations.
  - `code_to_optimize/`: Example and test code for optimization, including various sorting and utility scripts.
  - `tests/`: Unit and integration tests for core features and utilities.
  - `docs/`: Docusaurus-based documentation site (see `docs/README.md`).
- **Configuration:** Centralized in `pyproject.toml` after running `codeflash init`.


## Developer Workflows
- **Install & Manage Dependencies:**

  - Use [`uv`](https://docs.astral.sh/uv/) for all dependency management and environment setup.
  - To install dependencies: `uv sync` (reads from `pyproject.toml` and `uv.lock`).
  - To add a new package: `uv add <package>` (updates lockfile and installs).
  - To update dependencies, modify `pyproject.toml` and run `uv sync` to apply changes. Use `uv add <package>` to add new dependencies.
  - **Environment Management:** uv automatically manages virtual environments when you install, add, or run dependencies—no need to manually create or activate a venv. The `.venv` directory is created and used automatically.
  - Always commit `uv.lock` to version control for reproducible builds.
  - To run scripts with project dependencies: `uv run <script.py>`
  - To pin/install a Python version: `uv python pin <version>` and `uv python install <version>`
- **Initialize Project:** Run `codeflash init` in the project root to set up config, API keys, and GitHub integration.
- **Optimize Codebase:**
  - All files: `codeflash --all`
  - Single script: `codeflash optimize <script.py>`
- **Testing:**
  - Tests are in `tests/` and `code_to_optimize/tests/`.
  - Use standard Python test runners (pytest is typical, but check for custom scripts in `tests/`).
- **Docs Site:**
  - Install: `npm install` in `docs/`
  - Dev server: `npm run start`
  - Build: `npm run build`

## Project-Specific Conventions
- **Optimization Targets:** Most optimization logic and examples are in `code_to_optimize/`. Use these for benchmarking and agent training.
- **Config Location:** All config is stored in `pyproject.toml` (not a custom YAML/JSON file).
- **PR Automation:** GitHub Actions and the Codeflash GitHub App automate PR creation for optimizations.
- **API Keys:** Required for LLM access; prompt user if missing.
- **Benchmarking:** Results and traces are managed in `codeflash/benchmarking/` and `codeflash/tracing/`.

## Integration Points & Patterns
- **External Services:** Integrates with GitHub (PRs, Actions) and Codeflash LLM APIs.
- **Cross-Component Communication:** Most logic flows through the `codeflash/` package; tracing and benchmarking are modularized.
- **Testing Utilities:** Helper scripts in `code_to_optimize/` and `tests/` are used for dependency injection and mocking.

## Examples & References
- **Key Files:**
  - `codeflash/main.py`: Entry point for CLI and core logic.
  - `codeflash/tracer.py`, `codeflash/optimization/`: Tracing and optimization modules.
  - `code_to_optimize/bubble_sort.py`, `code_to_optimize/book_catalog.py`: Example targets for optimization.
  - `pyproject.toml`: Project configuration and Codeflash settings.
  - `tests/test_codeflash_trace_decorator.py`: Example of testing tracing logic.

## Patterns to Follow
- Prefer modular, testable code in `codeflash/`.
- Use `pyproject.toml` for all config changes.
- When adding new optimization targets, place them in `code_to_optimize/`.
- For documentation, update `docs/` and use Docusaurus conventions.

---

For more details, see [docs.codeflash.ai](https://docs.codeflash.ai) and the main `README.md`.