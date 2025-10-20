# CodeFlash AI Agent Instructions

This file provides comprehensive guidance to any coding agent (Warp, GitHub Copilot, Claude, Gemini, etc.) when working with the CodeFlash repository.

## Project Overview

CodeFlash is an AI-powered Python code optimizer that automatically improves code performance while maintaining correctness. It uses LLMs to analyze code, generate optimization ideas, validate correctness through comprehensive testing, benchmark performance improvements, and create merge-ready pull requests.

**Key Capabilities:**
- Optimize entire codebases with `codeflash --all`
- Optimize specific files or functions with targeted commands
- End-to-end workflow optimization with `codeflash optimize script.py`
- Automated GitHub Actions integration for CI/CD pipelines
- Comprehensive benchmarking and performance analysis
- Git worktree isolation for safe optimization

## Core Architecture

### Data Flow Pipeline
Discovery → Context → Optimization → Verification → Benchmarking → PR

1. **Discovery** (`codeflash/discovery/`) - Find optimizable functions via static analysis or execution tracing
2. **Context Extraction** (`codeflash/context/`) - Extract dependencies, imports, and related code
3. **Optimization** (`codeflash/optimization/`) - Generate optimized code via AI service calls
4. **Verification** (`codeflash/verification/`) - Run deterministic tests with custom pytest plugin
5. **Benchmarking** (`codeflash/benchmarking/`) - Performance measurement and comparison
6. **GitHub Integration** (`codeflash/github/`) - Automated PR creation with detailed analysis

### Key Components

**Main Entry Points:**
- `codeflash/main.py` - CLI entry point and main orchestration
- `codeflash/cli_cmds/cli.py` - Command-line argument parsing and validation

**Core Optimization Pipeline:**
- `codeflash/optimization/optimizer.py` - Main optimization orchestrator
- `codeflash/optimization/function_optimizer.py` - Individual function optimization
- `codeflash/tracing/` - Function call tracing and profiling

**Code Analysis & Manipulation:**
- `codeflash/code_utils/` - Code parsing, AST manipulation, static analysis
- `codeflash/context/` - Code context extraction and analysis
- `codeflash/verification/` - Code correctness verification through testing

**External Integrations:**
- `codeflash/api/aiservice.py` - LLM communication with rate limiting and retries
- `codeflash/github/` - GitHub integration for PR creation
- `codeflash/benchmarking/` - Performance benchmarking and measurement

**Supporting Systems:**
- `codeflash/models/models.py` - Pydantic models and type definitions
- `codeflash/telemetry/` - Usage analytics (PostHog) and error reporting (Sentry)
- `codeflash/ui/` - User interface components (Rich console output)
- `codeflash/lsp/` - Language Server Protocol support for IDE integration

### Key Optimization Workflows

**1. Full Codebase Optimization (`--all`)**
- Discovers all optimizable functions in the project
- Runs benchmarks if configured
- Optimizes functions in parallel
- Creates PRs for successful optimizations

**2. Targeted Optimization (`--file`, `--function`)**
- Focuses on specific files or functions
- Performs detailed analysis and context extraction
- Applies targeted optimizations

**3. Workflow Tracing (`optimize`)**
- Traces Python script execution
- Identifies performance bottlenecks
- Generates optimizations for traced functions
- Uses checkpoint system to resume interrupted runs

## Critical Development Patterns

### Package Management with uv (NOT pip)
```bash
# Always use uv, never pip
uv sync                    # Install dependencies
uv sync --group dev        # Install dev dependencies
uv run pytest              # Run commands
uv add package             # Add new packages
uv build                   # Build package
```

### Code Manipulation with LibCST (NOT ast)
Always use `libcst` for code parsing/modification to preserve formatting:
```python
from libcst import parse_module, PartialPythonCodeGen
# Never use ast module for code transformations
```

### Testing with Deterministic Execution
Custom pytest plugin (`codeflash/verification/pytest_plugin.py`) ensures reproducible tests:
- Patches time, random, uuid for deterministic behavior
- Environment variables: `CODEFLASH_TEST_MODULE`, `CODEFLASH_TEST_CLASS`, `CODEFLASH_TEST_FUNCTION`
- Always use `uv run pytest`, never `python -m pytest`

### Git Worktree Isolation
Optimizations run in isolated git worktrees to avoid affecting main repo:
```python
from codeflash.code_utils.git_utils import create_detached_worktree, remove_worktree
# Pattern: create_detached_worktree() → optimize → create_diff_patch_from_worktree()
```

### Error Handling with Either Pattern
Use functional error handling instead of exceptions:
```python
from codeflash.either import is_successful, Either
result = aiservice_client.call_llm(...)
if is_successful(result):
    optimized_code = result.value
else:
    error = result.error
```

## Configuration

All configuration in `pyproject.toml` under `[tool.codeflash]`:
```toml
[tool.codeflash]
module-root = "codeflash"           # Source code location
tests-root = "tests"                # Test directory
benchmarks-root = "tests/benchmarks" # Benchmark tests
test-framework = "pytest"          # Always pytest
formatter-cmds = [                  # Auto-formatting commands
    "uvx ruff check --exit-zero --fix $file",
    "uvx ruff format $file",
]
```

## Development Commands

### Environment Setup
```bash
# Install dependencies (always use uv)
uv sync

# Install development dependencies
uv sync --group dev

# Install pre-commit hooks
uv run pre-commit install
```

### Code Quality & Linting
```bash
# Run linting and formatting with ruff (primary tool)
uv run ruff check --fix .
uv run ruff format .

# Type checking with mypy (strict mode)
uv run mypy .

# Clean Python cache files
uvx pyclean .
```

### Testing
```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run coverage run -m pytest tests/

# Run specific test file
uv run pytest tests/test_code_utils.py

# Run tests with verbose output
uv run pytest -v

# Run benchmarks
uv run pytest tests/benchmarks/

# Run end-to-end tests
uv run pytest tests/scripts/

# Run with specific markers
uv run pytest -m "not ci_skip"
```

### Running CodeFlash
```bash
# Initialize CodeFlash in a project
uv run codeflash init

# Optimize entire codebase
uv run codeflash --all

# Optimize specific file
uv run codeflash --file path/to/file.py

# Optimize specific function
uv run codeflash --file path/to/file.py --function function_name

# Trace and optimize a workflow
uv run codeflash optimize script.py

# Verify setup with test optimization
uv run codeflash --verify-setup

# Run with verbose logging
uv run codeflash --verbose --all

# Run with benchmarking enabled
uv run codeflash --benchmark --file target_file.py

# Use replay tests for debugging
uv run codeflash --replay-test tests/specific_test.py
```

## Development Guidelines

### Code Style
- Uses Ruff for linting and formatting (configured in pyproject.toml)
- Strict mypy type checking enabled
- Pre-commit hooks enforce code quality
- Line length: 120 characters
- Python 3.10+ syntax

### Testing Strategy
- Primary test framework: pytest
- Tests located in `tests/` directory
- End-to-end tests in `tests/scripts/`
- Benchmarks in `tests/benchmarks/`
- Extensive use of `@pytest.mark.parametrize`
- Shared fixtures in conftest.py
- Test isolation via custom pytest plugin
- **Trace File Colocation**: When running `codeflash optimize --trace-only -m pytest`, trace files are automatically placed in the tests_root directory alongside replay tests for better organization

### Key Dependencies
- **Core**: `libcst`, `jedi`, `gitpython`, `pydantic`
- **Testing**: `pytest`, `coverage`, `crosshair-tool`
- **Performance**: `line_profiler`, `timeout-decorator`
- **UI**: `rich`, `inquirer`, `click`
- **AI**: Custom API client for LLM interactions

### Data Models & Types
- `codeflash/models/models.py` - Pydantic models for all data structures
- Extensive use of `@dataclass(frozen=True)` for immutable data
- Core types: `FunctionToOptimize`, `ValidCode`, `BenchmarkKey`

## AI Service Integration

### Rate Limiting & Retries
- Built-in rate limiting and exponential backoff
- Handle `Either` return types for error handling
- AI service endpoint: `codeflash/api/aiservice.py`

### Telemetry & Monitoring
- **Sentry**: Error tracking with `codeflash.telemetry.sentry`
- **PostHog**: Usage analytics with `codeflash.telemetry.posthog_cf`
- **Environment Variables**: `CODEFLASH_EXPERIMENT_ID` for testing modes

## Performance & Benchmarking

### Line Profiler Integration
- Uses `line_profiler` for detailed performance analysis
- Instruments functions with `@profile` decorator
- Generates before/after profiling reports
- Calculates precise speedup measurements

### Benchmark Test Framework
- Custom benchmarking in `tests/benchmarks/`
- Generates replay tests from execution traces
- Validates performance improvements statistically

## Debugging & Development

### Verbose Logging
```bash
uv run codeflash --verbose --file target_file.py
```

### Important Environment Variables
- `CODEFLASH_TEST_MODULE` - Current test module during verification
- `CODEFLASH_TEST_CLASS` - Current test class during verification
- `CODEFLASH_TEST_FUNCTION` - Current test function during verification
- `CODEFLASH_LOOP_INDEX` - Current iteration in pytest loops
- `CODEFLASH_EXPERIMENT_ID` - Enables local AI service for testing

### LSP Integration
Language Server Protocol support in `codeflash/lsp/` enables IDE integration during optimization.

### Common Debugging Patterns
1. Use verbose logging to trace optimization flow
2. Check git worktree operations for isolation issues
3. Verify deterministic test execution with environment variables
4. Use replay tests to debug specific optimization scenarios
5. Monitor AI service calls with rate limiting logs

## Best Practices

### Path Handling
- Always use absolute paths
- Handle encoding explicitly (UTF-8)
- Extensive path validation and cleanup utilities in `codeflash/code_utils/`

### Git Operations
- All optimizations run in isolated worktrees
- Never modify the main repository directly
- Use git utilities in `codeflash/code_utils/git_utils.py`

### Code Transformations
- Always use libcst, never ast module
- Preserve code formatting and comments
- Validate transformations with deterministic tests

### Error Handling
- Use Either pattern for functional error handling
- Log errors to Sentry for monitoring
- Provide clear user feedback via Rich console

### Performance Optimization
- Profile before and after changes
- Use benchmarks to validate improvements
- Generate detailed performance reports