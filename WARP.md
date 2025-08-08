# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Codeflash is a general-purpose optimizer for Python that helps improve code performance while maintaining correctness. It uses advanced LLMs to generate optimization ideas, tests them for correctness, and benchmarks them for performance, then creates merge-ready pull requests.

## Development Environment Setup

### Prerequisites
- Python 3.9+ (project uses uv for dependency management)
- Git (for version control and PR creation)
- Codeflash API key (for AI services)

### Initial Setup
```bash
# Install dependencies using uv (preferred over pip)
uv sync

# Initialize codeflash configuration
uv run codeflash init
```

## Core Development Commands

### Code Quality & Linting
```bash
# Format code with ruff (includes check and format)
uv run ruff check --fix codeflash/
uv run ruff format codeflash/

# Type checking with mypy
uv run mypy codeflash/

# Pre-commit hooks (ruff check + format)
uv run pre-commit run --all-files
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_specific_file.py

# Run tests matching pattern
uv run pytest -k "pattern"

```

### Running Codeflash
```bash
# Optimize entire codebase
uv run codeflash --all

# Optimize specific file
uv run codeflash --file path/to/file.py

# Optimize specific function
uv run codeflash --function "module.function"

# Optimize a script end-to-end
uv run codeflash optimize script.py

# Run with benchmarking
uv run codeflash --benchmark

# Verify setup
uv run codeflash --verify-setup
```

## Architecture Overview

### Main Components

**Core Modules:**
- `codeflash/main.py` - CLI entry point and command coordination
- `codeflash/cli_cmds/` - Command-line interface implementations
- `codeflash/optimization/` - Core optimization engine and algorithms
- `codeflash/verification/` - Code correctness verification
- `codeflash/benchmarking/` - Performance measurement and comparison
- `codeflash/discovery/` - Code analysis and function discovery
- `codeflash/tracing/` - Runtime tracing and profiling
- `codeflash/context/` - Code context extraction and analysis
- `codeflash/result/` - Result processing, PR creation, and explanations

**Supporting Systems:**
- `codeflash/api/` - Backend API communication
- `codeflash/github/` - GitHub integration for PR creation
- `codeflash/models/` - Data models and schemas
- `codeflash/telemetry/` - Analytics and error reporting
- `codeflash/code_utils/` - Code parsing, formatting, and manipulation utilities

### Key Workflows

1. **Code Discovery**: Analyzes codebase to identify optimization candidates
2. **Context Extraction**: Extracts relevant code context and dependencies
3. **Optimization Generation**: Uses LLMs to generate optimization candidates
4. **Verification**: Tests optimizations for correctness using existing tests
5. **Benchmarking**: Measures performance improvements
6. **Result Processing**: Creates explanations and pull requests

### Configuration

Configuration is stored in `pyproject.toml` under `[tool.codeflash]`:
- `module-root` - Source code location (default: "codeflash")
- `tests-root` - Test location (default: "tests") 
- `benchmarks-root` - Benchmark location (default: "tests/benchmarks")
- `test-framework` - Testing framework ("pytest" or "unittest")
- `formatter-cmds` - Commands for code formatting

## Project Structure

```
codeflash/
├── api/                 # Backend API communication
├── benchmarking/        # Performance measurement
├── cli_cmds/           # CLI command implementations
├── code_utils/         # Code analysis and manipulation
├── context/            # Code context extraction
├── discovery/          # Function and test discovery  
├── github/             # GitHub API integration
├── lsp/                # Language server protocol support
├── models/             # Data models and schemas
├── optimization/       # Core optimization engine
├── result/             # Result processing and PR creation
├── telemetry/          # Analytics and monitoring
├── tracing/            # Runtime tracing and profiling
├── verification/       # Correctness verification
└── main.py            # CLI entry point

tests/                  # Test suite
├── benchmarks/         # Performance benchmarks
└── scripts/           # Test utilities

docs/                   # Documentation
code_to_optimize/       # Example code for optimization
codeflash-benchmark/    # Benchmark workspace member
```

## Development Notes

### Code Style
- Uses ruff for linting and formatting (configured in pyproject.toml)
- Strict mypy type checking enabled
- Pre-commit hooks enforce code quality

### Testing
- pytest-based test suite with extensive coverage
- Parameterized tests for multiple scenarios
- Benchmarking tests for performance validation
- Test discovery supports both pytest and unittest frameworks

### Workspace Structure
- Uses uv workspace with `codeflash-benchmark` as a member
- Dependencies managed through uv.lock
- Dynamic versioning from git tags using uv-dynamic-versioning

### Build & Distribution
- Uses hatchling as build backend
- BSL-1.1 license
- Excludes development files from distribution packages

### CI/CD Integration
- GitHub Actions workflow for automatic optimization of PR code
- Pre-commit hooks for code quality enforcement
- Automated testing and benchmarking

## Important Patterns

### Error Handling
- Uses `either.py` for functional error handling patterns
- Comprehensive error tracking through Sentry integration
- Graceful degradation when AI services are unavailable

### Instrumentation
- Extensive tracing capabilities for performance analysis
- Line profiler integration for detailed performance metrics
- Custom tracer implementation for code execution analysis

### AI Integration
- Structured prompts and response handling for LLM interactions
- Critic module for evaluating optimization quality
- Context-aware code generation and explanation

### Git Integration 
- GitPython for repository operations
- Automated PR creation with detailed explanations
- Branch management for optimization experiments
