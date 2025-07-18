# CodeFlash Development Makefile
# This Makefile provides convenient commands for common development tasks

.PHONY: help install install-dev setup test test-verbose test-file lint format typecheck clean build publish dev-setup docs-serve docs-build verify-setup all-checks pre-commit

# Default target
.DEFAULT_GOAL := help

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

help: ## Show this help message
	@echo "$(CYAN)CodeFlash Development Commands$(RESET)"
	@echo "=============================="
	@echo ""
	@echo "$(GREEN)Installation & Setup:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(install|setup|dev-setup)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Development:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(test|lint|format|typecheck|verify-setup)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Build & Release:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(build|publish|clean)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Documentation:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(docs-)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Quality Assurance:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(all-checks|pre-commit)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-15s$(RESET) %s\n", $$1, $$2}'

install: ## Install CodeFlash for end users
	@echo "$(GREEN)Installing CodeFlash...$(RESET)"
	uv pip install .

install-dev: ## Install CodeFlash with development dependencies
	@echo "$(GREEN)Installing CodeFlash with development dependencies...$(RESET)"
	uv sync --all-extras

setup: install-dev ## Complete development setup (install + configure)
	@echo "$(GREEN)Setting up CodeFlash development environment...$(RESET)"
	@echo "$(YELLOW)Running initial setup...$(RESET)"
	uv run python -c "import codeflash; print('✅ CodeFlash installed successfully')"
	@echo "$(GREEN)✅ Development setup complete!$(RESET)"

dev-setup: setup ## Alias for setup (complete development setup)

test: ## Run all tests
	@echo "$(GREEN)Running tests...$(RESET)"
	uv run pytest tests/ -v

test-verbose: ## Run tests with verbose output
	@echo "$(GREEN)Running tests with verbose output...$(RESET)"
	uv run pytest tests/ -v -s

test-file: ## Run tests for a specific file (usage: make test-file FILE=path/to/test.py)
	@echo "$(GREEN)Running tests for $(FILE)...$(RESET)"
	uv run pytest $(FILE) -v

test-unit: ## Run unit tests only (exclude benchmarks)
	@echo "$(GREEN)Running unit tests only...$(RESET)"
	uv run pytest tests/ -v -k "not benchmark"

test-benchmark: ## Run benchmark tests only
	@echo "$(GREEN)Running benchmark tests...$(RESET)"
	uv run pytest tests/benchmarks/ -v

lint: ## Run linting checks
	@echo "$(GREEN)Running linting checks...$(RESET)"
	uv run ruff check codeflash/

lint-fix: ## Run linting and fix issues automatically
	@echo "$(GREEN)Running linting with auto-fix...$(RESET)"
	uv run ruff check --fix codeflash/

format: ## Format code
	@echo "$(GREEN)Formatting code...$(RESET)"
	uv run ruff format codeflash/

format-check: ## Check if code is formatted correctly
	@echo "$(GREEN)Checking code formatting...$(RESET)"
	uv run ruff format --check codeflash/

typecheck: ## Run type checking
	@echo "$(GREEN)Running type checking...$(RESET)"
	uv run mypy codeflash/

verify-setup: ## Verify CodeFlash setup by running bubble sort optimization
	@echo "$(GREEN)Verifying CodeFlash setup...$(RESET)"
	cd code_to_optimize && uv run codeflash --verify-setup

clean: ## Clean build artifacts and cache
	@echo "$(GREEN)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +

build: clean ## Build the package
	@echo "$(GREEN)Building CodeFlash package...$(RESET)"
	uv build

publish: build ## Publish to PyPI (requires authentication)
	@echo "$(YELLOW)Publishing to PyPI...$(RESET)"
	@echo "$(RED)⚠️  Make sure you have proper authentication configured!$(RESET)"
	uv publish

publish-test: build ## Publish to TestPyPI
	@echo "$(YELLOW)Publishing to TestPyPI...$(RESET)"
	uv publish --index-url https://test.pypi.org/simple/

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Starting documentation server...$(RESET)"
	cd docs && npm start

docs-build: ## Build documentation
	@echo "$(GREEN)Building documentation...$(RESET)"
	cd docs && npm run build

docs-install: ## Install documentation dependencies
	@echo "$(GREEN)Installing documentation dependencies...$(RESET)"
	cd docs && npm install

# Quality assurance targets
all-checks: lint typecheck test ## Run all quality checks (linting, type checking, tests)
	@echo "$(GREEN)✅ All quality checks passed!$(RESET)"

pre-commit: format-check lint typecheck test-unit ## Run checks suitable for pre-commit hook
	@echo "$(GREEN)✅ Pre-commit checks passed!$(RESET)"

# Advanced development targets
init-example: ## Initialize CodeFlash in an example project
	@echo "$(GREEN)Initializing CodeFlash in example project...$(RESET)"
	cd code_to_optimize && uv run codeflash init

optimize-example: ## Run optimization on example bubble sort
	@echo "$(GREEN)Running optimization on example...$(RESET)"
	cd code_to_optimize && uv run codeflash --file bubble_sort.py --function sorter

trace-example: ## Trace example workload
	@echo "$(GREEN)Tracing example workload...$(RESET)"
	cd code_to_optimize && uv run codeflash optimize workload.py

# File handle leak test
test-file-handles: ## Test file handle leak fixes
	@echo "$(GREEN)Testing file handle leak fixes...$(RESET)"
	uv run python test_file_handle_fixes.py

# Development utilities
shell: ## Open development shell with CodeFlash environment
	@echo "$(GREEN)Opening development shell...$(RESET)"
	uv shell

update-deps: ## Update dependencies
	@echo "$(GREEN)Updating dependencies...$(RESET)"
	uv sync --upgrade

check-deps: ## Check for dependency issues
	@echo "$(GREEN)Checking dependencies...$(RESET)"
	uv tree

# Debug targets
debug-install: ## Debug installation issues
	@echo "$(GREEN)Debugging installation...$(RESET)"
	uv run python -c "import sys; print('Python:', sys.version)"
	uv run python -c "import codeflash; print('CodeFlash version:', codeflash.__version__)"
	uv run python -c "from codeflash.main import main; print('✅ Main function importable')"

# Quick development workflow
quick-check: format lint typecheck test-unit ## Quick development check (format, lint, typecheck, unit tests)
	@echo "$(GREEN)✅ Quick development check passed!$(RESET)"

# Installation verification
verify-install: ## Verify installation works correctly
	@echo "$(GREEN)Verifying CodeFlash installation...$(RESET)"
	uv run codeflash --version
	uv run python -c "import codeflash; print('✅ CodeFlash can be imported')"
	@echo "$(GREEN)✅ Installation verification complete!$(RESET)"

# Performance testing
benchmark-discovery: ## Benchmark test discovery performance
	@echo "$(GREEN)Benchmarking test discovery...$(RESET)"
	uv run pytest tests/benchmarks/test_benchmark_discover_unit_tests.py -v

# Help for common use cases
help-dev: ## Show help for common development workflows
	@echo "$(CYAN)Common Development Workflows$(RESET)"
	@echo "============================"
	@echo ""
	@echo "$(GREEN)First time setup:$(RESET)"
	@echo "  make setup"
	@echo ""
	@echo "$(GREEN)Before committing:$(RESET)"
	@echo "  make pre-commit"
	@echo ""
	@echo "$(GREEN)Quick development cycle:$(RESET)"
	@echo "  make quick-check"
	@echo ""
	@echo "$(GREEN)Full quality assurance:$(RESET)"
	@echo "  make all-checks"
	@echo ""
	@echo "$(GREEN)Test specific functionality:$(RESET)"
	@echo "  make test-file FILE=tests/test_something.py"
	@echo "  make verify-setup"
	@echo "  make test-file-handles"
	@echo ""
	@echo "$(GREEN)Build and publish:$(RESET)"
	@echo "  make build"
	@echo "  make publish-test  # Test on TestPyPI first"
	@echo "  make publish       # Publish to PyPI"