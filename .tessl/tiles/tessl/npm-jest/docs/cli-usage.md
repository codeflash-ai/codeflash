# CLI Usage

Jest provides a comprehensive command-line interface with over 70 options for controlling test execution, coverage collection, watch mode, output formatting, and project configuration.

## Basic CLI Usage

Jest can be run from the command line with various options:

```bash
# Run all tests
jest

# Run tests matching a pattern
jest MyComponent

# Run tests in a specific directory
jest src/components

# Run tests with coverage
jest --coverage

# Run tests in watch mode
jest --watch
```

## CLI Options Reference

### Test Execution Options

Control how tests are discovered, executed, and filtered.

```bash
# Stop after N test failures
jest --bail
jest --bail=3

# Find tests related to specified files
jest --findRelatedTests src/utils.js src/components/Button.js

# List all tests Jest will run without executing them
jest --listTests

# Only run tests related to changed files (requires git)
jest --onlyChanged
jest -o

# Run only tests that failed in previous execution
jest --onlyFailures
jest -f

# Don't fail when no tests are found
jest --passWithNoTests

# Run tests serially in current process instead of parallel workers
jest --runInBand
jest -i

# Use exact file paths instead of patterns
jest --runTestsByPath path/to/test1.js path/to/test2.js

# Run tests matching name pattern (regex)
jest --testNamePattern="should add"
jest -t "user login"

# Regexp patterns for test file paths
jest --testPathPatterns="__tests__.*\.js$"

# Override testRegex configuration
jest --testRegex=".*\.(test|spec)\.(js|ts)$"

# Set default test timeout in milliseconds
jest --testTimeout=10000
```

### Watch Mode Options

Configure file watching and automatic test re-execution.

```bash
# Watch files and rerun related tests on changes
jest --watch

# Watch files and rerun all tests on changes
jest --watchAll

# Ignore patterns for watch mode
jest --watchPathIgnorePatterns="node_modules" --watchPathIgnorePatterns="build"
```

## Testing Options

### Test Selection and Filtering

```bash
# Run only tests that failed in the previous run
jest --onlyFailures

# Run only tests related to changed files (requires git)
jest --onlyChanged

# Run tests related to specific files
jest --findRelatedTests src/utils.js src/components/Button.js

# Run tests matching a name pattern
jest --testNamePattern="should render correctly"

# List all tests Jest would run without executing them
jest --listTests

# Run tests by exact file paths instead of patterns
jest --runTestsByPath
```

### Test Execution Control

```bash
# Stop after N test failures
jest --bail
jest --bail=3

# Run tests serially in the current process
jest --runInBand

# Set maximum number of worker processes
jest --maxWorkers=4
jest --maxWorkers=50%

# Set default timeout for tests
jest --testTimeout=10000

# Don't fail when no tests are found
jest --passWithNoTests
```

## Coverage Options

### Basic Coverage

```bash
# Collect and report test coverage
jest --coverage
jest --collectCoverage  # equivalent

# Specify coverage output directory
jest --coverageDirectory=coverage-report

# Choose coverage provider
jest --coverageProvider=babel
jest --coverageProvider=v8
```

### Coverage Configuration

```bash
# Collect coverage from specific files
jest --collectCoverageFrom="src/**/*.js" --collectCoverageFrom="!src/**/*.test.js"

# Ignore patterns for coverage
jest --coveragePathIgnorePatterns=node_modules --coveragePathIgnorePatterns=build

# Specify coverage reporters
jest --coverageReporters=text --coverageReporters=html --coverageReporters=lcov

# Set coverage thresholds (will fail if not met)
jest --coverageThreshold='{"global":{"branches":80,"functions":80,"lines":80,"statements":80}}'
```

## Output and Reporting

### Output Formats

```bash
# Output results in JSON format
jest --json

# Write JSON output to a file
jest --json --outputFile=test-results.json

# Verbose output showing individual test results
jest --verbose

# Silent mode - prevent tests from printing to console
jest --silent

# Disable stack traces in test output
jest --noStackTrace

# Force colored output
jest --colors
```

### Advanced Output Control

```bash
# Log heap usage after each test
jest --logHeapUsage

# Detect and report memory leaks (experimental)
jest --detectLeaks

# Detect handles that prevent Jest from exiting
jest --detectOpenHandles

# Force Jest to exit after tests complete
jest --forceExit
```

## Configuration Options

### Configuration Sources

```bash
# Use specific config file
jest --config=jest.config.js
jest --config=package.json

# Provide configuration as JSON string
jest --config='{"testEnvironment":"node"}'

# Set root directory
jest --rootDir=/path/to/project

# Specify multiple root directories
jest --roots=src --roots=lib

# Run tests for multiple projects
jest --projects=project1 --projects=project2
```

### Environment and Setup

```bash
# Specify test environment
jest --testEnvironment=node
jest --testEnvironment=jsdom

# Clear all mocks before each test
jest --clearMocks

# Reset module registry before each test
jest --resetModules

# Restore mocks after each test
jest --restoreMocks
```

## Cache Management

```bash
# Use the transform cache (default: true)
jest --cache

# Disable the transform cache
jest --no-cache

# Specify cache directory
jest --cacheDirectory=/tmp/jest-cache

# Clear cache and exit (useful for CI)
jest --clearCache
```

## Advanced Options

### Development and Debugging

```bash
# Enable debug mode
jest --debug

# Throw error on deprecated API usage
jest --errorOnDeprecated

# Update snapshots
jest --updateSnapshot

# Randomize test order within files
jest --randomize

# Set seed for test randomization
jest --seed=12345
```

### Performance and Concurrency

```bash
# Set maximum concurrent tests when using test.concurrent
jest --maxConcurrency=5

# Use specific number of workers
jest --maxWorkers=2

# Run tests in band (single process) for debugging
jest -i  # short for --runInBand
```

## Watch Mode Patterns

When running in watch mode (`--watch` or `--watchAll`), Jest provides an interactive interface:

```bash
# Watch mode commands (available during watch mode):
# Press 'a' to run all tests
# Press 'f' to run only failed tests  
# Press 'o' to only run tests related to changed files
# Press 'p' to filter by a filename regex pattern
# Press 't' to filter by a test name regex pattern
# Press 'q' to quit watch mode
# Press 'Enter' to trigger a test run
```

### Watch Mode Configuration

```bash
# Ignore patterns in watch mode
jest --watch --watchPathIgnorePatterns=node_modules --watchPathIgnorePatterns=build

# Watch all files (not just tracked by git)
jest --watchAll

# Combine with other options
jest --watch --coverage --verbose
```

## CLI Argument Building Programmatically

```typescript { .api }
import { buildArgv } from 'jest';

function buildArgv(maybeArgv?: Array<string>): Promise<Config.Argv>
```

You can build CLI arguments programmatically using the `buildArgv` function:

```typescript
import { buildArgv, runCLI } from 'jest';

// Build arguments for CI environment
const ciArgv = await buildArgv([
  '--ci',
  '--coverage',
  '--json',
  '--runInBand',
  '--passWithNoTests',
  '--outputFile=test-results.json'
]);

// Build arguments for development
const devArgv = await buildArgv([
  '--watch',
  '--verbose',
  '--testPathPatterns=src/'
]);

// Build arguments with coverage thresholds
const strictArgv = await buildArgv([
  '--coverage',
  '--coverageThreshold={"global":{"branches":90,"functions":90,"lines":90,"statements":90}}'
]);
```

## Common CLI Workflows

### Continuous Integration

```bash
# Typical CI command
jest --ci --coverage --json --runInBand --passWithNoTests --outputFile=results.json

# With coverage enforcement
jest --ci --coverage --coverageThreshold='{"global":{"branches":80,"functions":80,"lines":80,"statements":80}}'
```

### Development Workflow

```bash
# Start development with watch mode
jest --watch --verbose

# Run tests for specific feature
jest --watch --testPathPatterns=src/features/auth

# Debug failing tests
jest --runInBand --verbose --testNamePattern="failing test name"
```

### Pre-commit Checks

```bash
# Run only tests related to staged files
jest --onlyChanged --passWithNoTests

# Full validation with coverage
jest --coverage --bail=1
```

### Performance Analysis

```bash
# Analyze test performance
jest --verbose --logHeapUsage --detectLeaks

# Profile with single worker for consistency  
jest --runInBand --logHeapUsage
```

The Jest CLI provides comprehensive control over test execution, making it suitable for development, continuous integration, and automated testing workflows.