# CLI and Configuration

Command-line interface and comprehensive configuration system for running tests from the command line with file watching, parallel execution, and extensive customization options.

## Capabilities

### Command Line Interface

Main CLI executable for running tests from the command line.

```bash { .api }
# Basic usage
mocha [options] [files]

# Common usage patterns
mocha                           # Run all tests in test/ directory
mocha test/**/*.spec.js         # Run specific test files
mocha --grep "User"             # Run tests matching pattern
mocha --reporter json           # Use specific reporter
mocha --timeout 5000            # Set global timeout
mocha --watch                   # Watch files for changes
mocha --parallel               # Run tests in parallel
```

### CLI Options

Comprehensive command-line options for test configuration.

```bash { .api }
# Test Selection and Filtering
--grep <pattern>               # Filter tests by pattern (string or regex)
--fgrep <string>              # Filter tests by fixed string
--invert                      # Invert grep pattern
--recursive                   # Look for tests in subdirectories

# Test Execution  
--timeout <ms>                # Set global timeout (default: 2000ms)
--slow <ms>                   # Set slow test threshold (default: 75ms)
--retries <count>             # Set retry count for failed tests
--bail                        # Bail on first test failure
--parallel                    # Run tests in parallel
--jobs <count>                # Number of parallel jobs (default: CPU count - 1)

# Interfaces and Reporters
--ui <name>                   # Set interface: bdd, tdd, qunit, exports
--reporter <name>             # Set reporter (default: spec)
--reporter-option <key=value> # Pass options to reporter

# Test Behavior
--async-only                  # Force tests to be async
--allow-uncaught             # Allow uncaught exceptions to propagate
--delay                      # Delay test execution until run() is called
--dry-run                    # Report tests without executing them
--exit                       # Force exit after tests complete
--forbid-only                # Fail if .only tests are present
--forbid-pending             # Fail if .skip tests are present
--full-trace                 # Display full stack traces

# Global Variables and Leaks
--check-leaks                # Check for global variable leaks  
--globals <names>            # Specify global variables (comma-separated)

# Output and Formatting
--colors                     # Force color output
--no-colors                  # Disable color output
--diff                       # Show diff on test failure
--inline-diffs              # Show inline diffs
--sort                      # Sort test files alphabetically

# File Operations
--watch                      # Watch files for changes and re-run tests
--watch-files <globs>        # Specify files to watch (comma-separated)
--watch-ignore <globs>       # Specify files to ignore when watching
--file <file>                # Include file before other test files
--require <module>           # Require module before running tests
--loader <loader>            # Use custom loader for test files

# Configuration Files
--config <path>              # Specify config file path
--package <path>             # Specify package.json path  
--opts <path>                # Specify mocha.opts file (deprecated)

# Node.js Specific
--inspect                    # Enable Node.js inspector
--inspect-brk               # Enable inspector and break before start
--node-option <option>       # Pass option to Node.js

# Miscellaneous
--version                    # Show version
--help                       # Show help
--reporter-options <options> # (deprecated, use --reporter-option)
```

### Configuration Files

Mocha supports multiple configuration file formats.

```javascript { .api }
/**
 * Configuration file formats and locations
 */

// .mocharc.json - JSON configuration
{
  "ui": "bdd",
  "reporter": "spec", 
  "timeout": 5000,
  "slow": 100,
  "recursive": true,
  "require": ["test/setup.js"],
  "grep": "User"
}

// .mocharc.yml - YAML configuration  
ui: bdd
reporter: spec
timeout: 5000
slow: 100
recursive: true
require:
  - test/setup.js
grep: User

// .mocharc.js - JavaScript configuration
module.exports = {
  ui: 'bdd',
  reporter: 'spec',
  timeout: 5000,
  slow: 100,  
  recursive: true,
  require: ['test/setup.js'],
  grep: 'User'
};

// package.json - mocha field
{
  "mocha": {
    "ui": "bdd",
    "reporter": "spec",
    "timeout": 5000,
    "recursive": true
  }
}
```

### Configuration Options Interface

Complete configuration options available programmatically and via config files.

```javascript { .api }
/**
 * Complete Mocha configuration options
 */
interface MochaOptions {
  // Test Selection
  grep?: string | RegExp;        // Filter tests by pattern
  fgrep?: string;               // Filter by fixed string
  invert?: boolean;             // Invert grep pattern
  
  // Test Execution
  timeout?: number;             // Global timeout in ms
  slow?: number;               // Slow test threshold in ms
  retries?: number;            // Retry count for failed tests
  bail?: boolean;              // Bail on first failure
  parallel?: boolean;          // Enable parallel execution
  jobs?: number;               // Number of parallel jobs
  
  // Interfaces and Reporting
  ui?: string;                 // Test interface
  reporter?: string | Function; // Reporter name or constructor
  reporterOption?: object;     // Reporter options
  reporterOptions?: object;    // Reporter options (legacy)
  
  // Test Behavior  
  asyncOnly?: boolean;         // Require async tests
  allowUncaught?: boolean;     // Allow uncaught exceptions
  delay?: boolean;             // Delay execution
  dryRun?: boolean;           // Don't execute tests
  exit?: boolean;             // Force exit after completion
  forbidOnly?: boolean;       // Forbid .only tests
  forbidPending?: boolean;    // Forbid .skip tests
  fullTrace?: boolean;        // Show full stack traces
  
  // Global Variables
  checkLeaks?: boolean;        // Check for global leaks
  globals?: string[];         // Global variables to ignore
  
  // Output and Formatting
  color?: boolean;            // Enable colored output
  colors?: boolean;           // Alias for color
  diff?: boolean;             // Show diff on failure
  inlineDiffs?: boolean;      // Show inline diffs
  sort?: boolean;             // Sort test files
  
  // File Operations
  watch?: boolean;            // Watch for file changes
  watchFiles?: string[];      // Files to watch
  watchIgnore?: string[];     // Files to ignore
  file?: string[];           // Files to include first
  require?: string[];        // Modules to require
  loader?: string;           // Custom loader
  recursive?: boolean;       // Search subdirectories
  
  // Configuration
  config?: string;           // Config file path
  package?: string;          // package.json path
  opts?: string;             // mocha.opts file (deprecated)
  
  // Root Hooks and Global Setup
  rootHooks?: MochaRootHookObject; // Root hooks
  globalSetup?: string | string[]; // Global setup functions
  globalTeardown?: string | string[]; // Global teardown functions
  enableGlobalSetup?: boolean;     // Enable global setup
  enableGlobalTeardown?: boolean;  // Enable global teardown
  
  // Advanced Options
  isWorker?: boolean;        // Running in worker process
  serializer?: string;       // Custom serializer for parallel mode
}

/**
 * Root hooks object for global setup/teardown
 */
interface MochaRootHookObject {
  beforeAll?: Function | Function[];  // Global before hooks
  beforeEach?: Function | Function[]; // Global beforeEach hooks  
  afterAll?: Function | Function[];   // Global after hooks
  afterEach?: Function | Function[];  // Global afterEach hooks
}
```

### File Watching

Automatic test re-execution when files change.

```bash { .api }
# Basic file watching
mocha --watch

# Watch specific files
mocha --watch --watch-files "src/**/*.js,test/**/*.js"

# Ignore files when watching
mocha --watch --watch-ignore "node_modules/**,dist/**"

# Watch with grep pattern
mocha --watch --grep "Unit"
```

```javascript { .api }
/**
 * Programmatic file watching
 */
const mocha = new Mocha({
  watch: true,
  watchFiles: ['src/**/*.js', 'test/**/*.js'],
  watchIgnore: ['node_modules/**', 'dist/**']
});
```

### Parallel Execution Configuration

Configure parallel test execution for improved performance.

```bash { .api }
# Enable parallel execution
mocha --parallel

# Set number of workers
mocha --parallel --jobs 4

# Parallel with other options
mocha --parallel --jobs 2 --timeout 10000 --reporter spec
```

```javascript { .api }
/**
 * Parallel execution options
 */
interface ParallelOptions {
  parallel: boolean;           // Enable parallel execution
  jobs?: number;              // Number of worker processes
  timeout?: number;           // Worker timeout
  workerTimeout?: number;     // Worker-specific timeout
}

const mocha = new Mocha({
  parallel: true,
  jobs: 4,
  timeout: 10000
});
```

### Module Loading and Requirements

Load modules and setup files before tests.

```bash { .api }
# Require modules before tests
mocha --require test/setup.js --require should

# Multiple requires
mocha --require babel-register --require test/helpers.js

# Include files before test files
mocha --file test/globals.js --file test/setup.js
```

```javascript { .api }
/**
 * Module loading configuration
 */
interface ModuleLoadingOptions {
  require?: string[];          // Modules to require before tests
  file?: string[];            // Files to include before test files
  loader?: string;            // Custom loader for test files
}

// Example setup file (test/setup.js)
const chai = require('chai');
const sinon = require('sinon');

// Global setup
global.expect = chai.expect;
global.sinon = sinon;

// Configure chai
chai.config.includeStack = true;
chai.config.truncateThreshold = 0;
```

### Environment Variables

Environment variables that affect Mocha behavior.

```bash { .api }
# Common environment variables
MOCHA_COLORS=1               # Enable colors
MOCHA_GREP="pattern"         # Set grep pattern
MOCHA_TIMEOUT=5000          # Set timeout
MOCHA_REPORTER=json         # Set reporter
NODE_ENV=test               # Set Node environment
DEBUG=mocha:*               # Enable debug output

# Usage examples
MOCHA_TIMEOUT=10000 mocha test/
DEBUG=mocha:runner mocha --grep "slow tests"
```

### Configuration Precedence

Order of configuration precedence (highest to lowest):

```javascript { .api }
/**
 * Configuration precedence order
 * 1. Command line arguments (highest)
 * 2. Environment variables  
 * 3. Configuration files (.mocharc.*)
 * 4. package.json "mocha" field
 * 5. Default values (lowest)
 */

// Example: timeout value resolution
// 1. --timeout 3000 (CLI)
// 2. MOCHA_TIMEOUT=4000 (env)
// 3. { "timeout": 5000 } (.mocharc.json)
// 4. { "mocha": { "timeout": 6000 } } (package.json)
// 5. 2000 (default)
// Result: 3000ms (CLI wins)
```

### Advanced CLI Usage

Complex CLI usage patterns and examples.

```bash { .api }
# Complex test execution
mocha test/unit/**/*.spec.js \
  --require test/setup.js \
  --require babel-register \
  --grep "User" \
  --reporter json \
  --timeout 5000 \
  --slow 100 \
  --bail \
  --check-leaks

# Parallel execution with custom options
mocha --parallel \
  --jobs 4 \
  --timeout 10000 \
  --reporter spec \
  --require test/setup.js \
  "test/**/*.spec.js"

# Watch mode with filtering
mocha --watch \
  --watch-files "src/**/*.js" \
  --watch-ignore "dist/**" \
  --grep "integration" \
  --reporter min

# Browser test preparation
mocha --reporter json \
  --timeout 30000 \
  --slow 5000 \
  test/browser/**/*.js > browser-test-results.json

# Debug mode with inspector
mocha --inspect-brk \
  --timeout 0 \
  --grep "specific test" \
  test/debug.spec.js
```

### Legacy mocha.opts (Deprecated)

Legacy configuration file format (now deprecated in favor of .mocharc files).

```bash { .api }
# test/mocha.opts (deprecated)
--require test/setup.js
--require should
--reporter spec
--ui bdd
--timeout 5000
--colors
--recursive
test/**/*.spec.js
```

### Configuration Validation

Mocha validates configuration and provides helpful error messages.

```javascript { .api }
/**
 * Configuration validation examples
 */

// Invalid reporter
mocha --reporter nonexistent
// Error: invalid reporter "nonexistent"

// Invalid timeout
mocha --timeout abc
// Error: timeout must be a number

// Conflicting options
mocha --forbid-only test/with-only.js
// Error: .only tests found but forbidden

// Invalid parallel configuration  
mocha --parallel --bail
// Warning: --bail not supported in parallel mode
```