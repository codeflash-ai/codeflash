# Test Execution and Runner

Core test execution functionality providing the Mocha class for configuration and the Runner class for test execution with comprehensive lifecycle management and parallel execution support.

## Capabilities

### Mocha Class

Main test framework class for configuration and execution.

```javascript { .api }
/**
 * Mocha constructor - creates a new test framework instance
 * @param options - Configuration options object
 */
class Mocha {
  constructor(options);
  
  /**
   * Add a test file to be loaded
   * @param filepath - Path to test file
   * @returns {Mocha} this - for chaining
   */
  addFile(filepath);
  
  /**
   * Set the reporter for output formatting
   * @param name - Reporter name or constructor function
   * @param options - Reporter-specific options
   * @returns {Mocha} this - for chaining
   */
  reporter(name, options);
  
  /**
   * Set the test interface/UI
   * @param name - Interface name: 'bdd', 'tdd', 'qunit', 'exports'
   * @returns {Mocha} this - for chaining
   */
  ui(name);
  
  /**
   * Set global timeout for all tests
   * @param ms - Timeout in milliseconds, 0 to disable
   * @returns {Mocha} this - for chaining
   */
  timeout(ms);
  
  /**
   * Set threshold for slow test warnings
   * @param ms - Slow threshold in milliseconds
   * @returns {Mocha} this - for chaining
   */
  slow(ms);
  
  /**
   * Set global retry count for failed tests
   * @param count - Number of retries
   * @returns {Mocha} this - for chaining
   */
  retries(count);
  
  /**
   * Set grep pattern to filter tests
   * @param pattern - String or RegExp pattern
   * @returns {Mocha} this - for chaining
   */
  grep(pattern);
  
  /**
   * Set fixed grep string (non-regex)
   * @param string - Fixed string to match
   * @returns {Mocha} this - for chaining
   */
  fgrep(string);
  
  /**
   * Invert grep pattern matching
   * @param invert - Whether to invert pattern matching
   * @returns {Mocha} this - for chaining
   */
  invert(invert);
  
  /**
   * Bail on first test failure
   * @param bail - Whether to bail on first failure
   * @returns {Mocha} this - for chaining
   */
  bail(bail);
  
  /**
   * Enable/disable global leak detection
   * @param checkLeaks - Whether to check for global leaks
   * @returns {Mocha} this - for chaining
   */
  checkLeaks(checkLeaks);
  
  /**
   * Set global variables to ignore during leak detection
   * @param globals - Array of global variable names
   * @returns {Mocha} this - for chaining
   */
  globals(globals);
  
  /**
   * Run tests asynchronously only (no sync tests)
   * @param asyncOnly - Whether to require async tests
   * @returns {Mocha} this - for chaining
   */
  asyncOnly(asyncOnly);
  
  /**
   * Allow uncaught exceptions to propagate
   * @param allowUncaught - Whether to allow uncaught exceptions
   * @returns {Mocha} this - for chaining
   */
  allowUncaught(allowUncaught);
  
  /**
   * Add delay before running tests
   * @param delay - Whether to delay test execution
   * @returns {Mocha} this - for chaining
   */
  delay(delay);
  
  /**
   * Forbid exclusive tests (.only)
   * @param forbidOnly - Whether to forbid .only tests
   * @returns {Mocha} this - for chaining
   */
  forbidOnly(forbidOnly);
  
  /**
   * Forbid pending tests (.skip)
   * @param forbidPending - Whether to forbid .skip tests
   * @returns {Mocha} this - for chaining
   */
  forbidPending(forbidPending);
  
  /**
   * Show full stack traces
   * @param fullTrace - Whether to show full stack traces
   * @returns {Mocha} this - for chaining
   */
  fullTrace(fullTrace);
  
  /**
   * Enable colored output
   * @param color - Whether to enable colored output
   * @returns {Mocha} this - for chaining
   */
  color(color);
  
  /**
   * Show inline diffs
   * @param inlineDiffs - Whether to show inline diffs
   * @returns {Mocha} this - for chaining
   */
  inlineDiffs(inlineDiffs);
  
  /**
   * Show diff on test failure
   * @param diff - Whether to show diffs
   * @returns {Mocha} this - for chaining
   */
  diff(diff);
  
  /**
   * Perform dry run (don't execute tests)
   * @param dryRun - Whether to perform dry run
   * @returns {Mocha} this - for chaining
   */
  dryRun(dryRun);
  
  /**
   * Enable parallel test execution
   * @param parallel - Whether to run tests in parallel
   * @returns {Mocha} this - for chaining
   */
  parallelMode(parallel);
  
  /**
   * Set root hooks (global setup/teardown)
   * @param hooks - Root hook functions
   * @returns {Mocha} this - for chaining
   */
  rootHooks(hooks);
  
  /**
   * Set global setup function
   * @param fn - Global setup function
   * @returns {Mocha} this - for chaining
   */
  globalSetup(fn);
  
  /**
   * Set global teardown function
   * @param fn - Global teardown function
   * @returns {Mocha} this - for chaining
   */
  globalTeardown(fn);
  
  /**
   * Load test files into memory
   * @returns {Mocha} this - for chaining
   */
  loadFiles();
  
  /**
   * Load test files asynchronously
   * @returns {Promise<Mocha>} Promise resolving to this instance
   */
  loadFilesAsync();
  
  /**
   * Unload test files from memory
   * @returns {Mocha} this - for chaining
   */
  unloadFiles();
  
  /**
   * Run all loaded tests
   * @param callback - Completion callback receiving failure count
   * @returns {Runner} Runner instance
   */
  run(callback);
  
  /**
   * Dispose of this Mocha instance
   */
  dispose();
}
```

**Usage Example:**

```javascript
const Mocha = require('mocha');

const mocha = new Mocha({
  ui: 'bdd',
  reporter: 'spec',
  timeout: 5000,
  slow: 100
});

// Add test files
mocha.addFile('./test/unit/helpers.js');
mocha.addFile('./test/unit/models.js');

// Configure additional options
mocha
  .grep('User')
  .bail(true)
  .checkLeaks(true);

// Run tests
mocha.run(function(failures) {
  console.log(`Tests completed with ${failures} failures`);
  process.exitCode = failures ? 1 : 0;
});
```

### Runner Class

Test execution engine that manages the test lifecycle and emits events.

```javascript { .api }
/**
 * Runner class - manages test execution
 * Extends EventEmitter for test lifecycle events
 */
class Runner extends EventEmitter {
  /**
   * Run all tests
   * @param callback - Completion callback
   * @returns {Runner} this - for chaining
   */
  run(callback);
  
  /**
   * Abort test execution
   * @returns {Runner} this - for chaining
   */  
  abort();
  
  /**
   * Set grep pattern for filtering tests
   * @param pattern - String or RegExp pattern
   * @returns {Runner} this - for chaining
   */
  grep(pattern);
  
  /**
   * Get current test count statistics
   * @returns {Object} Test count statistics
   */
  stats;
  
  /**
   * Check if runner is running
   * @returns {boolean} Whether runner is currently executing
   */
  isRunning();
}
```

### Runner Events

The Runner emits events throughout the test lifecycle that reporters use:

```javascript { .api }
// Test execution lifecycle events
const EVENTS = {
  EVENT_RUN_BEGIN: 'start',           // Test run starts
  EVENT_RUN_END: 'end',               // Test run ends
  EVENT_SUITE_BEGIN: 'suite',         // Suite starts
  EVENT_SUITE_END: 'suite end',       // Suite ends  
  EVENT_TEST_BEGIN: 'test',           // Individual test starts
  EVENT_TEST_END: 'test end',         // Individual test ends
  EVENT_TEST_PASS: 'pass',            // Test passes
  EVENT_TEST_FAIL: 'fail',            // Test fails
  EVENT_TEST_PENDING: 'pending',      // Test is pending/skipped
  EVENT_HOOK_BEGIN: 'hook',           // Hook starts
  EVENT_HOOK_END: 'hook end'          // Hook ends
};
```

**Usage Example:**

```javascript
const runner = mocha.run();

runner.on('start', function() {
  console.log('Test run started');
});

runner.on('pass', function(test) {
  console.log(`✓ ${test.title}`);
});

runner.on('fail', function(test, err) {
  console.log(`✗ ${test.title}: ${err.message}`);
});

runner.on('end', function() {
  console.log('Test run completed');
});
```

### Parallel Execution

Mocha supports parallel test execution for improved performance:

```javascript { .api }
/**
 * Enable parallel execution
 * @param options - Parallel execution options
 */
const mocha = new Mocha({
  parallel: true,
  jobs: 4  // Number of worker processes
});

/**
 * Parallel execution configuration
 */
interface ParallelOptions {
  parallel: boolean;     // Enable parallel execution
  jobs?: number;         // Number of workers (default: CPU count - 1)
  timeout?: number;      // Worker timeout
}
```

### Asynchronous Test Support

Mocha supports multiple patterns for asynchronous tests:

```javascript { .api }
/**
 * Test function signatures for different async patterns
 */

// Promise-based tests
function testFunction(): Promise<any>;

// Callback-based tests  
function testFunction(done: DoneCB): void;

// Async/await tests
async function testFunction(): Promise<any>;

type DoneCB = (error?: any) => void;
```

**Usage Examples:**

```javascript
// Promise-based
it('should handle promises', function() {
  return new Promise((resolve) => {
    setTimeout(resolve, 100);
  });
});

// Callback-based
it('should handle callbacks', function(done) {
  setTimeout(() => {
    done();
  }, 100);
});

// Async/await
it('should handle async/await', async function() {
  await new Promise(resolve => setTimeout(resolve, 100));
});
```

### Context and Test State

Each test receives a context object with utilities:

```javascript { .api }
/**
 * Test context object (this in test functions)
 */
interface Context {
  test?: Test;           // Current test object
  currentTest?: Test;    // Alias for test
  timeout(ms?: number): number | Context;  // Set/get timeout
  slow(ms?: number): number | Context;     // Set/get slow threshold
  skip(): never;         // Skip current test
  retries(count?: number): number | Context; // Set/get retry count
}
```

**Usage Example:**

```javascript
it('should use context methods', function() {
  this.timeout(10000);  // Set timeout for this test
  this.slow(1000);      // Set slow threshold
  
  // Conditionally skip test
  if (process.env.SKIP_SLOW) {
    this.skip();
  }
  
  // Retry on failure
  this.retries(3);
});
```

### Test File Loading

Mocha provides methods for loading and managing test files:

```javascript { .api }
/**
 * Load test files synchronously
 */
mocha.loadFiles();

/**
 * Load test files asynchronously  
 * @returns {Promise<Mocha>} Promise resolving when files are loaded
 */
mocha.loadFilesAsync();

/**
 * Unload test files from require cache
 */
mocha.unloadFiles();

/**
 * Add files to be loaded
 * @param filepath - Path to test file
 */
mocha.addFile(filepath);

/**
 * Get list of files to be loaded
 * @returns {string[]} Array of file paths
 */
mocha.files;
```