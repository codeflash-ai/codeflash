# Reporters and Output

Comprehensive reporting system with built-in reporters for various output formats and support for custom reporters. Reporters listen to Runner events and format test results.

## Capabilities

### Base Reporter Class

Foundation class for all reporters providing shared functionality.

```javascript { .api }
/**
 * Base reporter class that all reporters extend
 * @param runner - Runner instance that emits test events
 * @param options - Reporter-specific options
 */
class Base {
  constructor(runner, options);
  
  /**
   * Called when test run completes
   * @param failures - Number of failed tests
   * @param callback - Completion callback
   */
  done(failures, callback);
  
  /**
   * Output test run summary/epilogue
   */
  epilogue();
  
  /**
   * Get test statistics
   * @returns {Object} Statistics object
   */
  stats;
}

/**
 * Statistics object structure
 */
interface Stats {
  suites: number;      // Number of suites
  tests: number;       // Total number of tests
  passes: number;      // Number of passing tests
  pending: number;     // Number of pending tests
  failures: number;    // Number of failing tests
  duration: number;    // Total execution time in ms
  start: Date;         // Test run start time
  end: Date;          // Test run end time
}
```

### Built-in Reporters

#### Spec Reporter (Default)

Hierarchical output showing test structure and results.

```javascript { .api }
/**
 * Spec reporter - hierarchical test output
 */
class Spec extends Base {
  constructor(runner, options);
}
```

**Output Example:**
```
Calculator
  #add()
    ✓ should add positive numbers
    ✓ should handle negative numbers
    - should handle decimals (pending)
  #multiply()
    ✓ should multiply numbers
    1) should handle zero

1) Calculator #multiply() should handle zero:
   Error: Expected 0 but got NaN
```

#### Dot Reporter

Minimal dot-based progress output.

```javascript { .api }
/**
 * Dot reporter - minimal dot progress
 */
class Dot extends Base {
  constructor(runner, options);
}
```

**Output Example:**
```
..·..

  4 passing (12ms)
  1 pending
```

#### TAP Reporter

Test Anything Protocol output.

```javascript { .api }
/**
 * TAP reporter - Test Anything Protocol format
 */
class TAP extends Base {
  constructor(runner, options);
}
```

**Output Example:**
```
1..5
ok 1 Calculator #add() should add positive numbers
ok 2 Calculator #add() should handle negative numbers
ok 3 Calculator #multiply() should multiply numbers # SKIP
not ok 4 Calculator #multiply() should handle zero
  ---
  message: Expected 0 but got NaN
  severity: fail
  ...
```

#### JSON Reporter

Machine-readable JSON output.

```javascript { .api }
/**
 * JSON reporter - structured JSON output
 */
class JSON extends Base {
  constructor(runner, options);
}
```

#### HTML Reporter (Browser)

Browser-specific HTML output with interactive features.

```javascript { .api }
/**
 * HTML reporter - browser HTML output with DOM integration
 * Only available in browser environments
 */
class HTML extends Base {
  constructor(runner, options);
}
```

#### List Reporter

Simple list format showing all tests.

```javascript { .api }
/**
 * List reporter - simple list of all tests
 */
class List extends Base {
  constructor(runner, options);
}
```

#### Min Reporter

Minimal output showing only summary.

```javascript { .api }
/**
 * Min reporter - minimal summary output
 */
class Min extends Base {
  constructor(runner, options);
}
```

#### Nyan Reporter

Colorful Nyan Cat progress reporter.

```javascript { .api }
/**
 * Nyan reporter - colorful cat progress animation
 */
class Nyan extends Base {
  constructor(runner, options);
}
```

#### XUnit Reporter

XML output compatible with JUnit/xUnit format.

```javascript { .api }
/**
 * XUnit reporter - XML output for CI systems
 */
class XUnit extends Base {
  constructor(runner, options);
}
```

#### Progress Reporter

Progress bar with test count information.

```javascript { .api }
/**
 * Progress reporter - progress bar with counters
 */
class Progress extends Base {
  constructor(runner, options);
}
```

#### Landing Reporter

Landing strip style progress indicator.

```javascript { .api }
/**
 * Landing reporter - landing strip progress
 */
class Landing extends Base {
  constructor(runner, options);
}
```

#### JSON Stream Reporter

Streaming JSON output for real-time processing.

```javascript { .api }
/**
 * JSONStream reporter - streaming JSON events
 */
class JSONStream extends Base {
  constructor(runner, options);
}
```

### Reporter Selection and Configuration

```javascript { .api }
/**
 * Set reporter for a Mocha instance
 * @param name - Reporter name or constructor function
 * @param options - Reporter-specific options
 */
mocha.reporter(name, options);

/**
 * Available built-in reporters
 */
const reporters = {
  Base: Base,
  base: Base,
  Dot: Dot,
  dot: Dot,
  Doc: Doc,
  doc: Doc,
  TAP: TAP,
  tap: TAP,
  JSON: JSON,
  json: JSON,
  HTML: HTML,
  html: HTML,
  List: List,
  list: List,
  Min: Min,
  min: Min,
  Spec: Spec,
  spec: Spec,
  Nyan: Nyan,
  nyan: Nyan,
  XUnit: XUnit,
  xunit: XUnit,
  Markdown: Markdown,
  markdown: Markdown,
  Progress: Progress,
  progress: Progress,
  Landing: Landing,
  landing: Landing,
  JSONStream: JSONStream,
  'json-stream': JSONStream
};
```

**Usage Examples:**

```javascript
// Using built-in reporter by name
const mocha = new Mocha({
  reporter: 'spec'
});

// With reporter options
mocha.reporter('xunit', {
  output: './test-results.xml'
});

// Using reporter constructor
const CustomReporter = require('./custom-reporter');
mocha.reporter(CustomReporter);

// Programmatically
mocha.reporter('json').reporter('tap'); // Last one wins
```

### Custom Reporters

Create custom reporters by extending the Base class:

```javascript { .api }
/**
 * Custom reporter implementation
 */
class CustomReporter extends Base {
  constructor(runner, options) {
    super(runner, options);
    
    // Listen to runner events
    runner.on('start', () => {
      console.log('Tests starting...');
    });
    
    runner.on('pass', (test) => {
      console.log(`✓ ${test.fullTitle()}`);
    });
    
    runner.on('fail', (test, err) => {
      console.log(`✗ ${test.fullTitle()}: ${err.message}`);
    });
    
    runner.on('end', () => {
      this.epilogue();
    });
  }
}
```

### Reporter Events and Data

Reporters receive these events with associated data:

```javascript { .api }
/**
 * Runner events available to reporters
 */
const events = [
  'start',      // Test run begins
  'end',        // Test run ends  
  'suite',      // Suite begins
  'suite end',  // Suite ends
  'test',       // Test begins
  'test end',   // Test ends
  'pass',       // Test passes
  'fail',       // Test fails
  'pending',    // Test is pending
  'hook',       // Hook begins
  'hook end'    // Hook ends
];

/**
 * Test object structure passed to reporter events
 */
interface Test {
  title: string;           // Test title
  fullTitle(): string;     // Full hierarchical title
  duration: number;        // Test execution time
  state: 'passed' | 'failed' | 'pending';
  err?: Error;            // Error if test failed
  parent: Suite;          // Parent suite
  pending: boolean;       // Whether test is pending
  timeout(): number;      // Test timeout value
  slow(): number;         // Test slow threshold
}

/**
 * Suite object structure
 */
interface Suite {
  title: string;           // Suite title
  fullTitle(): string;     // Full hierarchical title
  parent?: Suite;         // Parent suite
  tests: Test[];          // Child tests
  suites: Suite[];        // Child suites
  pending: boolean;       // Whether suite is pending
  timeout(): number;      // Suite timeout value
  slow(): number;         // Suite slow threshold
}
```

### Reporter Utilities

Base reporter provides utility methods:

```javascript { .api }
/**
 * Utility methods available in Base reporter
 */
class Base {
  /**
   * Get color function for terminal output
   * @param name - Color name
   * @returns {Function} Color function
   */
  color(name);
  
  /**
   * Generate cursor movement for terminal
   * @returns {Object} Cursor utilities
   */
  cursor;
  
  /**
   * Check if output supports color
   * @returns {boolean} Whether colors are supported
   */
  useColors;
  
  /**
   * Get window size for formatting
   * @returns {Object} Window dimensions
   */
  window;
  
  /**
   * Get symbols for different output types
   * @returns {Object} Symbol definitions
   */
  symbols;
}

/**
 * Available color names
 */
const colors = [
  'pass',     // Green
  'fail',     // Red  
  'bright pass',  // Bright green
  'bright fail',  // Bright red
  'bright yellow', // Bright yellow
  'pending',  // Cyan
  'suite',    // Blue
  'error title',   // Red background
  'error message', // Red text
  'error stack',   // Gray
  'checkmark',     // Green
  'fast',     // Gray
  'medium',   // Yellow
  'slow',     // Red
  'green',    // Green
  'light',    // Gray
  'diff gutter',   // Gray
  'diff added',    // Green
  'diff removed'   // Red
];
```

### Reporter Configuration Options

Different reporters accept various configuration options:

```javascript { .api }
/**
 * Common reporter options
 */
interface ReporterOptions {
  output?: string;         // Output file path
  reporterOptions?: any;   // Reporter-specific options
}

/**
 * XUnit reporter specific options
 */
interface XUnitOptions {
  output?: string;         // XML output file
  suiteName?: string;      // Test suite name in XML
}

/**
 * JSON reporter specific options  
*/
interface JSONOptions {
  output?: string;         // JSON output file
}

/**
 * HTML reporter specific options
 */
interface HTMLOptions {
  inline?: boolean;        // Inline CSS/JS
  timeout?: number;        // Test timeout
}
```

**Configuration Examples:**

```javascript
// XUnit with file output
mocha.reporter('xunit', {
  reporterOptions: {
    output: './test-results.xml',
    suiteName: 'My Test Suite'
  }
});

// JSON with custom formatting
mocha.reporter('json', {
  reporterOptions: {
    output: './results.json'
  }
});

// Multiple reporters (using third-party libraries)
const MultiReporter = require('mocha-multi-reporters');
mocha.reporter(MultiReporter, {
  reporterOptions: {
    configFile: './reporter-config.json'
  }
});
```