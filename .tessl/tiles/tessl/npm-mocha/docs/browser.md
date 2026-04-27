# Browser Support

Browser-specific functionality for running Mocha tests in web browsers with DOM integration, process shims, and browser-optimized features.

## Capabilities

### Browser Setup

Initialize Mocha for browser environments with configuration and DOM integration.

```javascript { .api }
/**
 * Browser-specific mocha setup function
 * @param options - Browser configuration options
 * @returns {mocha} Global mocha instance
 */
mocha.setup(options);

/**
 * Browser setup options
 */
interface BrowserSetupOptions {
  ui?: string;           // Interface: 'bdd', 'tdd', 'qunit', 'exports'
  reporter?: string;     // Reporter name (defaults to 'html')
  timeout?: number;      // Global timeout in milliseconds
  slow?: number;         // Slow test threshold
  grep?: string;         // Test filter pattern
  fgrep?: string;        // Fixed string filter
  invert?: boolean;      // Invert grep pattern
  bail?: boolean;        // Bail on first failure
  checkLeaks?: boolean;  // Check for global leaks
  globals?: string[];    // Global variables to ignore
  delay?: boolean;       // Delay test execution
  noHighlighting?: boolean; // Disable syntax highlighting
}
```

**Usage Examples:**

```html
<!DOCTYPE html>
<html>
<head>
  <title>Mocha Tests</title>
  <link rel="stylesheet" href="node_modules/mocha/mocha.css">
</head>
<body>
  <div id="mocha"></div>
  
  <script src="node_modules/mocha/mocha.js"></script>
  <script>
    // Setup Mocha for browser
    mocha.setup({
      ui: 'bdd',
      reporter: 'html',
      timeout: 5000,
      slow: 100
    });
  </script>
  
  <!-- Load test files -->
  <script src="test/browser-tests.js"></script>
  
  <script>
    // Run tests when page loads
    mocha.run();
  </script>
</body>
</html>
```

```javascript
// String-based setup (shorthand for ui)
mocha.setup('bdd');

// Object-based setup with full options
mocha.setup({
  ui: 'tdd',
  reporter: 'html',
  timeout: 10000,
  globals: ['MY_GLOBAL']
});
```

### Browser Test Execution

Execute tests in browser environment with DOM integration and result display.

```javascript { .api }
/**
 * Run tests in browser environment
 * @param callback - Optional completion callback
 * @returns {Runner} Runner instance
 */
mocha.run(callback);

/**
 * Callback function signature
 * @param failures - Number of failed tests
 */
type RunCallback = (failures: number) => void;
```

**Usage Examples:**

```javascript
// Basic execution
mocha.run();

// With completion callback
mocha.run(function(failures) {
  console.log('Tests completed');
  console.log(`Failed tests: ${failures}`);
  
  // Report results to parent window or test runner
  if (window.parent !== window) {
    window.parent.postMessage({
      type: 'test-results',
      failures: failures
    }, '*');
  }
});

// Get runner instance for event handling
const runner = mocha.run();
runner.on('end', function() {
  console.log('All tests finished');
});
```

### Browser Error Handling

Enhanced error handling for browser environments with assertion integration.

```javascript { .api }
/**
 * Throw error directly into Mocha's error handling system
 * Useful for integration with assertion libraries
 * @param error - Error to throw
 */
mocha.throwError(error);
```

**Usage Example:**

```javascript
// Integration with assertion libraries
function customAssert(condition, message) {
  if (!condition) {
    const error = new Error(message);
    error.name = 'AssertionError';
    mocha.throwError(error);
  }
}

// Usage in tests
it('should handle custom assertions', function() {
  customAssert(2 + 2 === 4, 'Math should work');
  customAssert(true === true, 'Truth should be true');
});
```

### Process Shim

Browser-compatible process object for Node.js compatibility.

```javascript { .api }
/**
 * Browser process shim - limited process object for compatibility
 */
interface BrowserProcess {
  /**
   * Add event listener for uncaught exceptions
   * @param event - Event name ('uncaughtException')
   * @param handler - Error handler function
   */
  on(event: 'uncaughtException', handler: (error: Error) => void): void;
  
  /**
   * Remove event listener
   * @param event - Event name
   * @param handler - Handler function to remove
   */
  removeListener(event: string, handler: Function): void;
  
  /**
   * Get listener count for event
   * @param event - Event name
   * @returns {number} Number of listeners
   */
  listenerCount(event: string): number;
  
  /**
   * Get all listeners for event
   * @param event - Event name
   * @returns {Function[]} Array of listener functions
   */
  listeners(event: string): Function[];
  
  /**
   * Standard output stream (browser-stdout shim)
   */
  stdout: any;
}

/**
 * Access browser process shim
 */
const process = Mocha.process;
```

### Global Functions Export

Browser-specific global function exports for ES module compatibility.

```javascript { .api }
/**
 * Global functions available in browser after setup
 * These are automatically attached to window/global scope
 */

// BDD interface functions (when ui: 'bdd')
function describe(title, fn);
function context(title, fn);   // alias for describe
function it(title, fn);
function specify(title, fn);   // alias for it

// Skip functions
function xdescribe(title, fn); // skip suite
function xcontext(title, fn);  // skip suite  
function xit(title, fn);       // skip test
function xspecify(title, fn);  // skip test

// Hook functions
function before(fn);           // before all tests in suite
function beforeEach(fn);       // before each test
function after(fn);            // after all tests in suite
function afterEach(fn);        // after each test

/**
 * ES module exports for import usage
 * Available when using module bundlers
 */
export {
  describe, context, it, specify,
  xdescribe, xcontext, xit, xspecify,
  before, beforeEach, after, afterEach
};
```

### Browser-Specific Features

Features and optimizations specific to browser environments.

```javascript { .api }
/**
 * High-performance timer override for browser
 * Optimized immediate execution scheduling
 */
Mocha.Runner.immediately = function(callback) {
  // Browser-optimized immediate execution
};

/**
 * URL query parameter parsing for browser test configuration
 * Automatically applied when mocha.run() is called
 */
interface URLQueryOptions {
  grep?: string;      // Filter tests by pattern
  fgrep?: string;     // Filter tests by fixed string
  invert?: boolean;   // Invert filter pattern
}

// Example URL: test.html?grep=User&invert=true
// Automatically applies grep: 'User', invert: true
```

### HTML Reporter Integration

Browser-specific HTML reporter with DOM integration and syntax highlighting.

```javascript { .api }
/**
 * HTML reporter automatically integrates with DOM
 * Requires <div id="mocha"></div> element
 */

/**
 * HTML reporter features
 */
interface HTMLReporterFeatures {
  /**
   * Automatic syntax highlighting for code blocks
   * Controlled by noHighlighting option
   */
  syntaxHighlighting: boolean;
  
  /**
   * Interactive test result filtering
   */
  interactiveFiltering: boolean;
  
  /**
   * Collapsible test suites
   */
  collapsibleSuites: boolean;
  
  /**
   * Real-time progress indication
   */
  progressIndicator: boolean;
}

/**
 * HTML reporter DOM structure
 */
interface HTMLReporterDOM {
  container: HTMLElement;      // #mocha container
  stats: HTMLElement;          // Test statistics
  tests: HTMLElement;          // Test results
  progress: HTMLElement;       // Progress indicator
}
```

### Browser Loading Patterns

Different approaches for loading Mocha in browsers.

```javascript { .api }
/**
 * Script tag loading (UMD build)
 */
// <script src="node_modules/mocha/mocha.js"></script>
// Creates global Mocha and mocha objects

/**
 * ES module loading (with bundler)
 */
import { describe, it, before, after } from 'mocha';

/**
 * CommonJS loading (with bundler like Browserify)
 */
const { describe, it } = require('mocha');

/**
 * AMD loading (with RequireJS)
 */
define(['mocha'], function(mocha) {
  mocha.setup('bdd');
  return mocha;
});
```

### Browser Compatibility

Browser support and compatibility information.

```javascript { .api }
/**
 * Supported browsers (as of Mocha 11.7.2)
 */
interface BrowserSupport {
  chrome: '>=60';      // Chrome 60+
  firefox: '>=55';     // Firefox 55+
  safari: '>=10';      // Safari 10+
  edge: '>=79';        // Chromium-based Edge
  ie: false;           // Internet Explorer not supported
}

/**
 * Required browser features
 */
interface RequiredFeatures {
  es6: true;           // ES6/ES2015 support required
  promises: true;      // Native Promise support
  eventEmitter: true;  // EventEmitter pattern support
  json: true;          // JSON parsing/stringifying
  setTimeout: true;    // Timer functions
  console: true;       // Console logging
}
```

### Browser Test Organization

Best practices and patterns for organizing browser tests.

```javascript { .api }
/**
 * Recommended browser test structure
 */

// test/browser/setup.js
mocha.setup({
  ui: 'bdd',
  reporter: 'html',
  timeout: 5000
});

// test/browser/utils.js  
function waitForElement(selector) {
  return new Promise(resolve => {
    const check = () => {
      const el = document.querySelector(selector);
      if (el) resolve(el);
      else setTimeout(check, 10);
    };
    check();
  });
}

// test/browser/dom-tests.js
describe('DOM Tests', function() {
  beforeEach(function() {
    document.body.innerHTML = '<div id="app"></div>';
  });
  
  afterEach(function() {
    document.body.innerHTML = '';
  });
  
  it('should create DOM elements', async function() {
    const app = document.getElementById('app');
    app.innerHTML = '<button>Click me</button>';
    
    const button = await waitForElement('button');
    assert(button.textContent === 'Click me');
  });
});

// test/browser/run.js
mocha.run(function(failures) {
  console.log(`Browser tests completed: ${failures} failures`);
});
```