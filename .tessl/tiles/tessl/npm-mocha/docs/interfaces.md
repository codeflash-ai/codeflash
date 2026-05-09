# Test Organization and Interfaces

Mocha supports multiple interfaces for organizing and writing tests. Each interface provides a different syntax style and mental model for structuring test suites and cases.

## Capabilities

### BDD Interface (Default)

Behavior Driven Development interface using `describe` and `it` functions. This is the default interface and most commonly used.

```javascript { .api }
/**
 * Define a test suite
 * @param title - Suite title/description
 * @param fn - Suite implementation function
 */
function describe(title, fn);

/**
 * Define a test case
 * @param title - Test title/description  
 * @param fn - Test implementation function
 */
function it(title, fn);

/**
 * Define aliases for describe and it
 */
function context(title, fn); // alias for describe
function specify(title, fn); // alias for it

/**
 * Define hooks that run before/after suites and tests
 */
function before(fn); // runs once before all tests in suite
function after(fn);  // runs once after all tests in suite
function beforeEach(fn); // runs before each test
function afterEach(fn);  // runs after each test

/**
 * Exclusive execution - only run marked tests/suites
 */
function describe.only(title, fn);
function it.only(title, fn);
function context.only(title, fn);
function specify.only(title, fn);

/**
 * Skip tests/suites - do not execute
 */
function describe.skip(title, fn);
function it.skip(title, fn);
function context.skip(title, fn);
function specify.skip(title, fn);

/**
 * Skip aliases using x prefix
 */
function xdescribe(title, fn); // alias for describe.skip
function xit(title, fn);       // alias for it.skip
function xcontext(title, fn);  // alias for context.skip
function xspecify(title, fn);  // alias for specify.skip
```

**Usage Example:**

```javascript
const assert = require('assert');

describe('Calculator', function() {
  let calculator;
  
  before(function() {
    // Setup before all tests
    calculator = new Calculator();
  });
  
  beforeEach(function() {
    // Reset state before each test
    calculator.reset();
  });
  
  describe('#add()', function() {
    it('should add two positive numbers', function() {
      const result = calculator.add(2, 3);
      assert.equal(result, 5);
    });
    
    it('should handle negative numbers', function() {
      const result = calculator.add(-1, 1);
      assert.equal(result, 0);
    });
    
    it.skip('should handle decimal numbers', function() {
      // This test is skipped
    });
  });
  
  describe.only('#multiply()', function() {
    // Only this suite will run when using .only
    it('should multiply two numbers', function() {
      const result = calculator.multiply(3, 4);
      assert.equal(result, 12);
    });
  });
});
```

### TDD Interface

Test Driven Development interface using `suite` and `test` functions.

```javascript { .api }
/**
 * Define a test suite (equivalent to describe)
 * @param title - Suite title/description
 * @param fn - Suite implementation function
 */
function suite(title, fn);

/**
 * Define a test case (equivalent to it)
 * @param title - Test title/description
 * @param fn - Test implementation function
 */
function test(title, fn);

/**
 * Define hooks for setup and teardown
 */
function setup(fn);        // equivalent to beforeEach
function teardown(fn);     // equivalent to afterEach
function suiteSetup(fn);   // equivalent to before
function suiteTeardown(fn); // equivalent to after

/**
 * Exclusive execution modifiers
 */
function suite.only(title, fn);
function test.only(title, fn);

/**
 * Skip modifiers
 */
function suite.skip(title, fn);
function test.skip(title, fn);
```

**Usage Example:**

```javascript
const assert = require('assert');

suite('Calculator TDD', function() {
  let calculator;
  
  suiteSetup(function() {
    calculator = new Calculator();
  });
  
  setup(function() {
    calculator.reset();
  });
  
  suite('Addition', function() {
    test('should add positive numbers', function() {
      const result = calculator.add(2, 3);
      assert.equal(result, 5);
    });
    
    test('should add negative numbers', function() {
      const result = calculator.add(-2, -3);
      assert.equal(result, -5);
    });
  });
  
  teardown(function() {
    // Cleanup after each test
  });
});
```

### QUnit Interface

QUnit-style interface providing `suite` and `test` functions with QUnit-compatible hooks.

```javascript { .api }
/**
 * Define a test suite
 * @param title - Suite title/description
 * @param fn - Suite implementation function
 */
function suite(title, fn);

/**
 * Define a test case
 * @param title - Test title/description
 * @param fn - Test implementation function
 */
function test(title, fn);

/**
 * Define hooks
 */
function before(fn);     // runs before all tests in suite
function after(fn);      // runs after all tests in suite
function beforeEach(fn); // runs before each test
function afterEach(fn);  // runs after each test

/**
 * Exclusive and skip modifiers
 */
function suite.only(title, fn);
function test.only(title, fn);
function suite.skip(title, fn);
function test.skip(title, fn);
```

### Exports Interface

Node.js module.exports style interface where test structure is defined using object properties.

```javascript { .api }
/**
 * Exports interface uses object properties to define test structure
 * No global functions - tests are defined as object methods
 */

// Example structure:
module.exports = {
  'Calculator': {
    'before': function() {
      // Setup
    },
    
    '#add()': {
      'should add positive numbers': function() {
        // Test implementation
      },
      
      'should add negative numbers': function() {
        // Test implementation
      }
    },
    
    '#multiply()': {
      'should multiply numbers': function() {
        // Test implementation  
      }
    },
    
    'after': function() {
      // Teardown
    }
  }
};
```

**Usage Example:**

```javascript
const assert = require('assert');
const Calculator = require('./calculator');

module.exports = {
  'Calculator Tests': {
    before: function() {
      this.calculator = new Calculator();
    },
    
    'Addition Tests': {
      beforeEach: function() {
        this.calculator.reset();
      },
      
      'should add two positive numbers': function() {
        const result = this.calculator.add(2, 3);
        assert.equal(result, 5);
      },
      
      'should handle zero': function() {
        const result = this.calculator.add(5, 0);
        assert.equal(result, 5);
      }
    },
    
    'Multiplication Tests': {
      'should multiply positive numbers': function() {
        const result = this.calculator.multiply(3, 4);
        assert.equal(result, 12);
      }
    }
  }
};
```

### Interface Selection

```javascript { .api }
/**
 * Set the interface for a Mocha instance or globally
 * @param name - Interface name: 'bdd', 'tdd', 'qunit', 'exports'
 */
mocha.ui(name);

// Available interfaces
const interfaces = {
  bdd: require('mocha/lib/interfaces/bdd'),
  tdd: require('mocha/lib/interfaces/tdd'), 
  qunit: require('mocha/lib/interfaces/qunit'),
  exports: require('mocha/lib/interfaces/exports')
};
```

### Global Function Aliases

All interfaces provide these global function aliases for compatibility:

```javascript { .api }
// These functions delegate to the current interface
function describe(title, fn);  // maps to current interface's suite function
function it(title, fn);        // maps to current interface's test function
function before(fn);           // maps to current interface's before hook
function after(fn);            // maps to current interface's after hook
function beforeEach(fn);       // maps to current interface's beforeEach hook
function afterEach(fn);        // maps to current interface's afterEach hook

// TDD aliases (always available)
function suite(title, fn);     // alias for describe
function test(title, fn);      // alias for it
function setup(fn);            // alias for beforeEach
function teardown(fn);         // alias for afterEach
function suiteSetup(fn);       // alias for before
function suiteTeardown(fn);    // alias for after

// Skip aliases
function xdescribe(title, fn); // alias for describe.skip
function xit(title, fn);       // alias for it.skip

// Programmatic execution
function run();                // trigger test execution
```

### Interface Configuration

Interfaces can be configured when creating a Mocha instance:

```javascript
const mocha = new Mocha({
  ui: 'tdd',  // Use TDD interface
  // other options...
});

// Or set programmatically
mocha.ui('bdd');
```