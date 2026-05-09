# Mocha

Mocha is a feature-rich JavaScript testing framework that runs on Node.js and in browsers. It provides flexible test organization with BDD/TDD interfaces, extensive reporting options, asynchronous testing support, and parallel execution capabilities for improved performance.

## Package Information

- **Package Name**: mocha
- **Package Type**: npm  
- **Language**: JavaScript
- **Installation**: `npm install mocha`

## Core Imports

```javascript
const { Mocha } = require('mocha');
const mocha = require('mocha');
```

For ES modules:

```javascript
import Mocha from 'mocha';
import { describe, it, before, after, beforeEach, afterEach } from 'mocha';
```

Browser (via script tag):

```html
<script src="node_modules/mocha/mocha.js"></script>
<script>
  mocha.setup('bdd');
</script>
```

## Basic Usage

```javascript
const { describe, it } = require('mocha');
const assert = require('assert');

describe('Array', function() {
  describe('#indexOf()', function() {
    it('should return -1 when the value is not present', function() {
      assert.equal([1, 2, 3].indexOf(4), -1);
    });
    
    it('should return the correct index when value is present', function() {
      assert.equal([1, 2, 3].indexOf(2), 1);
    });
  });
});
```

## Architecture

Mocha is built around several key components:

- **Interfaces**: Different styles for writing tests (BDD, TDD, QUnit, Exports)
- **Test Organization**: Hierarchical structure with suites and tests
- **Execution Engine**: Runner class that manages test execution and events
- **Reporting System**: Pluggable reporters for different output formats
- **Hook System**: Before/after hooks for setup and teardown at various levels
- **Configuration**: Flexible options system supporting files, CLI args, and programmatic setup

## Capabilities

### Test Organization and Interfaces

Mocha supports multiple interfaces for organizing tests, with BDD being the default. Each interface provides different syntax styles for defining test suites and cases.

```javascript { .api }
// BDD Interface (default)
function describe(title, fn);
function it(title, fn);
function before(fn);
function after(fn); 
function beforeEach(fn);
function afterEach(fn);

// TDD Interface
function suite(title, fn);
function test(title, fn);
function setup(fn);
function teardown(fn);
function suiteSetup(fn);
function suiteTeardown(fn);
```

[Test Organization and Interfaces](./interfaces.md)

### Test Execution and Runner

Core test execution functionality with lifecycle management, event emission, and parallel execution support.

```javascript { .api }
class Mocha {
  constructor(options);
  run(callback);
  addFile(filepath);
  reporter(name, options);
  timeout(ms);
  slow(ms);
}

class Runner extends EventEmitter {
  run(callback);
  abort();
  grep(pattern);
}
```

[Test Execution and Runner](./execution.md)

### Reporters and Output

Comprehensive reporting system with built-in reporters and support for custom reporters.

```javascript { .api }
class Base {
  constructor(runner, options);
  done(failures, callback);
  epilogue();
}
```

[Reporters and Output](./reporters.md)

### Browser Support

Browser-specific functionality and setup for running tests in browser environments.

```javascript { .api }
// Browser global functions
mocha.setup(options);
mocha.run(callback);
mocha.throwError(error);
```

[Browser Support](./browser.md)

### CLI and Configuration

Command-line interface and configuration options for test execution.

```javascript { .api }
interface MochaOptions {
  ui?: string;
  reporter?: string;
  timeout?: number;
  slow?: number;
  grep?: string | RegExp;
  fgrep?: string;
  bail?: boolean;
  parallel?: boolean;
  jobs?: number;
}
```

[CLI and Configuration](./cli-config.md)

## Types

```javascript { .api }
interface MochaOptions {
  ui?: string;
  reporter?: string | Reporter;
  timeout?: number;
  slow?: number;
  grep?: string | RegExp;
  fgrep?: string;
  bail?: boolean;
  parallel?: boolean;
  jobs?: number;
  asyncOnly?: boolean;
  allowUncaught?: boolean;
  checkLeaks?: boolean;
  color?: boolean;
  delay?: boolean;
  diff?: boolean;
  dryRun?: boolean;
  fullTrace?: boolean;
  inlineDiffs?: boolean;
  invert?: boolean;
  retries?: number;
  forbidOnly?: boolean;
  forbidPending?: boolean;
  global?: string[];
  recursive?: boolean;
  sort?: boolean;
  exit?: boolean;
}

interface Suite {
  title: string;
  parent: Suite | null;
  pending: boolean;
  timeout(ms?: number): number | Suite;
  slow(ms?: number): number | Suite;
  bail(bail?: boolean): boolean | Suite;
}

interface Test {
  title: string;
  fn: Function;
  parent: Suite;
  pending: boolean;
  state: 'failed' | 'passed' | 'pending';
  timeout(ms?: number): number | Test;
  slow(ms?: number): number | Test;
}

interface Hook {
  title: string;
  fn: Function;
  parent: Suite;
  type: 'before' | 'after' | 'beforeEach' | 'afterEach';
}

interface Context {
  test?: Test;
  currentTest?: Test;
  timeout(ms?: number): number | Context;
  slow(ms?: number): number | Context;
  skip(): never;
  retries(count?: number): number | Context;
}

type DoneCB = (error?: any) => void;
type AsyncTestFunction = () => Promise<any>;
type TestFunction = (done?: DoneCB) => void | Promise<any>;

interface Reporter {
  new(runner: Runner, options?: any): Reporter;
}
```