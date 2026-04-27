# Browser Testing

Vitest provides native browser testing capabilities for running tests in real browser environments. Import from `vitest/browser`.

## Capabilities

### Browser Commands

File system commands available in browser context.

```typescript { .api }
interface BrowserCommands {
  /**
   * Read file from file system
   * @param path - File path
   * @param options - Read options
   * @returns Promise resolving to file contents
   */
  readFile(
    path: string,
    options?: { encoding?: BufferEncoding }
  ): Promise<string>;

  /**
   * Write file to file system
   * @param path - File path
   * @param content - File content
   * @param options - Write options
   * @returns Promise resolving when write completes
   */
  writeFile(
    path: string,
    content: string,
    options?: { encoding?: BufferEncoding }
  ): Promise<void>;

  /**
   * Remove file from file system
   * @param path - File path
   * @returns Promise resolving when file is removed
   */
  removeFile(path: string): Promise<void>;
}
```

**Usage:**

```typescript
import { test, expect } from 'vitest';
import { readFile, writeFile, removeFile } from 'vitest/browser';

test('file operations', async () => {
  await writeFile('./test.txt', 'Hello World');

  const content = await readFile('./test.txt');
  expect(content).toBe('Hello World');

  await removeFile('./test.txt');
});
```

### Browser Coverage

Coverage collection APIs for browser tests.

```typescript { .api }
/**
 * Start coverage collection in browser worker
 * @returns Promise resolving when coverage starts
 */
function startCoverageInsideWorker(): Promise<void>;

/**
 * Stop coverage collection in browser worker
 * @returns Promise resolving when coverage stops
 */
function stopCoverageInsideWorker(): Promise<void>;

/**
 * Take coverage snapshot in browser worker
 * @returns Promise resolving to coverage data
 */
function takeCoverageInsideWorker(): Promise<Coverage>;
```

**Usage:**

```typescript
import { test, expect } from 'vitest';
import {
  startCoverageInsideWorker,
  takeCoverageInsideWorker,
  stopCoverageInsideWorker
} from 'vitest/browser';

test('with coverage', async () => {
  await startCoverageInsideWorker();

  // Test code

  const coverage = await takeCoverageInsideWorker();
  expect(coverage).toBeDefined();

  await stopCoverageInsideWorker();
});
```

### Browser Utilities

Utility functions for browser testing.

```typescript { .api }
/**
 * Format values for display
 */
function format(...args: any[]): string;

/**
 * Inspect objects
 */
function inspect(value: any, options?: InspectOptions): string;

/**
 * Stringify values
 */
function stringify(value: any, options?: StringifyOptions): string;

/**
 * Process error objects
 */
function processError(error: unknown): Error;

/**
 * Get JavaScript type of value
 */
function getType(value: any): string;

/**
 * Get safe timer functions
 */
function getSafeTimers(): SafeTimers;

/**
 * Set safe timer functions
 */
function setSafeTimers(timers: SafeTimers): void;

/**
 * Get original position from source map
 */
function getOriginalPosition(map: DecodedMap, pos: Position): Position | null;

/**
 * Decoded source map class
 */
class DecodedMap {
  // Source map utilities
}

/**
 * Collect tests in browser
 */
function collectTests(): Promise<void>;

/**
 * Start tests in browser
 */
function startTests(): Promise<void>;
```

## Browser Configuration

Configure browser testing in vitest.config.ts:

```typescript
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    browser: {
      enabled: true,
      name: 'chrome', // or 'firefox', 'safari', 'edge'
      provider: 'playwright', // or 'webdriverio', 'preview'
      headless: true,
      screenshotFailures: true,
      viewport: {
        width: 1280,
        height: 720
      }
    }
  }
});
```

### Browser Providers

Three browser providers available:

1. **playwright** - Uses Playwright for browser automation
2. **webdriverio** - Uses WebdriverIO for browser automation
3. **preview** - Lightweight preview mode

```typescript
// Install provider
npm install -D @vitest/browser-playwright
// or
npm install -D @vitest/browser-webdriverio
```

### Browser Configuration Options

```typescript { .api }
interface BrowserConfigOptions {
  /** Enable browser mode */
  enabled?: boolean;

  /** Browser name */
  name?: 'chrome' | 'firefox' | 'safari' | 'edge';

  /** Browser provider */
  provider?: 'playwright' | 'webdriverio' | 'preview';

  /** Run in headless mode */
  headless?: boolean;

  /** Take screenshots on failure */
  screenshotFailures?: boolean;

  /** Viewport size */
  viewport?: {
    width: number;
    height: number;
  };

  /** Browser instance options */
  providerOptions?: Record<string, any>;

  /** API server configuration */
  api?: {
    port?: number;
    host?: string;
  };

  /** Scripts to load in browser */
  scripts?: BrowserScript[];

  /** Isolate tests */
  isolate?: boolean;

  /** File parallelism */
  fileParallelism?: boolean;
}

interface BrowserScript {
  /** Script content or path */
  content?: string;
  src?: string;

  /** Load timing */
  async?: boolean;
  type?: string;
}
```

## Browser-Specific Matchers

### Screenshot Matching

```typescript { .api }
interface Assertion<T> {
  /**
   * Match screenshot against baseline
   * @param options - Screenshot comparison options
   */
  toMatchScreenshot(options?: ToMatchScreenshotOptions): Promise<void>;
}

interface ToMatchScreenshotOptions {
  /** Threshold for image comparison (0-1) */
  threshold?: number;

  /** Baseline screenshot path */
  name?: string;

  /** Include only specific element */
  element?: HTMLElement;

  /** Viewport size for screenshot */
  viewport?: {
    width: number;
    height: number;
  };
}
```

**Usage:**

```typescript
import { test, expect } from 'vitest';

test('visual regression', async () => {
  const element = document.querySelector('.my-component');

  await expect(element).toMatchScreenshot({
    threshold: 0.1,
    name: 'my-component'
  });
});
```

## Common Patterns

### DOM Testing

```typescript
import { test, expect } from 'vitest';

test('DOM manipulation', () => {
  const button = document.createElement('button');
  button.textContent = 'Click me';
  button.onclick = () => {
    button.textContent = 'Clicked!';
  };

  document.body.appendChild(button);

  button.click();

  expect(button.textContent).toBe('Clicked!');

  button.remove();
});
```

### Browser API Testing

```typescript
import { test, expect } from 'vitest';

test('localStorage', () => {
  localStorage.setItem('key', 'value');

  expect(localStorage.getItem('key')).toBe('value');

  localStorage.removeItem('key');
  expect(localStorage.getItem('key')).toBeNull();
});

test('fetch API', async () => {
  const response = await fetch('/api/data');
  const data = await response.json();

  expect(data).toBeDefined();
});
```

### Component Testing

```typescript
import { test, expect } from 'vitest';
import { render } from '@testing-library/react';
import MyComponent from './MyComponent';

test('renders component', () => {
  const { container } = render(<MyComponent />);

  expect(container.querySelector('.my-component')).toBeTruthy();
});
```

### Multi-Browser Testing

Configure multiple browser projects using the `projects` option:

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    projects: [
      {
        test: {
          name: 'chrome',
          browser: {
            enabled: true,
            name: 'chrome'
          }
        }
      },
      {
        test: {
          name: 'firefox',
          browser: {
            enabled: true,
            name: 'firefox'
          }
        }
      },
      {
        test: {
          name: 'safari',
          browser: {
            enabled: true,
            name: 'safari'
          }
        }
      }
    ]
  }
});
```

## Running Browser Tests

```bash
# Run browser tests
vitest --browser

# Run in specific browser
vitest --browser.name=chrome

# Run with UI
vitest --browser --ui
```

## Type Definitions

```typescript { .api }
interface Coverage {
  functions: FunctionCoverage[];
  statements: StatementCoverage[];
  branches: BranchCoverage[];
}

interface FunctionCoverage {
  name: string;
  count: number;
  line: number;
  column: number;
}

interface SafeTimers {
  setTimeout: typeof setTimeout;
  setInterval: typeof setInterval;
  clearTimeout: typeof clearTimeout;
  clearInterval: typeof clearInterval;
  setImmediate: typeof setImmediate;
  clearImmediate: typeof clearImmediate;
}

interface Position {
  line: number;
  column: number;
}

interface InspectOptions {
  depth?: number;
  colors?: boolean;
  compact?: boolean;
}

interface StringifyOptions {
  indent?: number;
  maxLength?: number;
}
```

## Environment Setup

Browser tests can use different DOM implementations:

1. **Real Browser** (browser mode)
2. **JSDOM** (jsdom environment)
3. **Happy DOM** (happy-dom environment)

```typescript
// Real browser
export default defineConfig({
  test: {
    browser: {
      enabled: true,
      name: 'chrome'
    }
  }
});

// JSDOM (simulated)
export default defineConfig({
  test: {
    environment: 'jsdom'
  }
});

// Happy DOM (simulated, faster)
export default defineConfig({
  test: {
    environment: 'happy-dom'
  }
});
```
