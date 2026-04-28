# Vitest

Vitest is a blazing fast unit test framework powered by Vite with native ES modules support, TypeScript integration, and Jest-compatible APIs.

## Package Information

- **Package**: `vitest` (npm)
- **Install**: `npm install -D vitest`
- **Docs**: https://vitest.dev

## Quick Start

```typescript
import { test, describe, expect, vi, beforeEach, afterEach } from 'vitest';

// Basic test
test('adds numbers', () => {
  expect(1 + 2).toBe(3);
});

// Suite with lifecycle
describe('Math', () => {
  beforeEach(() => { /* setup */ });
  afterEach(() => { /* cleanup */ });

  test('multiply', () => expect(2 * 3).toBe(6));
});

// Async test
test('async operation', async () => {
  await expect(Promise.resolve(42)).resolves.toBe(42);
});

// Mock function
test('with mock', () => {
  const fn = vi.fn(() => 'result');
  expect(fn()).toBe('result');
  expect(fn).toHaveBeenCalledTimes(1);
});
```

## Entry Points

| Import | Use Case |
|--------|----------|
| `vitest` | Test APIs (test, expect, vi, etc.) |
| `vitest/config` | Configuration (defineConfig, defineProject) |
| `vitest/node` | Programmatic APIs (createVitest, reporters) |
| `vitest/reporters` | Reporter classes |
| `vitest/browser` | Browser testing utilities |

## Core Imports

```typescript
// ESM
import { test, describe, expect, vi, beforeEach, afterEach } from 'vitest';

// Globals (with globals: true in config)
test('no import needed', () => expect(true).toBe(true));
```

## API Quick Reference

### Test Definition → [Details](./test-definition.md)

```typescript { .api }
function test(name: string, fn: TestFunction, options?: TestOptions): void;
function describe(name: string, fn: () => void, options?: TestOptions): void;

// Modifiers
test.only | test.skip | test.todo | test.skipIf(cond) | test.runIf(cond)
test.concurrent | test.sequential | test.fails | test.each(cases)

// Lifecycle
function beforeEach(fn: () => void | Promise<void>, timeout?: number): void;
function afterEach(fn: () => void | Promise<void>, timeout?: number): void;
function beforeAll(fn: () => void | Promise<void>, timeout?: number): void;
function afterAll(fn: () => void | Promise<void>, timeout?: number): void;
```

### Assertions → [Details](./assertions.md)

```typescript { .api }
function expect<T>(actual: T): Assertion<T>;
function createExpect(test?: TaskPopulated): ExpectStatic;

// Key methods
.toBe() | .toEqual() | .toStrictEqual()
.toBeTruthy() | .toBeFalsy() | .toBeDefined() | .toBeUndefined()
.toBeGreaterThan() | .toBeLessThan() | .toBeCloseTo()
.toMatch() | .toContain() | .toHaveLength() | .toHaveProperty()
.toThrow() | .toMatchObject()
.resolves | .rejects | .not

// Snapshots
.toMatchSnapshot() | .toMatchInlineSnapshot() | .toMatchFileSnapshot()

// Async
expect.poll(fn, options) | expect.soft(value)
```

### Mocking → [Details](./mocking.md)

```typescript { .api }
vi.fn(impl?) // Create mock function
vi.spyOn(obj, method, accessType?) // Spy on method
vi.mock(path, factory?) // Mock module (hoisted)
vi.doMock(path, factory?) // Mock module (not hoisted)
vi.importActual(path) | vi.importMock(path)
vi.mocked(item, deep?) | vi.mockObject(obj, options?)

// Cleanup
vi.clearAllMocks() | vi.resetAllMocks() | vi.restoreAllMocks()

// Globals
vi.stubGlobal(name, value) | vi.stubEnv(name, value)
vi.unstubAllGlobals() | vi.unstubAllEnvs()

// Wait utilities
vi.waitFor(callback, options?) | vi.waitUntil(callback, options?)
```

### Timers → [Details](./timers.md)

```typescript { .api }
vi.useFakeTimers(config?) | vi.useRealTimers()
vi.advanceTimersByTime(ms) | vi.advanceTimersToNextTimer()
vi.runOnlyPendingTimers() | vi.runAllTimers() | vi.runAllTicks()
vi.setSystemTime(time) | vi.getMockedSystemTime() | vi.getRealSystemTime()
vi.getTimerCount() | vi.clearAllTimers()
```

### Configuration → [Details](./configuration.md)

```typescript { .api }
import { defineConfig, defineProject } from 'vitest/config';

export default defineConfig({
  test: {
    globals: boolean;
    environment: 'node' | 'jsdom' | 'happy-dom' | 'edge-runtime';
    include: string[];
    exclude: string[];
    testTimeout: number;
    hookTimeout: number;
    setupFiles: string | string[];
    coverage: CoverageOptions;
    reporters: Array<Reporter>;
  }
});
```

## Common Patterns

### Test Organization

| Pattern | Code | Use When |
|---------|------|----------|
| Single test | `test('name', () => {...})` | Simple, isolated test |
| Test suite | `describe('name', () => {...})` | Grouping related tests |
| Nested suites | `describe('A', () => describe('B', ...))` | Hierarchical organization |
| Shared setup | `beforeEach(() => {...})` | Common test setup |

### Assertion Patterns

| Goal | Matcher | Example |
|------|---------|---------|
| Exact equality | `.toBe()` | `expect(x).toBe(5)` |
| Deep equality | `.toEqual()` | `expect(obj).toEqual({a: 1})` |
| Truthiness | `.toBeTruthy()` | `expect(value).toBeTruthy()` |
| Array contains | `.toContain()` | `expect([1,2]).toContain(1)` |
| Object has prop | `.toHaveProperty()` | `expect(obj).toHaveProperty('key')` |
| Throws error | `.toThrow()` | `expect(() => fn()).toThrow()` |
| Async success | `.resolves` | `expect(promise).resolves.toBe(x)` |
| Async failure | `.rejects` | `expect(promise).rejects.toThrow()` |

### Mocking Strategies

| Scenario | Approach | Code |
|----------|----------|------|
| Track calls | Mock function | `const fn = vi.fn()` |
| Keep original | Spy | `vi.spyOn(obj, 'method')` |
| Replace module | Module mock | `vi.mock('./module')` |
| Partial mock | Import actual + mock | `await vi.importActual()` |
| Global APIs | Stub global | `vi.stubGlobal('fetch', mockFetch)` |

### Time Control

| Need | Method | Use Case |
|------|--------|----------|
| Control time | `vi.useFakeTimers()` | Deterministic testing |
| Jump forward | `vi.advanceTimersByTime(ms)` | Skip delays |
| Next timer | `vi.advanceTimersToNextTimer()` | Step through |
| Run all | `vi.runAllTimers()` | Complete all async |
| Set date | `vi.setSystemTime(date)` | Mock Date.now() |

## Advanced Features

### Benchmarking → [Details](./benchmarking.md)

```typescript
import { bench } from 'vitest';

bench('operation', () => {
  // Code to benchmark
}, { time: 5000, iterations: 100 });
```

### Type Testing → [Details](./type-testing.md)

```typescript
import { expectTypeOf, assertType } from 'vitest';

expectTypeOf<string>().toBeString();
expectTypeOf(fn).returns.toBeNumber();
assertType<number>(value);
```

### Browser Testing → [Details](./browser-testing.md)

```typescript
import { readFile, writeFile } from 'vitest/browser';

test('browser APIs', async () => {
  await writeFile('./test.txt', 'content');
  expect(await readFile('./test.txt')).toBe('content');
});
```

### Reporters → [Details](./reporters.md)

Built-in: `default`, `verbose`, `dot`, `tree`, `json`, `junit`, `tap`, `github-actions`

```typescript
export default defineConfig({
  test: {
    reporters: ['default', 'json'],
    outputFile: './results.json'
  }
});
```

### Node APIs → [Details](./node-apis.md)

```typescript
import { createVitest } from 'vitest/node';

const vitest = await createVitest('test', { watch: false });
await vitest.start();
await vitest.close();
```

## Configuration Defaults

```typescript
configDefaults.include // ['**/*.{test,spec}.?(c|m)[jt]s?(x)']
configDefaults.exclude // ['**/node_modules/**', '**/dist/**', ...]
defaultBrowserPort // 63315
```

## Decision Guides

### When to Use

| Feature | Choose When |
|---------|-------------|
| `test.only` | Debugging specific test |
| `test.skip` | Temporarily disable test |
| `test.concurrent` | Independent tests that can run in parallel |
| `test.sequential` | Tests with shared state |
| `describe.skip` | Disable entire suite |
| `beforeEach` | Setup needed for every test |
| `beforeAll` | Expensive setup once per suite |

### Mocking Decision Tree

```
Need to track calls?
├─ Yes, keep original behavior?
│  └─ Yes: vi.spyOn(obj, 'method')
│  └─ No: vi.fn(newImpl)
└─ No, replace entire module?
   └─ Yes: vi.mock('./module')
   └─ No: Use real implementation
```

### Environment Selection

| Environment | Use Case | APIs Available |
|-------------|----------|----------------|
| `node` | Server code, CLI tools | Node.js APIs |
| `jsdom` | Browser simulation | DOM, window, document |
| `happy-dom` | Faster browser simulation | DOM (lighter) |
| `edge-runtime` | Edge functions | Subset of Web APIs |
| Browser mode | Real browser testing | Full browser APIs |

## Common Gotchas

1. **Module mocks are hoisted** - `vi.mock()` runs before imports
2. **Use `vi.hoisted()` for shared mock values**
3. **`vi.mock()` is hoisted, `vi.doMock()` is not**
4. **Restore mocks** - Use `afterEach(() => vi.restoreAllMocks())`
5. **Async timers** - Use `*Async()` versions with async code
6. **Globals require config** - Set `globals: true` in config

## Additional Exports

```typescript
inject<T>(key: string | symbol): T // Dependency injection
chai, assert, should // Chai assertion library
```
